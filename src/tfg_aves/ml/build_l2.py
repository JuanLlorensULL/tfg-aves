"""Orquestador único del pipeline L2 (modelo de dos etapas) de O4 causal."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L2V1_DIR
from .evaluate import (
    compute_markov_baseline,
    compute_persistence_baseline,
    dist_median_km,
    evaluate_moves_only,
    top_k_accuracy,
)
from .features import (
    FEATURES_HMM,
    FEATURES_KINEMATIC,
    build_feature_matrix,
    compute_causal_kinematics,
    split_temporal_per_bird,
)
from .hmm_causal import decode_causal_states, fit_causal_hmm
from .train import train_random_forest, train_xgboost
from .two_stage import (
    calibrate_prefit,
    combine_hard,
    combine_soft,
    derive_y_move,
    expand_proba_to_full,
    predictions_from_proba,
    sweep_tau,
    train_move_rf,
    train_move_xgb,
)

_FAMILIES = ("rf", "xgb")
_MODES = ("personalizado", "poblacional")


@dataclass
class BuildO4L2Result:
    """Resumen serializable de la ejecución de build_o4_l2."""

    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    n_moves_train: int
    tau_star: dict[str, float] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_paths: dict[str, Path] = field(default_factory=dict)
    metrics_path: Path = Path()


def _prepare_causal_splits(
    features_o3: pd.DataFrame, cells: pd.DataFrame, seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Replica el ensamblaje causal de build_o4 (cinemática entrante + HMM
    causal filtrado forward-only) y devuelve {mode: (train, val, test)} con
    state_b_causal y posterior_b_migracion_causal ya adjuntados.

    Se replica (en vez de extraer un helper de build.py) para NO tocar el
    build.py recién estabilizado; es leak-safe porque usa exactamente las
    mismas funciones leak-free.
    """
    kin = compute_causal_kinematics(features_o3)
    matrices = {
        "personalizado": build_feature_matrix(kin, cells, include_bird_id=True),
        "poblacional": build_feature_matrix(kin, cells, include_bird_id=False),
    }
    raw_splits = {m: split_temporal_per_bird(mat) for m, mat in matrices.items()}

    train_ref = raw_splits["poblacional"][0]
    cutoff_by_bird = (
        train_ref.assign(_d=pd.to_datetime(train_ref["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )
    hmm_model, hmm_labels = fit_causal_hmm(
        kin, cutoff_by_bird, n_restarts=10, seed=seed,
    )
    states = decode_causal_states(hmm_model, hmm_labels, kin)

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(states, on=["bird_id", "date_utc"], how="left", validate="m:1")
        if merged[FEATURES_HMM].isna().any().any():
            raise ValueError("Filas candidatas sin estado HMM causal tras el merge.")
        return merged

    return {m: tuple(attach(d) for d in raw_splits[m]) for m in _MODES}


def _feature_cols(mode: str) -> list[str]:
    base = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    return ["bird_id", *base] if mode == "personalizado" else list(base)


def build_o4_l2(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L2V1_DIR,
    seed: int = 0,
) -> BuildO4L2Result:
    """Pipeline L2-v1 completo sobre el feature set CAUSAL (ver spec §6.1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    splits = _prepare_causal_splits(features_o3, cells, seed)

    # y_move se deriva del modo poblacional (filas idénticas entre modos).
    train_pob, val_pob, test_pob = splits["poblacional"]
    feat_pob = _feature_cols("poblacional")
    y_move_train = derive_y_move(train_pob).to_numpy().astype(int)
    y_move_val = derive_y_move(val_pob).to_numpy().astype(int)
    y_move_test = derive_y_move(test_pob).to_numpy().astype(int)
    moves_mask_train = y_move_train.astype(bool)

    # Espacio de clases común para la combinación: unión de todas las celdas
    # vistas en train, val y test para garantizar que clf_dest (entrenado sobre
    # train) quede siempre dentro del espacio de classes_full.
    classes_full = np.array(sorted(
        set(train_pob["cell_id_t_next"].astype(str))
        | set(train_pob["cell_id_t"].astype(str))
        | set(val_pob["cell_id_t_next"].astype(str))
        | set(val_pob["cell_id_t"].astype(str))
        | set(test_pob["cell_id_t_next"].astype(str))
        | set(test_pob["cell_id_t"].astype(str)),
    ))
    full_index = {c: j for j, c in enumerate(classes_full)}
    n_classes = len(classes_full)
    # classes_full incluye todas las celdas de test, así que map nunca es NaN;
    # fillna(0) + int por paridad con cell_t_idx_val y seguridad explícita.
    cell_t_idx_test = (
        test_pob["cell_id_t"].astype(str).map(full_index).fillna(0)
        .to_numpy().astype(int)
    )

    model_paths: dict[str, Path] = {}
    tau_star_by_combo: dict[str, float] = {}
    metric_rows: list[dict] = []
    preds_soft_frames: list[pd.DataFrame] = []
    preds_hard_frames: list[pd.DataFrame] = []
    tau_tables: list[pd.DataFrame] = []

    for family in _FAMILIES:
        # --- Etapa 1: clf_move (poblacional) ---
        X_train_move = train_pob[feat_pob]
        X_val_move = val_pob[feat_pob]
        X_test_move = test_pob[feat_pob]
        if family == "rf":
            base_move = train_move_rf(X_train_move, y_move_train, seed=seed)
        else:
            base_move = train_move_xgb(X_train_move, y_move_train, seed=seed)
        clf_move = calibrate_prefit(base_move, X_val_move, y_move_val)

        # clf_move es siempre poblacional (F3): p_move_test/val se calculan una
        # vez por familia y se comparten entre ambos modos del bucle interior.
        pos_col = list(clf_move.classes_).index(1)
        p_move_test = clf_move.predict_proba(X_test_move)[:, pos_col]
        p_move_val = clf_move.predict_proba(X_val_move)[:, pos_col]

        mp = out_dir / f"model_clf_move_{family}.pkl"
        joblib.dump({"model": clf_move, "feature_cols": feat_pob}, mp)
        model_paths[f"clf_move_{family}"] = mp

        for mode in _MODES:
            train_m, val_m, test_m = splits[mode]
            feat_m = _feature_cols(mode)
            cat_cols = ["bird_id"] if "bird_id" in feat_m else []

            # --- Etapa 2B: clf_dest sobre filas con y_move=1 ---
            train_moves = train_m[moves_mask_train].reset_index(drop=True)
            le_dest = LabelEncoder().fit(train_moves["cell_id_t_next"].astype(str))
            y_dest = le_dest.transform(train_moves["cell_id_t_next"].astype(str))
            X_dest_train = train_moves[feat_m]

            if family == "rf":
                clf_dest = train_random_forest(
                    X_dest_train, y_dest, categorical_cols=cat_cols, seed=seed,
                )
            else:
                val_moves = val_m[
                    derive_y_move(val_m).to_numpy().astype(bool)
                ].reset_index(drop=True)
                known = set(le_dest.classes_)
                fallback = le_dest.classes_[0]
                y_dest_val = le_dest.transform([
                    c if c in known else fallback
                    for c in val_moves["cell_id_t_next"].astype(str)
                ])
                clf_dest = train_xgboost(
                    X_dest_train, y_dest, val_moves[feat_m], y_dest_val,
                    categorical_cols=cat_cols, seed=seed,
                )

            md = out_dir / f"model_clf_dest_{family}_{mode}.pkl"
            joblib.dump({
                "model": clf_dest, "label_encoder_y": le_dest,
                "feature_cols": feat_m, "categorical_cols": cat_cols,
            }, md)
            model_paths[f"clf_dest_{family}_{mode}"] = md

            # --- Combinación sobre test ---
            classes_dest = le_dest.inverse_transform(clf_dest.classes_)
            p2b_test = expand_proba_to_full(
                clf_dest.predict_proba(test_m[feat_m]), classes_dest, classes_full,
            )
            p2b_val = expand_proba_to_full(
                clf_dest.predict_proba(val_m[feat_m]), classes_dest, classes_full,
            )
            cell_t_idx_val = (
                val_m["cell_id_t"].astype(str).map(full_index).fillna(0)
                .to_numpy().astype(int)
            )
            y_true_idx_val = (
                val_m["cell_id_t_next"].astype(str).map(full_index).fillna(0)
                .to_numpy().astype(int)
            )
            tau_star, tau_table = sweep_tau(
                p_move_val, p2b_val, cell_t_idx_val, y_true_idx_val, n_classes,
            )
            tau_table["modelo"] = family
            tau_table["modo"] = mode
            tau_tables.append(tau_table)
            tau_star_by_combo[f"{family}_{mode}"] = tau_star

            proba_soft = combine_soft(p_move_test, p2b_test, cell_t_idx_test, n_classes)
            proba_hard = combine_hard(p_move_test, p2b_test, cell_t_idx_test, tau_star)

            for rule, proba in (("soft", proba_soft), ("hard", proba_hard)):
                preds = predictions_from_proba(proba, classes_full, test_m, cells=cells)
                if rule == "soft":
                    ll = float(log_loss(
                        test_m["cell_id_t_next"].astype(str).to_numpy(),
                        proba, labels=list(classes_full),
                    ))
                else:
                    ll = np.nan
                row = {
                    "modelo": family, "modo": mode, "regla": rule,
                    "top1": top_k_accuracy(preds, k=1),
                    "top3": top_k_accuracy(preds, k=3),
                    "log_loss": ll,
                    "dist_median_km": dist_median_km(preds),
                }
                if rule == "soft":
                    # evaluate_moves_only lee preds.attrs["_proba"] para el
                    # log_loss del subset; debe ir ANTES de limpiar attrs abajo.
                    mo = evaluate_moves_only(preds, y_move_test)
                    row["top1_moves"] = mo["top1"]
                    row["dist_median_km_moves"] = mo["dist_median_km"]
                metric_rows.append(row)

                preds_out = preds.copy()
                for k in list(preds_out.attrs):
                    preds_out.attrs.pop(k, None)
                preds_out["modelo"] = family
                preds_out["modo"] = mode
                preds_out["regla"] = rule
                preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
                (preds_soft_frames if rule == "soft" else preds_hard_frames).append(preds_out)

    # --- Baselines sobre el test poblacional ---
    persistence = compute_persistence_baseline(test_pob, cells=cells)
    markov = compute_markov_baseline(train_pob, test_pob, cells=cells)
    for name, bl in (("persistencia", persistence), ("markov", markov)):
        metric_rows.append({
            "modelo": name, "modo": "—", "regla": "—",
            "top1": top_k_accuracy(bl, k=1),
            "top3": top_k_accuracy(bl, k=3),
            "log_loss": np.nan,
            "dist_median_km": dist_median_km(bl),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path)

    soft_path = out_dir / "predictions_test_soft.parquet"
    hard_path = out_dir / "predictions_test_hard.parquet"
    pd.concat(preds_soft_frames, ignore_index=True).to_parquet(soft_path)
    pd.concat(preds_hard_frames, ignore_index=True).to_parquet(hard_path)
    pd.concat(tau_tables, ignore_index=True).to_parquet(out_dir / "tau_sweep.parquet")

    return BuildO4L2Result(
        n_rows_train=len(train_pob),
        n_rows_val=len(val_pob),
        n_rows_test=len(test_pob),
        n_moves_train=int(moves_mask_train.sum()),
        tau_star=tau_star_by_combo,
        model_paths=model_paths,
        predictions_paths={"soft": soft_path, "hard": hard_path},
        metrics_path=metrics_path,
    )
