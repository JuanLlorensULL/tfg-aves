"""Orquestador del pipeline L4 (exploratorio): L3 con estado del HMM A +
vegetación cruda como feature del regresor (sin daylight, redundante con
lat + día del año). Línea de curiosidad, fuera del trabajo fundamental."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L4_DIR
from .evaluate import (
    compute_persistence_baseline,
    evaluate_by_state,
    evaluate_moves_only,
)
from .features import (
    FEATURES_KINEMATIC,
    attach_o3_state_and_split,
    build_feature_matrix,
)
from .quantile import (
    INDIVIDUAL_BIRD_ID,
    QUANTILES,
    build_regression_predictions,
    derive_displacement_target,
    fit_quantile_axis,
    interval_coverage,
    pinball_loss,
    predict_quantiles,
)

# Estado del HMM A (cinemático) + vegetación cruda. Sin daylight (redundante
# con lat + sin/cos_doy, ya presentes en FEATURES_KINEMATIC).
_FEATURES_HMM_A = ["state_a_causal", "posterior_a_migracion_causal"]
_FEATURES_VEG = ["veg_low", "veg_high"]
_FEATURES = [*FEATURES_KINEMATIC, *_FEATURES_HMM_A, *_FEATURES_VEG]
_STATE_COL = "state_a_causal"
_FAMILIES_DEFAULT = ("xgb", "lgbm", "rf")


def _modes_for(family: str) -> tuple[str, ...]:
    """XGBoost conserva poblacional + individual; RF/LGBM solo poblacional."""
    return ("poblacional", "individual") if family == "xgb" else ("poblacional",)


@dataclass
class BuildO4L4Result:
    """Resumen serializable de build_o4_l4."""

    n_rows_train_pob: int
    n_rows_val_pob: int
    n_rows_test_pob: int
    n_rows_train_ind: int
    n_rows_val_ind: int
    n_rows_test_ind: int
    individual_bird_id: str
    n_crossings: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_path: Path = Path()
    metrics_path: Path = Path()


def _prepare_poblacional_split(
    features_o3: pd.DataFrame, cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) poblacional con estado del HMM A + vegetación cruda.

    El estado A (``state_a_causal``, ``posterior_a_migracion_causal``) y el split
    se leen de O3 vía attach(suffix="a"); ``veg_low``/``veg_high`` se pegan con
    un merge m:1 adicional desde O3.
    """
    matrix = build_feature_matrix(features_o3, cells, include_bird_id=False)
    matrix = attach_o3_state_and_split(matrix, features_o3, suffix="a")
    veg = features_o3[["bird_id", "date_utc", "veg_low", "veg_high"]]
    matrix = matrix.merge(veg, on=["bird_id", "date_utc"], how="left", validate="m:1")
    if matrix[_FEATURES_VEG].isna().any().any():
        raise ValueError("Filas candidatas sin vegetación tras el merge.")
    return tuple(
        matrix[matrix["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
    )


def _y_move(df: pd.DataFrame) -> np.ndarray:
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).to_numpy()


def _metric_rows(
    preds: pd.DataFrame, y_move: np.ndarray, modo_label: str, familia: str,
    *, pinball_lat: float, pinball_lon: float, cov_lat: float, cov_lon: float,
) -> list[dict]:
    """Filas de métrica por estado (global/estacionario/migración) + moves."""
    rows: list[dict] = []
    by = evaluate_by_state(preds, state_col=_STATE_COL)
    for _, r in by.iterrows():
        scope = r["state"]
        if scope == "estacionario":
            sub = preds[preds[_STATE_COL] == 0]
        elif scope == "migración":
            sub = preds[preds[_STATE_COL] == 1]
        else:
            sub = preds
        native = float(np.median(sub["dist_native_km"])) if len(sub) else np.nan
        is_global = scope == "global"
        rows.append({
            "familia": familia, "modo": modo_label, "scope": scope,
            "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": native,
            "pinball_lat": pinball_lat if is_global else np.nan,
            "pinball_lon": pinball_lon if is_global else np.nan,
            "coverage_lat": cov_lat if is_global else np.nan,
            "coverage_lon": cov_lon if is_global else np.nan,
        })
    mask = np.asarray(y_move).astype(bool)
    mo = evaluate_moves_only(preds, mask)
    native_moves = (
        float(np.median(preds[mask]["dist_native_km"])) if mask.any() else np.nan
    )
    rows.append({
        "familia": familia, "modo": modo_label, "scope": "moves",
        "n_obs": mo["n_obs"],
        "top1": mo["top1"], "top3": mo["top3"],
        "dist_centroide_km": mo["dist_median_km"], "dist_nativa_km": native_moves,
        "pinball_lat": np.nan, "pinball_lon": np.nan,
        "coverage_lat": np.nan, "coverage_lon": np.nan,
    })
    return rows


def _baseline_rows(preds: pd.DataFrame, modo_label: str, familia: str) -> list[dict]:
    by = evaluate_by_state(preds, state_col=_STATE_COL)
    rows = []
    for _, r in by.iterrows():
        rows.append({
            "familia": familia, "modo": modo_label, "scope": r["state"],
            "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": np.nan,
            "pinball_lat": np.nan, "pinball_lon": np.nan,
            "coverage_lat": np.nan, "coverage_lon": np.nan,
        })
    return rows


def build_o4_l4(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L4_DIR,
    seed: int = 0,
    *,
    families: tuple[str, ...] = _FAMILIES_DEFAULT,
    individual_bird_id: str | None = None,
) -> BuildO4L4Result:
    """Pipeline L4 (exploratorio): regresor de cuantiles multi-familia con
    estado del HMM A + vegetación cruda. Espejo de build_o4_l3."""
    if individual_bird_id is None:
        individual_bird_id = INDIVIDUAL_BIRD_ID

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    train_pob, val_pob, test_pob = _prepare_poblacional_split(features_o3, cells)

    model_paths: dict[str, Path] = {}
    n_crossings: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict[str, float]] = {}
    metric_rows: list[dict] = []
    preds_frames: list[pd.DataFrame] = []

    splits_by_mode = {
        "poblacional": (train_pob, val_pob, test_pob),
        "individual": tuple(
            d[d["bird_id"] == individual_bird_id].reset_index(drop=True)
            for d in (train_pob, val_pob, test_pob)
        ),
    }

    xgb_pob_full: pd.DataFrame | None = None

    for family in families:
        for mode in _modes_for(family):
            train_m, val_m, test_m = splits_by_mode[mode]
            tgt_tr = derive_displacement_target(train_m)
            tgt_va = derive_displacement_target(val_m)

            axis_lat = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlat"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlat"].to_numpy(),
                family=family, seed=seed,
            )
            axis_lon = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlon"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlon"].to_numpy(),
                family=family, seed=seed,
            )

            qp_test, crossings = predict_quantiles(axis_lat, axis_lon, test_m[_FEATURES])
            key = f"{family}_{mode}"
            n_crossings[key] = crossings
            preds = build_regression_predictions(
                qp_test, test_m, cells, state_col=_STATE_COL,
            )

            tgt_te = derive_displacement_target(test_m)
            pin_lat = float(np.mean([
                pinball_loss(tgt_te["y_dlat"].to_numpy(),
                             qp_test[f"dlat_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            pin_lon = float(np.mean([
                pinball_loss(tgt_te["y_dlon"].to_numpy(),
                             qp_test[f"dlon_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            cov_lat = interval_coverage(
                tgt_te["y_dlat"].to_numpy(),
                qp_test["dlat_p10"].to_numpy(), qp_test["dlat_p90"].to_numpy(),
            )
            cov_lon = interval_coverage(
                tgt_te["y_dlon"].to_numpy(),
                qp_test["dlon_p10"].to_numpy(), qp_test["dlon_p90"].to_numpy(),
            )
            coverage[key] = {"lat": cov_lat, "lon": cov_lon}

            metric_rows.extend(_metric_rows(
                preds, _y_move(test_m), mode, family,
                pinball_lat=pin_lat, pinball_lon=pin_lon,
                cov_lat=cov_lat, cov_lon=cov_lon,
            ))

            for axis_name, axis_model in (("dlat", axis_lat), ("dlon", axis_lon)):
                mp = out_dir / f"model_{family}_{mode}_{axis_name}.pkl"
                joblib.dump({
                    "model": axis_model, "feature_cols": _FEATURES,
                    "familia": family, "mode": mode, "axis": axis_name,
                    "individual_bird_id": (
                        individual_bird_id if mode == "individual" else None
                    ),
                }, mp)
                model_paths[f"{family}_{mode}_{axis_name}"] = mp

            preds_out = preds.copy()
            preds_out["familia"] = family
            preds_out["modo"] = mode
            preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
            preds_frames.append(preds_out)
            if family == "xgb" and mode == "poblacional":
                xgb_pob_full = preds.copy()

    if xgb_pob_full is not None:
        pob_at_ind = xgb_pob_full[
            xgb_pob_full["bird_id"] == individual_bird_id
        ].reset_index(drop=True)
        if len(pob_at_ind) > 0:
            test_ind = splits_by_mode["individual"][2]
            metric_rows.extend(_metric_rows(
                pob_at_ind, _y_move(test_ind), f"poblacional@{individual_bird_id}", "xgb",
                pinball_lat=np.nan, pinball_lon=np.nan, cov_lat=np.nan, cov_lon=np.nan,
            ))

    persistence = compute_persistence_baseline(
        test_pob, cells=cells, state_col=_STATE_COL,
    )
    metric_rows.extend(_baseline_rows(persistence, "persistencia", "—"))
    pers_ind = persistence[persistence["bird_id"] == individual_bird_id]
    if len(pers_ind) > 0:
        metric_rows.extend(_baseline_rows(
            pers_ind.reset_index(drop=True), f"persistencia@{individual_bird_id}", "—"))

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path)

    predictions_df = pd.concat(preds_frames, ignore_index=True)
    predictions_path = out_dir / "predictions_test.parquet"
    predictions_df.to_parquet(predictions_path)

    ind_train = splits_by_mode["individual"][0]
    ind_test = splits_by_mode["individual"][2]
    return BuildO4L4Result(
        n_rows_train_pob=len(train_pob),
        n_rows_val_pob=len(val_pob),
        n_rows_test_pob=len(test_pob),
        n_rows_train_ind=len(ind_train),
        n_rows_val_ind=len(splits_by_mode["individual"][1]),
        n_rows_test_ind=len(ind_test),
        individual_bird_id=individual_bird_id,
        n_crossings=n_crossings,
        coverage=coverage,
        model_paths=model_paths,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
