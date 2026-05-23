"""Orquestador único de O4."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..meteo._paths import WIND_PER_FIX_PARQUET
from ._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
    O4_L1V1_DIR,
    O4_OUT_DIR,
)
from .evaluate import (
    compare_models,
    compute_markov_baseline,
    compute_persistence_baseline,
    dist_median_km,
    evaluate_global,
    predict_with_meta,
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
from .train import (
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)


@dataclass
class BuildO4Result:
    """Resumen serializable de la ejecución de ``build_o4``."""

    n_birds: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_path: Path = Path()
    metrics_path: Path = Path()


_FAMILIES = ("rf", "xgb", "lgbm")
_MODES = ("personalizado", "poblacional")


def _safe_encode_cells(
    series: pd.Series, le: LabelEncoder, known: set,
) -> np.ndarray:
    """Codifica ``series`` con ``le``; cell_ids desconocidos se mapean a la clase 0."""
    fallback = le.classes_[0]
    mapped = [c if c in known else fallback for c in series.astype(str)]
    return le.transform(mapped)


def _train_one(
    family: str,
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    *, categorical_cols: list[str], seed: int,
):
    """Delega el entrenamiento al trainer correspondiente por familia."""
    if family == "rf":
        return train_random_forest(
            X_train, y_train, categorical_cols=categorical_cols, seed=seed,
        )
    if family == "xgb":
        return train_xgboost(
            X_train, y_train, X_val, y_val,
            categorical_cols=categorical_cols, seed=seed,
        )
    if family == "lgbm":
        return train_lightgbm(
            X_train, y_train, X_val, y_val,
            categorical_cols=categorical_cols, seed=seed,
        )
    raise ValueError(f"Familia desconocida: {family}")


def build_o4(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    output_dir: Path | None = None,
    seed: int = 0,
    *,
    with_wind: bool = False,
    wind_path: Path | None = None,
) -> BuildO4Result:
    """Pipeline completa de O4 (§5.4 del spec).

    Pasos (idénticos a O4 base más el merge opcional de viento):
        1. Carga features.parquet y cells.parquet.
        2. (Opcional, si with_wind=True) Carga wind_per_fix.parquet
           y lo pasa a build_feature_matrix vía el parámetro wind_df.
        3. Construye matriz para ambos modos.
        4. Split temporal por ave.
        5. Entrena las combinaciones (RF/XGB × 2 modos; LightGBM sólo
           cuando with_wind=False — L1 mantiene F7 de O4 base).
        6. Computa métricas globales en train y test.
        7. Computa baselines (persistencia + Markov(1)) sobre el mismo split.
        8. Guarda artefactos en output_dir (defecto: O4_OUT_DIR si
           with_wind=False, O4_L1V1_DIR si with_wind=True).

    Args:
        features_path: ruta a features.parquet de O3.
        cells_path: ruta a cells.parquet de O2.
        output_dir: directorio destino. Por defecto se resuelve según
            with_wind para evitar pisar artefactos de L1-v0.
        seed: semilla global.
        with_wind: si True, fusiona wind features y escribe a L1V1_DIR.
        wind_path: ruta al wind_per_fix.parquet. Por defecto
            WIND_PER_FIX_PARQUET. Sólo se lee cuando with_wind=True.
    """
    if output_dir is None:
        output_dir = O4_L1V1_DIR if with_wind else O4_OUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    wind_df: pd.DataFrame | None = None
    if with_wind:
        wind_p = Path(wind_path) if wind_path is not None else WIND_PER_FIX_PARQUET
        if not wind_p.exists():
            raise FileNotFoundError(
                f"with_wind=True pero {wind_p} no existe. Ejecuta build_wind primero.",
            )
        wind_df = pd.read_parquet(wind_p)

    # Cinemática causal: añade step_in_km, bearing_in, cos_turning_in y
    # la máscara is_hmm_obs_valid. Requiere veg_low/veg_high/daylight_hours.
    kin = compute_causal_kinematics(features_o3)

    # --- Matrices de features para ambos modos ---
    matrices = {
        "personalizado": build_feature_matrix(
            kin, cells, include_bird_id=True, wind_df=wind_df,
        ),
        "poblacional": build_feature_matrix(
            kin, cells, include_bird_id=False, wind_df=wind_df,
        ),
    }
    splits = {mode: split_temporal_per_bird(m) for mode, m in matrices.items()}

    # --- HMM causal: fit sobre el train temporal, decode filtrado de todo ---
    # Las filas de ambos modos son idénticas salvo bird_id, así que el cutoff
    # por ave y el decodificado se calculan una sola vez (modo poblacional).
    train_ref = splits["poblacional"][0]
    cutoff_by_bird = (
        train_ref.assign(_d=pd.to_datetime(train_ref["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )
    hmm_model, hmm_labels = fit_causal_hmm(kin, cutoff_by_bird, n_restarts=10, seed=seed)
    states = decode_causal_states(hmm_model, hmm_labels, kin)

    def _attach_states(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(states, on=["bird_id", "date_utc"], how="left", validate="m:1")
        if merged[FEATURES_HMM].isna().any().any():
            raise AssertionError("Filas candidatas sin estado HMM causal tras el merge.")
        return merged

    metrics_per_model: dict[str, dict] = {}
    predictions_all: list[pd.DataFrame] = []
    model_paths: dict[str, Path] = {}

    for mode in _MODES:
        train, val, test = (_attach_states(d) for d in splits[mode])
        base = [*FEATURES_KINEMATIC, *FEATURES_HMM]
        feature_cols = ["bird_id", *base] if mode == "personalizado" else list(base)
        categorical_cols = ["bird_id"] if "bird_id" in feature_cols else []

        # LabelEncoder ajustado SÓLO sobre el conjunto de entrenamiento. Esto
        # garantiza que np.unique(y_train) == [0, 1, ..., N-1], requisito de
        # XGBoost/LightGBM. Las celdas de val no vistas en train se mapean a
        # la clase 0 (dummy) para y_val; evaluate_global usa
        # meta["cell_id_t_next"] directamente para log_loss, por lo que el
        # dummy no afecta a las métricas.
        le_train = LabelEncoder().fit(train["cell_id_t_next"].astype(str))
        known_cells = set(le_train.classes_)

        X_train = train[feature_cols]
        X_val = val[feature_cols]
        X_test = test[feature_cols]
        y_train = le_train.transform(train["cell_id_t_next"].astype(str))
        y_val = _safe_encode_cells(val["cell_id_t_next"], le_train, known_cells)

        families = _FAMILIES if not with_wind else ("rf", "xgb")
        for family in families:
            model = _train_one(
                family, X_train, y_train, X_val, y_val,
                categorical_cols=categorical_cols, seed=seed,
            )

            # Métricas en test y train: evaluate_global usa meta["cell_id_t_next"]
            # directamente para log_loss, por lo que le_train es suficiente
            metrics_test = evaluate_global(
                model, X_test, test,
                cells=cells, label_encoder_y=le_train,
            )
            metrics_test["split"] = "test"

            metrics_train = evaluate_global(
                model, X_train, train,
                cells=cells, label_encoder_y=le_train,
            )
            metrics_train["split"] = "train"

            key = f"{mode}_{family}"
            metrics_per_model[f"{key}::test"] = metrics_test
            metrics_per_model[f"{key}::train"] = metrics_train

            model_path = output_dir / f"model_{mode}_{family}.pkl"
            joblib.dump({
                "model": model,
                "label_encoder_y": le_train,
                "feature_cols": feature_cols,
                "categorical_cols": categorical_cols,
                "mode": mode,
                "family": family,
            }, model_path)
            model_paths[key] = model_path

            preds = predict_with_meta(
                model, X_test, test, cells=cells, label_encoder_y=le_train,
            )
            # Eliminar attrs no serializables a parquet
            for k in list(preds.attrs):
                preds.attrs.pop(k, None)
            preds["modelo"] = family
            preds["modo"] = mode
            predictions_all.append(preds)

    # --- Baselines sobre el split temporal de O4 ---
    train_p = _attach_states(splits["personalizado"][0])
    test_p = _attach_states(splits["personalizado"][2])
    persistence = compute_persistence_baseline(test_p, cells=cells)
    markov = compute_markov_baseline(train_p, test_p, cells=cells)
    persistence["modelo"] = "persistencia"
    persistence["modo"] = "—"
    markov["modelo"] = "markov"
    markov["modo"] = "—"
    predictions_all.extend([persistence, markov])

    baselines_metrics: dict[str, dict] = {
        "persistencia": {
            "top1": top_k_accuracy(persistence, k=1),
            "top3": top_k_accuracy(persistence, k=3),
            "dist_median_km": dist_median_km(persistence),
            "log_loss": float("nan"),
            "split": "test",
        },
        "markov": {
            "top1": top_k_accuracy(markov, k=1),
            "top3": top_k_accuracy(markov, k=3),
            "dist_median_km": dist_median_km(markov),
            "log_loss": float("nan"),
            "split": "test",
        },
    }

    # --- Tabla comparativa ---
    metrics_test_dict = {
        k.replace("::test", ""): v
        for k, v in metrics_per_model.items() if k.endswith("::test")
    }
    metrics_train_dict = {
        k.replace("::train", ""): v
        for k, v in metrics_per_model.items() if k.endswith("::train")
    }
    table_test = compare_models(metrics_test_dict, baselines=baselines_metrics)
    table_train = compare_models(metrics_train_dict, baselines={})
    metrics_table = pd.concat([table_test, table_train], ignore_index=True)

    # --- Guardar artefactos ---
    predictions_df = pd.concat(predictions_all, ignore_index=True)
    # Convertir listas a objeto Python puro para serialización pyarrow
    predictions_df["pred_cell_topk"] = predictions_df["pred_cell_topk"].apply(
        lambda x: list(x) if x is not None else []
    )
    predictions_path = output_dir / "predictions_test.parquet"
    predictions_df.to_parquet(predictions_path)

    metrics_path = output_dir / "metrics.parquet"
    metrics_table.to_parquet(metrics_path)

    return BuildO4Result(
        n_birds=int(features_o3["bird_id"].nunique()),
        n_rows_train=len(splits["personalizado"][0]),
        n_rows_val=len(splits["personalizado"][1]),
        n_rows_test=len(splits["personalizado"][2]),
        model_paths=model_paths,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
