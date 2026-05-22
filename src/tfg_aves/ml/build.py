"""Orquestador único de O4."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tfg_aves.ml._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
    O4_OUT_DIR,
)
from tfg_aves.ml.evaluate import (
    compare_models,
    compute_markov_baseline,
    compute_persistence_baseline,
    dist_median_km,
    evaluate_global,
    predict_with_meta,
    top_k_accuracy,
)
from tfg_aves.ml.features import (
    build_feature_matrix,
    split_temporal_per_bird,
)
from tfg_aves.ml.train import (
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
    output_dir: Path = O4_OUT_DIR,
    seed: int = 0,
) -> BuildO4Result:
    """Pipeline completa de O4 (§5.4 del spec).

    Pasos:
        1. Carga ``features.parquet`` y ``cells.parquet``.
        2. Construye matriz para ambos modos (personalizado / poblacional).
        3. Split temporal por ave (72 % / 8 % / 20 %).
        4. Entrena las 6 combinaciones (3 familias × 2 modos).
        5. Computa métricas globales en train y test.
        6. Computa baselines (persistencia + Markov(1)) sobre el mismo split.
        7. Guarda artefactos en ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    # --- Matrices de features para ambos modos ---
    matrices = {
        "personalizado": build_feature_matrix(features_o3, cells, include_bird_id=True),
        "poblacional": build_feature_matrix(features_o3, cells, include_bird_id=False),
    }
    splits = {mode: split_temporal_per_bird(m) for mode, m in matrices.items()}

    metrics_per_model: dict[str, dict] = {}
    predictions_all: list[pd.DataFrame] = []
    model_paths: dict[str, Path] = {}

    for mode in _MODES:
        train, val, test = splits[mode]
        feature_cols = train.attrs["_features"]
        categorical_cols = ["bird_id"] if "bird_id" in feature_cols else []

        # LabelEncoder ajustado SÓLO sobre el conjunto de entrenamiento. Esto
        # garantiza que np.unique(y_train) == [0, 1, ..., N-1], requisito de
        # XGBoost/LightGBM. Las celdas de val o test no vistas en train se
        # mapean a la clase 0 (dummy) para y_val/y_test; evaluate_global usa
        # meta["cell_id_t_next"] directamente para log_loss, por lo que el
        # dummy no afecta a las métricas.
        le_train = LabelEncoder().fit(train["cell_id_t_next"].astype(str))
        known_cells = set(le_train.classes_)

        X_train = train[feature_cols]
        X_val = val[feature_cols]
        X_test = test[feature_cols]
        y_train = le_train.transform(train["cell_id_t_next"].astype(str))
        y_val = _safe_encode_cells(val["cell_id_t_next"], le_train, known_cells)
        y_test = _safe_encode_cells(test["cell_id_t_next"], le_train, known_cells)

        for family in _FAMILIES:
            model = _train_one(
                family, X_train, y_train, X_val, y_val,
                categorical_cols=categorical_cols, seed=seed,
            )

            # Métricas en test y train: evaluate_global usa meta["cell_id_t_next"]
            # directamente para log_loss, por lo que le_train es suficiente
            metrics_test = evaluate_global(
                model, X_test, y_test, test,
                cells=cells, label_encoder_y=le_train,
            )
            metrics_test["split"] = "test"

            metrics_train = evaluate_global(
                model, X_train, y_train, train,
                cells=cells, label_encoder_y=le_train,
            )
            metrics_train["split"] = "train"

            key = f"{mode}_{family}"
            metrics_per_model[f"{key}::test"] = metrics_test
            metrics_per_model[f"{key}::train"] = metrics_train

            # Guardar modelo + metadatos
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

            # Predicciones del test para predictions_test.parquet
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
    train_p, _val_p, test_p = splits["personalizado"]
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
