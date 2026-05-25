"""Orquestador único de O4."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
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
    attach_o3_state_and_split,
    build_feature_matrix,
)
from .quantile import INDIVIDUAL_BIRD_ID
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
    individual_bird_id: str = ""
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_path: Path = Path()
    metrics_path: Path = Path()


_FAMILIES = ("rf", "xgb", "lgbm")
_FAMILIES_INDIVIDUAL = ("rf", "xgb")
_MODES = ("individual", "poblacional")


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
    *,
    individual_bird_id: str | None = None,
) -> BuildO4Result:
    """Pipeline de O4 — línea base discretizada (L1).

    Dos modos:
      - ``poblacional``: un modelo sobre las 82 aves (sin bird_id). Canónico.
        Familias RF/XGB/LGBM (LGBM se descarta en el análisis por divergencia).
      - ``individual``: un modelo entrenado y evaluado SOLO sobre
        ``individual_bird_id`` (por defecto 91916A). Familias RF/XGB.

    Se añade el corte ``poblacional@<ave>`` (el modelo poblacional restringido a
    las filas de la ave) para comparar manzanas con manzanas con el individual,
    y las baselines (persistencia, Markov) se recalculan también sobre esas filas.
    El patrón replica build_l3.py (modo individual + corte @ave).
    """
    if individual_bird_id is None:
        individual_bird_id = INDIVIDUAL_BIRD_ID
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    # Una sola matriz poblacional (sin bird_id); el individual sale de filtrarla.
    # El estado HMM causal y el split temporal se leen directamente de O3.
    matrix = build_feature_matrix(features_o3, cells, include_bird_id=False)
    matrix = attach_o3_state_and_split(matrix, features_o3)

    def _by_split(label: str) -> pd.DataFrame:
        return matrix[matrix["split"] == label].reset_index(drop=True)

    train_s, val_s, test_s = _by_split("train"), _by_split("val"), _by_split("test")

    def _filter_bird(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["bird_id"] == individual_bird_id].reset_index(drop=True)

    splits_by_mode = {
        "poblacional": (train_s, val_s, test_s),
        "individual": (_filter_bird(train_s), _filter_bird(val_s), _filter_bird(test_s)),
    }
    families_by_mode = {"poblacional": _FAMILIES, "individual": _FAMILIES_INDIVIDUAL}

    # Ningún modo usa bird_id: el poblacional nunca lo tuvo y el individual es una
    # sola ave (constante). feature_cols es el set causal de 10 features.
    feature_cols = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    categorical_cols: list[str] = []

    metrics_per_model: dict[str, dict] = {}
    predictions_all: list[pd.DataFrame] = []
    model_paths: dict[str, Path] = {}
    pob_bundle: dict[str, tuple] = {}  # family -> (model, le_train) para el corte @ave

    for mode in _MODES:
        train, val, test = splits_by_mode[mode]
        le_train = LabelEncoder().fit(train["cell_id_t_next"].astype(str))
        known_cells = set(le_train.classes_)

        X_train, X_val, X_test = train[feature_cols], val[feature_cols], test[feature_cols]
        y_train = le_train.transform(train["cell_id_t_next"].astype(str))
        y_val = _safe_encode_cells(val["cell_id_t_next"], le_train, known_cells)

        for family in families_by_mode[mode]:
            model = _train_one(
                family, X_train, y_train, X_val, y_val,
                categorical_cols=categorical_cols, seed=seed,
            )
            metrics_test = evaluate_global(
                model, X_test, test, cells=cells, label_encoder_y=le_train,
            )
            metrics_test["split"] = "test"
            metrics_train = evaluate_global(
                model, X_train, train, cells=cells, label_encoder_y=le_train,
            )
            metrics_train["split"] = "train"

            key = f"{mode}_{family}"
            metrics_per_model[f"{key}::test"] = metrics_test
            metrics_per_model[f"{key}::train"] = metrics_train

            model_path = output_dir / f"model_{mode}_{family}.pkl"
            joblib.dump({
                "model": model, "label_encoder_y": le_train,
                "feature_cols": feature_cols, "categorical_cols": categorical_cols,
                "mode": mode, "family": family,
                "individual_bird_id": individual_bird_id if mode == "individual" else None,
            }, model_path)
            model_paths[key] = model_path

            preds = predict_with_meta(model, X_test, test, cells=cells, label_encoder_y=le_train)
            for k in list(preds.attrs):
                preds.attrs.pop(k, None)
            preds["modelo"] = family
            preds["modo"] = mode
            predictions_all.append(preds)

            if mode == "poblacional" and family in _FAMILIES_INDIVIDUAL:
                pob_bundle[family] = (model, le_train)

    # --- Corte poblacional@<ave>: el modelo poblacional sobre las filas de la ave ---
    train_pi, test_pi = _filter_bird(train_s), _filter_bird(test_s)
    for family, (model, le_train) in pob_bundle.items():
        for split_name, sub in (("train", train_pi), ("test", test_pi)):
            if len(sub) == 0:
                continue
            m = evaluate_global(
                model, sub[feature_cols], sub, cells=cells, label_encoder_y=le_train,
            )
            m["split"] = split_name
            metrics_per_model[f"poblacional@{individual_bird_id}_{family}::{split_name}"] = m

    # --- Baselines globales sobre el split temporal de O4 ---
    persistence = compute_persistence_baseline(test_s, cells=cells)
    markov = compute_markov_baseline(train_s, test_s, cells=cells)
    persistence["modelo"] = "persistencia"
    persistence["modo"] = "—"
    markov["modelo"] = "markov"
    markov["modo"] = "—"
    predictions_all.extend([persistence, markov])

    def _baseline_metrics(frame: pd.DataFrame) -> dict:
        return {
            "top1": top_k_accuracy(frame, k=1),
            "top3": top_k_accuracy(frame, k=3),
            "dist_median_km": dist_median_km(frame),
            "log_loss": float("nan"),
            "split": "test",
        }

    baselines_metrics: dict[str, dict] = {
        "persistencia": _baseline_metrics(persistence),
        "markov": _baseline_metrics(markov),
    }
    pers_i = persistence[persistence["bird_id"] == individual_bird_id].reset_index(drop=True)
    markov_i = markov[markov["bird_id"] == individual_bird_id].reset_index(drop=True)
    if len(pers_i) > 0:
        baselines_metrics[f"persistencia@{individual_bird_id}"] = _baseline_metrics(pers_i)
    if len(markov_i) > 0:
        baselines_metrics[f"markov@{individual_bird_id}"] = _baseline_metrics(markov_i)

    # --- Tabla comparativa ---
    metrics_test_dict = {
        k.replace("::test", ""): v for k, v in metrics_per_model.items() if k.endswith("::test")
    }
    metrics_train_dict = {
        k.replace("::train", ""): v for k, v in metrics_per_model.items() if k.endswith("::train")
    }
    table_test = compare_models(metrics_test_dict, baselines=baselines_metrics)
    table_train = compare_models(metrics_train_dict, baselines={})
    metrics_table = pd.concat([table_test, table_train], ignore_index=True)

    # --- Guardar artefactos ---
    predictions_df = pd.concat(predictions_all, ignore_index=True)
    predictions_df["pred_cell_topk"] = predictions_df["pred_cell_topk"].apply(
        lambda x: list(x) if x is not None else []
    )
    predictions_path = output_dir / "predictions_test.parquet"
    predictions_df.to_parquet(predictions_path)
    metrics_path = output_dir / "metrics.parquet"
    metrics_table.to_parquet(metrics_path)

    return BuildO4Result(
        n_birds=int(features_o3["bird_id"].nunique()),
        n_rows_train=len(train_s),
        n_rows_val=len(val_s),
        n_rows_test=len(test_s),
        individual_bird_id=individual_bird_id,
        model_paths=model_paths,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
