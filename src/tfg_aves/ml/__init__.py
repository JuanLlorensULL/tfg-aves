"""O4 — Modelos supervisados (RF, XGBoost, LightGBM) para predecir la celda
0,5° del día siguiente.

Pipeline:
    features.parquet (O3, con estado HMM causal + split) + cells.parquet (O2)
        -> build_feature_matrix (gap-aware)
        -> attach_o3_state_and_split (estado y split leídos de O3)
        -> {train_random_forest, train_xgboost, train_lightgbm}
        -> evaluate_global + evaluate_by_state + baselines
        -> data/processed/o4/{models, predictions, metrics}
"""

from tfg_aves.ml.build import BuildO4Result, build_o4
from tfg_aves.ml.build_l3 import BuildO4L3Result, build_o4_l3
from tfg_aves.ml.build_l4 import BuildO4L4Result, build_o4_l4
from tfg_aves.ml.evaluate import (
    compare_models,
    compute_markov_baseline,
    compute_persistence_baseline,
    dist_median_km,
    evaluate_by_state,
    evaluate_global,
    predict_with_meta,
    top_k_accuracy,
)
from tfg_aves.ml.features import (
    FEATURES_O4_CAUSAL,
    add_cyclic_doy,
    assign_cells_to_features,
    attach_o3_state_and_split,
    build_feature_matrix,
)
from tfg_aves.ml.train import (
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)

__all__ = [
    "BuildO4L3Result",
    "BuildO4L4Result",
    "BuildO4Result",
    "FEATURES_O4_CAUSAL",
    "add_cyclic_doy",
    "assign_cells_to_features",
    "attach_o3_state_and_split",
    "build_feature_matrix",
    "build_o4",
    "build_o4_l3",
    "build_o4_l4",
    "compare_models",
    "compute_markov_baseline",
    "compute_persistence_baseline",
    "dist_median_km",
    "evaluate_by_state",
    "evaluate_global",
    "predict_with_meta",
    "top_k_accuracy",
    "train_lightgbm",
    "train_random_forest",
    "train_xgboost",
]
