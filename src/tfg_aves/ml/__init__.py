"""O4 — Modelos supervisados (RF, XGBoost, LightGBM) para predecir la celda
0,5° del día siguiente.

Pipeline:
    features.parquet (O3) + cells.parquet (O2)
        -> build_feature_matrix (gap-aware)
        -> split_temporal_per_bird (80/10/20)
        -> {train_random_forest, train_xgboost, train_lightgbm}
        -> evaluate_global + evaluate_by_state + baselines
        -> data/processed/o4/{models, predictions, metrics}
"""

from tfg_aves.ml.build import BuildO4Result, build_o4
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
    build_feature_matrix,
    compute_causal_kinematics,
    split_temporal_per_bird,
)
from tfg_aves.ml.hmm_causal import decode_causal_states, fit_causal_hmm
from tfg_aves.ml.train import (
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)

__all__ = [
    "BuildO4Result",
    "FEATURES_O4_CAUSAL",
    "add_cyclic_doy",
    "assign_cells_to_features",
    "build_feature_matrix",
    "build_o4",
    "compare_models",
    "compute_causal_kinematics",
    "compute_markov_baseline",
    "compute_persistence_baseline",
    "decode_causal_states",
    "dist_median_km",
    "evaluate_by_state",
    "evaluate_global",
    "fit_causal_hmm",
    "predict_with_meta",
    "split_temporal_per_bird",
    "top_k_accuracy",
    "train_lightgbm",
    "train_random_forest",
    "train_xgboost",
]
