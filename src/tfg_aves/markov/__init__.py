"""O2 — Cadenas de Markov visibles: tablas de transición por mes y celda."""
from __future__ import annotations

from ._paths import DAILY_PARQUET, O2_OUT_DIR, ROOT
from .build import BuildO2Result, build_o2
from .discretize import (
    assign_cell,
    cell_centroid,
    discretize_dataframe,
    haversine_km,
)
from .evaluate import (
    aggregate_metrics,
    lobo_predictions,
    persistence_predictions,
)
from .predict import (
    predict_distribution,
    prediction_distance_km,
    topk_from_distribution,
)
from .smooth import laplace_smooth, marginal_distribution
from .transition import build_counts, build_transitions

__all__ = [
    "DAILY_PARQUET",
    "O2_OUT_DIR",
    "ROOT",
    "BuildO2Result",
    "aggregate_metrics",
    "assign_cell",
    "build_counts",
    "build_o2",
    "build_transitions",
    "cell_centroid",
    "discretize_dataframe",
    "haversine_km",
    "laplace_smooth",
    "lobo_predictions",
    "marginal_distribution",
    "persistence_predictions",
    "predict_distribution",
    "prediction_distance_km",
    "topk_from_distribution",
]
