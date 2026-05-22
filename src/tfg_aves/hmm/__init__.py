"""O3 — Detección de comportamiento con HMM (estacionario vs migración)."""
from __future__ import annotations

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV, ROOT
from .build import BuildO3Result, build_o3
from .evaluate import (
    ab_agreement,
    biological_coherence_table,
    log_likelihood_per_obs,
    viterbi_per_bird,
)
from .features import (
    bearing_rad,
    compute_observation_features,
    daylight_hours,
    load_vegetation_from_raw,
)
from .fit import (
    build_sequences,
    fit_hmm_with_restarts,
    relabel_states,
    stratified_holdout_split,
)

__all__ = [
    "DAILY_PARQUET",
    "O3_OUT_DIR",
    "RAW_CSV",
    "ROOT",
    "BuildO3Result",
    "ab_agreement",
    "bearing_rad",
    "biological_coherence_table",
    "build_o3",
    "build_sequences",
    "compute_observation_features",
    "daylight_hours",
    "fit_hmm_with_restarts",
    "load_vegetation_from_raw",
    "log_likelihood_per_obs",
    "relabel_states",
    "stratified_holdout_split",
    "viterbi_per_bird",
]
