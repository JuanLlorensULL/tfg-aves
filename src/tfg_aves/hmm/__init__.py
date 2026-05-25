"""O3 — Detección de comportamiento con HMM (estacionario vs migración)."""
from __future__ import annotations

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV, ROOT
from .build import BuildO3Result, build_o3
from .causal import (
    HMM_EMISSION_COLS_A,
    HMM_EMISSION_COLS_B,
    build_hmm_sequences,
    compute_causal_kinematics,
    decode_causal_states,
    fit_causal_hmm,
    forward_filtered_posteriors,
)
from .evaluate import (
    ab_agreement_causal,
    log_likelihood_per_obs,
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
)

__all__ = [
    "DAILY_PARQUET",
    "HMM_EMISSION_COLS_A",
    "HMM_EMISSION_COLS_B",
    "O3_OUT_DIR",
    "RAW_CSV",
    "ROOT",
    "BuildO3Result",
    "ab_agreement_causal",
    "bearing_rad",
    "build_hmm_sequences",
    "build_o3",
    "build_sequences",
    "compute_causal_kinematics",
    "compute_observation_features",
    "daylight_hours",
    "decode_causal_states",
    "fit_causal_hmm",
    "fit_hmm_with_restarts",
    "forward_filtered_posteriors",
    "load_vegetation_from_raw",
    "log_likelihood_per_obs",
]
