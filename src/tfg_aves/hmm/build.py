"""Orquestación end-to-end de O3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV


@dataclass
class BuildO3Result:
    features_path: Path
    models_path: Path
    metrics_path: Path
    n_birds_train: int
    n_birds_holdout: int
    n_observations: int
    ll_per_obs_a: float
    ll_per_obs_b: float
    pct_agreement_ab: float


def build_o3(
    *,
    holdout_frac: float = 0.20,
    n_restarts: int = 10,
    random_state: int = 0,
    daily_path: Path = DAILY_PARQUET,
    raw_csv: Path = RAW_CSV,
    out_dir: Path = O3_OUT_DIR,
) -> BuildO3Result:
    """Pipeline completa de O3: features → split → fit A y B → evaluar → escribir."""
    raise NotImplementedError
