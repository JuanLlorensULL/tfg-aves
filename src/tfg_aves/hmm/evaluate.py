"""Evaluación de los HMM: LL en holdout y acuerdo A-B causal."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def log_likelihood_per_obs(
    hmm: GaussianHMM,
    X_raw: np.ndarray,
    lengths: list[int],
) -> float:
    """Devuelve hmm.score(X_raw) / len(X_raw) sin estandarización previa."""
    if len(X_raw) == 0:
        return float("nan")
    return float(hmm.score(X_raw, lengths) / len(X_raw))


def ab_agreement_causal(df: pd.DataFrame) -> dict[str, float]:
    """Acuerdo entre state_a_causal y state_b_causal sobre días HMM-válidos."""
    valid = df[
        df["is_hmm_obs_valid"]
        & df["state_a_causal"].notna()
        & df["state_b_causal"].notna()
    ]
    a = valid["state_a_causal"].astype(int)
    b = valid["state_b_causal"].astype(int)
    n = len(valid)
    return {
        "pct_agreement": float((a == b).mean() * 100.0) if n else 0.0,
        "n": float(n),
    }
