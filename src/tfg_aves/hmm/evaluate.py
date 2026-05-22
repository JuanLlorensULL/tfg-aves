"""Evaluación de los HMM: LL en holdout, coherencia biológica, acuerdo A-B."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def log_likelihood_per_obs(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_raw: np.ndarray,
    lengths: list[int],
) -> float:
    """Aplica scaler a X_raw y devuelve hmm.score(X) / len(X)."""
    raise NotImplementedError


def viterbi_per_bird(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    label_map: dict[int, str],
    model_suffix: str,
) -> pd.DataFrame:
    """Añade state_<suffix>, posterior_<suffix>_estacionario, posterior_<suffix>_migracion."""
    raise NotImplementedError


def biological_coherence_table(
    df_features: pd.DataFrame, state_col: str
) -> pd.DataFrame:
    """Cross-tabula estado vs mes, vs latitud (cuartil), vs fotoperiodo."""
    raise NotImplementedError


def ab_agreement(df_features: pd.DataFrame) -> dict[str, float | pd.DataFrame]:
    """Matriz confusion 2×2 entre state_a y state_b + sub-DataFrame de desacuerdos."""
    raise NotImplementedError
