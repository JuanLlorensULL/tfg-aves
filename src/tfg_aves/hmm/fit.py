"""Entrenamiento de HMM con k-means init + EM y múltiples restarts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def stratified_holdout_split(
    df_features: pd.DataFrame,
    holdout_frac: float = 0.20,
    random_state: int = 0,
) -> tuple[list[str], list[str]]:
    """Reparte aves en train/holdout estratificando por nº de días válidos."""
    raise NotImplementedError


def build_sequences(
    df_features: pd.DataFrame,
    bird_ids: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Concatena tramos consecutivos válidos por ave en (X, lengths) para hmmlearn."""
    raise NotImplementedError


def fit_hmm_with_restarts(
    X_train: np.ndarray,
    lengths_train: list[int],
    n_components: int = 2,
    n_restarts: int = 10,
    random_state: int = 0,
) -> tuple[GaussianHMM, StandardScaler, float, list[float]]:
    """Ajusta StandardScaler en train, ejecuta k-means + EM n_restarts veces,
    devuelve (mejor modelo, scaler, mejor LL, lista de todas las LL)."""
    raise NotImplementedError


def relabel_states(
    hmm: GaussianHMM, scaler: StandardScaler, feature_cols: list[str]
) -> dict[int, str]:
    """Re-etiqueta: el estado con menor μ[log_displacement_km] = 'estacionario'."""
    raise NotImplementedError
