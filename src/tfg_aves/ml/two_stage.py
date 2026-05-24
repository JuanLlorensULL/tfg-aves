"""Funciones puras del pipeline L2 (modelo de dos etapas) de O4 causal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from tfg_aves.markov.discretize import haversine_km


def derive_y_move(df: pd.DataFrame) -> pd.Series:
    """Devuelve y_move = (cell_id_t_next != cell_id_t) como Series booleana."""
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).rename("y_move")


def combine_soft(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    n_classes: int,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla soft canónica con renormalización (§1 del spec).

    Para cada fila i:
        p_final[i, cell_t]      = 1 - p_move[i]
        p_final[i, cell≠cell_t] = p_move[i] * p_2b[i, cell] / (1 - p_2b[i, cell_t])

    El denominador se evalúa como max(1 - p_2b[i, cell_t], eps). Se aplica
    una normalización final por fila para garantizar suma 1.0 incluso en
    el caso degenerado en que clf_dest colapsa sobre cell_t.
    """
    n = p_move.shape[0]
    rows = np.arange(n)
    p2b_cellt = p_2b[rows, cell_t_idx]
    denom = np.maximum(1.0 - p2b_cellt, eps)

    out = p_move[:, None] * p_2b / denom[:, None]
    out[rows, cell_t_idx] = 1.0 - p_move  # sobrescribe la masa de movimiento en cell_t

    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return out / row_sums
