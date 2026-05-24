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


def combine_hard(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    tau: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla hard: cell_t si p_move<tau, si no argmax(p_2b).

    Devuelve (n, n_classes) con masa 1-eps en la celda predicha y
    eps/(n_classes-1) repartida en el resto. El clipping existe sólo por
    compatibilidad con sklearn.metrics.log_loss; el log-loss numérico de
    hard NO es una métrica honesta (cada fallo de argmax suma ~22.86).
    """
    n, n_classes = p_2b.shape
    pred_idx = np.where(p_move < tau, cell_t_idx, p_2b.argmax(axis=1))
    out = np.full((n, n_classes), eps / (n_classes - 1), dtype=np.float64)
    out[np.arange(n), pred_idx] = 1.0 - eps
    return out


def sweep_tau(
    p_move_val: np.ndarray,
    p_2b_val: np.ndarray,
    cell_t_idx_val: np.ndarray,
    y_true_idx_val: np.ndarray,
    n_classes: int,
    taus: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> tuple[float, pd.DataFrame]:
    """Barre tau y devuelve (tau*, tabla) maximizando top-1 sobre val.

    Se usa top-1 (NO log-loss) porque hard devuelve one-hot y su log-loss
    quedaría dominado por el clipping. Empates: gana el tau menor (primero).
    """
    records = []
    for tau in taus:
        out = combine_hard(p_move_val, p_2b_val, cell_t_idx_val, tau)
        top1 = float((out.argmax(axis=1) == y_true_idx_val).mean())
        records.append({"tau": tau, "top1": top1})
    table = pd.DataFrame(records)
    tau_star = float(table.loc[table["top1"].idxmax(), "tau"])
    return tau_star, table
