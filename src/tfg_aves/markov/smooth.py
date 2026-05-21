"""Suavizado Laplace y marginales."""
from __future__ import annotations

import numpy as np


def laplace_smooth(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Aplica add-α por fila a cada matriz mensual. ``alpha`` debe ser > 0."""
    if alpha <= 0:
        raise ValueError(f"alpha debe ser > 0; se recibió {alpha}")
    n_cells = counts.shape[-1]
    row_sums = counts.sum(axis=2, keepdims=True)  # (12, n_cells, 1)
    return (counts + alpha) / (row_sums + alpha * n_cells)


def marginal_distribution(counts: np.ndarray) -> np.ndarray:
    """Marginal del destino por mes ``π_m[j] = sum_i counts[m,i,j] / sum counts[m]``.

    Si un mes tiene 0 counts totales, devuelve uniforme ``1/n_cells``.
    """
    n_months, _, n_cells = counts.shape
    pi = np.zeros((n_months, n_cells), dtype=np.float64)
    totals = counts.sum(axis=(1, 2))  # (12,)
    for m in range(n_months):
        if totals[m] == 0:
            pi[m] = 1.0 / n_cells
        else:
            pi[m] = counts[m].sum(axis=0) / totals[m]
    return pi
