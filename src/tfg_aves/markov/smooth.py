"""Suavizado Laplace y marginales."""
from __future__ import annotations

import numpy as np


def laplace_smooth(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Aplica add-α por fila a cada matriz mensual. ``alpha`` debe ser > 0."""
    raise NotImplementedError


def marginal_distribution(counts: np.ndarray) -> np.ndarray:
    """Marginal del destino por mes ``π_m[j] = sum_i counts[m,i,j] / sum counts[m]``.

    Si un mes tiene 0 counts totales, devuelve uniforme ``1/n_cells``.
    """
    raise NotImplementedError
