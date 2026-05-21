"""Predicción a partir de las matrices suavizadas."""
from __future__ import annotations

import numpy as np


def predict_distribution(
    P: np.ndarray,
    cell_from: str,
    month_int: int,
    cells: list[str],
    marginal: np.ndarray | None = None,
) -> np.ndarray:
    """Distribución sobre celdas para un origen y mes dados.

    Si ``cell_from`` no está en ``cells``:
    - usa ``marginal[month_int]`` si se ha pasado;
    - lanza ``KeyError`` si ``marginal is None``.
    """
    raise NotImplementedError


def topk_from_distribution(
    distribution: np.ndarray, k: int, cells: list[str]
) -> list[str]:
    """Las k celdas con mayor probabilidad, ordenadas descendentes."""
    raise NotImplementedError


def prediction_distance_km(
    cell_pred: str, lat_real: float, lon_real: float, cell_deg: float
) -> float:
    """Haversine entre centroide(``cell_pred``) y ``(lat_real, lon_real)``."""
    raise NotImplementedError
