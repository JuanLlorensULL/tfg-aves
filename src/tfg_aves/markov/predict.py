"""Predicción a partir de las matrices suavizadas."""
from __future__ import annotations

import numpy as np

from .discretize import _parse_cell_id, cell_centroid, haversine_km


def predict_distribution(
    P: np.ndarray,
    cell_from: str,
    month_int: int,
    cells: list[str],
    marginal: np.ndarray | None = None,
) -> np.ndarray:
    """Distribución sobre celdas para un origen y mes dados."""
    m = month_int - 1  # convención 1-12 → 0-11
    try:
        i = cells.index(cell_from)
    except ValueError:
        if marginal is None:
            raise KeyError(
                f"cell_from={cell_from!r} no está en cells y no se pasó marginal"
            ) from None
        return marginal[m]
    return P[m, i, :]


def topk_from_distribution(
    distribution: np.ndarray, k: int, cells: list[str]
) -> list[str]:
    """Las k celdas con mayor probabilidad, ordenadas descendentes."""
    idx = np.argsort(distribution)[::-1][:k]
    return [cells[i] for i in idx]


def prediction_distance_km(
    cell_pred: str, lat_real: float, lon_real: float, cell_deg: float
) -> float:
    """Haversine entre centroide(``cell_pred``) y ``(lat_real, lon_real)``."""
    i, j = _parse_cell_id(cell_pred)
    lat_c, lon_c = cell_centroid((i, j), cell_deg)
    return float(haversine_km(lat_c, lon_c, lat_real, lon_real))
