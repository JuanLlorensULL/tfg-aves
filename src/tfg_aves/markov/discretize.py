"""Discretización del espacio en celdas regulares lat/lon."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    """Devuelve ``(cell_lat_idx, cell_lon_idx)`` para un punto.

    El grid tiene origen fijo en ``(0, 0)`` y paso ``cell_deg``.
    """
    raise NotImplementedError


def cell_centroid(cell_id: tuple[int, int], cell_deg: float) -> tuple[float, float]:
    """Inverso de :func:`assign_cell`: centro geométrico de una celda."""
    raise NotImplementedError


def discretize_dataframe(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    """Añade columnas ``cell_lat_idx``, ``cell_lon_idx`` y ``cell_id`` al DataFrame.

    Filas con ``is_valid=False`` reciben ``cell_id=None``.
    """
    raise NotImplementedError


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Distancia gran-círculo en km. Vectorizada sobre arrays NumPy."""
    raise NotImplementedError
