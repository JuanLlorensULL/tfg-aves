"""Discretización del espacio en celdas regulares lat/lon."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0


def assign_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    """Devuelve ``(cell_lat_idx, cell_lon_idx)`` para un punto.

    El grid tiene origen fijo en ``(0, 0)`` y paso ``cell_deg``.
    """
    return int(np.floor(lat / cell_deg)), int(np.floor(lon / cell_deg))


def cell_centroid(cell_id: tuple[int, int], cell_deg: float) -> tuple[float, float]:
    """Inverso de :func:`assign_cell`: centro geométrico de una celda."""
    i, j = cell_id
    return (i + 0.5) * cell_deg, (j + 0.5) * cell_deg


def _format_cell_id(idx_lat: int, idx_lon: int) -> str:
    return f"{idx_lat}_{idx_lon}"


def _parse_cell_id(cell_id: str) -> tuple[int, int]:
    i_str, j_str = cell_id.split("_")
    return int(i_str), int(j_str)


def discretize_dataframe(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    """Añade columnas ``cell_lat_idx``, ``cell_lon_idx`` y ``cell_id`` al DataFrame.

    Filas con ``is_valid=False`` reciben ``cell_id=None``.
    """
    out = df.copy()
    valid = out["is_valid"].to_numpy()
    lat = out["lat"].to_numpy()
    lon = out["lon"].to_numpy()

    n = len(out)
    cell_lat_idx = np.full(n, np.nan)
    cell_lon_idx = np.full(n, np.nan)
    # Se construye como dict para que pd.Series preserve None en celdas inválidas.
    cell_ids_dict: dict[int, str | None] = {k: None for k in range(n)}

    for k in range(n):
        if valid[k]:
            i = int(np.floor(lat[k] / cell_deg))
            j = int(np.floor(lon[k] / cell_deg))
            cell_lat_idx[k] = i
            cell_lon_idx[k] = j
            cell_ids_dict[k] = _format_cell_id(i, j)

    out = out.reset_index(drop=True)
    out["cell_lat_idx"] = pd.array(cell_lat_idx, dtype="Int64")
    out["cell_lon_idx"] = pd.array(cell_lon_idx, dtype="Int64")
    # Asignar como Series (no .values) para preservar None en filas inválidas.
    out["cell_id"] = pd.Series(cell_ids_dict, dtype=object)
    return out


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Distancia gran-círculo en km. Vectorizada sobre arrays NumPy."""
    lat1_r = np.deg2rad(lat1)
    lat2_r = np.deg2rad(lat2)
    dlat = np.deg2rad(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return _EARTH_RADIUS_KM * c
