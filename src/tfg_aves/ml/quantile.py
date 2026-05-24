"""Regresión de cuantiles del desplazamiento (L3 de O4).

Modela el target continuo (Δlat, Δlon) en grados con XGBoost
(`objective='reg:quantileerror'`), tres cuantiles {p10,p50,p90} por eje.
Las predicciones imitan el esquema de las de clasificación para reutilizar
``tfg_aves.ml.evaluate`` sin cambios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from tfg_aves.markov.discretize import _format_cell_id, assign_cell, haversine_km

QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
INDIVIDUAL_BIRD_ID: str = "91916A"   # ave con más histórico (rank 1, 2025 filas)
CELL_DEG: float = 0.5


def derive_displacement_target(matrix: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con y_dlat, y_dlon en grados.

    y_dlat = lat_t_next − lat ; y_dlon = lon_t_next − lon. No tiene fuga:
    ambas definen la etiqueta, no son features de entrada.
    """
    return pd.DataFrame({
        "y_dlat": matrix["lat_t_next"].to_numpy() - matrix["lat"].to_numpy(),
        "y_dlon": matrix["lon_t_next"].to_numpy() - matrix["lon"].to_numpy(),
    })
