"""Construcción de features observacionales (cinemáticas + contextuales)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def bearing_rad(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Rumbo inicial del trayecto (lat1, lon1) → (lat2, lon2), en radianes [-π, π]."""
    raise NotImplementedError


def daylight_hours(lat: float, day_of_year: int) -> float:
    """Horas de luz al mediodía local, dada latitud y día del año (1-366)."""
    raise NotImplementedError


def load_vegetation_from_raw(
    raw_csv: Path, event_ids: pd.Series
) -> pd.DataFrame:
    """Lee el CSV crudo, extrae veg_low + veg_high indexado por event_id."""
    raise NotImplementedError


def compute_observation_features(
    df_daily: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Para cada (bird_id, date_utc), calcula las 5 features observacionales.

    Devuelve DataFrame con esquema del entregable (sección 6.1 del spec).
    Marca con ``is_observation_valid=False`` los días sin triplete consecutivo.
    """
    raise NotImplementedError
