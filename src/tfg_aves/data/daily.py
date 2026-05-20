"""Resample a una fila por (bird_id, date_utc) con huecos explícitos."""
from __future__ import annotations

import pandas as pd


def coverage_by_hour(df: pd.DataFrame, tolerance_min: float) -> pd.DataFrame:
    """Cobertura % de (ave, día) con fix en [h±tol] para h en 0..23."""
    raise NotImplementedError


def pick_reference_hour(
    df: pd.DataFrame, tolerance_min: float
) -> tuple[int, pd.DataFrame]:
    """Devuelve la hora UTC con cobertura máxima y la tabla completa."""
    raise NotImplementedError


def build_daily(
    df: pd.DataFrame,
    reference_hour_utc: int,
    tolerance_min: float,
) -> pd.DataFrame:
    """Colapsa fixes a una fila por (bird_id, date_utc) con huecos explícitos."""
    raise NotImplementedError


def filter_birds_by_validity(
    df_daily: pd.DataFrame, min_valid_days: int
) -> pd.DataFrame:
    """Descarta individuos con < min_valid_days días válidos."""
    raise NotImplementedError
