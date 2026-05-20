"""Filtros de outliers sobre fixes GPS de Movebank."""
from __future__ import annotations

import pandas as pd


def drop_movebank_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes marcados como inválidos por Movebank."""
    raise NotImplementedError


def drop_invalid_coords_and_dupes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta coordenadas inválidas y duplicados por (bird_id, timestamp)."""
    raise NotImplementedError


def drop_speed_outliers(
    df: pd.DataFrame, max_speed_kmh: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes que implican velocidad > max_speed_kmh respecto al previo."""
    raise NotImplementedError
