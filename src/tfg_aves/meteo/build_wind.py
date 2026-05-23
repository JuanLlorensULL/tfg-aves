"""Orquestador único del pipeline de viento L1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import WIND_PER_FIX_PARQUET, WIND_RAW_DIR


def build_wind(
    daily_path: Path,
    wind_raw_dir: Path = WIND_RAW_DIR,
    out_path: Path = WIND_PER_FIX_PARQUET,
) -> pd.DataFrame:
    """Construye wind_per_fix.parquet desde daily.parquet y los .nc.

    Pasos:
        1. Lee daily.parquet (filas (bird_id, date_utc, lat, lon)).
        2. Determina años únicos del dataset; carga sólo esos .nc.
        3. Interpola viento al punto (lat, lon) de cada fila.
        4. Persiste el resultado en out_path.
        5. Devuelve el DataFrame para uso opcional en notebooks.

    Args:
        daily_path: ruta a data/processed/daily.parquet.
        wind_raw_dir: directorio con los .nc anuales.
        out_path: destino del parquet resultante.

    Returns:
        DataFrame con (bird_id, date_utc, wind_u_850, wind_v_850,
        wind_speed_850).
    """
    raise NotImplementedError
