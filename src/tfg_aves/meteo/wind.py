"""Carga y match espacio-temporal de viento ECMWF a 850 hPa para L1.

Conexión entre los .nc crudos (reanalysis ECMWF, 850 hPa, 0,5°,
12:00 UTC diario) y los fixes Movebank, mediante interpolación bilinear
espacial.

Spec: docs/superpowers/specs/2026-05-23-o4l1-features-design.md §6.1.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def load_wind_dataset(
    years: list[int],
    base_dir: Path,
) -> xr.Dataset:
    """Carga y concatena los archivos .nc de viento de los años pedidos.

    Args:
        years: lista de años a cargar (e.g., [2010, 2011]).
        base_dir: directorio con archivos `wind_{YYYY}.nc`.

    Returns:
        Dataset xarray con dims (valid_time, latitude, longitude) y
        variables `u`, `v` en m/s, concatenado en el eje temporal.
        La dimensión `pressure_level` (siempre tamaño 1) se elimina.

    Raises:
        FileNotFoundError: si algún archivo no existe.
    """
    base_dir = Path(base_dir)
    paths = [base_dir / f"wind_{year}.nc" for year in years]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    datasets = [xr.open_dataset(str(p), decode_times=True) for p in paths]
    ds = xr.concat(datasets, dim="valid_time")
    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds.load()


def interpolate_wind_to_fixes(
    wind_ds: xr.Dataset,
    fixes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Interpola bilinealmente el viento del día a cada fix.

    Para cada fila (bird_id, date_utc, lat, lon) de fixes_df, selecciona
    el snapshot con valid_time = date_utc 12:00 UTC e interpola U, V al
    punto (lat, lon). Calcula wind_speed_850 = sqrt(u² + v²).

    Args:
        wind_ds: Dataset cargado con load_wind_dataset.
        fixes_df: DataFrame con columnas bird_id, date_utc (date), lat,
            lon (floats con posibles NaN).

    Returns:
        DataFrame con columnas bird_id, date_utc, wind_u_850,
        wind_v_850, wind_speed_850. Mismo número de filas que fixes_df.
        NaN propagado donde lat/lon NaN o fuera del bbox del wind_ds.
    """
    raise NotImplementedError
