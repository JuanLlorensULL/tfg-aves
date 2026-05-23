"""Carga y match espacio-temporal de viento ECMWF a 850 hPa para L1.

Conexión entre los .nc crudos (reanalysis ECMWF, 850 hPa, 0,5°,
12:00 UTC diario) y los fixes Movebank, mediante interpolación bilinear
espacial.

Spec: docs/superpowers/specs/2026-05-23-o4l1-features-design.md §6.1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
    if not {"bird_id", "date_utc", "lat", "lon"}.issubset(fixes_df.columns):
        raise ValueError(
            "fixes_df necesita columnas bird_id, date_utc, lat, lon.",
        )

    out = pd.DataFrame({
        "bird_id": fixes_df["bird_id"].to_numpy(),
        "date_utc": fixes_df["date_utc"].to_numpy(),
        "wind_u_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
        "wind_v_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
        "wind_speed_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
    })

    # Rango espacial del .nc para detectar out-of-bbox.
    lat_min = float(wind_ds["latitude"].min())
    lat_max = float(wind_ds["latitude"].max())
    lon_min = float(wind_ds["longitude"].min())
    lon_max = float(wind_ds["longitude"].max())

    # Marca filas válidas (lat, lon no-NaN y dentro del bbox).
    lat_arr = pd.to_numeric(fixes_df["lat"], errors="coerce").to_numpy()
    lon_arr = pd.to_numeric(fixes_df["lon"], errors="coerce").to_numpy()
    valid_mask = (
        ~np.isnan(lat_arr)
        & ~np.isnan(lon_arr)
        & (lat_arr >= lat_min) & (lat_arr <= lat_max)
        & (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )

    if not valid_mask.any():
        return out

    valid_dates = pd.to_datetime(
        fixes_df.loc[valid_mask, "date_utc"].to_numpy(),
    ) + pd.Timedelta(hours=12)
    valid_lats = lat_arr[valid_mask]
    valid_lons = lon_arr[valid_mask]

    # Interpolación vectorizada vía xarray (linear en lat, lon, nearest en tiempo).
    interp = wind_ds.interp(
        valid_time=xr.DataArray(valid_dates, dims="points"),
        latitude=xr.DataArray(valid_lats, dims="points"),
        longitude=xr.DataArray(valid_lons, dims="points"),
        method="linear",
        kwargs={"fill_value": np.nan},
    )

    u_vals = interp["u"].values.astype(np.float64)
    v_vals = interp["v"].values.astype(np.float64)
    speed_vals = np.sqrt(u_vals**2 + v_vals**2)

    out.loc[valid_mask, "wind_u_850"] = u_vals
    out.loc[valid_mask, "wind_v_850"] = v_vals
    out.loc[valid_mask, "wind_speed_850"] = speed_vals

    return out
