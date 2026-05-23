"""Tests para src/tfg_aves/meteo/wind.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tfg_aves.meteo.wind import (
    load_wind_dataset,
)


def _make_synthetic_wind_nc(
    path: Path,
    year: int,
    *,
    lat_vals: np.ndarray | None = None,
    lon_vals: np.ndarray | None = None,
    u_value: float = 1.0,
    v_value: float = 2.0,
) -> None:
    """Crea un .nc sintético con la estructura de v2 (u, v, 850 hPa, daily)."""
    if lat_vals is None:
        lat_vals = np.array([60.0, 59.5, 59.0], dtype=np.float64)
    if lon_vals is None:
        lon_vals = np.array([10.0, 10.5, 11.0], dtype=np.float64)
    times = pd.date_range(
        f"{year}-01-01 12:00", f"{year}-12-31 12:00", freq="1D",
    )
    n_t, n_lat, n_lon = len(times), len(lat_vals), len(lon_vals)
    u = np.full((n_t, 1, n_lat, n_lon), u_value, dtype=np.float64)
    v = np.full((n_t, 1, n_lat, n_lon), v_value, dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "pressure_level": np.array([850.0]),
            "latitude": lat_vals,
            "longitude": lon_vals,
        },
    )
    ds.to_netcdf(path)


def test_load_wind_dataset_combines_years(tmp_path):
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    _make_synthetic_wind_nc(tmp_path / "wind_2011.nc", 2011)

    ds = load_wind_dataset([2010, 2011], base_dir=tmp_path)

    assert "u" in ds.data_vars
    assert "v" in ds.data_vars
    assert "pressure_level" not in ds.dims, "Se esperaba squeeze de pressure_level"
    assert int(ds.sizes["valid_time"]) == 365 + 365
    assert pd.Timestamp(ds["valid_time"].values[0]) == pd.Timestamp("2010-01-01 12:00")
    assert pd.Timestamp(ds["valid_time"].values[-1]) == pd.Timestamp("2011-12-31 12:00")


def test_load_wind_dataset_missing_file_raises(tmp_path):
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)

    with pytest.raises(FileNotFoundError):
        load_wind_dataset([2010, 2099], base_dir=tmp_path)
