"""Tests para src/tfg_aves/meteo/wind.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tfg_aves.meteo.wind import (
    interpolate_wind_to_fixes,
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


def test_interpolate_wind_known_point(tmp_path):
    """En el centro de cuatro nodos del grid, bilinear devuelve la media."""
    # Grid 2x2 con valores distintos para que la media sea no-trivial.
    lat_vals = np.array([60.0, 59.5], dtype=np.float64)
    lon_vals = np.array([10.0, 10.5], dtype=np.float64)

    # Construimos un .nc con u y v variando por nodo.
    times = pd.date_range("2010-07-01 12:00", "2010-07-31 12:00", freq="1D")
    n_t = len(times)
    u_grid = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (lat, lon)
    v_grid = np.array([[[10.0, 20.0], [30.0, 40.0]]])
    u = np.broadcast_to(u_grid, (n_t, 1, 2, 2)).copy()
    v = np.broadcast_to(v_grid, (n_t, 1, 2, 2)).copy()

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
    nc_path = tmp_path / "wind_2010.nc"
    ds.to_netcdf(nc_path)

    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    # Fix exactamente en el centro del grid (lat=59.75, lon=10.25).
    fixes = pd.DataFrame({
        "bird_id": ["X"],
        "date_utc": [pd.Timestamp("2010-07-15").date()],
        "lat": [59.75],
        "lon": [10.25],
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert out.shape == (1, 5)
    assert set(out.columns) == {
        "bird_id", "date_utc", "wind_u_850", "wind_v_850", "wind_speed_850",
    }
    # Bilinear media de (1, 2, 3, 4) = 2.5; de (10, 20, 30, 40) = 25.0.
    assert out["wind_u_850"].iloc[0] == pytest.approx(2.5)
    assert out["wind_v_850"].iloc[0] == pytest.approx(25.0)
    expected_speed = float(np.sqrt(2.5 ** 2 + 25.0 ** 2))
    assert out["wind_speed_850"].iloc[0] == pytest.approx(expected_speed)


def test_interpolate_wind_out_of_bbox(tmp_path):
    """Fix fuera del bbox del grid → NaN en las tres features."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    fixes = pd.DataFrame({
        "bird_id": ["X"],
        "date_utc": [pd.Timestamp("2010-07-15").date()],
        "lat": [80.0],   # fuera del rango [59, 60] del .nc sintético
        "lon": [50.0],   # fuera del rango [10, 11]
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert np.isnan(out["wind_u_850"].iloc[0])
    assert np.isnan(out["wind_v_850"].iloc[0])
    assert np.isnan(out["wind_speed_850"].iloc[0])


def test_interpolate_wind_nan_input(tmp_path):
    """Fix con lat=NaN o lon=NaN → NaN propagado."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    fixes = pd.DataFrame({
        "bird_id": ["X", "Y"],
        "date_utc": [
            pd.Timestamp("2010-07-15").date(),
            pd.Timestamp("2010-07-16").date(),
        ],
        "lat": [np.nan, 59.5],
        "lon": [10.5, np.nan],
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert out.shape == (2, 5)
    assert out["wind_u_850"].isna().all()
    assert out["wind_v_850"].isna().all()
    assert out["wind_speed_850"].isna().all()


def test_interpolate_wind_preserves_row_count(tmp_path):
    """Número de filas no cambia tras la interpolación."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    n = 100
    fixes = pd.DataFrame({
        "bird_id": [f"B{i % 5}" for i in range(n)],
        "date_utc": [pd.Timestamp(f"2010-07-{(i % 28) + 1:02d}").date() for i in range(n)],
        "lat": np.linspace(59.0, 60.0, n),
        "lon": np.linspace(10.0, 11.0, n),
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)
    assert len(out) == n
