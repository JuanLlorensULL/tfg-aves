"""Tests para src/tfg_aves/meteo/build_wind.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tfg_aves.meteo.build_wind import build_wind


def _make_synthetic_wind_nc(path: Path, year: int) -> None:
    times = pd.date_range(
        f"{year}-01-01 12:00", f"{year}-12-31 12:00", freq="1D",
    )
    n_t = len(times)
    u = np.full((n_t, 1, 3, 3), 1.5, dtype=np.float64)
    v = np.full((n_t, 1, 3, 3), -0.5, dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "pressure_level": np.array([850.0]),
            "latitude": np.array([60.0, 59.5, 59.0]),
            "longitude": np.array([10.0, 10.5, 11.0]),
        },
    )
    ds.to_netcdf(path)


def _make_synthetic_daily(path: Path) -> pd.DataFrame:
    df = pd.DataFrame({
        "bird_id": ["A", "A", "B", "B"],
        "date_utc": [
            pd.Timestamp("2010-03-15").date(),
            pd.Timestamp("2010-08-20").date(),
            pd.Timestamp("2011-04-10").date(),
            pd.Timestamp("2011-09-05").date(),
        ],
        "lat": [59.5, 59.7, 60.0, 59.2],
        "lon": [10.5, 10.7, 10.9, 10.1],
        "is_valid": [True, True, True, True],
    })
    df.to_parquet(path)
    return df


def test_build_wind_produces_expected_columns(tmp_path):
    wind_dir = tmp_path / "raw" / "wind"
    wind_dir.mkdir(parents=True)
    _make_synthetic_wind_nc(wind_dir / "wind_2010.nc", 2010)
    _make_synthetic_wind_nc(wind_dir / "wind_2011.nc", 2011)

    daily_path = tmp_path / "daily.parquet"
    daily_df = _make_synthetic_daily(daily_path)

    out_path = tmp_path / "wind_per_fix.parquet"
    result = build_wind(
        daily_path=daily_path,
        wind_raw_dir=wind_dir,
        out_path=out_path,
    )

    assert out_path.exists()
    assert set(result.columns) == {
        "bird_id", "date_utc", "wind_u_850", "wind_v_850", "wind_speed_850",
    }
    assert len(result) == len(daily_df)
    # Valores constantes en el .nc sintético → 1.5, -0.5, sqrt(1.5²+0.5²).
    assert result["wind_u_850"].to_numpy() == pytest.approx(1.5)
    assert result["wind_v_850"].to_numpy() == pytest.approx(-0.5)
    expected_speed = float(np.sqrt(1.5**2 + 0.5**2))
    assert result["wind_speed_850"].iloc[0] == pytest.approx(expected_speed)


def test_build_wind_idempotent(tmp_path):
    wind_dir = tmp_path / "raw" / "wind"
    wind_dir.mkdir(parents=True)
    _make_synthetic_wind_nc(wind_dir / "wind_2010.nc", 2010)
    _make_synthetic_wind_nc(wind_dir / "wind_2011.nc", 2011)
    _make_synthetic_daily(tmp_path / "daily.parquet")

    out_path = tmp_path / "wind_per_fix.parquet"
    df1 = build_wind(tmp_path / "daily.parquet", wind_dir, out_path)
    df2 = build_wind(tmp_path / "daily.parquet", wind_dir, out_path)

    pd.testing.assert_frame_equal(df1, df2)
