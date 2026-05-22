"""Tests de tfg_aves.hmm.features."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfg_aves.hmm.features import (
    bearing_rad,
    compute_observation_features,
    daylight_hours,
    load_vegetation_from_raw,
)


def _make_daily(rows: list[dict]) -> pd.DataFrame:
    """Construye un daily-like DataFrame con las columnas necesarias."""
    df = pd.DataFrame(rows)
    df["is_valid"] = df["lat"].notna() & df["lon"].notna()
    return df


def test_bearing_rad_norte() -> None:
    # Movimiento exactamente hacia el norte: bearing = 0 rad.
    b = bearing_rad(40.0, 0.0, 41.0, 0.0)
    assert abs(b) < 1e-6


def test_turning_angle_linea_recta() -> None:
    """Tres puntos en línea recta hacia el norte → turning_angle ≈ 0."""
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 1), "lat": 50.0, "lon": 0.0, "source_event_id": 1},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 2), "lat": 50.5, "lon": 0.0, "source_event_id": 2},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 3), "lat": 51.0, "lon": 0.0, "source_event_id": 3},
        ]
    )
    out = compute_observation_features(df, df_raw=None)
    # El día central (índice 1) es el que tiene turning angle definido.
    valid = out[out["is_observation_valid"]]
    assert len(valid) == 1
    assert abs(valid.iloc[0]["abs_turning_angle_rad"]) < 1e-3


def test_turning_angle_giro_180() -> None:
    """Norte-norte-sur: turning_angle ≈ π."""
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 1), "lat": 50.0, "lon": 0.0, "source_event_id": 1},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 2), "lat": 50.5, "lon": 0.0, "source_event_id": 2},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 3), "lat": 50.0, "lon": 0.0, "source_event_id": 3},
        ]
    )
    out = compute_observation_features(df, df_raw=None)
    valid = out[out["is_observation_valid"]]
    assert len(valid) == 1
    assert valid.iloc[0]["abs_turning_angle_rad"] == pytest.approx(np.pi, abs=1e-3)


def test_daylight_equinoccio_y_solsticio() -> None:
    # Equinoccio primavera aprox día 80, en latitud 0: ~12h.
    d_eq = daylight_hours(lat=0.0, day_of_year=80)
    assert d_eq == pytest.approx(12.0, abs=0.2)

    # Solsticio verano aprox día 172, lat 60°: ~18-19h.
    d_sols = daylight_hours(lat=60.0, day_of_year=172)
    assert 17.5 < d_sols < 19.5

    # Solsticio verano lat -60° (hemisferio sur): ~5-7h (días cortos).
    d_inv = daylight_hours(lat=-60.0, day_of_year=172)
    assert 4.0 < d_inv < 7.0


def test_huecos_rompen_triplete() -> None:
    """Día con vecino inválido → is_observation_valid=False."""
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 1), "lat": 50.0, "lon": 0.0, "source_event_id": 1},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 2), "lat": np.nan, "lon": np.nan, "source_event_id": 2},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 3), "lat": 50.5, "lon": 0.0, "source_event_id": 3},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 4), "lat": 51.0, "lon": 0.0, "source_event_id": 4},
            {"bird_id": "A", "date_utc": dt.date(2010, 6, 5), "lat": 51.5, "lon": 0.0, "source_event_id": 5},
        ]
    )
    out = compute_observation_features(df, df_raw=None)
    # Sólo día 4 tiene triplete (3, 4, 5 todos válidos).
    valid = out[out["is_observation_valid"]]
    assert len(valid) == 1
    assert valid.iloc[0]["date_utc"] == dt.date(2010, 6, 4)


def test_load_vegetation_join(tmp_path: Path) -> None:
    """El join por event_id rescata veg_low y veg_high del CSV crudo."""
    # CSV crudo mínimo con las dos columnas que necesitamos.
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text(
        '"event-id","ECMWF Interim Full Daily Invariant Low Vegetation Cover",'
        '"ECMWF Interim Full Daily Invariant High Vegetation Cover"\n'
        '100,0.42,0.10\n'
        '101,0.30,0.55\n'
        '102,0.00,0.00\n'
    )
    event_ids = pd.Series([100, 102, 999], name="source_event_id")  # 999 no existe
    veg = load_vegetation_from_raw(raw_csv, event_ids)
    assert set(veg.columns) >= {"event_id", "veg_low", "veg_high"}
    veg_by_id = veg.set_index("event_id")
    assert veg_by_id.loc[100, "veg_low"] == pytest.approx(0.42)
    assert veg_by_id.loc[100, "veg_high"] == pytest.approx(0.10)
    assert veg_by_id.loc[102, "veg_low"] == pytest.approx(0.0)
    # 999 no aparece (lo gestiona el caller).
    assert 999 not in veg_by_id.index
