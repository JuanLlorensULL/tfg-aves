"""Tests de tfg_aves.markov.discretize."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tfg_aves.markov.discretize import (
    assign_cell,
    cell_centroid,
    discretize_dataframe,
    haversine_km,
)


def test_assign_cell_quadrants() -> None:
    # cell_deg=1.0 → cada celda es un grado.
    assert assign_cell(lat=0.5, lon=0.5, cell_deg=1.0) == (0, 0)
    assert assign_cell(lat=10.2, lon=20.7, cell_deg=1.0) == (10, 20)
    # Hemisferio sur / oeste: floor → índices negativos correctos.
    assert assign_cell(lat=-1.5, lon=-0.5, cell_deg=1.0) == (-2, -1)


def test_assign_cell_borde_va_a_celda_superior() -> None:
    # Punto exactamente sobre la línea cae en la celda cuyo idx = floor.
    # 10.0 / 0.5 = 20.0 → idx 20 (la celda [10.0, 10.5)).
    assert assign_cell(lat=10.0, lon=0.0, cell_deg=0.5) == (20, 0)


def test_assign_cell_monotonia() -> None:
    deg = 0.5
    idxs = [assign_cell(lat=lat, lon=0.0, cell_deg=deg)[0] for lat in np.arange(0.0, 5.0, 0.1)]
    # Monotonía no decreciente.
    assert idxs == sorted(idxs)


def test_cell_centroid_inverso() -> None:
    deg = 0.5
    # assign_cell(lat=10.2, lon=20.7, deg=0.5) = (20, 41).
    # centroid → (20+0.5)*0.5, (41+0.5)*0.5 = (10.25, 20.75).
    assert cell_centroid((20, 41), cell_deg=deg) == pytest.approx((10.25, 20.75))


def test_haversine_paris_madrid() -> None:
    # París (48.8566, 2.3522) ↔ Madrid (40.4168, -3.7038) ≈ 1054 km.
    d = haversine_km(48.8566, 2.3522, 40.4168, -3.7038)
    assert d == pytest.approx(1054, rel=0.01)


def test_haversine_vectorizada() -> None:
    lat1 = np.array([48.8566, 0.0])
    lon1 = np.array([2.3522, 0.0])
    lat2 = np.array([40.4168, 0.0])
    lon2 = np.array([-3.7038, 0.0])
    d = haversine_km(lat1, lon1, lat2, lon2)
    assert d.shape == (2,)
    assert d[0] == pytest.approx(1054, rel=0.01)
    assert d[1] == pytest.approx(0.0, abs=1e-6)


def test_discretize_dataframe_basico() -> None:
    df = pd.DataFrame(
        {
            "bird_id": ["A", "A", "B"],
            "date_utc": [dt.date(2010, 1, 1), dt.date(2010, 1, 2), dt.date(2010, 1, 1)],
            "lat": [50.2, np.nan, 30.1],
            "lon": [4.7, np.nan, -1.5],
            "is_valid": [True, False, True],
        }
    )
    out = discretize_dataframe(df, cell_deg=0.5)
    assert "cell_id" in out.columns
    assert "cell_lat_idx" in out.columns
    assert "cell_lon_idx" in out.columns
    # Fila inválida → cell_id = None.
    assert out.loc[1, "cell_id"] is None
    # Fila válida → cell_id string "i_j".
    assert out.loc[0, "cell_id"] == "100_9"  # (50.2/0.5, 4.7/0.5) = (100, 9)
