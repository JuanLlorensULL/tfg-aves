"""Fixtures sintéticas para los tests de tfg_aves.data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Cabecera Movebank reducida a las columnas que usamos + algunas
# 'visible' duplicadas como en el CSV real.
_MOVEBANK_HEADER = (
    '"event-id","visible","timestamp","location-long","location-lat",'
    '"manually-marked-outlier","visible","sensor-type",'
    '"individual-taxon-canonical-name","tag-local-identifier",'
    '"individual-local-identifier","study-name"'
)


def _movebank_row(
    *,
    event_id: int,
    timestamp: str,
    lon: float,
    lat: float,
    outlier: bool = False,
    visible: bool = True,
    bird_id: str = "A",
) -> str:
    outlier_cell = "true" if outlier else ""
    visible_cell = "true" if visible else "false"
    return (
        f'"{event_id}","{visible_cell}","{timestamp}",'
        f'"{lon}","{lat}","{outlier_cell}","{visible_cell}","gps",'
        f'"Larus fuscus","91732","{bird_id}","study"'
    )


@pytest.fixture
def mini_movebank_csv(tmp_path: Path) -> Path:
    """CSV pequeño con formato Movebank y casos representativos."""
    rows = [
        _movebank_row(event_id=1, timestamp="2010-04-02 12:00:00.000",
                      lon=24.5, lat=61.2),
        _movebank_row(event_id=2, timestamp="2010-04-02 13:00:00.000",
                      lon=24.6, lat=61.3),
        _movebank_row(event_id=3, timestamp="2010-04-01 12:00:00.000",
                      lon=24.4, lat=61.1),
        _movebank_row(event_id=4, timestamp="2010-04-02 14:00:00.000",
                      lon=24.7, lat=61.4, visible=False),
        _movebank_row(event_id=5, timestamp="2010-04-02 15:00:00.000",
                      lon=24.8, lat=61.5, outlier=True),
    ]
    csv = tmp_path / "mini_movebank.csv"
    csv.write_text(_MOVEBANK_HEADER + "\n" + "\n".join(rows) + "\n",
                   encoding="utf-8")
    return csv


@pytest.fixture
def df_flags() -> pd.DataFrame:
    """DataFrame cargado para tests de drop_movebank_flags."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                ],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.7, 24.8],
            "lat": [61.1, 61.2, 61.3, 61.4],
            "manually_marked_outlier": [False, True, False, False],
            "visible": [True, True, False, True],
            "sensor_type": ["gps"] * 4,
            "bird_id": ["A"] * 4,
        }
    )


@pytest.fixture
def df_coords_dupes() -> pd.DataFrame:
    """DataFrame con coords inválidas y duplicado por (bird_id, timestamp)."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 13:00",  # duplicado con #2
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                ],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.61, 200.0, 24.8],   # 200 es inválida
            "lat": [61.1, 61.2, 61.21, 61.4, 95.0],    # 95 es inválida
            "manually_marked_outlier": [False] * 5,
            "visible": [True] * 5,
            "sensor_type": ["gps"] * 5,
            "bird_id": ["A"] * 5,
        }
    )


@pytest.fixture
def df_speed() -> pd.DataFrame:
    """Cinco fixes consecutivos donde el #3 implica un salto imposible.

    Distancias aproximadas (haversine) entre consecutivos a 1 hora:
      1→2: ~10 km (≈10 km/h) — OK
      2→3: ~5800 km (≈5800 km/h) — OUTLIER
      3→4: ~5800 km (≈5800 km/h) — OUTLIER si #3 sigue presente
      4→5: ~10 km — OK
    Al descartar iterativamente, el algoritmo debe quitar #3 y reconocer
    que 2→4 es de nuevo razonable (~10 km en 2 h ≈ 5 km/h).
    """
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                    "2010-04-01 16:00",
                ],
                utc=True,
            ),
            "lon": [24.50, 24.60, 90.0, 24.70, 24.80],
            "lat": [61.10, 61.15, 30.0, 61.25, 61.30],
            "manually_marked_outlier": [False] * 5,
            "visible": [True] * 5,
            "sensor_type": ["gps"] * 5,
            "bird_id": ["A"] * 5,
        }
    )
