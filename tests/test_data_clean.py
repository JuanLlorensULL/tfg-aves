"""Tests de tfg_aves.data.clean."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import (
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
)


def test_drop_movebank_flags(df_flags):
    out, report = drop_movebank_flags(df_flags)

    # event_id 2 (outlier=True) y 3 (visible=False) descartados; 1 y 4 quedan.
    assert sorted(out["event_id"].tolist()) == [1, 4]
    assert report == {
        "discarded_movebank_outlier": 1,
        "discarded_visible_false": 1,
    }


def test_drop_invalid_coords_and_dupes(df_coords_dupes):
    out, report = drop_invalid_coords_and_dupes(df_coords_dupes)

    # event_id 4 (lon=200) y 5 (lat=95) inválidos; #3 es duplicado de #2.
    # Sobreviven 1 y 2.
    assert sorted(out["event_id"].tolist()) == [1, 2]
    assert report == {
        "discarded_invalid_coords": 2,
        "discarded_duplicates": 1,
    }


def test_drop_speed_outliers_removes_impossible_fix(df_speed):
    out, report = drop_speed_outliers(df_speed, max_speed_kmh=200.0)

    # El fix #3 está a ~3700 km de los vecinos en 1 h → debe descartarse.
    assert 3 not in out["event_id"].tolist()
    # Los inocentes 1, 2, 4, 5 sobreviven.
    assert sorted(out["event_id"].tolist()) == [1, 2, 4, 5]
    assert report["discarded_speed"] == 1


def test_drop_speed_outliers_no_outliers_returns_intact():
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                ["2010-04-01 12:00", "2010-04-01 13:00", "2010-04-01 14:00"],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.7],
            "lat": [61.1, 61.15, 61.20],
            "manually_marked_outlier": [False] * 3,
            "visible": [True] * 3,
            "sensor_type": ["gps"] * 3,
            "bird_id": ["A"] * 3,
        }
    )
    out, report = drop_speed_outliers(df, max_speed_kmh=200.0)

    assert out["event_id"].tolist() == [1, 2, 3]
    assert report == {"discarded_speed": 0}
