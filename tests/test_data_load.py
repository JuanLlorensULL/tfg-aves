"""Tests de tfg_aves.data.load."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import load_raw


def test_load_raw_normalises_schema(mini_movebank_csv):
    df = load_raw(mini_movebank_csv)

    expected_cols = {
        "event_id",
        "timestamp",
        "lon",
        "lat",
        "manually_marked_outlier",
        "visible",
        "sensor_type",
        "bird_id",
    }
    assert set(df.columns) == expected_cols, df.columns.tolist()

    # timestamp tipado UTC.
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert str(df["timestamp"].dt.tz) == "UTC"

    # Ordenado por (bird_id, timestamp).
    sorted_df = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sorted_df)

    # Sin columnas ambientales.
    assert not any("ECMWF" in c or "NCEP" in c for c in df.columns)

    # Booleanos normalizados (no strings).
    assert df["visible"].dtype == bool
    assert df["manually_marked_outlier"].dtype == bool

    # event_id #4 vino con visible=false → debe seguir presente (load no filtra)
    # pero su valor visible es False.
    row_4 = df[df["event_id"] == 4].iloc[0]
    assert not row_4["visible"]

    # event_id #5 vino con outlier=true.
    row_5 = df[df["event_id"] == 5].iloc[0]
    assert row_5["manually_marked_outlier"]
