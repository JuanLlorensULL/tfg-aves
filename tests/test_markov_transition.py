"""Tests de tfg_aves.markov.transition."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from tfg_aves.markov.discretize import discretize_dataframe
from tfg_aves.markov.transition import build_counts, build_transitions


def _make_daily(rows: list[dict]) -> pd.DataFrame:
    """Construye un daily-like DataFrame con columnas esperadas."""
    df = pd.DataFrame(rows)
    df["is_valid"] = df["lat"].notna() & df["lon"].notna()
    return df


def test_par_consecutivo_emitido() -> None:
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 1), "lat": 50.0, "lon": 5.0},
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 2), "lat": 50.5, "lon": 5.5},
        ]
    )
    df = discretize_dataframe(df, cell_deg=0.5)
    pairs = build_transitions(df)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["bird_id"] == "A"
    assert row["month_int"] == 1
    assert row["date_t"] == dt.date(2010, 1, 1)


def test_hueco_rompe_par() -> None:
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 1), "lat": 50.0, "lon": 5.0},
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 2), "lat": np.nan, "lon": np.nan},
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 3), "lat": 50.5, "lon": 5.5},
        ]
    )
    df = discretize_dataframe(df, cell_deg=0.5)
    pairs = build_transitions(df)
    # Ningún par válido (1-2 inválido, 2-3 inválido, 1-3 no consecutivo).
    assert len(pairs) == 0


def test_cambio_de_ave_no_emite_par() -> None:
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 1), "lat": 50.0, "lon": 5.0},
            {"bird_id": "B", "date_utc": dt.date(2010, 1, 2), "lat": 30.0, "lon": -1.0},
        ]
    )
    df = discretize_dataframe(df, cell_deg=0.5)
    pairs = build_transitions(df)
    assert len(pairs) == 0


def test_mes_del_origen_aunque_cruce_fin_de_mes() -> None:
    df = _make_daily(
        [
            {"bird_id": "A", "date_utc": dt.date(2010, 1, 31), "lat": 50.0, "lon": 5.0},
            {"bird_id": "A", "date_utc": dt.date(2010, 2, 1), "lat": 50.5, "lon": 5.5},
        ]
    )
    df = discretize_dataframe(df, cell_deg=0.5)
    pairs = build_transitions(df)
    assert len(pairs) == 1
    assert pairs.iloc[0]["month_int"] == 1  # mes del origen


def test_build_counts_valores_conocidos() -> None:
    # Construir transiciones sintéticas con celdas conocidas y verificar counts.
    transitions = pd.DataFrame(
        [
            {"bird_id": "A", "date_t": dt.date(2010, 1, 1), "month_int": 1,
             "cell_from": "0_0", "cell_to": "1_0",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.75, "lon_to": 0.25},
            {"bird_id": "A", "date_t": dt.date(2010, 1, 2), "month_int": 1,
             "cell_from": "0_0", "cell_to": "1_0",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.75, "lon_to": 0.25},
            {"bird_id": "A", "date_t": dt.date(2010, 2, 1), "month_int": 2,
             "cell_from": "0_0", "cell_to": "0_1",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.25, "lon_to": 0.75},
        ]
    )
    cells = ["0_0", "0_1", "1_0"]
    counts = build_counts(transitions, cells=cells)
    assert counts.shape == (12, 3, 3)
    # Mes 1: 2 transiciones de 0_0 → 1_0.
    assert counts[0, 0, 2] == 2  # idx 0 = "0_0", idx 2 = "1_0"
    # Mes 2: 1 transición de 0_0 → 0_1.
    assert counts[1, 0, 1] == 1
    # Resto a 0.
    assert counts.sum() == 3


def test_build_counts_mes_sin_transiciones() -> None:
    transitions = pd.DataFrame(
        [
            {"bird_id": "A", "date_t": dt.date(2010, 5, 1), "month_int": 5,
             "cell_from": "0_0", "cell_to": "0_0",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.25, "lon_to": 0.25},
        ]
    )
    cells = ["0_0"]
    counts = build_counts(transitions, cells=cells)
    # Sólo mes 5 (índice 4) tiene contenido.
    assert counts[4, 0, 0] == 1
    assert counts.sum() == 1
