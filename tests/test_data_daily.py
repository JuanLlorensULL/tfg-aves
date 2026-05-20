"""Tests de tfg_aves.data.daily."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import (
    build_daily,
    coverage_by_hour,
    filter_birds_by_validity,
    pick_reference_hour,
)


def test_coverage_by_hour_peaks_at_fixed_hour(df_hourly_coverage):
    cov = coverage_by_hour(df_hourly_coverage, tolerance_min=30)

    assert set(cov["hour"]) == set(range(24))
    cov_by_hour = dict(zip(cov["hour"], cov["coverage"], strict=True))
    # 5 días con fix a 12:00 y 5 con fix a 18:00 → cobertura máxima en
    # h=12 y h=18, mínima fuera.
    assert cov_by_hour[12] == 1.0
    assert cov_by_hour[18] == 1.0
    assert cov_by_hour[0] == 0.0


def test_pick_reference_hour_returns_argmax(df_hourly_coverage):
    hour, cov = pick_reference_hour(df_hourly_coverage, tolerance_min=30)

    # Empate entre 12 y 18: la función debe devolver la primera de ellas
    # (la de menor índice horario).
    assert hour == 12
    assert set(cov["hour"]) == set(range(24))


def test_build_daily_picks_nearest_to_reference(df_daily_three_fixes_one_day):
    daily = build_daily(
        df_daily_three_fixes_one_day,
        reference_hour_utc=12,
        tolerance_min=120,
    )

    assert list(daily.columns) == [
        "bird_id",
        "date_utc",
        "lat",
        "lon",
        "is_valid",
        "source_event_id",
        "delta_minutes",
    ]
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["bird_id"] == "A"
    assert bool(row["is_valid"]) is True
    assert row["source_event_id"] == 11           # fix de 12:10 elegido
    assert row["delta_minutes"] == 10.0
    assert row["lon"] == 24.50
    assert row["lat"] == 61.20


def test_build_daily_creates_explicit_gap(df_daily_with_gap):
    daily = build_daily(
        df_daily_with_gap, reference_hour_utc=12, tolerance_min=60
    )

    # 3 filas: 01, 02 (hueco), 03.
    assert len(daily) == 3
    dates = pd.to_datetime(daily["date_utc"]).dt.date.tolist()
    assert dates == [
        pd.Timestamp("2010-04-01").date(),
        pd.Timestamp("2010-04-02").date(),
        pd.Timestamp("2010-04-03").date(),
    ]

    gap = daily[daily["date_utc"] == pd.Timestamp("2010-04-02").date()].iloc[0]
    assert bool(gap["is_valid"]) is False
    assert pd.isna(gap["lat"]) and pd.isna(gap["lon"])
    assert pd.isna(gap["source_event_id"])
    assert pd.isna(gap["delta_minutes"])


def test_build_daily_respects_tolerance(df_daily_far_from_reference):
    # Fix a 200 min de la hora de referencia: con tol=120 → inválido.
    daily_strict = build_daily(
        df_daily_far_from_reference,
        reference_hour_utc=12,
        tolerance_min=120,
    )
    assert len(daily_strict) == 1
    assert bool(daily_strict.iloc[0]["is_valid"]) is False

    # Con tol=300 → válido.
    daily_loose = build_daily(
        df_daily_far_from_reference,
        reference_hour_utc=12,
        tolerance_min=300,
    )
    assert bool(daily_loose.iloc[0]["is_valid"]) is True
    assert daily_loose.iloc[0]["delta_minutes"] == 200.0


def test_filter_birds_by_validity_drops_low_coverage(df_daily_two_birds_unequal):
    # Construimos primero la tabla diaria.
    daily = build_daily(
        df_daily_two_birds_unequal,
        reference_hour_utc=12,
        tolerance_min=60,
    )
    out = filter_birds_by_validity(daily, min_valid_days=2)

    assert set(out["bird_id"]) == {"A"}
    assert (out["bird_id"] == "B").sum() == 0
