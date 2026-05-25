"""Tests de integración de build_o3."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.hmm.build import build_o3


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Sintetiza un daily.parquet con 8 aves × 60 días + raw CSV con vegetación."""
    rows_daily: list[dict] = []
    rows_raw: list[dict] = []
    rng = np.random.default_rng(0)
    event_id = 1000
    for b in range(8):
        bird_id = f"BIRD{b:02d}"
        for d in range(60):
            state = 0 if d < 30 else 1
            base_lat = 50.0 + b * 0.5
            if state == 0:
                lat = base_lat + rng.normal(0, 0.05)
                lon = rng.normal(0, 0.05)
            else:
                lat = base_lat - d * 1.5
                lon = rng.normal(-d * 0.1, 0.05)
            rows_daily.append({
                "bird_id": bird_id,
                "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=d),
                "lat": lat,
                "lon": lon,
                "is_valid": True,
                "source_event_id": event_id,
                "delta_minutes": 0.0,
            })
            rows_raw.append({
                "event-id": event_id,
                "ECMWF Interim Full Daily Invariant Low Vegetation Cover": (
                    0.4 if state == 0 else 0.05
                ),
                "ECMWF Interim Full Daily Invariant High Vegetation Cover": (
                    0.3 if state == 0 else 0.05
                ),
            })
            event_id += 1

    daily_path = tmp_path / "daily.parquet"
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(rows_daily).to_parquet(daily_path, index=False)
    pd.DataFrame(rows_raw).to_csv(raw_path, index=False)
    return daily_path, raw_path


def test_build_o3_produce_ficheros_esperados(tmp_path: Path) -> None:
    daily_path, raw_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o3"

    result = build_o3(
        n_restarts=2,
        random_state=0,
        daily_path=daily_path,
        raw_csv=raw_path,
        out_dir=out_dir,
    )
    assert result.features_path.is_file()
    assert result.models_path.is_file()
    assert result.metrics_path.is_file()
    assert result.n_observations > 0
    assert np.isfinite(result.ll_per_obs_a)
    assert np.isfinite(result.ll_per_obs_b)
    assert 0.0 <= result.pct_agreement_ab <= 100.0

    feats = pd.read_parquet(result.features_path)
    assert set(feats["split"].dropna().unique()).issubset({"train", "val", "test"})


def test_build_o3_esquema_features_releen(tmp_path: Path) -> None:
    daily_path, raw_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o3"
    result = build_o3(
        n_restarts=2, random_state=0,
        daily_path=daily_path, raw_csv=raw_path, out_dir=out_dir,
    )
    feats = pd.read_parquet(result.features_path)
    expected_cols = {
        "bird_id", "date_utc", "lat", "lon",
        "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
        "daylight_hours", "veg_low", "veg_high",
        "is_hmm_obs_valid", "split",
        "state_a_causal", "state_b_causal",
        "posterior_a_estacionario", "posterior_a_migracion",
        "posterior_b_estacionario", "posterior_b_migracion",
    }
    assert expected_cols.issubset(set(feats.columns))
