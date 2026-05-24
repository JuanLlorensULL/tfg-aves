"""Tests de integración de build_o4_l3 con un fixture sintético causal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos consecutivos sobre celdas activas (espejo de
    tests/test_ml_build_l2.py)."""
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=90)
    rows = []
    for b in birds:
        lat0 = 40.0 + rng.uniform(-1, 1)
        lon0 = -3.0 + rng.uniform(-1, 1)
        for i, d in enumerate(dates):
            rows.append({
                "bird_id": b, "date_utc": d,
                "lat": lat0 + 0.05 * i + rng.normal(0, 0.05),
                "lon": lon0 + 0.05 * i + rng.normal(0, 0.05),
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
                "is_observation_valid": True,
            })
    feat_path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(feat_path)

    cells = []
    for i in range(78, 86):
        for j in range(-9, -1):
            cells.append({
                "cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5, "n_obs_total": 30,
            })
    cells_path = tmp_path / "cells.parquet"
    pd.DataFrame(cells).to_parquet(cells_path)
    return feat_path, cells_path


def test_prepare_poblacional_split_attaches_hmm(tmp_path):
    from tfg_aves.ml.build_l3 import _prepare_poblacional_split
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    features_o3 = pd.read_parquet(feat_path)
    cells = pd.read_parquet(cells_path)
    train, val, test = _prepare_poblacional_split(features_o3, cells, seed=0)
    for df in (train, val, test):
        assert "state_b_causal" in df.columns
        assert "posterior_b_migracion_causal" in df.columns
        assert "bird_id" not in [c for c in df.columns if c == "bird_id_feature"]
    assert len(train) > 0 and len(test) > 0
