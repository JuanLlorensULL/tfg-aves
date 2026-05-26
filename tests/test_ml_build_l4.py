"""Tests de integración de build_o4_l4 (HMM A + vegetación cruda)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import write_synthetic_o3_features


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos (espejo de test_ml_build_l3.py)."""
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
            })
    feat_path = tmp_path / "features.parquet"
    write_synthetic_o3_features(pd.DataFrame(rows), feat_path)

    cells = []
    for i in range(76, 92):
        for j in range(-11, 7):
            cells.append({
                "cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5, "n_obs_total": 30,
            })
    cells_path = tmp_path / "cells.parquet"
    pd.DataFrame(cells).to_parquet(cells_path)
    return feat_path, cells_path


def test_l4_feature_set():
    """L4 usa estado A + posterior A + vegetación; nunca B ni daylight."""
    from tfg_aves.ml.build_l4 import _FEATURES
    assert "state_a_causal" in _FEATURES
    assert "posterior_a_migracion_causal" in _FEATURES
    assert "veg_low" in _FEATURES and "veg_high" in _FEATURES
    assert "state_b_causal" not in _FEATURES
    assert "posterior_b_migracion_causal" not in _FEATURES
    assert "daylight_hours" not in _FEATURES


def test_prepare_poblacional_split_attaches_state_a(tmp_path):
    from tfg_aves.ml.build_l4 import _prepare_poblacional_split
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    features_o3 = pd.read_parquet(feat_path)
    cells = pd.read_parquet(cells_path)
    train, val, test = _prepare_poblacional_split(features_o3, cells)
    for df in (train, val, test):
        assert "state_a_causal" in df.columns
        assert "posterior_a_migracion_causal" in df.columns
        assert "veg_low" in df.columns and "veg_high" in df.columns
        assert "state_b_causal" not in df.columns
    assert len(train) > 0 and len(test) > 0


def test_build_o4_l4_artifacts(tmp_path):
    from tfg_aves.ml.build_l4 import build_o4_l4
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l4"
    result = build_o4_l4(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    for family in ("xgb", "lgbm", "rf"):
        for axis in ("dlat", "dlon"):
            assert (out_dir / f"model_{family}_poblacional_{axis}.pkl").exists()
    for axis in ("dlat", "dlon"):
        assert (out_dir / f"model_xgb_individual_{axis}.pkl").exists()
    assert (out_dir / "predictions_test.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"familia", "modo", "scope", "top1", "dist_centroide_km",
            "coverage_lat"}.issubset(metrics.columns)
    assert {"xgb", "lgbm", "rf"}.issubset(set(metrics["familia"]))
    assert "individual" in set(metrics["modo"])          # solo xgb
    assert "persistencia" in set(metrics["modo"])
    assert result.n_rows_test_pob > 0


def test_build_o4_l4_no_leakage(tmp_path):
    """feature_cols == las 12 features de L4, sin columnas de t+1 ni de B/daylight."""
    import joblib

    from tfg_aves.ml.build_l4 import _FEATURES, build_o4_l4
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l4"
    build_o4_l4(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon",
                 "state_b_causal", "posterior_b_migracion_causal", "daylight_hours"}
    for family in ("xgb", "lgbm", "rf"):
        payload = joblib.load(out_dir / f"model_{family}_poblacional_dlat.pkl")
        assert payload["feature_cols"] == _FEATURES
        assert forbidden.isdisjoint(set(payload["feature_cols"]))


def test_build_o4_l4_exported():
    import tfg_aves.ml as ml
    assert hasattr(ml, "build_o4_l4")
    assert hasattr(ml, "BuildO4L4Result")
