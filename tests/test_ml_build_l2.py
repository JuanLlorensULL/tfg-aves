"""Test de integración de build_o4_l2 con un fixture sintético causal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.ml.build_l2 import build_o4_l2


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos consecutivos sobre celdas activas.

    Mismo patrón que tests/test_ml_build.py pero con más días para
    garantizar suficientes filas con y_move=1 tras la máscara de racha
    de 4 días del HMM causal.
    """
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=90)
    rows = []
    for b in birds:
        lat0 = 40.0 + rng.uniform(-1, 1)
        lon0 = -3.0 + rng.uniform(-1, 1)
        for i, d in enumerate(dates):
            lat = lat0 + 0.05 * i + rng.normal(0, 0.05)
            lon = lon0 + 0.05 * i + rng.normal(0, 0.05)
            rows.append({
                "bird_id": b, "date_utc": d, "lat": lat, "lon": lon,
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
                "is_observation_valid": True,
            })
    feat = pd.DataFrame(rows)
    feat_path = tmp_path / "features.parquet"
    feat.to_parquet(feat_path)

    cells = []
    for i in range(78, 86):
        for j in range(-9, -1):
            cells.append({
                "cell_id": f"{i}_{j}",
                "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5,
                "n_obs_total": 30,
            })
    cells_df = pd.DataFrame(cells)
    cells_path = tmp_path / "cells.parquet"
    cells_df.to_parquet(cells_path)
    return feat_path, cells_path


def test_build_o4_l2_artifacts(tmp_path):
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l2_v1"
    result = build_o4_l2(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0,
    )
    # 6 modelos: 2 clf_move (rf, xgb) + 4 clf_dest (rf/xgb × pers/pob)
    assert (out_dir / "model_clf_move_rf.pkl").exists()
    assert (out_dir / "model_clf_move_xgb.pkl").exists()
    assert (out_dir / "model_clf_dest_rf_personalizado.pkl").exists()
    assert (out_dir / "model_clf_dest_xgb_poblacional.pkl").exists()
    assert (out_dir / "predictions_test_soft.parquet").exists()
    assert (out_dir / "predictions_test_hard.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()
    assert (out_dir / "tau_sweep.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"modelo", "modo", "regla", "top1", "log_loss"}.issubset(metrics.columns)
    assert len(metrics) >= 8

    # log_loss de hard debe ser NaN (no comparable)
    hard_rows = metrics[metrics["regla"] == "hard"]
    assert hard_rows["log_loss"].isna().all()
    assert result.n_moves_train > 0


def test_build_o4_l2_idempotent(tmp_path):
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l2_v1"
    build_o4_l2(features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0)
    m1 = pd.read_parquet(out_dir / "metrics.parquet")
    build_o4_l2(features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0)
    m2 = pd.read_parquet(out_dir / "metrics.parquet")
    pd.testing.assert_frame_equal(
        m1.sort_values(["modelo", "modo", "regla"]).reset_index(drop=True),
        m2.sort_values(["modelo", "modo", "regla"]).reset_index(drop=True),
    )
