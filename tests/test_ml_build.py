"""Tests de integración del orquestador build_o4."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.ml import build_o4


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 60 días válidos consecutivos sobre 16 celdas activas."""
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=60)
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
    for i in range(78, 84):
        for j in range(-9, -3):
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


def test_build_o4_end_to_end(tmp_path: Path) -> None:
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o4"
    result = build_o4(
        features_path=feat_path, cells_path=cells_path,
        output_dir=out_dir, seed=0,
    )
    assert result.n_birds == 5
    assert result.n_rows_train > 0
    assert result.n_rows_test > 0
    assert set(result.model_paths.keys()) == {
        "personalizado_rf", "personalizado_xgb", "personalizado_lgbm",
        "poblacional_rf", "poblacional_xgb", "poblacional_lgbm",
    }
    for p in result.model_paths.values():
        assert p.exists()
    assert result.predictions_path.exists()
    assert result.metrics_path.exists()

    preds = pd.read_parquet(result.predictions_path)
    assert {
        "bird_id", "date_utc", "true_cell", "pred_cell_top1", "modelo", "modo"
    } <= set(preds.columns)
    metrics = pd.read_parquet(result.metrics_path)
    assert {
        "modelo", "modo", "top1", "top3", "log_loss", "dist_median_km", "split"
    } <= set(metrics.columns)


def test_build_o4_with_wind_writes_l1v1_artifacts(tmp_path, monkeypatch):
    """build_o4(with_wind=True) escribe a O4_L1V1_DIR sin pisar O4_OUT_DIR."""
    # Este test es un placeholder de integración. La verificación real
    # se hace ejecutando build_o4(with_wind=True) sobre los datos
    # reales en el step 6.5 del plan. Aquí sólo verificamos la firma.
    from inspect import signature

    from tfg_aves.ml.build import build_o4

    sig = signature(build_o4)
    assert "with_wind" in sig.parameters
    assert sig.parameters["with_wind"].default is False


def test_build_o4_idempotent(tmp_path: Path) -> None:
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o4"
    r1 = build_o4(features_path=feat_path, cells_path=cells_path, output_dir=out_dir, seed=0)
    m1 = pd.read_parquet(r1.metrics_path)
    r2 = build_o4(features_path=feat_path, cells_path=cells_path, output_dir=out_dir, seed=0)
    m2 = pd.read_parquet(r2.metrics_path)
    pd.testing.assert_frame_equal(
        m1.sort_values(["modelo", "modo", "split"]).reset_index(drop=True),
        m2.sort_values(["modelo", "modo", "split"]).reset_index(drop=True),
    )


def test_build_o4_columnas_causales(tmp_path: Path) -> None:
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o4"
    res = build_o4(
        features_path=feat_path, cells_path=cells_path,
        output_dir=out_dir, seed=0,
    )
    assert res.predictions_path.exists()
    assert res.metrics_path.exists()
    assert (out_dir / "model_personalizado_rf.pkl").exists()

    import joblib
    blob = joblib.load(out_dir / "model_personalizado_rf.pkl")
    cols = blob["feature_cols"]
    # Exactamente las 10 causales + bird_id; nada de columnas con fuga.
    assert "bird_id" in cols
    for c in ["step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
              "state_b_causal", "posterior_b_migracion_causal"]:
        assert c in cols
    for prohibida in ["step_length_km", "cos_turning_angle", "state_b",
                      "posterior_b_migracion"]:
        assert prohibida not in cols

    # Las predicciones llevan el estado causal (para el análisis por régimen).
    preds = pd.read_parquet(res.predictions_path)
    assert "state_b_causal" in preds.columns
    assert "state_b" not in preds.columns
    modelos = preds[preds["modo"] == "personalizado"]
    assert modelos["state_b_causal"].notna().any()
