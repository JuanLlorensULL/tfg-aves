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


def test_build_o4_l3_artifacts(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    result = build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    # 12 modelos: 2 modos × 2 ejes × 3 cuantiles.
    for mode in ("poblacional", "individual"):
        for axis in ("dlat", "dlon"):
            for tag in ("p10", "p50", "p90"):
                assert (out_dir / f"model_{mode}_{axis}_{tag}.pkl").exists()
    assert (out_dir / "predictions_test.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"modo", "scope", "top1", "dist_centroide_km", "coverage_lat"}.issubset(
        metrics.columns)
    # Hay filas para ambos modos, el corte @A, persistencia y persistencia@A.
    assert "poblacional" in set(metrics["modo"])
    assert "individual" in set(metrics["modo"])
    assert "poblacional@A" in set(metrics["modo"])
    assert "persistencia" in set(metrics["modo"])
    # La cobertura global está reportada (no NaN) para cada modo.
    glob = metrics[(metrics["modo"] == "poblacional") & (metrics["scope"] == "global")]
    assert not np.isnan(glob["coverage_lat"].iloc[0])
    assert result.n_rows_test_ind > 0


def test_build_o4_l3_individual_subset(tmp_path):
    """El test del modo individual coincide EXACTAMENTE con el subconjunto
    bird_id==A del test poblacional (mismo split temporal por ave)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    ind = preds[preds["modo"] == "individual"]
    pob_a = preds[(preds["modo"] == "poblacional") & (preds["bird_id"] == "A")]
    assert len(ind) == len(pob_a)
    assert set(ind["date_utc"]) == set(pob_a["date_utc"])


def test_build_o4_l3_no_leakage(tmp_path):
    """Ningún modelo entrena con columnas del futuro: feature_cols == las 10
    causales, sin lat_t_next/lon_t_next/cell_id_t_next/y_dlat/y_dlon."""
    import joblib
    from tfg_aves.ml.build_l3 import build_o4_l3
    from tfg_aves.ml.features import FEATURES_HMM, FEATURES_KINEMATIC
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    expected = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    payload = joblib.load(out_dir / "model_poblacional_dlat_p50.pkl")
    assert payload["feature_cols"] == expected
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon"}
    assert forbidden.isdisjoint(set(payload["feature_cols"]))


def test_build_o4_l3_idempotent(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_a,
                seed=0, individual_bird_id="A")
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_b,
                seed=0, individual_bird_id="A")
    ma = pd.read_parquet(out_a / "metrics.parquet").sort_values(["modo", "scope"]).reset_index(drop=True)
    mb = pd.read_parquet(out_b / "metrics.parquet").sort_values(["modo", "scope"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ma, mb)
