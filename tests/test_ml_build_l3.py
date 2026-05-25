"""Tests de integración de build_o4_l3 con un fixture sintético causal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import write_synthetic_o3_features


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos consecutivos sobre celdas activas (espejo de
    tests/test_ml_build_l2.py).

    El features.parquet lleva el esquema de O3 (estado HMM causal + split),
    que es lo que build_o4_l3 consume desde el rewire causal.
    """
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
    write_synthetic_o3_features(pd.DataFrame(rows), feat_path)

    # Rejilla amplia que cubre toda la deriva de las 5 aves a lo largo de los
    # 90 días (lat ~39..45, lon ~-4..2). Necesaria desde el rewire causal: el
    # split de O3 se aplica a todos los días HMM-válidos, así que las celdas
    # deben seguir activas también en los últimos días (val/test).
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


def test_prepare_poblacional_split_attaches_hmm(tmp_path):
    from tfg_aves.ml.build_l3 import _FEATURES, _prepare_poblacional_split
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    features_o3 = pd.read_parquet(feat_path)
    cells = pd.read_parquet(cells_path)
    train, val, test = _prepare_poblacional_split(features_o3, cells)
    assert "bird_id" not in _FEATURES
    for df in (train, val, test):
        assert "state_b_causal" in df.columns
        assert "posterior_b_migracion_causal" in df.columns
    assert len(train) > 0 and len(test) > 0


def test_build_o4_l3_artifacts(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    result = build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    # 8 modelos: xgb (pob+ind) + lgbm (pob) + rf (pob), 2 ejes cada combinación.
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
    assert "poblacional@A" in set(metrics["modo"])
    assert "persistencia" in set(metrics["modo"])
    # Cobertura global reportada (no NaN) para el poblacional de cada familia.
    for family in ("xgb", "lgbm", "rf"):
        glob = metrics[(metrics["familia"] == family)
                       & (metrics["modo"] == "poblacional")
                       & (metrics["scope"] == "global")]
        assert len(glob) == 1
        assert not np.isnan(glob["coverage_lat"].iloc[0])
    assert result.n_rows_test_ind > 0


def test_build_o4_l3_individual_subset(tmp_path):
    """El test individual (xgb) coincide con el subconjunto bird_id==A del
    poblacional xgb (mismo split temporal por ave)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    ind = preds[(preds["familia"] == "xgb") & (preds["modo"] == "individual")]
    pob_a = preds[(preds["familia"] == "xgb") & (preds["modo"] == "poblacional")
                  & (preds["bird_id"] == "A")]
    assert len(ind) == len(pob_a)
    assert set(ind["date_utc"]) == set(pob_a["date_utc"])


def test_build_o4_l3_rf_lgbm_poblacional_only(tmp_path):
    """RF y LightGBM solo se entrenan en modo poblacional (G3)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    for family in ("lgbm", "rf"):
        assert set(preds[preds["familia"] == family]["modo"]) == {"poblacional"}
        assert not (out_dir / f"model_{family}_individual_dlat.pkl").exists()


def test_build_o4_l3_no_leakage(tmp_path):
    """feature_cols == las 10 causales, sin columnas de t+1, en las 3 familias."""
    import joblib

    from tfg_aves.ml.build_l3 import build_o4_l3
    from tfg_aves.ml.features import FEATURES_HMM, FEATURES_KINEMATIC
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    expected = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon"}
    for family in ("xgb", "lgbm", "rf"):
        payload = joblib.load(out_dir / f"model_{family}_poblacional_dlat.pkl")
        assert payload["feature_cols"] == expected
        assert payload["familia"] == family
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
    ma = (pd.read_parquet(out_a / "metrics.parquet")
          .sort_values(["familia", "modo", "scope"]).reset_index(drop=True))
    mb = (pd.read_parquet(out_b / "metrics.parquet")
          .sort_values(["familia", "modo", "scope"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(ma, mb)


def test_build_o4_l3_exported():
    import tfg_aves.ml as ml
    assert hasattr(ml, "build_o4_l3")
    assert hasattr(ml, "BuildO4L3Result")
