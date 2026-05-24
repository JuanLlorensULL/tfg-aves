"""Tests de tfg_aves.viz.error."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.viz.error import error_by_cell, recover_origin


def _cells():
    return pd.DataFrame({
        "cell_id": ["80_-6", "80_-7", "81_-6"],
        "lat_c": [40.25, 40.25, 40.75],
        "lon_c": [-2.75, -3.25, -2.75],
    })


def _preds():
    return pd.DataFrame({
        "bird_id": ["A", "A", "B"],
        "date_utc": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-03-01"]),
        "pred_lat": [40.30, 40.40, 40.80],
        "pred_lon": [-2.70, -2.80, -2.70],
        "dlat_p50": [0.10, 0.10, 0.10],
        "dlon_p50": [0.05, 0.05, 0.05],
        "dist_native_km": [10.0, 30.0, 50.0],
        "true_cell": ["80_-6", "80_-6", "81_-6"],
        "pred_cell_top1": ["80_-6", "80_-7", "81_-6"],
        "state_b_causal": [0, 0, 1],
        "in_interval_lat": [True, True, False],
        "in_interval_lon": [True, False, True],
    })


def test_recover_origin_inverts_p50():
    out = recover_origin(_preds())
    assert np.allclose(out["lat_t"], [40.20, 40.30, 40.70])
    assert np.allclose(out["lon_t"], [-2.75, -2.85, -2.75])


def test_error_by_cell_groups_by_origin_cell_median():
    out = error_by_cell(_preds(), _cells())
    row_80 = out[out["cell_id"] == "80_-6"].iloc[0]
    assert row_80["n"] == 2
    assert row_80["median_error_km"] == 20.0
    row_81 = out[out["cell_id"] == "81_-6"].iloc[0]
    assert row_81["n"] == 1
    assert row_81["median_error_km"] == 50.0
    assert {"lat_c", "lon_c"}.issubset(out.columns)


def test_coverage_by_cell_marginal_mean():
    from tfg_aves.viz.error import coverage_by_cell
    out = coverage_by_cell(_preds(), _cells())
    row_80 = out[out["cell_id"] == "80_-6"].iloc[0]
    assert row_80["coverage_marginal"] == 0.75
    row_81 = out[out["cell_id"] == "81_-6"].iloc[0]
    assert row_81["coverage_marginal"] == 0.5


def test_metrics_by_regime_top1_and_coverage():
    from tfg_aves.viz.error import metrics_by_regime
    out = metrics_by_regime(_preds())
    estac = out[out["regimen"] == "estacionario"].iloc[0]
    assert estac["n"] == 2
    assert estac["top1"] == 0.5
    assert estac["dist_median_km"] == 20.0
    assert estac["coverage_joint"] == 0.5
    migr = out[out["regimen"] == "migración"].iloc[0]
    assert migr["n"] == 1
    assert migr["top1"] == 1.0


def test_metrics_by_month_groups_by_calendar_month():
    from tfg_aves.viz.error import metrics_by_month
    out = metrics_by_month(_preds())
    ene = out[out["mes"] == 1].iloc[0]
    assert ene["n"] == 2
    mar = out[out["mes"] == 3].iloc[0]
    assert mar["n"] == 1
