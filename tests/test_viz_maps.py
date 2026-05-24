"""Tests de tfg_aves.viz.maps (render folium — asserts ligeros)."""
from __future__ import annotations

import folium
import pandas as pd

from tfg_aves.viz.maps import (
    calibration_choropleth,
    error_choropleth,
    failure_vectors_map,
    multistep_demo_map,
    prediction_map,
)


def _error_df():
    return pd.DataFrame({
        "cell_id": ["80_-6", "81_-6"], "lat_c": [40.25, 40.75],
        "lon_c": [-2.75, -2.75], "n": [2, 1], "median_error_km": [20.0, 50.0],
    })


def _cov_df():
    return pd.DataFrame({
        "cell_id": ["80_-6", "81_-6"], "lat_c": [40.25, 40.75],
        "lon_c": [-2.75, -2.75], "n": [2, 1], "coverage_marginal": [0.75, 0.5],
    })


def test_error_choropleth_returns_map_with_cells():
    m = error_choropleth(_error_df(), cell_deg=0.5)
    assert isinstance(m, folium.Map)
    html = m.get_root().render()
    assert "leaflet" in html.lower()


def test_calibration_choropleth_returns_map():
    m = calibration_choropleth(_cov_df(), cell_deg=0.5)
    assert isinstance(m, folium.Map)
    assert m.get_root().render()


def test_prediction_map_one_bird_layers():
    preds = pd.DataFrame({
        "bird_id": ["A", "A"],
        "date_utc": pd.to_datetime(["2020-04-01", "2020-04-02"]),
        "pred_lat": [40.3, 40.5], "pred_lon": [-2.7, -2.6],
        "dlat_p10": [0.05, 0.05], "dlat_p50": [0.1, 0.1], "dlat_p90": [0.15, 0.15],
        "dlon_p10": [-0.05, -0.05], "dlon_p50": [0.0, 0.0], "dlon_p90": [0.05, 0.05],
    })
    daily = pd.DataFrame({
        "bird_id": ["A", "A", "A"],
        "date_utc": pd.to_datetime(["2020-04-01", "2020-04-02", "2020-04-03"]),
        "lat": [40.2, 40.4, 40.45], "lon": [-2.75, -2.65, -2.60],
        "is_valid": [True, True, True],
    })
    m = prediction_map(preds, daily, bird_id="A")
    assert m.get_root().render()


def test_failure_vectors_map_migration_only():
    preds = pd.DataFrame({
        "bird_id": ["A", "B"], "date_utc": pd.to_datetime(["2020-04-01", "2020-09-01"]),
        "pred_lat": [42.0, 50.0], "pred_lon": [-2.0, 3.0],
        "dlat_p50": [1.0, 1.0], "dlon_p50": [1.0, 1.0],
        "dist_native_km": [300.0, 500.0], "state_b_causal": [1, 1],
    })
    daily = pd.DataFrame({
        "bird_id": ["A", "B"], "date_utc": pd.to_datetime(["2020-04-02", "2020-09-02"]),
        "lat": [43.0, 52.0], "lon": [-1.0, 4.0], "is_valid": [True, True],
    })
    m = failure_vectors_map(preds, daily, top_n=2)
    assert m.get_root().render()


def test_multistep_demo_map_renders():
    chain = pd.DataFrame({
        "step": [1, 2], "lat": [40.2, 40.3], "lon": [-3.0, -3.0],
        "cone_halfwidth_lat": [0.05, 0.10], "cone_halfwidth_lon": [0.05, 0.10],
    })
    real = pd.DataFrame({"lat": [40.1, 40.15, 40.18], "lon": [-3.0, -2.98, -2.95]})
    m = multistep_demo_map(chain, real, start=(40.1, -3.0))
    assert m.get_root().render()


def test_prediction_map_colors_trajectory_by_year_when_multiyear():
    # Periodo de test que cruza dos años -> la leyenda lista ambos.
    preds = pd.DataFrame({
        "bird_id": ["A", "A"],
        "date_utc": pd.to_datetime(["2020-12-30", "2021-01-02"]),
        "pred_lat": [40.3, 40.5], "pred_lon": [-2.7, -2.6],
        "dlat_p10": [0.05, 0.05], "dlat_p50": [0.1, 0.1], "dlat_p90": [0.15, 0.15],
        "dlon_p10": [-0.05, -0.05], "dlon_p50": [0.0, 0.0], "dlon_p90": [0.05, 0.05],
    })
    daily = pd.DataFrame({
        "bird_id": ["A"] * 4,
        "date_utc": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"]),
        "lat": [40.2, 40.25, 40.3, 40.4], "lon": [-2.75, -2.72, -2.70, -2.65],
        "is_valid": [True, True, True, True],
    })
    html = prediction_map(preds, daily, bird_id="A").get_root().render()
    assert "trayectoria real 2020" in html
    assert "trayectoria real 2021" in html
