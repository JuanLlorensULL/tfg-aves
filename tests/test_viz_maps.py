"""Tests de tfg_aves.viz.maps (render folium — asserts ligeros)."""
from __future__ import annotations

import folium
import pandas as pd

from tfg_aves.viz.maps import calibration_choropleth, error_choropleth


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
