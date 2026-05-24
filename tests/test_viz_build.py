"""Test de orquestación de build_o5 sobre fixtures sintéticos."""
from __future__ import annotations

import pandas as pd

from tfg_aves.viz.build import build_o5_tables


def test_build_o5_tables_curated_and_regime(tmp_path):
    preds = pd.DataFrame({
        "bird_id": ["91916A"] * 4,
        "date_utc": pd.to_datetime(["2020-04-01", "2020-04-02", "2020-09-01", "2020-09-02"]),
        "pred_lat": [40.3, 40.4, 41.0, 41.1], "pred_lon": [-2.7, -2.8, -2.7, -2.6],
        "dlat_p50": [0.1, 0.1, 0.1, 0.1], "dlon_p50": [0.05, 0.05, 0.05, 0.05],
        "dist_native_km": [10.0, 30.0, 200.0, 400.0],
        "true_cell": ["80_-6", "80_-6", "82_-6", "82_-6"],
        "pred_cell_top1": ["80_-6", "80_-7", "82_-6", "82_-7"],
        "state_b_causal": [0, 0, 1, 1],
        "in_interval_lat": [True, True, False, True],
        "in_interval_lon": [True, False, True, False],
    })
    daily = pd.DataFrame({
        "bird_id": ["91916A"] * 4,
        "date_utc": pd.to_datetime(["2020-04-01", "2020-04-02", "2020-09-01", "2020-09-02"]),
        "lat": [40.2, 40.3, 40.9, 41.0], "lon": [-2.75, -2.85, -2.75, -2.65],
        "is_valid": [True, True, True, True],
    })
    features_o3 = pd.DataFrame({
        "bird_id": ["91916A"] * 4, "state_b": [0, 0, 1, 1],
    })
    tables = build_o5_tables(preds, daily=daily, features_o3=features_o3, project_root=tmp_path)
    assert set(tables) == {"aves_curadas", "por_regimen", "por_mes"}
    assert (tmp_path / "reports" / "tables" / "o5_tab02_error-por-regimen.csv").exists()
    assert len(tables["aves_curadas"]) == 4
    # 91916A: 4 días de test y 50 % de migración en el fixture (state_b=[0,0,1,1]).
    row = tables["aves_curadas"].set_index("bird_id").loc["91916A"]
    assert row["dias_test"] == 4
    assert row["pct_migracion"] == 50.0


def test_build_o5_maps_writes_html(tmp_path):
    from tfg_aves.viz.build import build_error_maps
    preds = pd.DataFrame({
        "bird_id": ["91916A"] * 3,
        "date_utc": pd.to_datetime(["2020-04-01", "2020-04-02", "2020-09-01"]),
        "pred_lat": [40.3, 40.4, 41.0], "pred_lon": [-2.7, -2.8, -2.7],
        "dlat_p50": [0.1, 0.1, 0.1], "dlon_p50": [0.05, 0.05, 0.05],
        "dist_native_km": [10.0, 30.0, 200.0],
        "true_cell": ["80_-6", "80_-6", "82_-6"],
        "pred_cell_top1": ["80_-6", "80_-7", "82_-6"],
        "state_b_causal": [0, 0, 1],
        "in_interval_lat": [True, True, False], "in_interval_lon": [True, False, True],
    })
    cells = pd.DataFrame({
        "cell_id": ["80_-6", "80_-7", "82_-6"], "lat_c": [40.25, 40.25, 41.25],
        "lon_c": [-2.75, -3.25, -2.75],
    })
    build_error_maps(preds, cells, out_dir=tmp_path)
    assert (tmp_path / "o5_fig20_error-por-celda.html").exists()
    assert (tmp_path / "o5_fig21_calibracion-por-celda.html").exists()


def test_build_multistep_demo_with_injected_axes(tmp_path):
    from tfg_aves.viz.build import build_multistep_demo

    class _FakeAxis:
        def predict_raw(self, X):
            return [[0.0, 0.1, 0.2]] * len(X)

    preds = pd.DataFrame({
        "bird_id": ["91916A", "91916A"],
        "date_utc": pd.to_datetime(["2020-09-01", "2020-09-02"]),
        "state_b_causal": [1, 1],
    })
    daily = pd.DataFrame({
        "bird_id": ["91916A"] * 4,
        "date_utc": pd.to_datetime(["2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02"]),
        "lat": [40.0, 40.1, 40.2, 40.3], "lon": [-3.0, -3.0, -3.0, -3.0],
        "is_valid": [True] * 4,
    })
    out = build_multistep_demo(preds, daily, out_dir=tmp_path,
                               axes=(_FakeAxis(), _FakeAxis()), k=3)
    assert out
    assert (tmp_path / "o5_fig30_demo-multipaso-91916A.html").exists()


def _app_fixtures():
    preds = pd.DataFrame({
        "bird_id": ["91916A"] * 2,
        "date_utc": pd.to_datetime(["2014-12-31", "2015-01-01"]),
        "pred_lat": [40.3, 41.0], "pred_lon": [-2.7, -2.6],
        "dlat_p10": [0.05, 0.05], "dlat_p50": [0.1, 0.1], "dlat_p90": [0.2, 0.2],
        "dlon_p10": [-0.05, -0.05], "dlon_p50": [0.0, 0.0], "dlon_p90": [0.05, 0.05],
        "dist_native_km": [12.0, 45.0],
        "state_b_causal": [0, 1],
    })
    daily = pd.DataFrame({
        "bird_id": ["91916A"] * 3,
        "date_utc": pd.to_datetime(["2014-12-31", "2015-01-01", "2015-01-02"]),
        "lat": [40.2, 40.9, 41.05], "lon": [-2.75, -2.75, -2.60],
        "is_valid": [True, True, True],
    })
    return preds, daily


def test_prediction_app_data_shape_and_origin_recovery():
    from tfg_aves.viz.build import prediction_app_data
    preds, daily = _app_fixtures()
    markov = {("91916A", "2014-12-31"): [40.55, -2.55]}
    data = prediction_app_data(preds, daily, markov_points=markov)
    assert [b["id"] for b in data["birds"]] == ["91916A"]
    days = data["birds"][0]["days"]
    assert len(days) == 2
    d0 = days[0]
    # origen recuperado = pred - p50 = (40.3-0.1, -2.7-0.0)
    assert d0["o"] == [40.2, -2.7]
    assert d0["year"] == 2014
    # banda [[sur,oeste],[norte,este]] con sur<norte y oeste<este
    (s, w), (n, e) = d0["band"]
    assert s < n and w < e
    # real t+1 = posición real del día siguiente (2015-01-01)
    assert d0["r"] == [40.9, -2.75]
    # punto de Markov por (ave, fecha); None si no hay
    assert d0["m"] == [40.55, -2.55]
    assert days[1]["m"] is None
    # error p50→real en km (de dist_native_km)
    assert d0["e"] == 12.0 and days[1]["e"] == 45.0
    # estado O3 por día (0 estacionario, 1 migración)
    assert d0["s"] == 0 and days[1]["s"] == 1
    # vista "Todas": rutas reales + estado por punto
    assert data["all"][0]["id"] == "91916A"
    assert len(data["all"][0]["track"]) == 2
    assert data["all"][0]["st"] == [0, 1]


def test_build_prediction_app_writes_html_with_data(tmp_path):
    from tfg_aves.viz.build import build_prediction_app
    preds, daily = _app_fixtures()
    p = build_prediction_app(preds, daily, out_dir=tmp_path)
    assert p.name == "o5_prediccion_app.html"
    html = p.read_text(encoding="utf-8")
    # placeholder sustituido por datos reales + UI presente
    assert "__PRED_DATA__" not in html
    assert '"id": "91916A"' in html or '"id":"91916A"' in html
    assert 'id="map"' in html and "Ventana de días" in html


def test_build_prediction_index_has_selector_and_iframe(tmp_path):
    from tfg_aves.viz.build import build_prediction_index
    paths = {
        "91916A": tmp_path / "o5_fig01_prediccion-91916A.html",
        "91752A": tmp_path / "o5_fig02_prediccion-91752A.html",
    }
    p = build_prediction_index(paths, out_dir=tmp_path)
    html = p.read_text(encoding="utf-8")
    assert "<select" in html and "<iframe" in html
    assert html.count("<option") == 2
    # Los src son relativos (basename), no rutas absolutas.
    assert 'src="o5_fig01_prediccion-91916A.html"' in html
    assert "o5_fig02_prediccion-91752A.html" in html
