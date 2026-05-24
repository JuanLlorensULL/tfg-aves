"""Tests unitarios del módulo de regresión de cuantiles (L3)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_o4_l3v1_dir_exists():
    from tfg_aves.ml._paths import O4_L3V1_DIR, O4_OUT_DIR
    assert O4_L3V1_DIR == O4_OUT_DIR / "l3_v1"


def test_derive_displacement_target():
    from tfg_aves.ml.quantile import derive_displacement_target
    m = pd.DataFrame({
        "lat": [40.0, 41.0], "lon": [-3.0, -2.5],
        "lat_t_next": [40.5, 41.2], "lon_t_next": [-2.0, -2.7],
    })
    out = derive_displacement_target(m)
    assert np.allclose(out["y_dlat"], [0.5, 0.2])
    assert np.allclose(out["y_dlon"], [1.0, -0.2])


def test_fit_quantile_axis_shapes_and_order():
    from tfg_aves.ml.quantile import QUANTILES, fit_quantile_axis
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=300)
    Xv = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    yv = (Xv["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=80)
    models = fit_quantile_axis(X, y, Xv, yv, seed=0)
    assert set(models) == set(QUANTILES)
    p10 = models[0.10].predict(Xv)
    p90 = models[0.90].predict(Xv)
    assert p10.shape == (80,)
    # En media, el cuantil 90 está por encima del 10.
    assert float(np.mean(p90 >= p10)) > 0.9


class _Const:
    """Modelo de juguete que predice una constante (para tests de monotonía)."""
    def __init__(self, v: float) -> None:
        self.v = v
    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.v, dtype=float)


def test_predict_quantiles_monotonic_and_crossings():
    from tfg_aves.ml.quantile import predict_quantiles
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    # lat: cuantiles DESORDENADOS (1.0, 0.0, 0.5) -> cruce en las 3 filas.
    models_lat = {0.10: _Const(1.0), 0.50: _Const(0.0), 0.90: _Const(0.5)}
    # lon: cuantiles ya ordenados (-0.5, 0.0, 0.5) -> sin cruces.
    models_lon = {0.10: _Const(-0.5), 0.50: _Const(0.0), 0.90: _Const(0.5)}
    preds, crossings = predict_quantiles(models_lat, models_lon, X)
    assert (preds["dlat_p10"] <= preds["dlat_p50"]).all()
    assert (preds["dlat_p50"] <= preds["dlat_p90"]).all()
    assert (preds["dlon_p10"] <= preds["dlon_p50"]).all()
    assert crossings["lat"] == 3
    assert crossings["lon"] == 0
    assert len(preds) == 3
