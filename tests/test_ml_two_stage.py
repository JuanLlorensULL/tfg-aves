"""Tests de las funciones puras de L2 (two_stage)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.ml import two_stage as ts


def test_derive_y_move_basic():
    df = pd.DataFrame({
        "cell_id_t":      ["A", "A", "B", "B", "A"],
        "cell_id_t_next": ["A", "B", "B", "A", "A"],
    })
    result = ts.derive_y_move(df).tolist()
    assert result == [False, True, False, True, False]


def test_combine_soft_extremes():
    # p_move=0 -> toda la masa en cell_t; p_move=1 -> nada en cell_t
    p_move = np.array([0.0, 1.0])
    p_2b = np.array([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]])
    cell_t_idx = np.array([0, 0])
    out = ts.combine_soft(p_move, p_2b, cell_t_idx, n_classes=3)
    assert out[0, 0] == 1.0
    assert out[0, 1] == 0.0 and out[0, 2] == 0.0
    assert out[1, 0] == 0.0  # nada en cell_t cuando p_move=1


def test_combine_soft_sums_to_one():
    rng = np.random.default_rng(0)
    p_move = rng.random(50)
    p_2b = rng.random((50, 8))
    p_2b /= p_2b.sum(axis=1, keepdims=True)
    cell_t_idx = rng.integers(0, 8, size=50)
    out = ts.combine_soft(p_move, p_2b, cell_t_idx, n_classes=8)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_combine_soft_renormalizes_when_p2b_has_mass_on_cellt():
    # p_2b asigna 0.3 a cell_t; con p_move=0.5 la suma debe seguir siendo 1
    # (NO 1 - 0.5*0.3 = 0.85)
    p_move = np.array([0.5])
    p_2b = np.array([[0.3, 0.5, 0.2]])
    cell_t_idx = np.array([0])
    out = ts.combine_soft(p_move, p_2b, cell_t_idx, n_classes=3)
    assert np.isclose(out.sum(), 1.0, atol=1e-9)
    assert np.isclose(out[0, 0], 0.5, atol=1e-9)  # 1 - p_move


def test_combine_soft_handles_p2b_cellt_near_one():
    p_move = np.array([0.5])
    p_2b = np.array([[1.0 - 1e-10, 5e-11, 5e-11]])
    cell_t_idx = np.array([0])
    out = ts.combine_soft(p_move, p_2b, cell_t_idx, n_classes=3)
    assert np.all(np.isfinite(out))
    assert np.isclose(out.sum(), 1.0, atol=1e-6)


def test_combine_hard_threshold():
    # fila 0: p_move<tau -> predice cell_t (idx 0)
    # fila 1: p_move>=tau -> predice argmax(p_2b) (idx 2)
    p_move = np.array([0.1, 0.9])
    p_2b = np.array([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
    cell_t_idx = np.array([0, 0])
    out = ts.combine_hard(p_move, p_2b, cell_t_idx, tau=0.5)
    assert out.argmax(axis=1).tolist() == [0, 2]


def test_combine_hard_is_distribution():
    rng = np.random.default_rng(1)
    p_move = rng.random(20)
    p_2b = rng.random((20, 5))
    p_2b /= p_2b.sum(axis=1, keepdims=True)
    cell_t_idx = rng.integers(0, 5, size=20)
    out = ts.combine_hard(p_move, p_2b, cell_t_idx, tau=0.5)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(out > 0.0)  # clipping garantiza positividad estricta


def test_sweep_tau_returns_best():
    # Construimos un val donde tau=0.5 maximiza top-1.
    # 4 filas: 2 estáticas (y=cell_t) con p_move bajo, 2 que se mueven
    # (y=argmax p_2b) con p_move alto.
    p_move = np.array([0.2, 0.2, 0.8, 0.8])
    p_2b = np.array([
        [0.1, 0.8, 0.1],   # argmax idx 1
        [0.1, 0.8, 0.1],   # argmax idx 1
        [0.1, 0.1, 0.8],   # argmax idx 2
        [0.1, 0.1, 0.8],   # argmax idx 2
    ])
    cell_t_idx = np.array([0, 0, 0, 0])
    y_true_idx = np.array([0, 0, 2, 2])  # estáticas->cell_t(0), móviles->idx2
    tau_star, table = ts.sweep_tau(
        p_move, p_2b, cell_t_idx, y_true_idx, n_classes=3,
        taus=(0.1, 0.5, 0.7),
    )
    assert tau_star == 0.5
    assert set(table.columns) == {"tau", "top1"}
    assert len(table) == 3


def test_expand_proba_to_full():
    # clf_dest sólo conoce clases ["B", "C"]; el espacio completo es
    # ["A", "B", "C"]. La columna "A" debe quedar a 0.
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    classes_str = np.array(["B", "C"])
    classes_full = np.array(["A", "B", "C"])
    out = ts.expand_proba_to_full(proba, classes_str, classes_full)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 0], 0.0)          # "A" ausente -> 0
    assert np.allclose(out[:, 1], [0.7, 0.4])   # "B"
    assert np.allclose(out[:, 2], [0.3, 0.6])   # "C"


def test_predictions_from_proba_columns_and_top1():
    proba_full = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    classes_full = np.array(["A", "B", "C"])
    meta = pd.DataFrame({
        "bird_id": ["x", "x"],
        "date_utc": pd.to_datetime(["2010-01-01", "2010-01-02"]).date,
        "cell_id_t_next": ["B", "A"],
        "lat_t_next": [10.0, 10.0],
        "lon_t_next": [0.0, 0.0],
        "state_b_causal": [1, 0],
    })
    cells = pd.DataFrame({
        "cell_id": ["A", "B", "C"],
        "lat_c": [10.0, 10.5, 11.0],
        "lon_c": [0.0, 0.0, 0.0],
    })
    preds = ts.predictions_from_proba(proba_full, classes_full, meta, cells=cells)
    assert list(preds["pred_cell_top1"]) == ["B", "A"]
    assert preds["true_cell"].tolist() == ["B", "A"]
    assert preds.attrs["_proba"].shape == (2, 3)
    assert "pred_dist_km" in preds.columns
    assert "state_b_causal" in preds.columns


def _toy_binary(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "lat": rng.normal(50, 5, n),
        "lon": rng.normal(0, 5, n),
        "state_b_causal": rng.integers(0, 2, n),
    })
    # y_move correlado con state_b_causal para que el modelo aprenda algo
    y = ((X["state_b_causal"] == 1) | (rng.random(n) < 0.1)).astype(int).to_numpy()
    return X, y


def test_train_move_rf_predicts_proba():
    X, y = _toy_binary()
    model = ts.train_move_rf(X, y, seed=0)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert list(model.classes_) == [0, 1]


def test_train_move_xgb_predicts_proba():
    X, y = _toy_binary()
    model = ts.train_move_xgb(X, y, seed=0)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_calibrate_prefit_returns_calibrated():
    X, y = _toy_binary(n=400, seed=1)
    Xv, yv = _toy_binary(n=160, seed=2)
    base = ts.train_move_rf(X, y, seed=1)
    cal = ts.calibrate_prefit(base, Xv, yv)
    proba = cal.predict_proba(Xv)
    assert proba.shape == (len(Xv), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
