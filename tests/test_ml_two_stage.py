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
