"""Tests de tfg_aves.markov.predict."""
from __future__ import annotations

import numpy as np
import pytest

from tfg_aves.markov.predict import (
    predict_distribution,
    prediction_distance_km,
    topk_from_distribution,
)


def _toy_p_matrix() -> tuple[np.ndarray, list[str]]:
    """Matriz de 12 meses × 3 celdas con valores conocidos para mes 0."""
    cells = ["a", "b", "c"]
    P = np.zeros((12, 3, 3), dtype=np.float64)
    P[0] = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]])
    return P, cells


def test_predict_distribution_celda_conocida() -> None:
    P, cells = _toy_p_matrix()
    d = predict_distribution(P, cell_from="a", month_int=1, cells=cells)
    np.testing.assert_allclose(d, [0.7, 0.2, 0.1])


def test_predict_distribution_celda_nueva_usa_marginal() -> None:
    P, cells = _toy_p_matrix()
    marginal = np.zeros((12, 3))
    marginal[0] = np.array([0.1, 0.1, 0.8])
    d = predict_distribution(P, cell_from="zz", month_int=1, cells=cells, marginal=marginal)
    np.testing.assert_allclose(d, [0.1, 0.1, 0.8])


def test_predict_distribution_celda_nueva_sin_marginal_lanza() -> None:
    P, cells = _toy_p_matrix()
    with pytest.raises(KeyError):
        predict_distribution(P, cell_from="zz", month_int=1, cells=cells, marginal=None)


def test_topk_ordenado_descendente() -> None:
    cells = ["a", "b", "c", "d"]
    dist = np.array([0.1, 0.5, 0.3, 0.1])
    top3 = topk_from_distribution(dist, k=3, cells=cells)
    assert top3 == ["b", "c", "a"] or top3 == ["b", "c", "d"]
    # Top-1 siempre "b" (max).
    assert top3[0] == "b"


def test_prediction_distance_centroide_coincide_con_real() -> None:
    # cell "0_0" con cell_deg=0.5 → centroide (0.25, 0.25).
    # Si real está en (0.25, 0.25), distancia = 0.
    d = prediction_distance_km(
        cell_pred="0_0", lat_real=0.25, lon_real=0.25, cell_deg=0.5
    )
    assert d == pytest.approx(0.0, abs=1e-6)
