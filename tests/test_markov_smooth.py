"""Tests de tfg_aves.markov.smooth."""
from __future__ import annotations

import numpy as np
import pytest

from tfg_aves.markov.smooth import laplace_smooth, marginal_distribution


def test_filas_suman_uno() -> None:
    counts = np.array([[[0, 1, 0], [2, 0, 0], [0, 0, 0]]], dtype=np.int32)
    P = laplace_smooth(counts, alpha=1.0)
    # 1 mes, 3 celdas, alpha=1 → cada fila debe sumar 1.
    np.testing.assert_allclose(P.sum(axis=2), 1.0, atol=1e-12)


def test_sin_entradas_cero_ni_negativas() -> None:
    counts = np.zeros((1, 3, 3), dtype=np.int32)
    P = laplace_smooth(counts, alpha=1.0)
    # Sin counts pero con alpha=1 todas las entradas son 1/3.
    np.testing.assert_allclose(P, 1.0 / 3.0)


def test_alpha_zero_lanza_value_error() -> None:
    counts = np.zeros((1, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        laplace_smooth(counts, alpha=0.0)


def test_alpha_grande_tiende_a_uniforme() -> None:
    counts = np.array([[[5, 0, 0]]], dtype=np.int32)
    counts = np.broadcast_to(counts, (1, 3, 3)).copy()
    P = laplace_smooth(counts, alpha=1e6)
    # Cada fila ≈ 1/3 modulo 1e-3.
    np.testing.assert_allclose(P, 1.0 / 3.0, atol=1e-3)


def test_marginal_distribution_mes_sin_counts_es_uniforme() -> None:
    counts = np.zeros((12, 4, 4), dtype=np.int32)
    counts[0] = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pi = marginal_distribution(counts)
    # Mes 0 tiene counts; marginal debe estar normalizada.
    np.testing.assert_allclose(pi[0].sum(), 1.0, atol=1e-12)
    # Mes 5 (índice 5) no tiene counts → uniforme 1/4.
    np.testing.assert_allclose(pi[5], 1.0 / 4.0)
