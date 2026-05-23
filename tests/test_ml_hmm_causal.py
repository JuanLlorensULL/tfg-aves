"""Tests del HMM causal de O4: filtrado forward-only y propiedad leak-free."""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from tfg_aves.ml.hmm_causal import forward_filtered_posteriors


def _known_hmm() -> GaussianHMM:
    """HMM 2 estados, 1 feature, parámetros conocidos (no entrenado)."""
    m = GaussianHMM(n_components=2, covariance_type="diag")
    m.startprob_ = np.array([0.6, 0.4])
    m.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    m.means_ = np.array([[0.0], [10.0]])
    m.covars_ = np.array([[1.0], [1.0]])
    return m


def _reference_filtered(m, X):
    """Recursión forward de referencia en espacio lineal (secuencia única)."""
    n = len(X)
    k = m.n_components
    # covars_ de hmmlearn puede venir (k,d) o (k,d,d); extraemos la varianza
    # escalar de la única feature (d=1) de forma robusta.
    var_s = [
        float(np.atleast_1d(np.diagonal(np.atleast_2d(m.covars_[s])))[0])
        for s in range(k)
    ]

    def emit(x, s):
        mu = m.means_[s, 0]
        v = var_s[s]
        return np.exp(-0.5 * ((x[0] - mu) ** 2) / v) / np.sqrt(2 * np.pi * v)

    post = np.zeros((n, k))
    alpha = np.array([m.startprob_[s] * emit(X[0], s) for s in range(k)])
    post[0] = alpha / alpha.sum()
    for t in range(1, n):
        alpha = np.array([
            emit(X[t], j) * sum(alpha[i] * m.transmat_[i, j] for i in range(k))
            for j in range(k)
        ])
        post[t] = alpha / alpha.sum()
    return post


def test_filtrado_coincide_con_referencia():
    m = _known_hmm()
    X = np.array([[0.1], [0.0], [9.8], [10.2], [0.2]])
    got = forward_filtered_posteriors(m, X, [len(X)])
    ref = _reference_filtered(m, X)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_filtrado_es_leak_free():
    """El posterior filtrado en t no cambia al alterar observaciones futuras."""
    m = _known_hmm()
    X = np.array([[0.0], [0.1], [10.0], [9.9], [0.0], [0.1]])
    post = forward_filtered_posteriors(m, X, [len(X)])
    t = 2
    X2 = X.copy()
    X2[t + 1:] = 999.0  # destroza el futuro
    post2 = forward_filtered_posteriors(m, X2, [len(X)])
    np.testing.assert_allclose(post[: t + 1], post2[: t + 1], rtol=1e-10)


def test_suavizado_si_cambia_con_el_futuro():
    """Contraste: predict_proba (forward-backward) SÍ cambia con el futuro."""
    m = _known_hmm()
    X = np.array([[0.0], [0.1], [10.0], [9.9], [0.0], [0.1]])
    sm = m.predict_proba(X)
    X2 = X.copy()
    X2[3:] = 999.0
    sm2 = m.predict_proba(X2)
    assert not np.allclose(sm[:3], sm2[:3])


def test_filtrado_respeta_segmentos():
    """Con dos segmentos, cada uno reinicia en startprob_."""
    m = _known_hmm()
    X = np.array([[0.0], [10.0], [0.0], [10.0]])
    post = forward_filtered_posteriors(m, X, [2, 2])
    # La primera fila de cada segmento usa sólo startprob_ + su emisión.
    one_step = forward_filtered_posteriors(m, X[0:1], [1])
    np.testing.assert_allclose(post[0], one_step[0])
    np.testing.assert_allclose(post[2], one_step[0])
