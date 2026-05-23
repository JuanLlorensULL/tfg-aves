"""HMM causal de O4: emisión entrante + decodificado por filtrado forward-only.

Reutiliza las primitivas de ``tfg_aves.hmm.fit`` (ajuste) sin modificar O3. El
estado se decodifica con el posterior FILTRADO ``P(estado_t | obs_1..t)``, que
sólo usa observaciones hasta t (sin el paso backward del suavizado), para no
introducir look-ahead en una feature predictiva.
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp


def _diag_log_emission(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log p(x_t | estado) gaussiana diagonal, devuelto como (n_obs, n_states).

    ``covars`` puede venir de hmmlearn como (k, d) varianzas o (k, d, d)
    matrices diagonales; se normaliza a varianzas (k, d).
    """
    if covars.ndim == 3:
        var = np.stack([np.diag(c) for c in covars])
    else:
        var = covars
    n, d = X.shape
    k = means.shape[0]
    log_prob = np.empty((n, k))
    const = d * np.log(2.0 * np.pi)
    for s in range(k):
        diff = X - means[s]
        log_prob[:, s] = -0.5 * (
            const + np.sum(np.log(var[s])) + np.sum(diff**2 / var[s], axis=1)
        )
    return log_prob


def forward_filtered_posteriors(
    model: GaussianHMM,
    X: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """Posterior filtrado ``P(estado_t | obs_1..t)`` por fila, segmento a segmento.

    Usa sólo atributos públicos del modelo ajustado (``startprob_``,
    ``transmat_``, ``means_``, ``covars_``); la recursión forward es propia.
    Cada segmento de ``lengths`` reinicia en ``startprob_``.
    """
    if len(X) == 0:
        return np.zeros((0, model.n_components))
    log_start = np.log(model.startprob_)
    log_trans = np.log(model.transmat_)
    # Acceso robusto a las varianzas: el property covars_ de hmmlearn requiere
    # que n_features esté inicializado (sólo ocurre tras fit()). En modelos
    # entrenados funciona siempre; para tests con parámetros fijos se usa el
    # atributo interno como alternativa.
    try:
        covars = model.covars_
    except AttributeError:
        covars = model._covars_
    log_emit_all = _diag_log_emission(X, model.means_, covars)

    out = np.empty((len(X), model.n_components))
    pos = 0
    for length in lengths:
        log_emit = log_emit_all[pos : pos + length]
        log_alpha = np.empty((length, model.n_components))
        log_alpha[0] = log_start + log_emit[0]
        for t in range(1, length):
            log_alpha[t] = log_emit[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_trans, axis=0
            )
        out[pos : pos + length] = np.exp(
            log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        )
        pos += length
    return out
