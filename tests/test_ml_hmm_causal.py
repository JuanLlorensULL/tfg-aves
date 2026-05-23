"""Tests del HMM causal de O4: filtrado forward-only y propiedad leak-free."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from tfg_aves.ml.features import HMM_EMISSION_COLS, compute_causal_kinematics
from tfg_aves.ml.hmm_causal import (
    build_hmm_sequences,
    decode_causal_states,
    fit_causal_hmm,
    forward_filtered_posteriors,
)


def _known_hmm() -> GaussianHMM:
    """HMM 2 estados, 1 feature, parámetros conocidos (no entrenado)."""
    m = GaussianHMM(n_components=2, covariance_type="diag")
    m.n_features = 1
    m.startprob_ = np.array([0.5, 0.5])
    m.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    m.means_ = np.array([[0.0], [4.0]])
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
    X = np.array([[0.1], [0.0], [3.8], [4.2], [0.2]])
    got = forward_filtered_posteriors(m, X, [len(X)])
    ref = _reference_filtered(m, X)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_filtrado_es_leak_free():
    """El posterior filtrado en t no cambia al alterar observaciones futuras."""
    m = _known_hmm()
    X = np.array([[0.0], [0.1], [4.0], [3.9], [0.0], [0.1]])
    post = forward_filtered_posteriors(m, X, [len(X)])
    t = 2
    X2 = X.copy()
    X2[t + 1:] = 999.0  # destroza el futuro
    post2 = forward_filtered_posteriors(m, X2, [len(X)])
    np.testing.assert_allclose(post[: t + 1], post2[: t + 1], rtol=1e-10)


def test_suavizado_si_cambia_con_el_futuro():
    """Contraste: predict_proba (forward-backward) SÍ usa el futuro.

    La observación presente (t=0) está en el punto medio entre las dos
    medias (2.0), así que es ambigua: el suavizado fija su estado mirando
    el futuro; el filtrado no puede.
    """
    m = _known_hmm()
    X = np.array([[2.0], [2.0]])
    X2 = np.array([[2.0], [4.0]])  # cambia sólo el futuro (t=1)
    sm = m.predict_proba(X)
    sm2 = m.predict_proba(X2)
    assert not np.allclose(sm[0], sm2[0])  # suavizado del presente cambia
    # Filtrado: P(s0 | o0) NO cambia porque o0 es idéntico.
    f = forward_filtered_posteriors(m, X, [2])
    f2 = forward_filtered_posteriors(m, X2, [2])
    np.testing.assert_allclose(f[0], f2[0])


def test_filtrado_respeta_segmentos():
    """Con dos segmentos, cada uno reinicia en startprob_."""
    m = _known_hmm()
    X = np.array([[0.0], [4.0], [0.0], [4.0]])
    post = forward_filtered_posteriors(m, X, [2, 2])
    # La primera fila de cada segmento usa sólo startprob_ + su emisión.
    one_step = forward_filtered_posteriors(m, X[0:1], [1])
    np.testing.assert_allclose(post[0], one_step[0])
    np.testing.assert_allclose(post[2], one_step[0])


def test_filtrado_rechaza_lengths_inconsistentes():
    m = _known_hmm()
    X = np.array([[0.0], [4.0], [0.0]])  # 3 filas
    with pytest.raises(ValueError):
        forward_filtered_posteriors(m, X, [2])  # suma 2 != 3


def _two_regime_bird(bird="A", n=40, seed=0):
    """Ave con dos regímenes: primera mitad pasos cortos, segunda larga."""
    dates = pd.date_range("2020-01-01", periods=n)
    lat, lon = 40.0, -3.0
    rows = []
    for i, d in enumerate(dates):
        step_deg = 0.01 if i < n // 2 else 0.6  # corto vs largo
        lat += step_deg
        rows.append({
            "bird_id": bird, "date_utc": d, "lat": lat, "lon": lon,
            "veg_low": 0.5, "veg_high": 0.5, "daylight_hours": 12.0,
            "is_observation_valid": True,
        })
    return pd.DataFrame(rows)


def test_build_hmm_sequences_solo_dias_validos():
    kin = compute_causal_kinematics(_two_regime_bird(n=20))
    X, lengths, idx = build_hmm_sequences(kin, cutoff_by_bird=None)
    n_valid = int(kin["is_hmm_obs_valid"].sum())
    assert len(X) == n_valid == sum(lengths)
    assert X.shape[1] == 5  # HMM_EMISSION_COLS


def test_fit_causal_hmm_relabel_por_step():
    kin = compute_causal_kinematics(_two_regime_bird(n=60))
    model, label_map = fit_causal_hmm(kin, cutoff_by_bird=None, n_restarts=4, seed=0)
    # El estado con mayor μ[step_in_km] debe ser 'migración'.
    step_idx = HMM_EMISSION_COLS.index("step_in_km")
    migr_state = int(np.argmax(model.means_[:, step_idx]))
    assert label_map[migr_state] == "migración"


def test_decode_causal_states_columnas_y_rango():
    kin = compute_causal_kinematics(_two_regime_bird(n=60))
    model, label_map = fit_causal_hmm(kin, cutoff_by_bird=None, n_restarts=4, seed=0)
    states = decode_causal_states(model, label_map, kin)
    assert set(states.columns) == {
        "bird_id", "date_utc", "state_b_causal", "posterior_b_migracion_causal",
    }
    assert states["state_b_causal"].isin([0, 1]).all()
    assert (states["posterior_b_migracion_causal"] >= 0).all()
    assert (states["posterior_b_migracion_causal"] <= 1).all()
    # Una fila por día HMM-válido.
    assert len(states) == int(kin["is_hmm_obs_valid"].sum())


def test_build_hmm_sequences_respeta_cutoff():
    """El cutoff por ave excluye días posteriores; un ave ausente del dict se omite."""
    kin = compute_causal_kinematics(
        pd.concat(
            [_two_regime_bird("A", n=30), _two_regime_bird("B", n=30)],
            ignore_index=True,
        )
    )
    cutoff = {"A": pd.Timestamp("2020-01-15")}  # B ausente del dict
    X, lengths, idx = build_hmm_sequences(kin, cutoff_by_bird=cutoff)
    aves = kin.loc[idx, "bird_id"]
    fechas = pd.to_datetime(kin.loc[idx, "date_utc"])
    assert (aves == "A").all()  # B se omite (no está en el dict)
    assert (fechas <= pd.Timestamp("2020-01-15")).all()  # respeta el cutoff de A
    assert len(X) == sum(lengths) == len(idx)
    assert len(X) > 0  # debe quedar al menos un día válido de A antes del cutoff
