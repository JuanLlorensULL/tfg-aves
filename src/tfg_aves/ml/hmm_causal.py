"""HMM causal de O4: emisión entrante + decodificado por filtrado forward-only.

Reutiliza las primitivas de ``tfg_aves.hmm.fit`` (ajuste) sin modificar O3. El
estado se decodifica con el posterior FILTRADO ``P(estado_t | obs_1..t)``, que
sólo usa observaciones hasta t (sin el paso backward del suavizado), para no
introducir look-ahead en una feature predictiva.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from tfg_aves.hmm.fit import fit_hmm_with_restarts
from tfg_aves.ml.features import HMM_EMISSION_COLS


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
    if sum(lengths) != len(X):
        raise ValueError(
            f"sum(lengths)={sum(lengths)} no coincide con len(X)={len(X)}.",
        )
    # log(0) = -inf en transiciones/arranques nulos es correcto: logsumexp lo maneja.
    with np.errstate(divide="ignore"):
        log_start = np.log(model.startprob_)
        log_trans = np.log(model.transmat_)
    log_emit_all = _diag_log_emission(X, model.means_, model.covars_)

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


def build_hmm_sequences(
    kin: pd.DataFrame,
    cutoff_by_bird: dict[str, pd.Timestamp] | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Construye (X, lengths, row_index) sobre los días HMM-válidos.

    Segmenta por ave en runs consecutivos de calendario (un hueco corta la
    secuencia). Si ``cutoff_by_bird`` se da, sólo incluye días con
    ``date_utc <= cutoff`` (ajuste sobre train); si es None, todos (decode).
    ``row_index`` son los índices de ``kin`` para mapear el resultado de vuelta.
    """
    # Segmentación por runs consecutivos, paralela a tfg_aves.hmm.fit.build_sequences
    # (no se toca O3); aquí además devolvemos row_index para el mapeo posterior.
    sub_all = kin[kin["is_hmm_obs_valid"]].sort_values(["bird_id", "date_utc"])
    X_parts: list[np.ndarray] = []
    lengths: list[int] = []
    idx_parts: list[np.ndarray] = []
    for bird_id, sub in sub_all.groupby("bird_id", sort=False):
        if cutoff_by_bird is not None:
            cutoff = cutoff_by_bird.get(bird_id)
            if cutoff is None:
                continue
            cutoff = pd.Timestamp(cutoff)
            sub = sub[pd.to_datetime(sub["date_utc"]) <= cutoff]
        if len(sub) == 0:
            continue
        dates = pd.to_datetime(sub["date_utc"]).reset_index(drop=True)
        gap = (dates.diff() != pd.Timedelta(days=1)).cumsum()
        for _, seg in sub.groupby(gap.values):
            X_parts.append(seg[HMM_EMISSION_COLS].to_numpy(dtype=np.float64))
            lengths.append(len(seg))
            idx_parts.append(seg.index.to_numpy())
    if not X_parts:
        return np.zeros((0, len(HMM_EMISSION_COLS))), [], np.array([], dtype=int)
    return np.vstack(X_parts), lengths, np.concatenate(idx_parts)


def _relabel_by_step(model: GaussianHMM) -> dict[int, str]:
    """El estado con menor μ[step_in_km] es 'estacionario'; el otro, 'migración'."""
    step_idx = HMM_EMISSION_COLS.index("step_in_km")
    estac = int(np.argmin(model.means_[:, step_idx]))
    return {
        i: ("estacionario" if i == estac else "migración")
        for i in range(model.n_components)
    }


def fit_causal_hmm(
    kin: pd.DataFrame,
    cutoff_by_bird: dict[str, pd.Timestamp] | None,
    *,
    n_restarts: int = 10,
    seed: int = 0,
) -> tuple[GaussianHMM, dict[int, str]]:
    """Ajusta el HMM causal (emisión R7) sobre los días HMM-válidos de train."""
    X, lengths, _ = build_hmm_sequences(kin, cutoff_by_bird=cutoff_by_bird)
    model, _, _ = fit_hmm_with_restarts(
        X, lengths, n_components=2, n_restarts=n_restarts, random_state=seed,
    )
    return model, _relabel_by_step(model)


def decode_causal_states(
    model: GaussianHMM,
    label_map: dict[int, str],
    kin: pd.DataFrame,
) -> pd.DataFrame:
    """Decodifica TODOS los días HMM-válidos por filtrado forward-only.

    Devuelve un DataFrame (bird_id, date_utc, state_b_causal,
    posterior_b_migracion_causal); state 1 = migración.
    """
    X, lengths, row_index = build_hmm_sequences(kin, cutoff_by_bird=None)
    migr_idx = next(i for i, lab in label_map.items() if lab == "migración")
    post = forward_filtered_posteriors(model, X, lengths)
    raw_state = np.argmax(post, axis=1)
    res = kin.loc[row_index, ["bird_id", "date_utc"]].copy()
    res["state_b_causal"] = np.where(raw_state == migr_idx, 1, 0).astype(np.int8)
    res["posterior_b_migracion_causal"] = post[:, migr_idx]
    return res.reset_index(drop=True)
