"""HMM causal: cinemática entrante + decodificado por filtrado forward-only.

Propiedad de O3. La emisión entrante (t-1 -> t) y el filtrado
``P(estado_t | obs_1..t)`` garantizan que el estado no observa el futuro,
condición para que alimente a O4 como feature predictiva sin fuga.
Generalizado a los dos modelos de la ablación de O3:
  - Modelo A (cinemático): emisión ``HMM_EMISSION_COLS_A``.
  - Modelo B (cinemático + contexto): emisión ``HMM_EMISSION_COLS_B`` (= L3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from tfg_aves.hmm.features import bearing_rad
from tfg_aves.hmm.fit import fit_hmm_with_restarts
from tfg_aves.markov.discretize import haversine_km

HMM_EMISSION_COLS_A = ["step_in_km", "cos_turning_in"]
HMM_EMISSION_COLS_B = ["step_in_km", "cos_turning_in", "veg_low", "veg_high", "daylight_hours"]


def compute_causal_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Añade cinemática ENTRANTE (tramo t-1 → t) y la máscara is_hmm_obs_valid.

    - ``step_in_km``      = haversine(pos(t-1), pos(t)).
    - ``sin/cos_bearing_in`` = sin/cos del rumbo del tramo t-1 → t (dirección).
    - ``cos_turning_in``  = cos(rumbo(t-1→t) − rumbo(t-2→t-1)) (variabilidad del rumbo).
    - ``is_hmm_obs_valid``: día t con (t-2, t-1, t) válidos y consecutivos en
      calendario y con veg_low/veg_high/daylight_hours presentes (lo que exige
      la emisión del HMM causal).

    Toda la cinemática es función exclusiva de posiciones hasta t inclusive:
    no hay look-ahead. Las filas sin la racha necesaria reciben NaN.
    """
    out = df.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    out["_date_dt"] = pd.to_datetime(out["date_utc"])

    valid = out["lat"].notna() & out["lon"].notna()
    same1 = out["bird_id"].shift(1) == out["bird_id"]
    same2 = out["bird_id"].shift(2) == out["bird_id"]
    consec1 = (out["_date_dt"] - out["_date_dt"].shift(1)) == pd.Timedelta(days=1)
    consec2 = (out["_date_dt"].shift(1) - out["_date_dt"].shift(2)) == pd.Timedelta(days=1)
    valid1 = valid.shift(1).fillna(False).astype(bool)
    valid2 = valid.shift(2).fillna(False).astype(bool)

    lat_t, lon_t = out["lat"].to_numpy(), out["lon"].to_numpy()
    lat_1, lon_1 = out["lat"].shift(1).to_numpy(), out["lon"].shift(1).to_numpy()
    lat_2, lon_2 = out["lat"].shift(2).to_numpy(), out["lon"].shift(2).to_numpy()

    # Tramo entrante t-1 → t: necesita t-1 válido y consecutivo.
    mask_in = (valid & valid1 & same1 & consec1).to_numpy()
    step_in = np.full(len(out), np.nan)
    step_in[mask_in] = np.asarray(haversine_km(lat_1, lon_1, lat_t, lon_t))[mask_in]

    # Los rumbos/distancias se calculan sobre el array completo (coords NaN
    # producen NaN); sólo las posiciones enmascaradas se escriben en la salida.
    # Patrón deliberado, paralelo a tfg_aves.hmm.features.compute_observation_features.
    bearing_in = np.asarray(bearing_rad(lat_1, lon_1, lat_t, lon_t))
    sin_b = np.full(len(out), np.nan)
    cos_b = np.full(len(out), np.nan)
    sin_b[mask_in] = np.sin(bearing_in)[mask_in]
    cos_b[mask_in] = np.cos(bearing_in)[mask_in]

    # Giro causal: necesita además t-2 válido y consecutivo.
    mask_turn = mask_in & (valid2 & same2 & consec2).to_numpy()
    bearing_prev = np.asarray(bearing_rad(lat_2, lon_2, lat_1, lon_1))
    turning = (bearing_in - bearing_prev + np.pi) % (2.0 * np.pi) - np.pi
    cos_turn = np.full(len(out), np.nan)
    cos_turn[mask_turn] = np.cos(turning)[mask_turn]

    out["step_in_km"] = step_in
    out["sin_bearing_in"] = sin_b
    out["cos_bearing_in"] = cos_b
    out["cos_turning_in"] = cos_turn

    veg_ok = (
        out["veg_low"].notna() & out["veg_high"].notna() & out["daylight_hours"].notna()
    ).to_numpy()
    out["is_hmm_obs_valid"] = mask_turn & veg_ok

    return out.drop(columns=["_date_dt"])


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
    emission_cols: list[str] = HMM_EMISSION_COLS_B,
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
            X_parts.append(seg[emission_cols].to_numpy(dtype=np.float64))
            lengths.append(len(seg))
            idx_parts.append(seg.index.to_numpy())
    if not X_parts:
        return np.zeros((0, len(emission_cols))), [], np.array([], dtype=int)
    return np.vstack(X_parts), lengths, np.concatenate(idx_parts)


def _relabel_by_step(model: GaussianHMM, emission_cols: list[str]) -> dict[int, str]:
    """El estado con menor μ[step_in_km] es 'estacionario'; el otro, 'migración'."""
    step_idx = emission_cols.index("step_in_km")
    estac = int(np.argmin(model.means_[:, step_idx]))
    return {
        i: ("estacionario" if i == estac else "migración")
        for i in range(model.n_components)
    }


def fit_causal_hmm(
    kin: pd.DataFrame,
    cutoff_by_bird: dict[str, pd.Timestamp] | None,
    *,
    emission_cols: list[str] = HMM_EMISSION_COLS_B,
    n_restarts: int = 10,
    seed: int = 0,
) -> tuple[GaussianHMM, dict[int, str]]:
    """Ajusta el HMM causal sobre los días HMM-válidos de train."""
    X, lengths, _ = build_hmm_sequences(
        kin, cutoff_by_bird=cutoff_by_bird, emission_cols=emission_cols,
    )
    model, _, _ = fit_hmm_with_restarts(
        X, lengths, n_components=2, n_restarts=n_restarts, random_state=seed,
    )
    return model, _relabel_by_step(model, emission_cols)


def decode_causal_states(
    model: GaussianHMM,
    label_map: dict[int, str],
    kin: pd.DataFrame,
    *,
    emission_cols: list[str] = HMM_EMISSION_COLS_B,
    suffix: str = "b",
) -> pd.DataFrame:
    """Decodifica todos los días HMM-válidos por filtrado forward-only.

    Devuelve (bird_id, date_utc, state_<suffix>_causal,
    posterior_<suffix>_estacionario, posterior_<suffix>_migracion).
    """
    X, lengths, row_index = build_hmm_sequences(
        kin, cutoff_by_bird=None, emission_cols=emission_cols,
    )
    migr_idx = next(i for i, lab in label_map.items() if lab == "migración")
    estac_idx = next(i for i, lab in label_map.items() if lab == "estacionario")
    post = forward_filtered_posteriors(model, X, lengths)
    raw_state = np.argmax(post, axis=1)
    res = kin.loc[row_index, ["bird_id", "date_utc"]].copy()
    res[f"state_{suffix}_causal"] = np.where(raw_state == migr_idx, 1, 0).astype(np.int8)
    res[f"posterior_{suffix}_estacionario"] = post[:, estac_idx]
    res[f"posterior_{suffix}_migracion"] = post[:, migr_idx]
    return res.reset_index(drop=True)
