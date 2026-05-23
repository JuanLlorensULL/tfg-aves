# Rework causal de O4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar la fuga de información de O4 redefiniendo la cinemática en sentido entrante (`t-1 → t`, inercia real) y reconstruyendo el estado HMM de forma causal (emisión entrante + filtrado forward-only).

**Architecture:** `tfg_aves.ml.features` recalcula la cinemática causal desde las posiciones y aplica las dos máscaras (observación del HMM y fila de O4). Un módulo nuevo `tfg_aves.ml.hmm_causal` ajusta un HMM sobre el train temporal con la emisión causal y lo decodifica por filtrado forward-only, reutilizando las primitivas de `tfg_aves.hmm.fit` sin modificarlas. `build_o4` orquesta: cinemática → split → fit+filtrado HMM → merge → entrenar los 6 modelos. O3 queda intacto.

**Tech Stack:** Python 3.12, numpy, pandas, scipy (`logsumexp`), hmmlearn 0.3.3 (`GaussianHMM`), scikit-learn, xgboost, lightgbm, pytest, ruff. Reutiliza `tfg_aves.hmm.fit.fit_hmm_with_restarts`, `tfg_aves.hmm.features.bearing_rad`, `tfg_aves.markov.discretize.haversine_km`.

**Spec:** `docs/superpowers/specs/2026-05-23-o4-rework-causal-design.md`

---

## Estructura de ficheros

| Fichero | Responsabilidad | Acción |
|---|---|---|
| `src/tfg_aves/ml/features.py` | Cinemática causal, máscaras, `build_feature_matrix`, split temporal | Modificar |
| `src/tfg_aves/ml/hmm_causal.py` | Fit + filtrado forward-only del HMM causal | **Crear** |
| `src/tfg_aves/ml/build.py` | Orquestación de `build_o4` con HMM causal | Modificar |
| `src/tfg_aves/ml/train.py` | Trainers RF/XGB/LGBM | Sin cambios |
| `src/tfg_aves/ml/evaluate.py` | Métricas y baselines | Modificar: `state_b` → `state_b_causal` (predicciones, baselines, desglose por estado) |
| `tests/test_ml_features.py` | Tests de cinemática causal y matriz | Modificar |
| `tests/test_ml_hmm_causal.py` | Tests del filtrado forward-only (leak-free) | **Crear** |
| `tests/test_ml_build.py` | Test integración end-to-end | Modificar |

**Convenciones a respetar:** identificadores en inglés, docstrings/comentarios en castellano, commits castellano imperativos sin trailer de IA. `tfg_aves.hmm` NO se toca.

---

## Task 1: Cinemática causal en `features.py`

**Files:**
- Modify: `src/tfg_aves/ml/features.py`
- Test: `tests/test_ml_features.py`

- [ ] **Step 1: Escribir el test de `compute_causal_kinematics`**

Añadir al final de `tests/test_ml_features.py`:

```python
from tfg_aves.ml.features import compute_causal_kinematics
from tfg_aves.markov.discretize import haversine_km


def _linear_bird(bird="A", n=10, lat0=40.0, lon0=-3.0, dlat=0.1, dlon=0.0):
    """Un ave con n días contiguos, posiciones en línea recta hacia el norte."""
    dates = pd.date_range("2020-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "bird_id": bird, "date_utc": d,
            "lat": lat0 + dlat * i, "lon": lon0 + dlon * i,
            "veg_low": 0.5, "veg_high": 0.5, "daylight_hours": 12.0,
            "is_observation_valid": True,
        })
    return pd.DataFrame(rows)


def test_step_in_km_es_haversine_entrante():
    df = _linear_bird(n=5, dlat=0.1)
    out = compute_causal_kinematics(df)
    # step_in_km en t = dist(pos(t-1), pos(t)); fila 0 es NaN (no hay t-1).
    esperado = haversine_km(40.0, -3.0, 40.1, -3.0)
    assert np.isnan(out.loc[0, "step_in_km"])
    assert out.loc[1, "step_in_km"] == pytest.approx(esperado, rel=1e-6)


def test_bearing_in_es_unitario():
    df = _linear_bird(n=5, dlat=0.1, dlon=0.1)
    out = compute_causal_kinematics(df)
    s = out.loc[2, "sin_bearing_in"]
    c = out.loc[2, "cos_bearing_in"]
    assert s**2 + c**2 == pytest.approx(1.0, rel=1e-6)


def test_cos_turning_in_es_causal():
    """Cambiar pos(t+1) no debe alterar cos_turning_in(t): sólo usa t-2,t-1,t."""
    df = _linear_bird(n=6, dlat=0.1)
    out1 = compute_causal_kinematics(df)
    df2 = df.copy()
    df2.loc[4, "lat"] = 99.0  # altera pos en t=4
    out2 = compute_causal_kinematics(df2)
    # cos_turning_in en t=3 sólo depende de t=1,2,3 → no cambia.
    assert out1.loc[3, "cos_turning_in"] == pytest.approx(out2.loc[3, "cos_turning_in"])


def test_vuelo_recto_da_cos_turning_uno():
    df = _linear_bird(n=5, dlat=0.1)  # recto hacia el norte
    out = compute_causal_kinematics(df)
    # primeras dos filas NaN (necesita t-2); de la 2 en adelante recto → +1.
    assert out.loc[2, "cos_turning_in"] == pytest.approx(1.0, abs=1e-6)


def test_is_hmm_obs_valid_requiere_racha_de_3():
    df = _linear_bird(n=5, dlat=0.1)
    out = compute_causal_kinematics(df)
    # Filas 0 y 1 no tienen t-2 → inválidas; 2,3,4 válidas.
    assert not out.loc[0, "is_hmm_obs_valid"]
    assert not out.loc[1, "is_hmm_obs_valid"]
    assert out.loc[2, "is_hmm_obs_valid"]
    assert out.loc[4, "is_hmm_obs_valid"]
```

- [ ] **Step 2: Ejecutar el test y verificar que falla**

Run: `uv run pytest tests/test_ml_features.py -k "causal or bearing or turning or hmm_obs or step_in" -v`
Expected: FAIL con `ImportError: cannot import name 'compute_causal_kinematics'`.

- [ ] **Step 3: Implementar `compute_causal_kinematics` y las constantes de features**

En `src/tfg_aves/ml/features.py`, sustituir el bloque de constantes superior:

```python
_FEATURES_BASE = [
    "lat", "lon", "sin_doy", "cos_doy",
    "step_length_km", "cos_turning_angle",
    "state_b", "posterior_b_migracion",
]

_FEATURES_WIND = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
]
```

por:

```python
from tfg_aves.hmm.features import bearing_rad
from tfg_aves.markov.discretize import haversine_km

# Features cinemáticas causales (conocidas el día t). Ver R6 del spec.
FEATURES_KINEMATIC = [
    "lat", "lon", "sin_doy", "cos_doy",
    "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
]
# Features derivadas del HMM causal, añadidas por build_o4 tras el filtrado.
FEATURES_HMM = ["state_b_causal", "posterior_b_migracion_causal"]
# Conjunto supervisado completo de O4 (10 features) + bird_id en personalizado.
FEATURES_O4_CAUSAL = [*FEATURES_KINEMATIC, *FEATURES_HMM]
# Emisión del HMM causal (5, paralela al Modelo B de O3, sin rumbo absoluto).
HMM_EMISSION_COLS = [
    "step_in_km", "cos_turning_in", "veg_low", "veg_high", "daylight_hours",
]

_FEATURES_WIND = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
]
```

(El import de `assign_cell`, `_format_cell_id` ya existe arriba; deja esa línea.)

Añadir la función nueva (por ejemplo, justo antes de `assign_cells_to_features`):

```python
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
```

- [ ] **Step 4: Ejecutar los tests y verificar que pasan**

Run: `uv run pytest tests/test_ml_features.py -k "causal or bearing or turning or hmm_obs or step_in" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/features.py tests/test_ml_features.py
git commit -m "Añadir cinemática causal entrante a O4 (step_in, rumbo, giro)"
```

---

## Task 2: `build_feature_matrix` causal + máscara de fila

**Files:**
- Modify: `src/tfg_aves/ml/features.py` (`build_feature_matrix`)
- Test: `tests/test_ml_features.py`

- [ ] **Step 1: Escribir el test de la matriz causal**

Añadir a `tests/test_ml_features.py`:

```python
from tfg_aves.ml.features import build_feature_matrix, FEATURES_KINEMATIC


def _cells_grid():
    cells = []
    for i in range(78, 84):
        for j in range(-9, -3):
            cells.append({"cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j})
    return pd.DataFrame(cells)


def test_build_feature_matrix_solo_filas_candidatas():
    kin = compute_causal_kinematics(_linear_bird(n=8, dlat=0.1))
    cells = _cells_grid()
    m = build_feature_matrix(kin, cells, include_bird_id=False)
    # Candidata = is_hmm_obs_valid(t) (t>=2) y cell_id_t_next no nulo (t+1 existe).
    # Con 8 días contiguos: t en {2..6} cumplen ambas (t=7 no tiene t+1).
    assert set(m["date_utc"]) == set(pd.date_range("2020-01-01", periods=8)[2:7])
    # Las 8 features cinemáticas presentes y sin NaN.
    assert FEATURES_KINEMATIC == m.attrs["_features"]
    assert not m[FEATURES_KINEMATIC].isna().any().any()


def test_build_feature_matrix_no_cruza_gap():
    df = _linear_bird(n=8, dlat=0.1)
    # Inserta un gap: día 4 inválido (lat NaN).
    df.loc[4, ["lat", "lon"]] = [np.nan, np.nan]
    kin = compute_causal_kinematics(df)
    m = build_feature_matrix(kin, _cells_grid(), include_bird_id=False)
    # Ninguna fila candidata puede tener t, t-1, t-2 o t+1 tocando el día 4.
    fechas = set(m["date_utc"])
    base = pd.date_range("2020-01-01", periods=8)
    assert base[4] not in fechas  # el propio gap
    assert base[3] not in fechas  # su t+1 cae en gap
    assert base[5] not in fechas and base[6] not in fechas  # necesitan t-1/t-2 en gap


def test_build_feature_matrix_incluye_bird_id():
    kin = compute_causal_kinematics(_linear_bird(n=8, dlat=0.1))
    m = build_feature_matrix(kin, _cells_grid(), include_bird_id=True)
    assert m.attrs["_features"][0] == "bird_id"
    assert "bird_id" in m.columns
```

- [ ] **Step 2: Ejecutar y verificar fallo**

Run: `uv run pytest tests/test_ml_features.py -k "feature_matrix" -v`
Expected: FAIL (la firma actual de `build_feature_matrix` filtra `is_observation_valid` y usa `_FEATURES_BASE`).

- [ ] **Step 3: Reescribir `build_feature_matrix`**

Sustituir la función `build_feature_matrix` completa en `src/tfg_aves/ml/features.py` por:

```python
def build_feature_matrix(
    kin: pd.DataFrame,
    cells: pd.DataFrame,
    include_bird_id: bool,
    *,
    wind_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye la matriz de filas candidatas de O4 a partir de la cinemática causal.

    ``kin`` es la salida de ``compute_causal_kinematics`` (todas las filas con la
    cinemática entrante e ``is_hmm_obs_valid``). Pasos:
        1. Asigna ``cell_id_t`` y ``cell_id_t_next`` (target) vía ``cells``.
        2. Añade ``sin_doy``/``cos_doy``.
        3. (Opcional) fusiona viento — rama L1, no ejercitada en el rework.
        4. Filtra a filas candidatas: ``is_hmm_obs_valid`` y target válido
           (``cell_id_t_next`` no nulo ⟺ t+1 consecutivo y celda activa). Es la
           máscara de racha de 4 días (t-2, t-1, t, t+1).

    Las columnas del HMM causal (``state_b_causal``, ``posterior_b_migracion_causal``)
    NO se añaden aquí: las inserta ``build_o4`` tras ajustar y filtrar el HMM.
    El atributo ``_features`` contiene de momento sólo las cinemáticas (+bird_id).
    """
    df = assign_cells_to_features(kin, cells)
    df = add_cyclic_doy(df)

    if wind_df is not None:
        df = merge_wind_features(df, wind_df)

    df = df[df["is_hmm_obs_valid"] & df["cell_id_t_next"].notna()].copy()

    if wind_df is not None:
        df = df.dropna(subset=_FEATURES_WIND).copy()

    base = list(FEATURES_KINEMATIC)
    if wind_df is not None:
        base = [*base, *_FEATURES_WIND]
    feature_cols = ["bird_id", *base] if include_bird_id else list(base)

    keep_cols = list(dict.fromkeys([
        "bird_id", "date_utc",
        *base,
        "cell_id_t", "cell_id_t_next",
        "lat_t_next", "lon_t_next",
    ]))
    out = df[keep_cols].reset_index(drop=True)
    out.attrs["_features"] = feature_cols
    return out
```

- [ ] **Step 4: Ejecutar y verificar que pasan**

Run: `uv run pytest tests/test_ml_features.py -k "feature_matrix" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/features.py tests/test_ml_features.py
git commit -m "Reescribir build_feature_matrix para filas candidatas causales"
```

---

## Task 3: Filtrado forward-only (`hmm_causal.py`)

**Files:**
- Create: `src/tfg_aves/ml/hmm_causal.py`
- Test: `tests/test_ml_hmm_causal.py` (nuevo)

- [ ] **Step 1: Escribir el test del filtrado (incl. propiedad leak-free)**

Crear `tests/test_ml_hmm_causal.py`:

```python
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
```

- [ ] **Step 2: Ejecutar y verificar fallo**

Run: `uv run pytest tests/test_ml_hmm_causal.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tfg_aves.ml.hmm_causal'`.

- [ ] **Step 3: Crear `hmm_causal.py` con el filtrado**

Crear `src/tfg_aves/ml/hmm_causal.py`:

```python
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
```

- [ ] **Step 4: Ejecutar y verificar que pasan**

Run: `uv run pytest tests/test_ml_hmm_causal.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/hmm_causal.py tests/test_ml_hmm_causal.py
git commit -m "Implementar filtrado forward-only leak-free del HMM causal"
```

---

## Task 4: Fit + decodificado del HMM causal

**Files:**
- Modify: `src/tfg_aves/ml/hmm_causal.py`
- Test: `tests/test_ml_hmm_causal.py`

- [ ] **Step 1: Escribir los tests de fit + decode**

Añadir a `tests/test_ml_hmm_causal.py`:

```python
import pandas as pd

from tfg_aves.ml.features import compute_causal_kinematics
from tfg_aves.ml.hmm_causal import (
    build_hmm_sequences,
    decode_causal_states,
    fit_causal_hmm,
)


def _two_regime_bird(bird="A", n=40, seed=0):
    """Ave con dos regímenes: primera mitad pasos cortos, segunda larga."""
    rng = np.random.default_rng(seed)
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
    step_idx = 0  # step_in_km es la 1.ª columna de HMM_EMISSION_COLS
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
```

- [ ] **Step 2: Ejecutar y verificar fallo**

Run: `uv run pytest tests/test_ml_hmm_causal.py -k "sequences or relabel or decode" -v`
Expected: FAIL con `ImportError` de `build_hmm_sequences`, `fit_causal_hmm`, `decode_causal_states`.

- [ ] **Step 3: Implementar fit + decode en `hmm_causal.py`**

Añadir a `src/tfg_aves/ml/hmm_causal.py` (imports y funciones):

```python
import pandas as pd

from tfg_aves.hmm.fit import fit_hmm_with_restarts
from tfg_aves.ml.features import HMM_EMISSION_COLS


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
    sub_all = kin[kin["is_hmm_obs_valid"]].sort_values(["bird_id", "date_utc"])
    X_parts: list[np.ndarray] = []
    lengths: list[int] = []
    idx_parts: list[np.ndarray] = []
    for bird_id, sub in sub_all.groupby("bird_id", sort=False):
        if cutoff_by_bird is not None:
            cutoff = cutoff_by_bird.get(bird_id)
            if cutoff is None:
                continue
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
```

- [ ] **Step 4: Ejecutar y verificar que pasan**

Run: `uv run pytest tests/test_ml_hmm_causal.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/hmm_causal.py tests/test_ml_hmm_causal.py
git commit -m "Ajuste y decodificado filtrado del HMM causal sobre train temporal"
```

---

## Task 5: Orquestación de `build_o4`

**Files:**
- Modify: `src/tfg_aves/ml/build.py`
- Modify: `src/tfg_aves/ml/evaluate.py` (`state_b` → `state_b_causal`; ver Step 3 (d))
- Test: `tests/test_ml_build.py`

- [ ] **Step 1: Reescribir el fixture y el test de integración**

En `tests/test_ml_build.py`, sustituir el cuerpo del fixture `_write_synthetic_inputs` para que las features sintéticas NO dependan de las columnas contaminadas (sólo lat/lon/veg/daylight/validez) y añadir el test de columnas causales. Reemplaza el bloque del `rows.append({...})` por:

```python
            rows.append({
                "bird_id": b, "date_utc": d, "lat": lat, "lon": lon,
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
                "is_observation_valid": True,
            })
```

Y añade al final del fichero:

```python
def test_build_o4_columnas_causales(tmp_path):
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "o4"
    res = build_o4(
        features_path=feat_path, cells_path=cells_path,
        output_dir=out_dir, seed=0,
    )
    assert res.predictions_path.exists()
    assert res.metrics_path.exists()
    assert (out_dir / "model_personalizado_rf.pkl").exists()

    import joblib
    blob = joblib.load(out_dir / "model_personalizado_rf.pkl")
    cols = blob["feature_cols"]
    # Exactamente las 10 causales + bird_id; nada de columnas con fuga.
    assert "bird_id" in cols
    for c in ["step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
              "state_b_causal", "posterior_b_migracion_causal"]:
        assert c in cols
    for prohibida in ["step_length_km", "cos_turning_angle", "state_b",
                      "posterior_b_migracion"]:
        assert prohibida not in cols

    # Las predicciones llevan el estado causal (para el análisis por régimen),
    # no la columna contaminada state_b.
    preds = pd.read_parquet(res.predictions_path)
    assert "state_b_causal" in preds.columns
    assert "state_b" not in preds.columns
    modelos = preds[preds["modo"] == "personalizado"]
    assert modelos["state_b_causal"].notna().any()
```

- [ ] **Step 2: Ejecutar y verificar fallo**

Run: `uv run pytest tests/test_ml_build.py::test_build_o4_columnas_causales -v`
Expected: FAIL (el `build_o4` actual usa el pipeline contaminado).

- [ ] **Step 3: Reescribir la sección de features de `build_o4`**

En `src/tfg_aves/ml/build.py`:

(a) Actualizar imports de `features`:

```python
from .features import (
    FEATURES_HMM,
    FEATURES_KINEMATIC,
    build_feature_matrix,
    compute_causal_kinematics,
    split_temporal_per_bird,
)
from .hmm_causal import decode_causal_states, fit_causal_hmm
```

(b) Sustituir el bloque de construcción de matrices y el cuerpo del bucle de modos. Reemplazar desde el comentario `# --- Matrices de features para ambos modos ---` hasta justo antes de `# --- Baselines sobre el split temporal de O4 ---` por:

```python
    # --- Cinemática causal (mode-independent, se calcula una vez) ---
    kin = compute_causal_kinematics(features_o3)

    matrices = {
        "personalizado": build_feature_matrix(
            kin, cells, include_bird_id=True, wind_df=wind_df,
        ),
        "poblacional": build_feature_matrix(
            kin, cells, include_bird_id=False, wind_df=wind_df,
        ),
    }
    splits = {mode: split_temporal_per_bird(m) for mode, m in matrices.items()}

    # --- HMM causal: fit sobre el train temporal, decode filtrado de todo ---
    # Las filas de ambos modos son idénticas salvo bird_id, así que el cutoff
    # por ave y el decodificado se calculan una sola vez (modo poblacional).
    train_ref = splits["poblacional"][0]
    cutoff_by_bird = (
        train_ref.assign(_d=pd.to_datetime(train_ref["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )
    hmm_model, hmm_labels = fit_causal_hmm(
        kin, cutoff_by_bird, n_restarts=10, seed=seed,
    )
    states = decode_causal_states(hmm_model, hmm_labels, kin)

    def _attach_states(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(states, on=["bird_id", "date_utc"], how="left", validate="m:1")
        if merged[FEATURES_HMM].isna().any().any():
            raise AssertionError("Filas candidatas sin estado HMM causal tras el merge.")
        return merged

    metrics_per_model: dict[str, dict] = {}
    predictions_all: list[pd.DataFrame] = []
    model_paths: dict[str, Path] = {}

    for mode in _MODES:
        train, val, test = (_attach_states(d) for d in splits[mode])
        base = [*FEATURES_KINEMATIC, *FEATURES_HMM]
        feature_cols = ["bird_id", *base] if mode == "personalizado" else list(base)
        categorical_cols = ["bird_id"] if "bird_id" in feature_cols else []

        le_train = LabelEncoder().fit(train["cell_id_t_next"].astype(str))
        known_cells = set(le_train.classes_)

        X_train = train[feature_cols]
        X_val = val[feature_cols]
        X_test = test[feature_cols]
        y_train = le_train.transform(train["cell_id_t_next"].astype(str))
        y_val = _safe_encode_cells(val["cell_id_t_next"], le_train, known_cells)

        families = _FAMILIES if not with_wind else ("rf", "xgb")
        for family in families:
            model = _train_one(
                family, X_train, y_train, X_val, y_val,
                categorical_cols=categorical_cols, seed=seed,
            )
            metrics_test = evaluate_global(
                model, X_test, test, cells=cells, label_encoder_y=le_train,
            )
            metrics_test["split"] = "test"
            metrics_train = evaluate_global(
                model, X_train, train, cells=cells, label_encoder_y=le_train,
            )
            metrics_train["split"] = "train"

            key = f"{mode}_{family}"
            metrics_per_model[f"{key}::test"] = metrics_test
            metrics_per_model[f"{key}::train"] = metrics_train

            model_path = output_dir / f"model_{mode}_{family}.pkl"
            joblib.dump({
                "model": model,
                "label_encoder_y": le_train,
                "feature_cols": feature_cols,
                "categorical_cols": categorical_cols,
                "mode": mode,
                "family": family,
            }, model_path)
            model_paths[key] = model_path

            preds = predict_with_meta(
                model, X_test, test, cells=cells, label_encoder_y=le_train,
            )
            for k in list(preds.attrs):
                preds.attrs.pop(k, None)
            preds["modelo"] = family
            preds["modo"] = mode
            predictions_all.append(preds)
```

(c) El bloque de baselines (desde `# --- Baselines sobre el split temporal de O4 ---` en adelante) **no se toca**: `train_p, _val_p, test_p = splits["personalizado"]` sigue siendo válido y los baselines usan `cell_id_t`/`cell_id_t_next` (presentes en la matriz), no las features. Confirmar que no requiere los estados HMM. NOTA: `train_p`/`test_p` deben tener el estado causal adjunto para que `compute_persistence_baseline`/`compute_markov_baseline` puedan reportarlo; usa los `train`/`test` ya pasados por `_attach_states` (no los crudos de `splits`).

(d) **Actualizar `evaluate.py` a `state_b_causal`.** En `src/tfg_aves/ml/evaluate.py`, la Task 2 dejó un parche temporal que hacía `state_b` opcional (→ NaN). Sustituirlo por uso directo del estado causal real, ya que ahora las matrices y el meta llevan `state_b_causal`:
- En `predict_with_meta`: la columna de salida pasa de `state_b` a `state_b_causal`, leyendo `meta["state_b_causal"].values` (quitar el fallback NaN).
- En `compute_persistence_baseline`: salida `state_b_causal` desde `matrix_test["state_b_causal"]`.
- En `compute_markov_baseline`: salida `state_b_causal` desde `r.state_b_causal`.
- En la función de desglose por estado (firma `error_by_state(predictions, state_col="state_b")` ~línea 51): cambiar el default a `state_col="state_b_causal"`.
Quitar los comentarios "state_b es opcional durante el rework" introducidos en la Task 2. El resultado: `evaluate.py` referencia exclusivamente `state_b_causal`, sin `state_b`.

(e) **Puente de `build.py` de la Task 2.** La Task 2 ya insertó `kin = compute_causal_kinematics(features_o3)` y cambió las llamadas a `build_feature_matrix(kin, ...)`. Al aplicar (b), reconciliar: el bloque reescrito ya incluye esa línea `kin = ...`, así que no debe quedar duplicada.

- [ ] **Step 4: Ejecutar el test de integración**

Run: `uv run pytest tests/test_ml_build.py -v`
Expected: PASS (incluido `test_build_o4_columnas_causales`).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/build.py tests/test_ml_build.py
git commit -m "Orquestar build_o4 con cinemática causal y HMM filtrado"
```

---

## Task 6: Limpieza, exports y suite completa

**Files:**
- Modify: `src/tfg_aves/ml/features.py`, `src/tfg_aves/ml/__init__.py` (si procede)
- Test: toda la suite

- [ ] **Step 1: Eliminar código muerto y exponer lo nuevo**

- En `src/tfg_aves/ml/features.py`, confirmar que `_FEATURES_BASE` ya no se usa en ningún sitio (`grep -rn "_FEATURES_BASE" src tests`); si no, eliminar su definición.
- En `src/tfg_aves/ml/__init__.py`, exportar lo público nuevo si el módulo expone símbolos (revisar el patrón existente):

```python
from .features import (
    FEATURES_O4_CAUSAL,
    build_feature_matrix,
    compute_causal_kinematics,
)
from .hmm_causal import decode_causal_states, fit_causal_hmm
```

(Mantener los exports previos que sigan siendo válidos; no romper `build_o4`.)

- [ ] **Step 2: Revisar tests obsoletos en `test_ml_features.py`**

Algunos tests antiguos usaban `build_feature_matrix(features_o3, ...)` con el fixture `_build_synthetic_o3_features` (que filtraba `is_observation_valid` y columnas contaminadas). Para cada test fallido:
- Si comprueba `assign_cells_to_features`/`add_cyclic_doy`/`split_temporal_per_bird` (sin tocar features contaminadas) → dejar igual.
- Si comprueba el viejo feature set (`step_length_km`, `state_b`, etc.) → reescribir para el set causal o eliminar (esa semántica ya no existe).

Limpiezas concretas pendientes de revisiones previas:
- Migrar `test_assign_cells_adds_columns` a `_linear_bird` y retirar los helpers obsoletos `_build_synthetic_o3_features` y `_build_synthetic_cells` (esquema O3 viejo, columnas no leídas).
- Mover `_cells_grid()` junto a `_linear_bird` (los helpers preceden a sus usuarios en este fichero).
- Reforzar asserts positivos: en `test_build_feature_matrix_no_cruza_gap` añadir qué fechas SÍ están; en `test_build_feature_matrix_filters_gap_aware` documentar/asegurar que los dos primeros días de cada ave también quedan fuera.

Run: `uv run pytest tests/test_ml_features.py -v`
Expected: PASS (ajustar hasta lograrlo).

- [ ] **Step 3: Ejecutar la suite completa y el linter**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: todos los tests PASAN, ruff limpio. (Si LightGBM diverge en sintético, el test de integración no debe depender de su convergencia — sólo de que produce un modelo con `predict_proba`; ver `test_ml_train.py`.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Limpiar código muerto de O4 y verificar suite completa causal"
```

---

## Task 7: Ejecución real, sanity-checks y evidencia (`save_artifact`)

**Files:**
- Modify: `notebooks/04_eda_o4.py`
- Genera: `reports/figures|tables|captions/o4_*`, `reports/INDEX.md`

> Esta tarea NO es TDD: ejecuta el pipeline sobre datos reales y produce la evidencia para la memoria. Requiere `data/processed/o3/features.parquet`, `o2/cells.parquet`.

- [ ] **Step 1: Regenerar O4 sobre datos reales**

Run: `uv run python -c "from tfg_aves.ml import build_o4; r = build_o4(); print(r.n_rows_train, r.n_rows_val, r.n_rows_test)"`
Expected: corre sin error y regenera `data/processed/o4/{model_*.pkl, predictions_test.parquet, metrics.parquet}`. Anotar las cifras de filas.

- [ ] **Step 2: Sanity-check del HMM causal (coherencia con O3 y biología)**

En el notebook `notebooks/04_eda_o4.py`, añadir celdas que:
- Carguen el HMM causal (refiteándolo vía `compute_causal_kinematics` + `fit_causal_hmm` sobre el cutoff real) e impriman `model.means_` por estado. **Verificar §6.1:** μ[step_in_km] ≈ 5,8 km (estacionario) y ≈ 164 km (migración), próximos a O3.
- Calculen el % de migración por mes (con `state_b_causal`) y lo contrasten con la fenología (jun-jul mínimo, abr y sep-oct picos). **Si chirría con la biología, parar y diagnosticar** (regla `feedback-biological-coherence`).

- [ ] **Step 3: Artefactos `save_artifact` sobre datos limpios**

Generar (numeración `o4_figNN` continuando el índice; confirmar el último NN en `reports/INDEX.md`):
- **Impacto de la máscara de 4 días**: nº de filas candidatas causales vs. la máscara de 3 días previa (tabla).
- **Tabla comparativa final**: 6 modelos + baselines (persistencia, Markov) en test — top-1, top-3, log-loss, dist mediana km. Reemplaza la tabla con fuga.
- **Error por estado HMM**: top-1 y dist mediana de los dos ganadores condicionados a `state_b_causal` (estacionario vs migración).
- **Importancia de features** de los ganadores: ¿entra el rumbo (`sin/cos_bearing_in`)? ¿y `state_b_causal`/`posterior_b_migracion_causal`?

Cada uno con `save_artifact(..., objective="o4", decision=..., caption_es=...)`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_eda_o4.py reports/figures reports/tables reports/captions reports/INDEX.md
git commit -m "Notebook y evidencia de O4 causal (sanity HMM + comparativa limpia)"
```

---

## Task 8: Documentación de cierre y tag

**Files:**
- Modify: `reports/memoria/06_o4_ml.md`, `CLAUDE.md`
- Create: `reports/ai-log/NNNN-o4-rework-causal.md`

- [ ] **Step 1: Actualizar las notas de memoria**

En `reports/memoria/06_o4_ml.md`: sustituir los números de O4 por los causales; redactar el marco metodológico como hallazgo **positivo** (cinemática entrante = inercia real; filtrado en línea para evitar look-ahead). Tono investigador (`feedback-memoria-tone`). **No** narrar la fuga.

- [ ] **Step 2: Actualizar `CLAUDE.md`**

En la sección O4 de `CLAUDE.md`: sustituir la tabla de resultados y los hallazgos por los causales. Mantener el aviso interno de la fuga ya presente (banner) pero actualizando "rework en curso" → "rework completado, tag `v0.4.2-o4-rework-causal`".

- [ ] **Step 3: Entrada en el ai-log**

Crear `reports/ai-log/NNNN-o4-rework-causal.md` (NN = siguiente número) según `reports/ai-log/README.md`, describiendo la **ingeniería de features causales de O4** (decisión del sentido entrante, rumbo, HMM por filtrado). Tono que ensalza la autoría del autor. (Trabajo sustantivo del TFG → sí se registra.)

- [ ] **Step 4: Commit y tag**

```bash
git add reports/memoria/06_o4_ml.md reports/ai-log
git commit -m "Cerrar rework causal de O4: notas de memoria y log de IA"
git tag v0.4.2-o4-rework-causal
git log --oneline -1 && git tag --list "v0.4*"
```

(`CLAUDE.md` está en `.gitignore`; no entra en el commit.)

---

## Notas de ejecución

- **Si la máscara de 4 días reduce mucho la muestra** (Task 7 Step 1): cuantificarlo y, si fuera severo (Riesgo R-C del spec), consultar al autor antes de seguir — la decisión de degradar a `cos_turning_in` opcional es reversible y documentada.
- **`tfg_aves.hmm` no se toca** en ninguna tarea. Todo el código causal vive en `ml/`.
- **Rama L1 (viento):** `with_wind=True` queda plumbed pero no se ejercita ni se testea aquí; su rework causal es trabajo aparte (§12 del spec).
