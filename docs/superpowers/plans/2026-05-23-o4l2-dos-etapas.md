# L2 — Modelo de dos etapas (O4 causal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠ ACTUALIZADO 2026-05-24 al pipeline CAUSAL de O4.** Tras detectarse
> data leakage en O4 base, se ejecutó el rework causal (tag
> `v0.4.2-o4-rework-causal`). L2 se construye sobre el **feature set
> causal de 10 features**, NO sobre las 8 antiguas (que filtraban el
> target). El baseline L2-v0 se re-deriva del O4 **causal**. Las features
> del HMM causal (`state_b_causal`, `posterior_b_migracion_causal`) NO las
> produce `build_feature_matrix`: se adjuntan tras el split vía
> `fit_causal_hmm` + `decode_causal_states`, igual que `build_o4`. Ver
> `feedback-causal-features-no-leakage` y el spec refrescado.

**Goal:** Implementar el pipeline supervisado de O4 causal en dos etapas (clasificador binario de movimiento `clf_move` + clasificador multiclase condicional `clf_dest`) y compararlo contra O4 causal monolítico (L2-v0) para atacar la causa estructural D1 (dominio de self-loops).

**Architecture:** Etapa 1 (`clf_move`, siempre poblacional) predice `p_move ∈ [0,1]`; etapa 2B (`clf_dest`, dos modos) predice destino condicionado a movimiento, entrenada sólo sobre filas con `y_move=1`. Dos reglas de combinación: soft (canónica, distribución probabilística renormalizada) y hard (ablación, one-hot con umbral τ). Reutiliza features causales, split, familias e hiperparámetros de O4 causal; la única variable manipulada es la estructura del decisor.

**Tech Stack:** Python 3.12, scikit-learn 1.8 (`RandomForestClassifier`, `CalibratedClassifierCV` + `FrozenEstimator`), xgboost (`XGBClassifier` binario y multiclase), pandas, numpy, pytest, ruff. Reutiliza `tfg_aves.ml.{features,train,evaluate,hmm_causal}` y `tfg_aves.markov.discretize`.

**Spec:** `docs/superpowers/specs/2026-05-23-o4l2-dos-etapas-design.md` (refrescado al pipeline causal).

**Feature set causal (10 features) — la fuente es `tfg_aves.ml.features`:**
- `FEATURES_KINEMATIC` = `lat, lon, sin_doy, cos_doy, step_in_km, sin_bearing_in, cos_bearing_in, cos_turning_in` (cinemática entrante `t-1→t`, leak-free).
- `FEATURES_HMM` = `state_b_causal, posterior_b_migracion_causal` (HMM causal filtrado forward-only, adjuntado tras el split).
- En modo personalizado se antepone `bird_id`.

**Refinamientos sobre el spec (decisiones "cómo", no "qué"):**
- §6.2 dice "reutiliza `train_xgboost` tal cual" — válido SÓLO para la etapa 2B (multiclase). NO para la etapa 1: `_XGBoostWrapper` tiene `objective='multi:softprob'` hardcodeado y no expone `scale_pos_weight`. La etapa 1 usa trainers binarios dedicados en `two_stage.py` (`objective='binary:logistic'`, `scale_pos_weight`, `n_estimators=300` fijos sin early stopping, para reservar `val` a la calibración).
- Calibración: `cv='prefit'` fue eliminado en sklearn 1.8 → se usa `CalibratedClassifierCV(FrozenEstimator(base), method='isotonic').fit(X_val, y_val_move)`. Misma intención de F8 (respetar estructura temporal), API correcta.
- `combine_soft` aplica una normalización final de fila para garantizar suma 1.0 incluso si el clipping defensivo se activa (clf_dest colapsado sobre cell_t).
- `predictions_from_proba` vive en `two_stage.py` (hermano de `predict_with_meta`); produce la columna `state_b_causal` para que `evaluate_by_state` (que ya usa `state_col="state_b_causal"`) y `evaluate_moves_only` funcionen sin tocar `evaluate.py` más allá del helper nuevo.
- El ensamblaje causal de features (cinemática + HMM filtrado) se replica en un helper privado `_prepare_causal_splits` dentro de `build_l2.py` reutilizando las funciones de `features`/`hmm_causal`. Se opta por replicar (~20 LOC) en vez de extraer un helper compartido de `build.py` para NO modificar el `build.py` recién estabilizado por el rework. Es leak-safe: usa exactamente las mismas funciones leak-free.

---

## File Structure

- **Create** `src/tfg_aves/ml/two_stage.py` — funciones puras de L2: `derive_y_move`, trainers binarios (`train_move_rf`, `train_move_xgb`), `calibrate_prefit`, `expand_proba_to_full`, `combine_soft`, `combine_hard`, `sweep_tau`, `predictions_from_proba`.
- **Create** `src/tfg_aves/ml/build_l2.py` — orquestador `build_o4_l2()` + helper `_prepare_causal_splits()` + dataclass `BuildO4L2Result`.
- **Modify** `src/tfg_aves/ml/_paths.py` — añadir `O4_L2V1_DIR` y `PREDICTIONS_O4_PARQUET`.
- **Modify** `src/tfg_aves/ml/evaluate.py` — añadir `evaluate_moves_only`.
- **Create** `tests/test_ml_two_stage.py` — tests de las funciones puras.
- **Modify** `tests/test_ml_evaluate.py` — test de `evaluate_moves_only`.
- **Create** `tests/test_ml_build_l2.py` — test de integración con fixture sintético causal (reusa el patrón de `tests/test_ml_build.py`).
- **Create** `notebooks/04l2_eda_o4l2.py` — notebook jupytext que ejecuta `build_o4_l2` y genera los 7 artefactos.

**Nota sobre trabajo en paralelo:** L1 y el rework causal ya cerraron. L2 NO modifica `features.py`, `build.py`, `hmm_causal.py` ni `train.py` (sólo los importa). Aun así, **antes de cada commit** ejecutar `git log --oneline -3` para confirmar HEAD y usar SIEMPRE commits nuevos (nunca `--amend`).

---

## Task 1: Scaffolding — rutas y módulo vacío

**Files:**
- Modify: `src/tfg_aves/ml/_paths.py`
- Create: `src/tfg_aves/ml/two_stage.py`

- [ ] **Step 1: Añadir rutas L2 a `_paths.py`**

Añadir al final de `src/tfg_aves/ml/_paths.py`:

```python
O4_L2V1_DIR: Path = ROOT / "data" / "processed" / "o4" / "l2_v1"
PREDICTIONS_O4_PARQUET: Path = O4_OUT_DIR / "predictions_test.parquet"
```

- [ ] **Step 2: Crear `two_stage.py` con cabecera e imports**

Crear `src/tfg_aves/ml/two_stage.py`:

```python
"""Funciones puras del pipeline L2 (modelo de dos etapas) de O4 causal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from tfg_aves.markov.discretize import haversine_km
```

- [ ] **Step 3: Verificar que el módulo importa**

Run: `uv run python -c "import tfg_aves.ml.two_stage; from tfg_aves.ml._paths import O4_L2V1_DIR; print('ok', O4_L2V1_DIR.name)"`
Expected: `ok l2_v1`

- [ ] **Step 4: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/_paths.py src/tfg_aves/ml/two_stage.py
git commit -m "L2: scaffolding de rutas y módulo two_stage"
```

---

## Task 2: `derive_y_move`

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

> `y_move` se deriva de `cell_id_t` y `cell_id_t_next` (posiciones; el
> target estructural, NO una feature). Es leak-free por construcción.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_two_stage.py::test_derive_y_move_basic -v`
Expected: FAIL con `AttributeError: module ... has no attribute 'derive_y_move'`

- [ ] **Step 3: Implementar `derive_y_move`**

Añadir a `two_stage.py`:

```python
def derive_y_move(df: pd.DataFrame) -> pd.Series:
    """Devuelve y_move = (cell_id_t_next != cell_id_t) como Series booleana."""
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).rename("y_move")
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `uv run pytest tests/test_ml_two_stage.py::test_derive_y_move_basic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: derive_y_move (target binario de la etapa 1)"
```

---

## Task 3: `combine_soft` (regla canónica, renormalizada)

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `uv run pytest tests/test_ml_two_stage.py -k combine_soft -v`
Expected: FAIL con `AttributeError: ... 'combine_soft'`

- [ ] **Step 3: Implementar `combine_soft`**

Añadir a `two_stage.py`:

```python
def combine_soft(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    n_classes: int,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla soft canónica con renormalización (§1 del spec).

    Para cada fila i:
        p_final[i, cell_t]      = 1 - p_move[i]
        p_final[i, cell≠cell_t] = p_move[i] * p_2b[i, cell] / (1 - p_2b[i, cell_t])

    El denominador se evalúa como max(1 - p_2b[i, cell_t], eps). Se aplica
    una normalización final por fila para garantizar suma 1.0 incluso en
    el caso degenerado en que clf_dest colapsa sobre cell_t.
    """
    n = p_move.shape[0]
    rows = np.arange(n)
    p2b_cellt = p_2b[rows, cell_t_idx]
    denom = np.maximum(1.0 - p2b_cellt, eps)

    out = p_move[:, None] * p_2b / denom[:, None]
    out[rows, cell_t_idx] = 1.0 - p_move  # sobrescribe la masa de movimiento en cell_t

    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return out / row_sums
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `uv run pytest tests/test_ml_two_stage.py -k combine_soft -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: combine_soft (regla canónica renormalizada)"
```

---

## Task 4: `combine_hard` (regla de ablación)

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `uv run pytest tests/test_ml_two_stage.py -k combine_hard -v`
Expected: FAIL con `AttributeError: ... 'combine_hard'`

- [ ] **Step 3: Implementar `combine_hard`**

Añadir a `two_stage.py`:

```python
def combine_hard(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    tau: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla hard: cell_t si p_move<tau, si no argmax(p_2b).

    Devuelve (n, n_classes) con masa 1-eps en la celda predicha y
    eps/(n_classes-1) repartida en el resto. El clipping existe sólo por
    compatibilidad con sklearn.metrics.log_loss; el log-loss numérico de
    hard NO es una métrica honesta (cada fallo de argmax suma ~22.86).
    """
    n, n_classes = p_2b.shape
    pred_idx = np.where(p_move < tau, cell_t_idx, p_2b.argmax(axis=1))
    out = np.full((n, n_classes), eps / (n_classes - 1), dtype=np.float64)
    out[np.arange(n), pred_idx] = 1.0 - eps
    return out
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `uv run pytest tests/test_ml_two_stage.py -k combine_hard -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: combine_hard (regla de ablación one-hot)"
```

---

## Task 5: `sweep_tau` (selección de τ por top-1)

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
        taus=(0.3, 0.5, 0.7),
    )
    assert tau_star == 0.5
    assert set(table.columns) == {"tau", "top1"}
    assert len(table) == 3
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_two_stage.py::test_sweep_tau_returns_best -v`
Expected: FAIL con `AttributeError: ... 'sweep_tau'`

- [ ] **Step 3: Implementar `sweep_tau`**

Añadir a `two_stage.py`:

```python
def sweep_tau(
    p_move_val: np.ndarray,
    p_2b_val: np.ndarray,
    cell_t_idx_val: np.ndarray,
    y_true_idx_val: np.ndarray,
    n_classes: int,
    taus: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> tuple[float, pd.DataFrame]:
    """Barre tau y devuelve (tau*, tabla) maximizando top-1 sobre val.

    Se usa top-1 (NO log-loss) porque hard devuelve one-hot y su log-loss
    quedaría dominado por el clipping. Empates: gana el tau menor (primero).
    """
    records = []
    for tau in taus:
        out = combine_hard(p_move_val, p_2b_val, cell_t_idx_val, tau)
        top1 = float((out.argmax(axis=1) == y_true_idx_val).mean())
        records.append({"tau": tau, "top1": top1})
    table = pd.DataFrame(records)
    tau_star = float(table.loc[table["top1"].idxmax(), "tau"])
    return tau_star, table
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `uv run pytest tests/test_ml_two_stage.py::test_sweep_tau_returns_best -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: sweep_tau (selección de umbral por top-1)"
```

---

## Task 6: `expand_proba_to_full` (proyección al espacio completo de clases)

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_two_stage.py::test_expand_proba_to_full -v`
Expected: FAIL con `AttributeError: ... 'expand_proba_to_full'`

- [ ] **Step 3: Implementar `expand_proba_to_full`**

Añadir a `two_stage.py`:

```python
def expand_proba_to_full(
    proba: np.ndarray,
    classes_str: np.ndarray,
    classes_full: np.ndarray,
) -> np.ndarray:
    """Proyecta proba (n, k) al espacio completo (n, m) de classes_full.

    Las columnas de classes_full ausentes en classes_str quedan a 0.
    classes_full debe ser un superconjunto ordenado de classes_str.
    """
    n = proba.shape[0]
    m = len(classes_full)
    out = np.zeros((n, m), dtype=np.float64)
    full_index = {c: j for j, c in enumerate(classes_full)}
    for src_col, c in enumerate(classes_str):
        out[:, full_index[c]] = proba[:, src_col]
    return out
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `uv run pytest tests/test_ml_two_stage.py::test_expand_proba_to_full -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: expand_proba_to_full (espacio común de clases)"
```

---

## Task 7: `predictions_from_proba` (DataFrame de evaluación desde una matriz de probabilidad)

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

> La columna de régimen es `state_b_causal` (coherente con
> `evaluate_by_state`, que usa `state_col="state_b_causal"`).

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_two_stage.py::test_predictions_from_proba_columns_and_top1 -v`
Expected: FAIL con `AttributeError: ... 'predictions_from_proba'`

- [ ] **Step 3: Implementar `predictions_from_proba`**

Añadir a `two_stage.py` (incluye helper de centroides local para no depender de `evaluate.py`):

```python
def _centroid_lookup(cells: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {row.cell_id: (row.lat_c, row.lon_c) for row in cells.itertuples()}


def _dist_to_centroid(
    pred_cell: str, lat_next: float, lon_next: float,
    centroids: dict[str, tuple[float, float]],
) -> float:
    if pd.isna(lat_next) or pd.isna(lon_next) or pred_cell not in centroids:
        return np.nan
    c_lat, c_lon = centroids[pred_cell]
    return haversine_km(c_lat, c_lon, lat_next, lon_next)


def predictions_from_proba(
    proba_full: np.ndarray,
    classes_full: np.ndarray,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Construye el DataFrame de predicciones estándar desde proba_full.

    Hermano de evaluate.predict_with_meta pero parte de una matriz de
    probabilidad ya combinada (no de un modelo). Columnas: bird_id,
    date_utc, true_cell, pred_cell_top1, pred_cell_topk, pred_prob_top1,
    pred_dist_km, state_b_causal. Adjunta proba_full y classes_full en attrs.
    """
    top1_idx = np.argmax(proba_full, axis=1)
    topk_idx = np.argsort(proba_full, axis=1)[:, -top_k:][:, ::-1]
    pred_top1 = classes_full[top1_idx]
    pred_topk = [classes_full[row].tolist() for row in topk_idx]
    prob_top1 = proba_full[np.arange(len(proba_full)), top1_idx]

    centroids = _centroid_lookup(cells)
    dists = [
        _dist_to_centroid(pc, lt, ln, centroids) for pc, lt, ln in
        zip(pred_top1, meta["lat_t_next"], meta["lon_t_next"], strict=True)
    ]

    out = pd.DataFrame({
        "bird_id": meta["bird_id"].values,
        "date_utc": meta["date_utc"].values,
        "true_cell": meta["cell_id_t_next"].values,
        "pred_cell_top1": pred_top1,
        "pred_cell_topk": pred_topk,
        "pred_prob_top1": prob_top1,
        "pred_dist_km": dists,
        "state_b_causal": meta["state_b_causal"].values,
    })
    out.attrs["_proba"] = proba_full
    out.attrs["_classes"] = classes_full
    return out
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `uv run pytest tests/test_ml_two_stage.py::test_predictions_from_proba_columns_and_top1 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: predictions_from_proba (eval desde matriz combinada)"
```

---

## Task 8: `evaluate_moves_only` en `evaluate.py`

**Files:**
- Modify: `src/tfg_aves/ml/evaluate.py`
- Test: `tests/test_ml_evaluate.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/test_ml_evaluate.py` (importar `numpy as np` y `pandas as pd` si no están):

```python
def test_evaluate_moves_only_filters_correctly():
    from tfg_aves.ml.evaluate import evaluate_moves_only
    preds = pd.DataFrame({
        "true_cell":      ["A", "B", "C", "D", "E"],
        "pred_cell_top1": ["A", "B", "X", "D", "Y"],
        "pred_cell_topk": [["A"], ["B"], ["X"], ["D"], ["Y"]],
        "pred_dist_km":   [0.0, 0.0, 50.0, 0.0, 80.0],
        "state_b_causal": [0, 1, 1, 1, 0],
    })
    # 3 filas se mueven (índices 1,2,3); de ellas aciertan top1 las 1 y 3
    y_move = np.array([False, True, True, True, False])
    result = evaluate_moves_only(preds, y_move)
    assert result["n_obs"] == 3
    assert np.isclose(result["top1"], 2 / 3)
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_evaluate.py::test_evaluate_moves_only_filters_correctly -v`
Expected: FAIL con `ImportError: cannot import name 'evaluate_moves_only'`

- [ ] **Step 3: Implementar `evaluate_moves_only`**

Añadir a `src/tfg_aves/ml/evaluate.py`:

```python
def evaluate_moves_only(
    predictions: pd.DataFrame, y_move_true: np.ndarray,
) -> dict[str, float]:
    """Métricas restringidas al subset donde verdaderamente y_move=1.

    Aísla la calidad de clf_dest sin que la persistencia trivial domine.
    Si predictions trae attrs["_proba"]/["_classes"], computa también
    log_loss sobre el subset.
    """
    mask = np.asarray(y_move_true).astype(bool)
    sub = predictions[mask].reset_index(drop=True)
    result: dict[str, float] = {
        "n_obs": int(mask.sum()),
        "top1": top_k_accuracy(sub, k=1),
        "top3": top_k_accuracy(sub, k=3),
        "dist_median_km": dist_median_km(sub),
    }
    proba = predictions.attrs.get("_proba")
    classes = predictions.attrs.get("_classes")
    if proba is not None and classes is not None and mask.any():
        y_str = sub["true_cell"].astype(str).to_numpy()
        result["log_loss"] = float(
            log_loss(y_str, np.asarray(proba)[mask], labels=list(classes)),
        )
    return result
```

- [ ] **Step 4: Ejecutar el test para verificar que pasa**

Run: `uv run pytest tests/test_ml_evaluate.py::test_evaluate_moves_only_filters_correctly -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/evaluate.py tests/test_ml_evaluate.py
git commit -m "L2: evaluate_moves_only (métricas sobre subset y_move=1)"
```

---

## Task 9: Trainers binarios + calibración de la etapa 1

**Files:**
- Modify: `src/tfg_aves/ml/two_stage.py`
- Test: `tests/test_ml_two_stage.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_ml_two_stage.py`:

```python
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
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `uv run pytest tests/test_ml_two_stage.py -k "train_move or calibrate" -v`
Expected: FAIL con `AttributeError: ... 'train_move_rf'`

- [ ] **Step 3: Implementar los trainers binarios y la calibración**

Añadir a `two_stage.py`:

```python
def train_move_rf(
    X_train: pd.DataFrame, y_move: np.ndarray, *, seed: int = 0,
) -> RandomForestClassifier:
    """Etapa 1 RF binaria. Config §8.6 + class_weight='balanced'.

    Poblacional: X_train sólo tiene columnas numéricas (las 8 cinemáticas
    + las 2 del HMM causal), sin bird_id, así que no necesita encoder.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, np.asarray(y_move).astype(int))
    return rf


def train_move_xgb(
    X_train: pd.DataFrame, y_move: np.ndarray, *, seed: int = 0,
) -> xgb.XGBClassifier:
    """Etapa 1 XGBoost binaria. Config §8.6 adaptada a binario.

    n_estimators=300 FIJOS (sin early stopping) para reservar X_val
    exclusivamente a la calibración (F8). objective='binary:logistic',
    scale_pos_weight = n_neg / n_pos calculado sobre y_move.
    """
    y = np.asarray(y_move).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y)
    return model


def calibrate_prefit(
    base: ClassifierMixin, X_val: pd.DataFrame, y_val_move: np.ndarray,
) -> CalibratedClassifierCV:
    """Calibra isotónicamente un modelo ya entrenado usando el val temporal.

    Usa FrozenEstimator (sklearn 1.8; reemplaza cv='prefit'). El base NO se
    reentrena: sólo se ajusta el calibrador sobre (X_val, y_val_move).
    """
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(X_val, np.asarray(y_val_move).astype(int))
    return cal
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `uv run pytest tests/test_ml_two_stage.py -k "train_move or calibrate" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Ejecutar toda la suite de two_stage + ruff**

Run: `uv run pytest tests/test_ml_two_stage.py -v && uv run ruff check src/tfg_aves/ml/two_stage.py`
Expected: todos PASS, ruff sin errores

- [ ] **Step 6: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/two_stage.py tests/test_ml_two_stage.py
git commit -m "L2: trainers binarios de etapa 1 + calibración isotonic"
```

---

## Task 10: Orquestador `build_o4_l2` (con ensamblaje causal)

**Files:**
- Create: `src/tfg_aves/ml/build_l2.py`
- Test: `tests/test_ml_build_l2.py`

- [ ] **Step 1: Escribir el test de integración que falla**

Crear `tests/test_ml_build_l2.py` (reusa el patrón de fixture causal de `tests/test_ml_build.py`: incluye `veg_low/veg_high/daylight_hours`, que la emisión del HMM causal exige):

```python
"""Test de integración de build_o4_l2 con un fixture sintético causal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.ml.build_l2 import build_o4_l2


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos consecutivos sobre celdas activas.

    Mismo patrón que tests/test_ml_build.py pero con más días para
    garantizar suficientes filas con y_move=1 tras la máscara de racha
    de 4 días del HMM causal.
    """
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=90)
    rows = []
    for b in birds:
        lat0 = 40.0 + rng.uniform(-1, 1)
        lon0 = -3.0 + rng.uniform(-1, 1)
        for i, d in enumerate(dates):
            lat = lat0 + 0.05 * i + rng.normal(0, 0.05)
            lon = lon0 + 0.05 * i + rng.normal(0, 0.05)
            rows.append({
                "bird_id": b, "date_utc": d, "lat": lat, "lon": lon,
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
                "is_observation_valid": True,
            })
    feat = pd.DataFrame(rows)
    feat_path = tmp_path / "features.parquet"
    feat.to_parquet(feat_path)

    cells = []
    for i in range(78, 86):
        for j in range(-9, -1):
            cells.append({
                "cell_id": f"{i}_{j}",
                "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5,
                "n_obs_total": 30,
            })
    cells_df = pd.DataFrame(cells)
    cells_path = tmp_path / "cells.parquet"
    cells_df.to_parquet(cells_path)
    return feat_path, cells_path


def test_build_o4_l2_artifacts(tmp_path):
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l2_v1"
    result = build_o4_l2(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0,
    )
    # 6 modelos: 2 clf_move (rf, xgb) + 4 clf_dest (rf/xgb × pers/pob)
    assert (out_dir / "model_clf_move_rf.pkl").exists()
    assert (out_dir / "model_clf_move_xgb.pkl").exists()
    assert (out_dir / "model_clf_dest_rf_personalizado.pkl").exists()
    assert (out_dir / "model_clf_dest_xgb_poblacional.pkl").exists()
    assert (out_dir / "predictions_test_soft.parquet").exists()
    assert (out_dir / "predictions_test_hard.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()
    assert (out_dir / "tau_sweep.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"modelo", "modo", "regla", "top1", "log_loss"}.issubset(metrics.columns)
    assert len(metrics) >= 8

    # log_loss de hard debe ser NaN (no comparable)
    hard_rows = metrics[metrics["regla"] == "hard"]
    assert hard_rows["log_loss"].isna().all()
    assert result.n_moves_train > 0


def test_build_o4_l2_idempotent(tmp_path):
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l2_v1"
    build_o4_l2(features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0)
    m1 = pd.read_parquet(out_dir / "metrics.parquet")
    build_o4_l2(features_path=feat_path, cells_path=cells_path, out_dir=out_dir, seed=0)
    m2 = pd.read_parquet(out_dir / "metrics.parquet")
    pd.testing.assert_frame_equal(
        m1.sort_values(["modelo", "modo", "regla"]).reset_index(drop=True),
        m2.sort_values(["modelo", "modo", "regla"]).reset_index(drop=True),
    )
```

- [ ] **Step 2: Ejecutar el test para verificar que falla**

Run: `uv run pytest tests/test_ml_build_l2.py::test_build_o4_l2_artifacts -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tfg_aves.ml.build_l2'`

- [ ] **Step 3: Implementar `build_l2.py`**

Crear `src/tfg_aves/ml/build_l2.py`:

```python
"""Orquestador único del pipeline L2 (modelo de dos etapas) de O4 causal."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L2V1_DIR
from .evaluate import (
    compute_markov_baseline,
    compute_persistence_baseline,
    dist_median_km,
    evaluate_moves_only,
    top_k_accuracy,
)
from .features import (
    FEATURES_HMM,
    FEATURES_KINEMATIC,
    build_feature_matrix,
    compute_causal_kinematics,
    split_temporal_per_bird,
)
from .hmm_causal import decode_causal_states, fit_causal_hmm
from .train import train_random_forest, train_xgboost
from .two_stage import (
    calibrate_prefit,
    combine_hard,
    combine_soft,
    derive_y_move,
    expand_proba_to_full,
    predictions_from_proba,
    sweep_tau,
    train_move_rf,
    train_move_xgb,
)

_FAMILIES = ("rf", "xgb")
_MODES = ("personalizado", "poblacional")


@dataclass
class BuildO4L2Result:
    """Resumen serializable de la ejecución de build_o4_l2."""

    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    n_moves_train: int
    tau_star: dict[str, float] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_paths: dict[str, Path] = field(default_factory=dict)
    metrics_path: Path = Path()


def _prepare_causal_splits(
    features_o3: pd.DataFrame, cells: pd.DataFrame, seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Replica el ensamblaje causal de build_o4 (cinemática entrante + HMM
    causal filtrado forward-only) y devuelve {mode: (train, val, test)} con
    state_b_causal y posterior_b_migracion_causal ya adjuntados.

    Se replica (en vez de extraer un helper de build.py) para NO tocar el
    build.py recién estabilizado; es leak-safe porque usa exactamente las
    mismas funciones leak-free.
    """
    kin = compute_causal_kinematics(features_o3)
    matrices = {
        "personalizado": build_feature_matrix(kin, cells, include_bird_id=True),
        "poblacional": build_feature_matrix(kin, cells, include_bird_id=False),
    }
    raw_splits = {m: split_temporal_per_bird(mat) for m, mat in matrices.items()}

    train_ref = raw_splits["poblacional"][0]
    cutoff_by_bird = (
        train_ref.assign(_d=pd.to_datetime(train_ref["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )
    hmm_model, hmm_labels = fit_causal_hmm(
        kin, cutoff_by_bird, n_restarts=10, seed=seed,
    )
    states = decode_causal_states(hmm_model, hmm_labels, kin)

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(states, on=["bird_id", "date_utc"], how="left", validate="m:1")
        if merged[FEATURES_HMM].isna().any().any():
            raise ValueError("Filas candidatas sin estado HMM causal tras el merge.")
        return merged

    return {m: tuple(attach(d) for d in raw_splits[m]) for m in _MODES}


def _feature_cols(mode: str) -> list[str]:
    base = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    return ["bird_id", *base] if mode == "personalizado" else list(base)


def build_o4_l2(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L2V1_DIR,
    seed: int = 0,
) -> BuildO4L2Result:
    """Pipeline L2-v1 completo sobre el feature set CAUSAL (ver spec §6.1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    splits = _prepare_causal_splits(features_o3, cells, seed)

    # y_move se deriva del modo poblacional (filas idénticas entre modos).
    train_pob, val_pob, test_pob = splits["poblacional"]
    feat_pob = _feature_cols("poblacional")
    y_move_train = derive_y_move(train_pob).to_numpy().astype(int)
    y_move_val = derive_y_move(val_pob).to_numpy().astype(int)
    y_move_test = derive_y_move(test_pob).to_numpy().astype(int)
    moves_mask_train = y_move_train.astype(bool)

    # Espacio de clases común para la combinación.
    classes_full = np.array(sorted(
        set(test_pob["cell_id_t_next"].astype(str))
        | set(test_pob["cell_id_t"].astype(str)),
    ))
    full_index = {c: j for j, c in enumerate(classes_full)}
    n_classes = len(classes_full)
    cell_t_idx_test = test_pob["cell_id_t"].astype(str).map(full_index).to_numpy()

    model_paths: dict[str, Path] = {}
    tau_star_by_combo: dict[str, float] = {}
    metric_rows: list[dict] = []
    preds_soft_frames: list[pd.DataFrame] = []
    preds_hard_frames: list[pd.DataFrame] = []
    tau_tables: list[pd.DataFrame] = []

    for family in _FAMILIES:
        # --- Etapa 1: clf_move (poblacional) ---
        X_train_move = train_pob[feat_pob]
        X_val_move = val_pob[feat_pob]
        X_test_move = test_pob[feat_pob]
        if family == "rf":
            base_move = train_move_rf(X_train_move, y_move_train, seed=seed)
        else:
            base_move = train_move_xgb(X_train_move, y_move_train, seed=seed)
        clf_move = calibrate_prefit(base_move, X_val_move, y_move_val)

        pos_col = list(clf_move.classes_).index(1)
        p_move_test = clf_move.predict_proba(X_test_move)[:, pos_col]
        p_move_val = clf_move.predict_proba(X_val_move)[:, pos_col]

        mp = out_dir / f"model_clf_move_{family}.pkl"
        joblib.dump({"model": clf_move, "feature_cols": feat_pob}, mp)
        model_paths[f"clf_move_{family}"] = mp

        for mode in _MODES:
            train_m, val_m, test_m = splits[mode]
            feat_m = _feature_cols(mode)
            cat_cols = ["bird_id"] if "bird_id" in feat_m else []

            # --- Etapa 2B: clf_dest sobre filas con y_move=1 ---
            train_moves = train_m[moves_mask_train].reset_index(drop=True)
            le_dest = LabelEncoder().fit(train_moves["cell_id_t_next"].astype(str))
            y_dest = le_dest.transform(train_moves["cell_id_t_next"].astype(str))
            X_dest_train = train_moves[feat_m]

            if family == "rf":
                clf_dest = train_random_forest(
                    X_dest_train, y_dest, categorical_cols=cat_cols, seed=seed,
                )
            else:
                val_moves = val_m[
                    derive_y_move(val_m).to_numpy().astype(bool)
                ].reset_index(drop=True)
                known = set(le_dest.classes_)
                fallback = le_dest.classes_[0]
                y_dest_val = le_dest.transform([
                    c if c in known else fallback
                    for c in val_moves["cell_id_t_next"].astype(str)
                ])
                clf_dest = train_xgboost(
                    X_dest_train, y_dest, val_moves[feat_m], y_dest_val,
                    categorical_cols=cat_cols, seed=seed,
                )

            md = out_dir / f"model_clf_dest_{family}_{mode}.pkl"
            joblib.dump({
                "model": clf_dest, "label_encoder_y": le_dest,
                "feature_cols": feat_m, "categorical_cols": cat_cols,
            }, md)
            model_paths[f"clf_dest_{family}_{mode}"] = md

            # --- Combinación sobre test ---
            classes_dest = le_dest.inverse_transform(clf_dest.classes_)
            p2b_test = expand_proba_to_full(
                clf_dest.predict_proba(test_m[feat_m]), classes_dest, classes_full,
            )
            p2b_val = expand_proba_to_full(
                clf_dest.predict_proba(val_m[feat_m]), classes_dest, classes_full,
            )
            cell_t_idx_val = (
                val_m["cell_id_t"].astype(str).map(full_index).fillna(0)
                .to_numpy().astype(int)
            )
            y_true_idx_val = (
                val_m["cell_id_t_next"].astype(str).map(full_index).fillna(0)
                .to_numpy().astype(int)
            )
            tau_star, tau_table = sweep_tau(
                p_move_val, p2b_val, cell_t_idx_val, y_true_idx_val, n_classes,
            )
            tau_table["modelo"] = family
            tau_table["modo"] = mode
            tau_tables.append(tau_table)
            tau_star_by_combo[f"{family}_{mode}"] = tau_star

            proba_soft = combine_soft(p_move_test, p2b_test, cell_t_idx_test, n_classes)
            proba_hard = combine_hard(p_move_test, p2b_test, cell_t_idx_test, tau_star)

            for rule, proba in (("soft", proba_soft), ("hard", proba_hard)):
                preds = predictions_from_proba(proba, classes_full, test_m, cells=cells)
                if rule == "soft":
                    ll = float(log_loss(
                        test_m["cell_id_t_next"].astype(str).to_numpy(),
                        proba, labels=list(classes_full),
                    ))
                else:
                    ll = np.nan
                row = {
                    "modelo": family, "modo": mode, "regla": rule,
                    "top1": top_k_accuracy(preds, k=1),
                    "top3": top_k_accuracy(preds, k=3),
                    "log_loss": ll,
                    "dist_median_km": dist_median_km(preds),
                }
                if rule == "soft":
                    mo = evaluate_moves_only(preds, y_move_test)
                    row["top1_moves"] = mo["top1"]
                    row["dist_median_km_moves"] = mo["dist_median_km"]
                metric_rows.append(row)

                preds_out = preds.copy()
                for k in list(preds_out.attrs):
                    preds_out.attrs.pop(k, None)
                preds_out["modelo"] = family
                preds_out["modo"] = mode
                preds_out["regla"] = rule
                preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
                (preds_soft_frames if rule == "soft" else preds_hard_frames).append(preds_out)

    # --- Baselines sobre el test poblacional ---
    persistence = compute_persistence_baseline(test_pob, cells=cells)
    markov = compute_markov_baseline(train_pob, test_pob, cells=cells)
    for name, bl in (("persistencia", persistence), ("markov", markov)):
        metric_rows.append({
            "modelo": name, "modo": "—", "regla": "—",
            "top1": top_k_accuracy(bl, k=1),
            "top3": top_k_accuracy(bl, k=3),
            "log_loss": np.nan,
            "dist_median_km": dist_median_km(bl),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path)

    soft_path = out_dir / "predictions_test_soft.parquet"
    hard_path = out_dir / "predictions_test_hard.parquet"
    pd.concat(preds_soft_frames, ignore_index=True).to_parquet(soft_path)
    pd.concat(preds_hard_frames, ignore_index=True).to_parquet(hard_path)
    pd.concat(tau_tables, ignore_index=True).to_parquet(out_dir / "tau_sweep.parquet")

    return BuildO4L2Result(
        n_rows_train=len(train_pob),
        n_rows_val=len(val_pob),
        n_rows_test=len(test_pob),
        n_moves_train=int(moves_mask_train.sum()),
        tau_star=tau_star_by_combo,
        model_paths=model_paths,
        predictions_paths={"soft": soft_path, "hard": hard_path},
        metrics_path=metrics_path,
    )
```

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `uv run pytest tests/test_ml_build_l2.py -v`
Expected: PASS (2 tests).

> **Nota de implementación:** si el baseline Markov o el HMM causal
> rompen con el fixture sintético (p.ej. celdas escasas o pocas filas con
> racha de 4 días), **ampliar el fixture** (más días/aves) hasta que corra.
> NO envolver en guardas defensivas: el código heredado (build_o4,
> hmm_causal, baselines) ya está testado; el problema sería del fixture.

- [ ] **Step 5: Ejecutar la suite completa + ruff**

Run: `uv run pytest -q && uv run ruff check src/tfg_aves/ml/`
Expected: todos los tests PASS (suite previa + nuevos), ruff sin errores

- [ ] **Step 6: Commit**

```bash
git log --oneline -3
git add src/tfg_aves/ml/build_l2.py tests/test_ml_build_l2.py
git commit -m "L2: orquestador build_o4_l2 causal (dos etapas + soft/hard + baselines)"
```

---

## Task 11: Notebook EDA + artefactos `save_artifact`

**Files:**
- Create: `notebooks/04l2_eda_o4l2.py`

> **Requisito:** este task necesita los datos reales en `data/processed/`
> (gitignored, regenerables) y el O4 **causal** ya regenerado en
> `data/processed/o4/` (tag `v0.4.2-o4-rework-causal`). Si no existen,
> regenerar con `build_o4()` (causal) antes de empezar. El notebook NO se
> ejecuta en CI; produce evidencia para la memoria.

- [ ] **Step 1: Crear el notebook jupytext (percent) con la fase de build**

Crear `notebooks/04l2_eda_o4l2.py` con cabecera jupytext y primera celda:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # O4 · L2 — Modelo de dos etapas (régimen → posición), pipeline causal
# Ejecuta `build_o4_l2` y genera los 7 artefactos (D1, D2, C1..C5)
# comparando L2-v1 contra O4 causal monolítico (L2-v0).

# %%
import pandas as pd

from tfg_aves.ml.build_l2 import build_o4_l2
from tfg_aves.ml._paths import O4_L2V1_DIR, O4_OUT_DIR

result = build_o4_l2()
print("filas train/val/test:", result.n_rows_train, result.n_rows_val, result.n_rows_test)
print("movimientos en train:", result.n_moves_train)
print("tau* por combo:", result.tau_star)

metrics_l2 = pd.read_parquet(O4_L2V1_DIR / "metrics.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")  # O4 causal
```

- [ ] **Step 2: Añadir celda del artefacto D1 (calidad de clf_move)**

Añadir celda que computa AUC/accuracy/Brier/precision/recall de cada
`clf_move` sobre test y lo guarda. **Usa el ensamblaje causal** para
reconstruir el test (mismo que `build_l2._prepare_causal_splits`):

```python
# %%
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score,
)
from tfg_aves.ml.build_l2 import _prepare_causal_splits, _feature_cols
from tfg_aves.ml.two_stage import derive_y_move
from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET
from tfg_aves.reporting import save_artifact

features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
splits = _prepare_causal_splits(features_o3, cells, seed=0)
_, _, test_pob = splits["poblacional"]
feat_pob = _feature_cols("poblacional")
y_move_test = derive_y_move(test_pob).to_numpy().astype(int)

rows = []
for fam in ("rf", "xgb"):
    bundle = joblib.load(O4_L2V1_DIR / f"model_clf_move_{fam}.pkl")
    clf = bundle["model"]
    pos = list(clf.classes_).index(1)
    p = clf.predict_proba(test_pob[feat_pob])[:, pos]
    yhat = (p >= 0.5).astype(int)
    rows.append({
        "familia": fam,
        "auc": roc_auc_score(y_move_test, p),
        "accuracy": accuracy_score(y_move_test, yhat),
        "brier": brier_score_loss(y_move_test, p),
        "precision_mov": precision_score(y_move_test, yhat, zero_division=0),
        "recall_mov": recall_score(y_move_test, yhat, zero_division=0),
    })
d1 = pd.DataFrame(rows).round(4)

save_artifact(
    "l2v1-clf-move-quality",
    objective="o4", num=17,
    decision="Calidad bruta del clasificador de movimiento (etapa 1)",
    caption_es=(
        "Métricas del clasificador binario de movimiento (etapa 1) sobre el "
        "test temporal: AUC, accuracy, Brier score y precision/recall de la "
        "clase movimiento, para RF y XGBoost. Justifica que la etapa 1 "
        "discrimina movimiento con calibración aceptable."
    ),
    table=d1,
)
```

> **Nota:** los `num=` de save_artifact (17, 18, ...) deben elegirse según
> el siguiente número libre de `o4_*` en `reports/INDEX.md` en el momento
> de ejecutar (L1 consumió hasta 16). Verificar con
> `grep "o4_fig\|o4_tab" reports/INDEX.md | tail` y ajustar.

- [ ] **Step 3: Añadir celda del artefacto D2 (barrido de τ)**

```python
# %%
import matplotlib.pyplot as plt

tau_df = pd.read_parquet(O4_L2V1_DIR / "tau_sweep.parquet")
fig, ax = plt.subplots(figsize=(7, 4))
for (fam, modo), sub in tau_df.groupby(["modelo", "modo"]):
    ax.plot(sub["tau"], sub["top1"], marker="o", label=f"{fam}-{modo}")
ax.set_xlabel("τ (umbral de p_move)")
ax.set_ylabel("top-1 sobre val")
ax.set_title("Barrido de τ — regla hard")
ax.legend(fontsize=8)
fig.tight_layout()

save_artifact(
    "l2v1-tau-sweep",
    objective="o4", num=18,
    decision="τ* de la regla hard elegido por top-1 sobre val",
    caption_es=(
        "Top-1 sobre el conjunto de validación en función del umbral τ de la "
        "regla hard, por familia y modo. Se selecciona el τ que maximiza top-1; "
        "el log-loss no se usa por ser degenerado para una regla one-hot."
    ),
    fig=fig, table=tau_df,
)
```

- [ ] **Step 4: Añadir celda del artefacto C1 (comparativa global L2-v0 vs L2-v1)**

L2-v0 = O4 **causal** monolítico (de `metrics_v0`, filas `split=="test"`).

```python
# %%
# C1: tabla central — L2-v0 (O4 causal) vs L2-v1 (soft/hard), globales.
v0 = metrics_v0[metrics_v0.get("split", "test") == "test"][
    ["modelo", "modo", "top1", "top3", "log_loss", "dist_median_km"]
].assign(version="L2-v0", regla="argmax")
v1 = metrics_l2[metrics_l2["regla"].isin(["soft", "hard"])][
    ["modelo", "modo", "regla", "top1", "top3", "log_loss", "dist_median_km"]
].assign(version="L2-v1")
c1 = pd.concat([v0, v1], ignore_index=True).round(4)

fig, ax = plt.subplots(figsize=(8, 4.5))
soft = c1[(c1["version"] == "L2-v1") & (c1["regla"] == "soft")]
base = c1[(c1["version"] == "L2-v0") & (c1["modo"].isin(["personalizado", "poblacional"]))]
labels = [f"{r.modelo}-{r.modo}" for r in base.itertuples()]
x = np.arange(len(labels))
soft_aligned = soft.set_index(["modelo", "modo"]).reindex(
    base.set_index(["modelo", "modo"]).index)["top1"].values
ax.bar(x - 0.2, base["top1"], width=0.4, label="L2-v0 (causal)")
ax.bar(x + 0.2, soft_aligned, width=0.4, label="L2-v1 (soft)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("top-1"); ax.set_title("L2-v0 (causal) vs L2-v1 (soft) — top-1 global")
ax.legend()
fig.tight_layout()

save_artifact(
    "l2v1-metrics-comparison",
    objective="o4", num=19,
    decision="Comparativa global L2-v0 (O4 causal) vs L2-v1 (entregable central de L2)",
    caption_es=(
        "Comparativa de métricas globales entre el modelo monolítico de O4 "
        "causal (L2-v0) y el modelo de dos etapas (L2-v1), con reglas soft y "
        "hard. El log-loss de la regla hard se omite por no ser comparable."
    ),
    fig=fig, table=c1,
)
```

- [ ] **Step 5: Añadir celdas de C2..C5**

Añadir cuatro celdas análogas que generen y guarden:
- **C2** (`l2v1-by-state-comparison`, num=20): tabla + figura de top-1/dist por estado HMM (estacionario/migración) para L2-v0 vs L2-v1 soft. Reutilizar `evaluate_by_state` de `evaluate.py` (usa `state_col="state_b_causal"` por defecto) aplicada a `predictions_test_soft.parquet` filtrado por combo, y a `predictions_test.parquet` de O4 causal. **Es el artefacto que mide el lift en migración.**
- **C3** (`l2v1-moves-only`, num=21): tabla con `top1_moves` y `dist_median_km_moves` (ya en `metrics.parquet`) frente al top-1 de O4 causal sobre el mismo subset. Aísla la calidad de clf_dest.
- **C4** (`l2v1-gap-train-test`, num=22): figura de gap train-test de clf_dest por combo. Recomputar top-1 de cada `clf_dest` sobre su propio train (filas y_move=1) y sobre test, graficar la diferencia frente al gap de O4 causal.
- **C5** (`l2v1-state-confusion`, num=23): matriz `state_b_causal verdad × (predice cambio de celda sí/no)` para L2-v1 soft (RF personalizado) vs O4 causal, vía `pd.crosstab`. Diagnóstico de cuándo L2 cambia su decisión.

Cada celda termina en un `save_artifact(...)` con `objective="o4"`, su `num` libre y un `caption_es` descriptivo en castellano. Patrón completo ya mostrado en C1 (steps 2-4).

- [ ] **Step 6: Sincronizar el `.ipynb` con jupytext y ejecutar de punta a punta**

Run: `uv run jupytext --to notebook notebooks/04l2_eda_o4l2.py && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04l2_eda_o4l2.ipynb`
Expected: ejecuta sin error; aparecen `reports/figures/o4_fig17..23_*.png`, `reports/tables/o4_tab17..23_*.csv`, `reports/captions/o4_*.md` y entradas nuevas en `reports/INDEX.md`.

- [ ] **Step 7: Verificar artefactos y commit**

Run: `ls reports/figures/o4_fig1[7-9]_* reports/figures/o4_fig2[0-3]_* 2>/dev/null && grep -c "l2v1" reports/INDEX.md`
Expected: ficheros presentes; `grep -c` ≥ 7

```bash
git log --oneline -3
git add notebooks/04l2_eda_o4l2.py notebooks/04l2_eda_o4l2.ipynb reports/
git commit -m "Notebook L2: dos etapas causal, 7 artefactos (D1-D2, C1-C5)"
```

---

## Cierre de L2 (fuera del bucle de tasks, decisión del autor)

Tras completar las 11 tasks, el autor decide:
- Notas de memoria en `reports/memoria/06_o4_ml.md` (sección "L2 — Mejora con dos etapas") según `feedback-memoria-tone`. **Narrativa esperada:** L2 no rompe el techo estructural (no añade señal), pero tiene un tiro honesto en recuperar una ventaja de calibración/log-loss sobre persistencia que el rework causal le quitó al monolítico — porque dos etapas es el sesgo inductivo correcto para un proceso 77 % self-loop. NO se narra la fuga de O4 (registro interno).
- Entrada en `reports/ai-log/` (`00NN-o4l2-dos-etapas.md`) según la regla de scope.
- Tag `v0.4.3-o4l2-dos-etapas`.
- Actualizar `CLAUDE.md` (estado de L2) y la memoria `project-o4-improvement-lines`.

Estos pasos NO son tasks de implementación: dependen de los resultados numéricos y del juicio del autor sobre la narrativa.

---

## Self-Review

**Spec coverage:**
- F1 (dos etapas) → Tasks 3-5, 9, 10. ✓
- F2 (10 features causales) → Task 10 `_feature_cols` = `FEATURES_KINEMATIC + FEATURES_HMM`; `_prepare_causal_splits` adjunta el HMM causal. ✓
- F3 (etapa 1 poblacional) → Task 10 entrena clf_move sólo sobre `splits["poblacional"]`. ✓
- F4 (etapa 2B dos modos) → Task 10 itera `_MODES`. ✓
- F5 (RF+XGB, sin LGBM) → Task 9/10 sólo `("rf", "xgb")`. ✓
- F6 (soft+hard, τ por top-1) → Tasks 3,4,5,10. ✓
- F7 (hiperparámetros §8.6 sin tuning) → Task 9 config fija; etapa 2B reusa trainers causales de O4. ✓
- F8 (class weighting + calibración prefit/FrozenEstimator sobre val) → Task 9. ✓
- F9 (split idéntico) → Task 10 usa `split_temporal_per_bird`. ✓
- F10 (y_move de columnas existentes) → Task 2. ✓
- §6.1 ensamblaje causal (kin → matrix → split → fit_causal_hmm → decode → attach) → Task 10 `_prepare_causal_splits`. ✓
- §8.1 tests → Tasks 2-10. ✓
- §8.2 artefactos D1,D2,C1-C5 → Task 11. ✓
- §8.3 tabla central con baseline causal → Task 11 C1 + `metrics.parquet`. ✓
- §6.2 `evaluate_moves_only` → Task 8. ✓

**Placeholder scan:** Task 11 step 5 (C2-C5) describe artefactos sin todo el código inline; deliberado (variaciones del patrón completo de C1, dependientes de números reales, reutilizan `evaluate_by_state`/`pd.crosstab`). Las funciones núcleo (Tasks 1-10) llevan código completo, sin placeholders.

**Type consistency:**
- `combine_soft`/`combine_hard` devuelven `(n, n_classes)`; `predictions_from_proba`, `sweep_tau` y `evaluate_moves_only` consumen ese shape. ✓
- Columna de régimen `state_b_causal` consistente: `predictions_from_proba` la produce, `evaluate_by_state` la espera (`state_col="state_b_causal"`). ✓
- `_feature_cols(mode)` = `[*FEATURES_KINEMATIC, *FEATURES_HMM]` (+bird_id), idéntico a `build_o4` (build.py:184-185). Las features del HMM causal se adjuntan tras el split vía `_prepare_causal_splits`, NO desde `attrs["_features"]` (que sólo tiene cinemáticas). ✓
- `clf_move.classes_` → `pos_col` localiza la clase 1; `predict_proba[:, pos_col]` es `p_move`. ✓
- `metrics.parquet` columnas (`modelo, modo, regla, top1, top3, log_loss, dist_median_km`) consistentes entre Task 10 (escritura), su test (asserts) y Task 11 C1 (lectura). ✓
- `evaluate_moves_only(preds, y_move_test)` recibe `preds` con attrs `_proba`/`_classes` (los borra `preds_out`, una copia; `preds` conserva attrs). ✓
- Fixture de integración (Task 10) incluye `veg_low/veg_high/daylight_hours` requeridos por la emisión del HMM causal, igual que `tests/test_ml_build.py`. ✓
