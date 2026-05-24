# L3 — Regresión espacial con cuantiles · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar L3-v1, un regresor de cuantiles XGBoost del desplazamiento continuo `(Δlat, Δlon)` en dos modos (poblacional + individual 91916A), que reutiliza el esquema de predicciones de clasificación para apoyarse en `evaluate.py` sin tocarlo.

**Architecture:** Dos módulos nuevos en `src/tfg_aves/ml/`: `quantile.py` (funciones puras + wrapper fino de XGBRegressor + puente punto→celda + métricas de cuantil) y `build_l3.py` (orquestador `build_o4_l3` que replica el ensamblaje causal de `build_o4`, entrena 6 regresores por modo y persiste artefactos en `data/processed/o4/l3_v1/`). Las predicciones imitan el esquema de las de clasificación, de modo que `top_k_accuracy`, `dist_median_km`, `evaluate_by_state` y `evaluate_moves_only` operan sin cambios.

**Tech Stack:** Python 3.12, xgboost 3.2.0 (`reg:quantileerror`), pandas, numpy, scikit-learn (`BaseEstimator`/`RegressorMixin`), joblib, pytest, ruff, jupytext.

**Spec:** `docs/superpowers/specs/2026-05-24-o4l3-regresion-design.md`.

**Restricción transversal:** features causales (10), sin observar el futuro. El target `(Δlat,Δlon)` usa `t+1` pero es etiqueta, no feature.

---

## File Structure

- Modify: `src/tfg_aves/ml/_paths.py` — añade `O4_L3V1_DIR`.
- Create: `src/tfg_aves/ml/quantile.py` — regresión de cuantiles, puente a celda, métricas.
- Create: `src/tfg_aves/ml/build_l3.py` — orquestador `build_o4_l3` + `BuildO4L3Result`.
- Modify: `src/tfg_aves/ml/__init__.py` — exporta `build_o4_l3`, `BuildO4L3Result`.
- Create: `tests/test_ml_quantile.py` — tests unitarios de `quantile.py`.
- Create: `tests/test_ml_build_l3.py` — test de integración con fixture sintético.
- Create: `notebooks/04l3_eda_o4l3.py` — ejecuta el build sobre datos reales y emite 6 artefactos.

Convención de commits (CLAUDE.md): castellano, imperativo, breve, sin trailers. Ejecuta `uv run` para todo.

---

### Task 1: Constante de ruta `O4_L3V1_DIR`

**Files:**
- Modify: `src/tfg_aves/ml/_paths.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

Crea `tests/test_ml_quantile.py` con:

```python
"""Tests unitarios del módulo de regresión de cuantiles (L3)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_o4_l3v1_dir_exists():
    from tfg_aves.ml._paths import O4_L3V1_DIR, O4_OUT_DIR
    assert O4_L3V1_DIR == O4_OUT_DIR / "l3_v1"
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_o4_l3v1_dir_exists -q`
Expected: FAIL con `ImportError: cannot import name 'O4_L3V1_DIR'`.

- [ ] **Step 3: Añadir la constante**

En `src/tfg_aves/ml/_paths.py`, tras la línea `O4_L2V1_DIR`:

```python
O4_L3V1_DIR: Path = ROOT / "data" / "processed" / "o4" / "l3_v1"
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_o4_l3v1_dir_exists -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/_paths.py tests/test_ml_quantile.py
git commit -m "L3: ruta O4_L3V1_DIR para artefactos del regresor de cuantiles"
```

---

### Task 2: `derive_displacement_target`

**Files:**
- Create: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

Añade a `tests/test_ml_quantile.py`:

```python
def test_derive_displacement_target():
    from tfg_aves.ml.quantile import derive_displacement_target
    m = pd.DataFrame({
        "lat": [40.0, 41.0], "lon": [-3.0, -2.5],
        "lat_t_next": [40.5, 41.2], "lon_t_next": [-2.0, -2.7],
    })
    out = derive_displacement_target(m)
    assert np.allclose(out["y_dlat"], [0.5, 0.2])
    assert np.allclose(out["y_dlon"], [1.0, -0.2])
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_derive_displacement_target -q`
Expected: FAIL con `ModuleNotFoundError: tfg_aves.ml.quantile`.

- [ ] **Step 3: Crear el módulo con la función**

Crea `src/tfg_aves/ml/quantile.py`:

```python
"""Regresión de cuantiles del desplazamiento (L3 de O4).

Modela el target continuo (Δlat, Δlon) en grados con XGBoost
(`objective='reg:quantileerror'`), tres cuantiles {p10,p50,p90} por eje.
Las predicciones imitan el esquema de las de clasificación para reutilizar
``tfg_aves.ml.evaluate`` sin cambios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from tfg_aves.markov.discretize import _format_cell_id, assign_cell, haversine_km

QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
INDIVIDUAL_BIRD_ID: str = "91916A"   # ave con más histórico (rank 1, 2025 filas)
CELL_DEG: float = 0.5


def derive_displacement_target(matrix: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con y_dlat, y_dlon en grados.

    y_dlat = lat_t_next − lat ; y_dlon = lon_t_next − lon. No tiene fuga:
    ambas definen la etiqueta, no son features de entrada.
    """
    return pd.DataFrame({
        "y_dlat": matrix["lat_t_next"].to_numpy() - matrix["lat"].to_numpy(),
        "y_dlon": matrix["lon_t_next"].to_numpy() - matrix["lon"].to_numpy(),
    })
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_derive_displacement_target -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: derive_displacement_target (Δlat, Δlon en grados)"
```

---

### Task 3: `_XGBQuantileRegressor` y `fit_quantile_axis`

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

```python
def test_fit_quantile_axis_shapes_and_order():
    from tfg_aves.ml.quantile import QUANTILES, fit_quantile_axis
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=300)
    Xv = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    yv = (Xv["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=80)
    models = fit_quantile_axis(X, y, Xv, yv, seed=0)
    assert set(models) == set(QUANTILES)
    p10 = models[0.10].predict(Xv)
    p90 = models[0.90].predict(Xv)
    assert p10.shape == (80,)
    # En media, el cuantil 90 está por encima del 10.
    assert float(np.mean(p90 >= p10)) > 0.9
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_fit_quantile_axis_shapes_and_order -q`
Expected: FAIL con `ImportError: cannot import name 'fit_quantile_axis'`.

- [ ] **Step 3: Implementar el wrapper y el fit por eje**

Añade a `src/tfg_aves/ml/quantile.py`:

```python
class _XGBQuantileRegressor(BaseEstimator, RegressorMixin):
    """Wrapper fino sobre XGBRegressor(objective='reg:quantileerror').

    Config conservadora §8.6 de O4 adaptada a regresión (F8 del spec),
    idéntica para todos los cuantiles y modos; sólo varía quantile_alpha.
    Early stopping sobre val con la pérdida pinball por defecto del objetivo.
    Ningún modo usa bird_id, así que no hace falta codificar categóricas.
    """

    def __init__(self, quantile: float, seed: int = 0) -> None:
        self.quantile = quantile
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "_XGBQuantileRegressor":
        self._reg = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.quantile,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
        self._reg.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._reg.predict(X), dtype=np.float64)

    @property
    def best_iteration(self) -> int | None:
        return getattr(self._reg, "best_iteration", None)


def fit_quantile_axis(
    X_train: pd.DataFrame,
    y_train_axis: np.ndarray,
    X_val: pd.DataFrame,
    y_val_axis: np.ndarray,
    *,
    seed: int = 0,
) -> dict[float, _XGBQuantileRegressor]:
    """Entrena los 3 regresores {p10,p50,p90} para UN eje (Δlat o Δlon)."""
    models: dict[float, _XGBQuantileRegressor] = {}
    for q in QUANTILES:
        models[q] = _XGBQuantileRegressor(quantile=q, seed=seed).fit(
            X_train, np.asarray(y_train_axis, dtype=np.float64),
            X_val, np.asarray(y_val_axis, dtype=np.float64),
        )
    return models
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_fit_quantile_axis_shapes_and_order -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: _XGBQuantileRegressor + fit_quantile_axis (config §8.6 a regresión)"
```

---

### Task 4: `predict_quantiles` (monotonía + conteo de cruces)

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

```python
class _Const:
    """Modelo de juguete que predice una constante (para tests de monotonía)."""
    def __init__(self, v: float) -> None:
        self.v = v
    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.v, dtype=float)


def test_predict_quantiles_monotonic_and_crossings():
    from tfg_aves.ml.quantile import predict_quantiles
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    # lat: cuantiles DESORDENADOS (1.0, 0.0, 0.5) -> cruce en las 3 filas.
    models_lat = {0.10: _Const(1.0), 0.50: _Const(0.0), 0.90: _Const(0.5)}
    # lon: cuantiles ya ordenados (-0.5, 0.0, 0.5) -> sin cruces.
    models_lon = {0.10: _Const(-0.5), 0.50: _Const(0.0), 0.90: _Const(0.5)}
    preds, crossings = predict_quantiles(models_lat, models_lon, X)
    assert (preds["dlat_p10"] <= preds["dlat_p50"]).all()
    assert (preds["dlat_p50"] <= preds["dlat_p90"]).all()
    assert (preds["dlon_p10"] <= preds["dlon_p50"]).all()
    assert crossings["lat"] == 3
    assert crossings["lon"] == 0
    assert len(preds) == 3
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_predict_quantiles_monotonic_and_crossings -q`
Expected: FAIL con `ImportError: cannot import name 'predict_quantiles'`.

- [ ] **Step 3: Implementar**

Añade a `quantile.py`:

```python
def _axis_quantiles_sorted(
    models: dict[float, _XGBQuantileRegressor], X: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Devuelve (matriz (n,3) ordenada por fila, nº de filas con cruce)."""
    raw = np.column_stack([models[q].predict(X) for q in QUANTILES])
    ordered = raw[:, 0] <= raw[:, 1]
    ordered &= raw[:, 1] <= raw[:, 2]
    n_crossings = int(np.sum(~ordered))
    sorted_q = np.sort(raw, axis=1)
    return sorted_q, n_crossings


def predict_quantiles(
    models_lat: dict[float, _XGBQuantileRegressor],
    models_lon: dict[float, _XGBQuantileRegressor],
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Predice los 6 cuantiles, fuerza monotonía por eje (np.sort) y devuelve
    (DataFrame con dlat_p10/50/90, dlon_p10/50/90, índice 0..n-1; dict de cruces
    ANTES de ordenar, para el artefacto C5).
    """
    lat_q, n_cross_lat = _axis_quantiles_sorted(models_lat, X)
    lon_q, n_cross_lon = _axis_quantiles_sorted(models_lon, X)
    out = pd.DataFrame({
        "dlat_p10": lat_q[:, 0], "dlat_p50": lat_q[:, 1], "dlat_p90": lat_q[:, 2],
        "dlon_p10": lon_q[:, 0], "dlon_p50": lon_q[:, 1], "dlon_p90": lon_q[:, 2],
    })
    return out, {"lat": n_cross_lat, "lon": n_cross_lon}
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_predict_quantiles_monotonic_and_crossings -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: predict_quantiles con monotonía forzada y conteo de cruces"
```

---

### Task 5: `point_to_cell` y `nearest_cells`

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

```python
def test_point_to_cell_known():
    from tfg_aves.ml.quantile import point_to_cell
    cells = pd.DataFrame({
        "cell_id": ["80_-6"], "cell_lat_idx": [80], "cell_lon_idx": [-6],
        "lat_c": [40.25], "lon_c": [-2.75],
    })
    # (40.3, -2.7): i=floor(40.3/0.5)=80, j=floor(-2.7/0.5)=-6 -> "80_-6".
    # (10.0, 10.0): celda "20_20", no presente en cells -> is_active False.
    out = point_to_cell(np.array([40.3, 10.0]), np.array([-2.7, 10.0]), cells)
    assert out["cell_id"].iloc[0] == "80_-6"
    assert bool(out["is_active"].iloc[0]) is True
    assert np.isclose(out["cent_lat"].iloc[0], 40.25)
    assert np.isclose(out["cent_lon"].iloc[0], -2.75)
    assert bool(out["is_active"].iloc[1]) is False


def test_nearest_cells_orders_by_distance():
    from tfg_aves.ml.quantile import nearest_cells
    cells = pd.DataFrame({
        "cell_id": ["A", "B", "C"],
        "lat_c": [40.0, 40.0, 50.0], "lon_c": [-3.0, -3.5, -3.0],
    })
    out = nearest_cells(40.0, -3.05, cells, k=2)
    assert out[0] == "A"
    assert set(out) == {"A", "B"}
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_point_to_cell_known tests/test_ml_quantile.py::test_nearest_cells_orders_by_distance -q`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementar**

Añade a `quantile.py`:

```python
def point_to_cell(
    lat_pred: np.ndarray, lon_pred: np.ndarray, cells: pd.DataFrame,
    cell_deg: float = CELL_DEG,
) -> pd.DataFrame:
    """Mapea cada punto a su celda contenedora (discretización al grid 0,5°).

    Devuelve un DataFrame con: cell_id (str del grid, activo o no), cent_lat,
    cent_lon (centroide analítico de la celda contenedora) e is_active (bool,
    si la celda está en ``cells``). El centroide es analítico para que la
    distancia vía centroide quede definida también fuera del grid activo.
    """
    lat_pred = np.asarray(lat_pred, dtype=np.float64)
    lon_pred = np.asarray(lon_pred, dtype=np.float64)
    i = np.floor(lat_pred / cell_deg).astype(int)
    j = np.floor(lon_pred / cell_deg).astype(int)
    cell_ids = [_format_cell_id(int(a), int(b)) for a, b in zip(i, j, strict=True)]
    cent_lat = (i + 0.5) * cell_deg
    cent_lon = (j + 0.5) * cell_deg
    active = set(cells["cell_id"].astype(str))
    is_active = np.array([c in active for c in cell_ids])
    return pd.DataFrame({
        "cell_id": cell_ids, "cent_lat": cent_lat, "cent_lon": cent_lon,
        "is_active": is_active,
    })


def nearest_cells(
    lat_pred: float, lon_pred: float, cells: pd.DataFrame, k: int = 3,
) -> list[str]:
    """k celdas activas cuyo centroide está más cerca del punto (proximidad)."""
    d = np.asarray(haversine_km(
        cells["lat_c"].to_numpy(), cells["lon_c"].to_numpy(),
        float(lat_pred), float(lon_pred),
    ))
    order = np.argsort(d)[:k]
    return [str(c) for c in cells["cell_id"].to_numpy()[order]]
```

Nota: `assign_cell` es escalar y aquí necesitamos vectorizar, por eso se
recalcula `floor` directamente (mismo cómputo, mismo origen del grid).

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_point_to_cell_known tests/test_ml_quantile.py::test_nearest_cells_orders_by_distance -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: point_to_cell (discretización) y nearest_cells (top-k por proximidad)"
```

---

### Task 6: `pinball_loss` e `interval_coverage`

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

```python
def test_pinball_loss_known():
    from tfg_aves.ml.quantile import pinball_loss
    y = np.array([1.0])
    # Sub-predicción de 1: q=0.9 penaliza 0.9; q=0.1 penaliza 0.1.
    assert np.isclose(pinball_loss(y, np.array([0.0]), 0.9), 0.9)
    assert np.isclose(pinball_loss(y, np.array([0.0]), 0.1), 0.1)


def test_interval_coverage():
    from tfg_aves.ml.quantile import interval_coverage
    y = np.arange(10).astype(float)          # 0..9
    p10 = np.full(10, 1.0)
    p90 = np.full(10, 8.0)                    # dentro de [1,8]: 1..8 = 8 valores
    assert np.isclose(interval_coverage(y, p10, p90), 0.8)
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_pinball_loss_known tests/test_ml_quantile.py::test_interval_coverage -q`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementar**

Añade a `quantile.py`:

```python
def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pérdida de cuantil (pinball) media para el cuantil q."""
    d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred_q, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def interval_coverage(
    y_true: np.ndarray, y_p10: np.ndarray, y_p90: np.ndarray,
) -> float:
    """Fracción de y_true dentro de [p10, p90]. Ideal ≈ 0.80."""
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y >= np.asarray(y_p10)) & (y <= np.asarray(y_p90))))
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_pinball_loss_known tests/test_ml_quantile.py::test_interval_coverage -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: pinball_loss e interval_coverage"
```

---

### Task 7: `build_regression_predictions` (esquema compatible con evaluate.py)

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test**

```python
def test_build_regression_predictions_schema():
    from tfg_aves.ml.evaluate import evaluate_by_state, evaluate_moves_only
    from tfg_aves.ml.quantile import build_regression_predictions
    cells = pd.DataFrame({
        "cell_id": ["80_-6", "81_-6"], "cell_lat_idx": [80, 81],
        "cell_lon_idx": [-6, -6], "lat_c": [40.25, 40.75], "lon_c": [-2.75, -2.75],
    })
    meta = pd.DataFrame({
        "bird_id": ["A", "A"],
        "date_utc": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "lat": [40.3, 40.3], "lon": [-2.7, -2.7],
        "lat_t_next": [40.8, 40.3], "lon_t_next": [-2.7, -2.7],
        "cell_id_t": ["80_-6", "80_-6"], "cell_id_t_next": ["81_-6", "80_-6"],
        "state_b_causal": [1, 0],
    })
    qp = pd.DataFrame({
        "dlat_p10": [0.1, -0.1], "dlat_p50": [0.5, 0.0], "dlat_p90": [0.9, 0.1],
        "dlon_p10": [-0.1, -0.1], "dlon_p50": [0.0, 0.0], "dlon_p90": [0.1, 0.1],
    })
    preds = build_regression_predictions(qp, meta, cells)
    for c in [
        "true_cell", "pred_cell_top1", "pred_cell_topk", "pred_prob_top1",
        "pred_dist_km", "state_b_causal", "pred_lat", "pred_lon",
        "dist_native_km", "in_interval_lat", "in_interval_lon",
    ]:
        assert c in preds.columns, f"falta columna {c}"
    # Fila 0: p50 desplaza +0.5 lat -> punto (40.8,-2.7) -> celda 81_-6 == verdad.
    assert preds["pred_cell_top1"].iloc[0] == "81_-6"
    # evaluate.py opera sin cambios sobre este esquema:
    tbl = evaluate_by_state(preds)
    assert {"state", "top1", "dist_median_km"}.issubset(tbl.columns)
    mo = evaluate_moves_only(preds, np.array([1, 0]))
    assert {"top1", "dist_median_km", "n_obs"}.issubset(mo)
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_build_regression_predictions_schema -q`
Expected: FAIL con `ImportError`.

- [ ] **Step 3: Implementar**

Añade a `quantile.py`:

```python
def build_regression_predictions(
    quantile_preds: pd.DataFrame,
    meta: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    k_top: int = 3,
) -> pd.DataFrame:
    """Ensambla predicciones con el MISMO esquema que la clasificación
    (true_cell, pred_cell_top1, pred_cell_topk, pred_prob_top1, pred_dist_km,
    state_b_causal) MÁS columnas de regresión (pred_lat, pred_lon,
    dist_native_km, dlat_p*/dlon_p*, in_interval_lat, in_interval_lon).

    - pred_cell_top1 = celda contenedora del punto p50 (discretización).
    - pred_cell_topk = k celdas activas más cercanas (proximidad).
    - pred_dist_km   = haversine(centroide de la celda contenedora, verdad t+1)
                       → comparable con dist_median_km de L1/L2.
    - dist_native_km = haversine(punto p50, verdad t+1) → métrica nativa.
    - in_interval_*  = el desplazamiento verdadero cae en [p10, p90] (por eje).
    """
    meta = meta.reset_index(drop=True)
    qp = quantile_preds.reset_index(drop=True)

    lat_t = meta["lat"].to_numpy(dtype=np.float64)
    lon_t = meta["lon"].to_numpy(dtype=np.float64)
    lat_next = meta["lat_t_next"].to_numpy(dtype=np.float64)
    lon_next = meta["lon_t_next"].to_numpy(dtype=np.float64)

    pred_lat = lat_t + qp["dlat_p50"].to_numpy()
    pred_lon = lon_t + qp["dlon_p50"].to_numpy()

    mapped = point_to_cell(pred_lat, pred_lon, cells)
    pred_dist_km = np.asarray(haversine_km(
        mapped["cent_lat"].to_numpy(), mapped["cent_lon"].to_numpy(),
        lat_next, lon_next,
    ))
    dist_native_km = np.asarray(haversine_km(pred_lat, pred_lon, lat_next, lon_next))

    topk = [
        nearest_cells(pl, pn, cells, k=k_top)
        for pl, pn in zip(pred_lat, pred_lon, strict=True)
    ]

    y_dlat_true = lat_next - lat_t
    y_dlon_true = lon_next - lon_t
    in_lat = (y_dlat_true >= qp["dlat_p10"].to_numpy()) & (
        y_dlat_true <= qp["dlat_p90"].to_numpy())
    in_lon = (y_dlon_true >= qp["dlon_p10"].to_numpy()) & (
        y_dlon_true <= qp["dlon_p90"].to_numpy())

    out = pd.DataFrame({
        "bird_id": meta["bird_id"].to_numpy(),
        "date_utc": meta["date_utc"].to_numpy(),
        "true_cell": meta["cell_id_t_next"].to_numpy(),
        "pred_cell_top1": mapped["cell_id"].to_numpy(),
        "pred_cell_topk": topk,
        "pred_prob_top1": np.nan,
        "pred_dist_km": pred_dist_km,
        "state_b_causal": meta["state_b_causal"].to_numpy(),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "dist_native_km": dist_native_km,
        "dlat_p10": qp["dlat_p10"].to_numpy(),
        "dlat_p50": qp["dlat_p50"].to_numpy(),
        "dlat_p90": qp["dlat_p90"].to_numpy(),
        "dlon_p10": qp["dlon_p10"].to_numpy(),
        "dlon_p50": qp["dlon_p50"].to_numpy(),
        "dlon_p90": qp["dlon_p90"].to_numpy(),
        "in_interval_lat": in_lat,
        "in_interval_lon": in_lon,
        "pred_cell_in_active_grid": mapped["is_active"].to_numpy(),
    })
    return out
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_quantile.py::test_build_regression_predictions_schema -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/tfg_aves/ml/quantile.py
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "L3: build_regression_predictions con esquema compatible con evaluate.py"
```

---

### Task 8: Orquestador `build_o4_l3`

**Files:**
- Create: `src/tfg_aves/ml/build_l3.py`
- Test: `tests/test_ml_build_l3.py` (se crea aquí con un único test rápido de humo sobre el helper; el test de integración completo va en la Task 9)

- [ ] **Step 1: Escribir el test de humo del helper de splits**

Crea `tests/test_ml_build_l3.py`:

```python
"""Tests de integración de build_o4_l3 con un fixture sintético causal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos consecutivos sobre celdas activas (espejo de
    tests/test_ml_build_l2.py)."""
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=90)
    rows = []
    for b in birds:
        lat0 = 40.0 + rng.uniform(-1, 1)
        lon0 = -3.0 + rng.uniform(-1, 1)
        for i, d in enumerate(dates):
            rows.append({
                "bird_id": b, "date_utc": d,
                "lat": lat0 + 0.05 * i + rng.normal(0, 0.05),
                "lon": lon0 + 0.05 * i + rng.normal(0, 0.05),
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
                "is_observation_valid": True,
            })
    feat_path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(feat_path)

    cells = []
    for i in range(78, 86):
        for j in range(-9, -1):
            cells.append({
                "cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5, "n_obs_total": 30,
            })
    cells_path = tmp_path / "cells.parquet"
    pd.DataFrame(cells).to_parquet(cells_path)
    return feat_path, cells_path


def test_prepare_poblacional_split_attaches_hmm(tmp_path):
    from tfg_aves.ml.build_l3 import _prepare_poblacional_split
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    features_o3 = pd.read_parquet(feat_path)
    cells = pd.read_parquet(cells_path)
    train, val, test = _prepare_poblacional_split(features_o3, cells, seed=0)
    for df in (train, val, test):
        assert "state_b_causal" in df.columns
        assert "posterior_b_migracion_causal" in df.columns
        assert "bird_id" not in [c for c in df.columns if c == "bird_id_feature"]
    assert len(train) > 0 and len(test) > 0
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_build_l3.py::test_prepare_poblacional_split_attaches_hmm -q`
Expected: FAIL con `ModuleNotFoundError: tfg_aves.ml.build_l3`.

- [ ] **Step 3: Implementar el orquestador**

Crea `src/tfg_aves/ml/build_l3.py`:

```python
"""Orquestador del pipeline L3 (regresión de cuantiles) de O4 causal."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V1_DIR
from .evaluate import (
    compute_persistence_baseline,
    dist_median_km,
    evaluate_by_state,
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
from .quantile import (
    QUANTILES,
    build_regression_predictions,
    derive_displacement_target,
    fit_quantile_axis,
    interval_coverage,
    pinball_loss,
    predict_quantiles,
)

_FEATURES = [*FEATURES_KINEMATIC, *FEATURES_HMM]
_MODES = ("poblacional", "individual")


@dataclass
class BuildO4L3Result:
    """Resumen serializable de build_o4_l3."""

    n_rows_train_pob: int
    n_rows_test_pob: int
    n_rows_train_ind: int
    n_rows_test_ind: int
    individual_bird_id: str
    n_crossings: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_path: Path = Path()
    metrics_path: Path = Path()


def _prepare_poblacional_split(
    features_o3: pd.DataFrame, cells: pd.DataFrame, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replica el ensamblaje causal de build_o4 SOLO para el modo poblacional
    (sin bird_id) y adjunta state_b_causal/posterior_b_migracion_causal del HMM
    causal global. Se replica (en vez de importar de build.py) para no tocar el
    build.py estabilizado; usa exactamente las mismas funciones leak-free.
    """
    kin = compute_causal_kinematics(features_o3)
    matrix = build_feature_matrix(kin, cells, include_bird_id=False)
    train, val, test = split_temporal_per_bird(matrix)

    cutoff_by_bird = (
        train.assign(_d=pd.to_datetime(train["date_utc"]))
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

    return attach(train), attach(val), attach(test)


def _y_move(df: pd.DataFrame) -> np.ndarray:
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).to_numpy()


def _metric_rows(
    preds: pd.DataFrame, y_move: np.ndarray, modo_label: str,
    *, pinball_lat: float, pinball_lon: float, cov_lat: float, cov_lon: float,
) -> list[dict]:
    """Filas de métrica por estado (global/estacionario/migración) + moves."""
    rows: list[dict] = []
    by = evaluate_by_state(preds)
    for _, r in by.iterrows():
        scope = r["state"]
        if scope == "estacionario":
            sub = preds[preds["state_b_causal"] == 0]
        elif scope == "migración":
            sub = preds[preds["state_b_causal"] == 1]
        else:
            sub = preds
        native = float(np.median(sub["dist_native_km"])) if len(sub) else np.nan
        is_global = scope == "global"
        rows.append({
            "modo": modo_label, "scope": scope, "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": native,
            "pinball_lat": pinball_lat if is_global else np.nan,
            "pinball_lon": pinball_lon if is_global else np.nan,
            "coverage_lat": cov_lat if is_global else np.nan,
            "coverage_lon": cov_lon if is_global else np.nan,
        })
    mask = np.asarray(y_move).astype(bool)
    mo = evaluate_moves_only(preds, mask)
    native_moves = (
        float(np.median(preds[mask]["dist_native_km"])) if mask.any() else np.nan
    )
    rows.append({
        "modo": modo_label, "scope": "moves", "n_obs": mo["n_obs"],
        "top1": mo["top1"], "top3": mo["top3"],
        "dist_centroide_km": mo["dist_median_km"], "dist_nativa_km": native_moves,
        "pinball_lat": np.nan, "pinball_lon": np.nan,
        "coverage_lat": np.nan, "coverage_lon": np.nan,
    })
    return rows


def _baseline_rows(preds: pd.DataFrame, modo_label: str) -> list[dict]:
    by = evaluate_by_state(preds)
    rows = []
    for _, r in by.iterrows():
        rows.append({
            "modo": modo_label, "scope": r["state"], "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": np.nan,
            "pinball_lat": np.nan, "pinball_lon": np.nan,
            "coverage_lat": np.nan, "coverage_lon": np.nan,
        })
    return rows


def build_o4_l3(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L3V1_DIR,
    seed: int = 0,
    *,
    individual_bird_id: str | None = None,
) -> BuildO4L3Result:
    """Pipeline L3-v1: regresor de cuantiles en modo poblacional + individual.

    individual_bird_id: ave del modo individual. Por defecto INDIVIDUAL_BIRD_ID
    (91916A). Se expone como parámetro para los tests con fixtures sintéticos.
    """
    from .quantile import INDIVIDUAL_BIRD_ID
    if individual_bird_id is None:
        individual_bird_id = INDIVIDUAL_BIRD_ID

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    train_pob, val_pob, test_pob = _prepare_poblacional_split(features_o3, cells, seed)

    model_paths: dict[str, Path] = {}
    n_crossings: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict[str, float]] = {}
    metric_rows: list[dict] = []
    preds_frames: list[pd.DataFrame] = []
    # Guardamos las preds poblacionales para el corte @individual.
    preds_pob_full: pd.DataFrame | None = None

    splits_by_mode = {
        "poblacional": (train_pob, val_pob, test_pob),
        "individual": tuple(
            d[d["bird_id"] == individual_bird_id].reset_index(drop=True)
            for d in (train_pob, val_pob, test_pob)
        ),
    }

    for mode in _MODES:
        train_m, val_m, test_m = splits_by_mode[mode]
        tgt_tr = derive_displacement_target(train_m)
        tgt_va = derive_displacement_target(val_m)

        models_lat = fit_quantile_axis(
            train_m[_FEATURES], tgt_tr["y_dlat"].to_numpy(),
            val_m[_FEATURES], tgt_va["y_dlat"].to_numpy(), seed=seed,
        )
        models_lon = fit_quantile_axis(
            train_m[_FEATURES], tgt_tr["y_dlon"].to_numpy(),
            val_m[_FEATURES], tgt_va["y_dlon"].to_numpy(), seed=seed,
        )

        qp_test, crossings = predict_quantiles(models_lat, models_lon, test_m[_FEATURES])
        n_crossings[mode] = crossings
        preds = build_regression_predictions(qp_test, test_m, cells)

        # Pinball (media sobre los 3 cuantiles) y cobertura por eje.
        tgt_te = derive_displacement_target(test_m)
        pin_lat = float(np.mean([
            pinball_loss(tgt_te["y_dlat"].to_numpy(), qp_test[f"dlat_p{int(q*100):02d}"].to_numpy(), q)
            for q in QUANTILES
        ]))
        pin_lon = float(np.mean([
            pinball_loss(tgt_te["y_dlon"].to_numpy(), qp_test[f"dlon_p{int(q*100):02d}"].to_numpy(), q)
            for q in QUANTILES
        ]))
        cov_lat = interval_coverage(
            tgt_te["y_dlat"].to_numpy(), qp_test["dlat_p10"].to_numpy(), qp_test["dlat_p90"].to_numpy())
        cov_lon = interval_coverage(
            tgt_te["y_dlon"].to_numpy(), qp_test["dlon_p10"].to_numpy(), qp_test["dlon_p90"].to_numpy())
        coverage[mode] = {"lat": cov_lat, "lon": cov_lon}

        y_move_m = _y_move(test_m)
        metric_rows.extend(_metric_rows(
            preds, y_move_m, mode,
            pinball_lat=pin_lat, pinball_lon=pin_lon, cov_lat=cov_lat, cov_lon=cov_lon,
        ))

        for q in QUANTILES:
            tag = f"p{int(q * 100):02d}"
            for axis, models in (("dlat", models_lat), ("dlon", models_lon)):
                mp = out_dir / f"model_{mode}_{axis}_{tag}.pkl"
                joblib.dump({
                    "model": models[q], "feature_cols": _FEATURES,
                    "mode": mode, "axis": axis, "quantile": q,
                }, mp)
                model_paths[f"{mode}_{axis}_{tag}"] = mp

        preds_out = preds.copy()
        preds_out["modo"] = mode
        preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
        preds_frames.append(preds_out)
        if mode == "poblacional":
            preds_pob_full = preds.copy()

    # --- Corte poblacional@individual: mismas filas del individual ---
    assert preds_pob_full is not None
    pob_at_ind = preds_pob_full[
        preds_pob_full["bird_id"] == individual_bird_id
    ].reset_index(drop=True)
    if len(pob_at_ind) > 0:
        test_ind = splits_by_mode["individual"][2]
        metric_rows.extend(_metric_rows(
            pob_at_ind, _y_move(test_ind), f"poblacional@{individual_bird_id}",
            pinball_lat=np.nan, pinball_lon=np.nan, cov_lat=np.nan, cov_lon=np.nan,
        ))

    # --- Baselines de persistencia (test completo + corte individual) ---
    persistence = compute_persistence_baseline(test_pob, cells=cells)
    metric_rows.extend(_baseline_rows(persistence, "persistencia"))
    pers_ind = persistence[persistence["bird_id"] == individual_bird_id]
    if len(pers_ind) > 0:
        metric_rows.extend(_baseline_rows(
            pers_ind.reset_index(drop=True), f"persistencia@{individual_bird_id}"))

    # --- Guardar artefactos ---
    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path)

    predictions_df = pd.concat(preds_frames, ignore_index=True)
    predictions_path = out_dir / "predictions_test.parquet"
    predictions_df.to_parquet(predictions_path)

    ind_train = splits_by_mode["individual"][0]
    ind_test = splits_by_mode["individual"][2]
    return BuildO4L3Result(
        n_rows_train_pob=len(train_pob),
        n_rows_test_pob=len(test_pob),
        n_rows_train_ind=len(ind_train),
        n_rows_test_ind=len(ind_test),
        individual_bird_id=individual_bird_id,
        n_crossings=n_crossings,
        coverage=coverage,
        model_paths=model_paths,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_build_l3.py::test_prepare_poblacional_split_attaches_hmm -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/tfg_aves/ml/build_l3.py
git add src/tfg_aves/ml/build_l3.py tests/test_ml_build_l3.py
git commit -m "L3: orquestador build_o4_l3 (poblacional + individual 91916A)"
```

---

### Task 9: Tests de integración de `build_o4_l3`

**Files:**
- Modify: `tests/test_ml_build_l3.py`

- [ ] **Step 1: Escribir los tests**

Añade a `tests/test_ml_build_l3.py`:

```python
def test_build_o4_l3_artifacts(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    result = build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    # 12 modelos: 2 modos × 2 ejes × 3 cuantiles.
    for mode in ("poblacional", "individual"):
        for axis in ("dlat", "dlon"):
            for tag in ("p10", "p50", "p90"):
                assert (out_dir / f"model_{mode}_{axis}_{tag}.pkl").exists()
    assert (out_dir / "predictions_test.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"modo", "scope", "top1", "dist_centroide_km", "coverage_lat"}.issubset(
        metrics.columns)
    # Hay filas para ambos modos, el corte @A, persistencia y persistencia@A.
    assert "poblacional" in set(metrics["modo"])
    assert "individual" in set(metrics["modo"])
    assert "poblacional@A" in set(metrics["modo"])
    assert "persistencia" in set(metrics["modo"])
    # La cobertura global está reportada (no NaN) para cada modo.
    glob = metrics[(metrics["modo"] == "poblacional") & (metrics["scope"] == "global")]
    assert not np.isnan(glob["coverage_lat"].iloc[0])
    assert result.n_rows_test_ind > 0


def test_build_o4_l3_individual_subset(tmp_path):
    """El test del modo individual coincide EXACTAMENTE con el subconjunto
    bird_id==A del test poblacional (mismo split temporal por ave)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    ind = preds[preds["modo"] == "individual"]
    pob_a = preds[(preds["modo"] == "poblacional") & (preds["bird_id"] == "A")]
    assert len(ind) == len(pob_a)
    assert set(ind["date_utc"]) == set(pob_a["date_utc"])


def test_build_o4_l3_no_leakage(tmp_path):
    """Ningún modelo entrena con columnas del futuro: feature_cols == las 10
    causales, sin lat_t_next/lon_t_next/cell_id_t_next/y_dlat/y_dlon."""
    import joblib
    from tfg_aves.ml.build_l3 import build_o4_l3
    from tfg_aves.ml.features import FEATURES_HMM, FEATURES_KINEMATIC
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v1"
    build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    expected = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    payload = joblib.load(out_dir / "model_poblacional_dlat_p50.pkl")
    assert payload["feature_cols"] == expected
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon"}
    assert forbidden.isdisjoint(set(payload["feature_cols"]))


def test_build_o4_l3_idempotent(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_a,
                seed=0, individual_bird_id="A")
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_b,
                seed=0, individual_bird_id="A")
    ma = pd.read_parquet(out_a / "metrics.parquet").sort_values(["modo", "scope"]).reset_index(drop=True)
    mb = pd.read_parquet(out_b / "metrics.parquet").sort_values(["modo", "scope"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ma, mb)
```

- [ ] **Step 2: Ejecutar y verificar que pasan**

Run: `uv run pytest tests/test_ml_build_l3.py -q`
Expected: PASS (5 passed). Puede tardar ~30-90 s por el HMM + 24 regresores sobre el fixture.

Si `test_build_o4_l3_idempotent` falla por diferencias mínimas de punto flotante, revisa que XGBoost use `tree_method="hist"` y `random_state=seed`; si persiste una diferencia no determinista, compara con `check_exact=False, atol=1e-6` en `assert_frame_equal`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ml_build_l3.py
git commit -m "L3: tests de integración de build_o4_l3 (artefactos, subset individual, leak-free, idempotencia)"
```

---

### Task 10: Exportar la API en `__init__.py`

**Files:**
- Modify: `src/tfg_aves/ml/__init__.py`
- Test: `tests/test_ml_build_l3.py`

- [ ] **Step 1: Escribir el test**

Añade a `tests/test_ml_build_l3.py`:

```python
def test_build_o4_l3_exported():
    import tfg_aves.ml as ml
    assert hasattr(ml, "build_o4_l3")
    assert hasattr(ml, "BuildO4L3Result")
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `uv run pytest tests/test_ml_build_l3.py::test_build_o4_l3_exported -q`
Expected: FAIL con `AttributeError`.

- [ ] **Step 3: Añadir el import y el `__all__`**

En `src/tfg_aves/ml/__init__.py`, añade tras el import de `build`:

```python
from tfg_aves.ml.build_l3 import BuildO4L3Result, build_o4_l3
```

Y añade `"BuildO4L3Result"` y `"build_o4_l3"` a la lista `__all__` (orden alfabético).

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `uv run pytest tests/test_ml_build_l3.py::test_build_o4_l3_exported -q`
Expected: PASS.

- [ ] **Step 5: Suite completa + ruff + commit**

```bash
uv run pytest -q
uv run ruff check src tests
git add src/tfg_aves/ml/__init__.py tests/test_ml_build_l3.py
git commit -m "L3: exporta build_o4_l3 y BuildO4L3Result en tfg_aves.ml"
```

Expected: toda la suite (≥158 tests) en verde, ruff limpio.

---

### Task 11: Notebook EDA + 6 artefactos sobre datos reales

**Files:**
- Create: `notebooks/04l3_eda_o4l3.py`

Este task ejecuta `build_o4_l3()` sobre los datos reales (tarda varios minutos por el HMM + 24 regresores) y emite los 6 artefactos D1, C1-C5 vía `save_artifact`. Como produce ficheros versionados en `reports/`, ejecuta el notebook como script.

- [ ] **Step 1: Crear el notebook (jupytext percent)**

Crea `notebooks/04l3_eda_o4l3.py`:

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
# # O4 · L3 — Regresión espacial con cuantiles
#
# Ejecuta `build_o4_l3` (modo poblacional + individual 91916A) y genera los
# 6 artefactos (D1, C1..C5). Ataca Mo1 (target categórico) y D3 (rutas
# individuales). Baseline categórico = O4 monolítico causal poblacional.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V1_DIR, O4_OUT_DIR
from tfg_aves.ml.build_l3 import _prepare_poblacional_split, build_o4_l3
from tfg_aves.ml.quantile import INDIVIDUAL_BIRD_ID, derive_displacement_target
from tfg_aves.reporting import save_artifact

result = build_o4_l3()
print("filas pob train/test:", result.n_rows_train_pob, result.n_rows_test_pob)
print("filas ind train/test:", result.n_rows_train_ind, result.n_rows_test_ind)
print("cobertura:", result.coverage)
print("cruces de cuantil:", result.n_crossings)

metrics = pd.read_parquet(O4_L3V1_DIR / "metrics.parquet")
preds = pd.read_parquet(O4_L3V1_DIR / "predictions_test.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")  # O4 causal (L3-v0)

# Splits reales para D1 (distribución del target y conteo de histórico por ave).
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
train_pob, _, _ = _prepare_poblacional_split(features_o3, cells, seed=0)
```

- [ ] **Step 2: Añadir D1 — distribución del target + histórico por ave**

```python
# %% [markdown]
# ## D1 — Distribución del target y selección del individuo
# %%
tgt = derive_displacement_target(train_pob)
hist = (
    train_pob.groupby("bird_id").size().sort_values(ascending=False)
    .head(8).rename("n_filas_train").reset_index()
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(tgt["y_dlat"], bins=80, alpha=0.7, label="Δlat")
axes[0].hist(tgt["y_dlon"], bins=80, alpha=0.7, label="Δlon")
axes[0].set_yscale("log")
axes[0].set_xlabel("desplazamiento (grados)"); axes[0].set_ylabel("frecuencia (log)")
axes[0].set_title("Distribución del target Δ"); axes[0].legend()
axes[1].barh(hist["bird_id"][::-1], hist["n_filas_train"][::-1], color="steelblue")
axes[1].set_xlabel("filas de entrenamiento")
axes[1].set_title(f"Histórico por ave (top 8) — individual: {INDIVIDUAL_BIRD_ID}")
fig.tight_layout()
save_artifact(
    "target-dist-bird-history", objective="o4", num=24,
    decision=("Target continuo concentrado en cero (justifica cuantiles + pinball); "
              f"{INDIVIDUAL_BIRD_ID} es el ave con más histórico (modo individual)."),
    caption_es=(
        "Izquierda: distribución del desplazamiento diario (Δlat, Δlon) en escala "
        "logarítmica, fuertemente concentrada en cero por el dominio de días "
        "estacionarios, con colas de migración. Justifica modelar tres cuantiles "
        "{p10, p50, p90} y evaluar con pérdida pinball. Derecha: número de filas de "
        f"entrenamiento por ave; {INDIVIDUAL_BIRD_ID} encabeza el histórico y se elige "
        "para el modelo individual de L3."),
    fig=fig, table=hist,
)
plt.close(fig)
```

- [ ] **Step 3: Añadir C1 — calibración (cobertura vs 80%)**

```python
# %% [markdown]
# ## C1 — Calibración de la incertidumbre
# %%
cov_rows = []
for mode in ("poblacional", "individual"):
    sub = preds[preds["modo"] == mode]
    cov_rows.append({"modo": mode,
                     "cobertura_lat": float(sub["in_interval_lat"].mean()),
                     "cobertura_lon": float(sub["in_interval_lon"].mean())})
cov_df = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(cov_df)); w = 0.35
ax.bar(x - w / 2, cov_df["cobertura_lat"], w, label="Δlat")
ax.bar(x + w / 2, cov_df["cobertura_lon"], w, label="Δlon")
ax.axhline(0.80, color="red", ls="--", label="nominal 80%")
ax.set_xticks(x); ax.set_xticklabels(cov_df["modo"])
ax.set_ylabel("cobertura empírica de [p10, p90]"); ax.set_ylim(0, 1)
ax.set_title("Calibración del intervalo de incertidumbre"); ax.legend()
fig.tight_layout()
save_artifact(
    "calibration-coverage", objective="o4", num=25,
    decision="Cobertura empírica del intervalo [p10,p90] cercana al 80% nominal.",
    caption_es=(
        "Cobertura empírica del intervalo de predicción [p10, p90] frente al 80% "
        "nominal, por modo y eje. Una cobertura próxima al 80% indica que la "
        "incertidumbre del desplazamiento está bien calibrada, contribución "
        "diferencial de la regresión de cuantiles frente al clasificador."),
    fig=fig, table=cov_df,
)
plt.close(fig)
```

- [ ] **Step 4: Añadir C2 — distribución de distancias (poblacional)**

```python
# %% [markdown]
# ## C2 — Distribución de distancias (ablación de target, poblacional)
# %%
preds_pob = preds[preds["modo"] == "poblacional"]
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
v0_xgb_pob = preds_v0[(preds_v0["modelo"] == "xgb") & (preds_v0["modo"] == "poblacional")]
v0_pers = preds_v0[preds_v0["modelo"] == "persistencia"]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 300, 60)
ax.hist(preds_pob["pred_dist_km"], bins=bins, alpha=0.6, density=True, label="L3 cuantil")
ax.hist(v0_xgb_pob["pred_dist_km"], bins=bins, alpha=0.6, density=True, label="L3-v0 categórico")
ax.hist(v0_pers["pred_dist_km"], bins=bins, alpha=0.4, density=True, label="persistencia")
ax.set_xlabel("distancia vía centroide (km)"); ax.set_ylabel("densidad")
ax.set_title("Distribución de distancias en test (poblacional)"); ax.legend()
fig.tight_layout()
dist_tbl = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["modo", "scope", "dist_centroide_km", "dist_nativa_km"]]
save_artifact(
    "distance-distribution", objective="o4", num=26,
    decision="Comparación de la distancia del error: L3 cuantil vs categórico vs persistencia.",
    caption_es=(
        "Distribución de la distancia (vía centroide) entre la predicción y la "
        "posición real en el conjunto de test (modo poblacional), comparando la "
        "regresión de cuantiles de L3, el clasificador categórico L3-v0 y la "
        "persistencia trivial. La ventaja de la regresión, si existe, se concentra "
        "en los días de movimiento (cola de la distribución)."),
    fig=fig, table=dist_tbl,
)
plt.close(fig)
```

- [ ] **Step 5: Añadir C3 — tablas comparativas A y B**

```python
# %% [markdown]
# ## C3 — Comparativas centrales (A: ablación de target; B: per-individuo)
# %%
tabla_a = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["modo", "scope", "n_obs", "top1", "top3", "dist_centroide_km",
     "dist_nativa_km", "pinball_lat", "pinball_lon", "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
ind_id = result.individual_bird_id
tabla_b = metrics[metrics["modo"].isin(
    ["individual", f"poblacional@{ind_id}", f"persistencia@{ind_id}"])][
    ["modo", "scope", "n_obs", "top1", "dist_centroide_km", "dist_nativa_km",
     "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
tabla_comparativa = pd.concat([
    tabla_a.assign(bloque="A_target_poblacional"),
    tabla_b.assign(bloque="B_per_individuo"),
], ignore_index=True)
print(tabla_comparativa.to_string())
save_artifact(
    "comparativa-l3", objective="o4", num=27,
    decision=("L3 cuantil vs categórico (A, target) y individual vs poblacional "
              "sobre 91916A (B, per-individuo)."),
    caption_es=(
        "Comparativas centrales de L3. Bloque A: efecto de reformular el target "
        "(categórico L3-v0 vs cuantil L3-v1) en el modo poblacional, sobre todo el "
        "test, por estado HMM y en días de movimiento. Bloque B: hipótesis "
        "per-individuo, modelo individual de 91916A frente al poblacional evaluado "
        "sobre las mismas filas. Sin columna log-loss: L3 no produce una "
        "distribución categórica, su veredicto es geométrico (distancia + top-1 "
        "mapeado)."),
    table=tabla_comparativa,
)
```

- [ ] **Step 6: Añadir C4 — mapa de vectores Δ de 91916A**

```python
# %% [markdown]
# ## C4 — Vectores de desplazamiento predichos vs reales (91916A)
# %%
ind = preds[preds["modo"] == "individual"].copy().sort_values("date_utc").reset_index(drop=True)
sample = ind.iloc[:: max(1, len(ind) // 40)].copy()  # ~40 días para legibilidad
fig, ax = plt.subplots(figsize=(7, 7))
# Real (gris) vs predicho p50 (azul) desde la posición de t.
true_lat = sample["pred_lat"] - (sample["dlat_p50"])  # = lat_t
true_lon = sample["pred_lon"] - (sample["dlon_p50"])  # = lon_t
ax.quiver(true_lon, true_lat,
          sample["pred_lon"] - true_lon, sample["pred_lat"] - true_lat,
          angles="xy", scale_units="xy", scale=1, color="steelblue",
          width=0.004, label="predicho p50")
# Banda de incertidumbre: rango p10-p90 como segmento en cada eje.
ax.errorbar(sample["pred_lon"], sample["pred_lat"],
            xerr=[(sample["dlon_p50"] - sample["dlon_p10"]).abs(),
                  (sample["dlon_p90"] - sample["dlon_p50"]).abs()],
            yerr=[(sample["dlat_p50"] - sample["dlat_p10"]).abs(),
                  (sample["dlat_p90"] - sample["dlat_p50"]).abs()],
            fmt="none", ecolor="orange", alpha=0.5, label="banda [p10,p90]")
ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
ax.set_title(f"Desplazamientos predichos de {result.individual_bird_id} (muestra)")
ax.legend()
fig.tight_layout()
save_artifact(
    "vectores-desplazamiento-91916A", objective="o4", num=28,
    decision="Vectores de desplazamiento p50 del modelo individual con banda de incertidumbre.",
    caption_es=(
        f"Vectores de desplazamiento diario predichos (mediana p50) por el modelo "
        f"individual de {result.individual_bird_id} sobre una muestra de su test, con "
        "la banda de incertidumbre [p10, p90] por eje. Ilustra la salida geométrica "
        "e interpretable de la regresión de cuantiles, no disponible en el "
        "clasificador categórico."),
    fig=fig, table=sample[["date_utc", "pred_lat", "pred_lon",
                           "dlat_p10", "dlat_p90", "dlon_p10", "dlon_p90"]],
)
plt.close(fig)
```

- [ ] **Step 7: Añadir C5 — incidencia de quantile crossing**

```python
# %% [markdown]
# ## C5 — Incidencia de quantile crossing
# %%
cross_rows = []
for mode, d in result.n_crossings.items():
    n_test = result.n_rows_test_pob if mode == "poblacional" else result.n_rows_test_ind
    cross_rows.append({"modo": mode, "n_test": n_test,
                       "cruces_lat": d["lat"], "cruces_lon": d["lon"],
                       "pct_lat": 100 * d["lat"] / max(1, n_test),
                       "pct_lon": 100 * d["lon"] / max(1, n_test)})
cross_df = pd.DataFrame(cross_rows)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(cross_df)); w = 0.35
ax.bar(x - w / 2, cross_df["pct_lat"], w, label="Δlat")
ax.bar(x + w / 2, cross_df["pct_lon"], w, label="Δlon")
ax.set_xticks(x); ax.set_xticklabels(cross_df["modo"])
ax.set_ylabel("% de filas con cruce (antes de ordenar)")
ax.set_title("Incidencia de quantile crossing"); ax.legend()
fig.tight_layout()
save_artifact(
    "quantile-crossing", objective="o4", num=29,
    decision="Incidencia de quantile crossing corregida por ordenación post-hoc.",
    caption_es=(
        "Porcentaje de filas donde los cuantiles predichos se cruzan (p10>p50 o "
        "p50>p90) antes de la corrección monótona post-hoc, por modo y eje. Una "
        "incidencia baja confirma que los regresores independientes producen "
        "cuantiles coherentes; la ordenación garantiza monotonía en todo caso."),
    fig=fig, table=cross_df,
)
plt.close(fig)
print("Artefactos L3 generados (D1, C1..C5).")
```

- [ ] **Step 8: Sincronizar a .ipynb y ejecutar el notebook completo**

```bash
uv run jupytext --to ipynb notebooks/04l3_eda_o4l3.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04l3_eda_o4l3.ipynb
```

Expected: ejecución sin errores (varios minutos). Se crean
`reports/figures/o4_fig24..29_*.png`, `reports/tables/o4_*24..29*.csv`,
`reports/captions/o4_*24..29*.md` y entradas nuevas en `reports/INDEX.md`.

Si `save_artifact` lanza por `num` ya usado (L1/L2 llegaron a 23), incrementa
los `num` a partir del primero libre y mantén la coherencia entre los 6.

- [ ] **Step 9: Commit**

```bash
git add notebooks/04l3_eda_o4l3.py notebooks/04l3_eda_o4l3.ipynb \
        reports/figures reports/tables reports/captions reports/INDEX.md
git commit -m "L3: notebook EDA y 6 artefactos (D1, C1-C5) del regresor de cuantiles"
```

---

### Task 12: Cierre — verificación final, notas de memoria, ai-log, CLAUDE.md, tag

**Files:**
- Modify: `reports/memoria/06_o4_ml.md` (sección "L3 — Regresión con cuantiles")
- Create: `reports/ai-log/00NN-o4l3-regresion.md`
- Modify: `CLAUDE.md` (estado de L3 → completado), `MEMORY.md` / memoria del asistente

- [ ] **Step 1: Verificación final (verification-before-completion)**

```bash
uv run pytest -q
uv run ruff check src tests
```

Expected: toda la suite en verde (≥159 tests con los nuevos), ruff limpio. Pega
la salida real; no afirmar éxito sin ver el output.

- [ ] **Step 2: Notas de memoria**

Añade a `reports/memoria/06_o4_ml.md` una sección "L3 — Regresión con
cuantiles" con: marco metodológico (target continuo, ataca Mo1+D3), los
números reales de las tablas A y B (rellenar desde `metrics.parquet`),
interpretación según §9 del spec (qué criterios se cumplieron), la
calibración (C1) y el cierre de la decisión global vs per-individuo.
Tono investigador, no apologético ([[feedback-memoria-tone]]).

- [ ] **Step 3: Entrada en ai-log**

Crea `reports/ai-log/00NN-o4l3-regresion.md` (NN = siguiente libre) según
`reports/ai-log/README.md`, enfatizando la autoría del autor (decisiones:
descartar personalizado, elegir el modo individual, fijar los hiperparámetros).

- [ ] **Step 4: Actualizar CLAUDE.md y la memoria del asistente**

En `CLAUDE.md`, marca L3 como completado (tag `v0.4.4-o4l3-regresion`) en la
sección de O4. Actualiza la memoria persistente `project_o4_improvement_lines.md`
con el resultado real de L3.

- [ ] **Step 5: Commit de cierre + tag**

```bash
git add reports/memoria/06_o4_ml.md reports/ai-log CLAUDE.md
git commit -m "L3: notas de memoria, ai-log y cierre (regresión de cuantiles)"
git tag v0.4.4-o4l3-regresion
git log --oneline -8
```

---

## Self-Review (autochequeo del plan)

**Cobertura del spec:**
- F1 target Δlat/Δlon → Task 2. F2 modos poblacional+individual → Task 8 (split_by_mode + corte @individual). F3 XGBoost único → Task 3. F4 tres cuantiles, un regresor por (eje×cuantil) → Tasks 3, 8. F5 monotonía post-hoc → Task 4. F6 sin log-loss, métricas geométricas → Tasks 6, 7, 8. F7 features causales → `_FEATURES` en Task 8 + test leak-free Task 9. F8 hiperparámetros §8.6 → Task 3. F9 split temporal → Task 8 (`split_temporal_per_bird`). F10 target sin fuga → Tasks 2, 9.
- Métricas (top-1 mapeado, top-3 proximidad, dist nativa + centroide, pinball, cobertura, moves-only, por estado) → Tasks 5, 6, 7, 8.
- Artefactos D1, C1-C5 → Task 11. Tests (8 unitarios + 5 integración) → Tasks 2-10. Entregables §11 → Tasks 10-12.

**Placeholder scan:** sin TBD/TODO; todos los pasos de código llevan el código completo. Los `00NN` del ai-log y los `num` de artefactos se resuelven en ejecución (instrucción explícita de buscar el primer libre).

**Consistencia de tipos:** `fit_quantile_axis` → `dict[float, _XGBQuantileRegressor]` consumido por `predict_quantiles` (Task 4, 8). `predict_quantiles` → `(DataFrame, dict)` desempacado en Task 8. `build_regression_predictions` produce el esquema que `evaluate_by_state`/`evaluate_moves_only` consumen (verificado en Task 7). Las columnas de cuantil `dlat_p10/50/90`, `dlon_p10/50/90` son consistentes entre Tasks 4, 7, 8 y el notebook (Task 11 usa `p{int(q*100):02d}`).
