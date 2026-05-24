# L3 multi-familia (RF + LightGBM + XGBoost cuantil) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extender L3 para que la regresión de cuantiles del desplazamiento `(Δlat, Δlon)` se entrene también con Random Forest (QRF) y LightGBM además de XGBoost, y producir la comparativa de las tres familias.

**Architecture:** Se generaliza `quantile.py` con una interfaz uniforme de "predictor de eje" (`predict_raw(X) -> (n,3)`) que envuelve, según familia, un `dict` de regresores single-quantile (XGBoost/LightGBM, pérdida pinball) o un único bosque QRF (`quantile-forest`, cuantiles desde las hojas). `build_l3.py` itera familias × modos, etiqueta métricas y predicciones con `familia`, y escribe en un directorio nuevo `l3_v2/`. El notebook extiende C1/C2/C5 a las tres familias y añade una tabla maestra.

**Tech Stack:** Python 3.12, scikit-learn, xgboost (`reg:quantileerror`), lightgbm (`objective='quantile'`), quantile-forest (`RandomForestQuantileRegressor`), pandas, numpy, pytest, jupytext.

**Spec:** `docs/superpowers/specs/2026-05-24-o4l3-multifamilia-design.md` (decisiones G1–G9).

**Restricción transversal:** las tres familias usan **exactamente** las 10 features causales y el mismo target sin fuga ([[feedback-causal-features-no-leakage]]).

---

### Task 1: Dependencia `quantile-forest` y constante de ruta `O4_L3V2_DIR`

**Files:**
- Modify: `pyproject.toml` (+ `uv.lock` regenerado por `uv add`)
- Modify: `src/tfg_aves/ml/_paths.py:13`
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Añadir la dependencia y verificar que importa y entrena**

Run:
```bash
uv add quantile-forest
uv run python -c "
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
X = np.random.RandomState(0).normal(size=(200, 3)); y = X[:,0]*2 + np.random.RandomState(1).normal(0,0.1,200)
m = RandomForestQuantileRegressor(n_estimators=50, min_samples_leaf=20, random_state=0, n_jobs=-1).fit(X, y)
p = m.predict(X, quantiles=[0.10, 0.50, 0.90])
print('shape', np.asarray(p).shape)
assert np.asarray(p).shape == (200, 3)
assert (p[:,0] <= p[:,1]).all() and (p[:,1] <= p[:,2]).all()
print('QRF OK, monotono por construccion')
"
```
Expected: imprime `shape (200, 3)` y `QRF OK, monotono por construccion` sin error. Si falla por incompatibilidad con el stack fijado (R3 del spec), parar y reportar antes de continuar (fallback documentado: QRF artesanal).

- [ ] **Step 2: Escribir el test de la nueva constante de ruta**

Añadir a `tests/test_ml_quantile.py`:
```python
def test_o4_l3v2_dir_exists():
    from tfg_aves.ml._paths import O4_L3V2_DIR, O4_OUT_DIR
    assert O4_L3V2_DIR == O4_OUT_DIR / "l3_v2"
```

- [ ] **Step 3: Ejecutar el test y verlo fallar**

Run: `uv run pytest tests/test_ml_quantile.py::test_o4_l3v2_dir_exists -v`
Expected: FAIL con `ImportError: cannot import name 'O4_L3V2_DIR'`.

- [ ] **Step 4: Añadir la constante**

En `src/tfg_aves/ml/_paths.py`, tras la línea 13 (`O4_L3V1_DIR = ...`):
```python
O4_L3V2_DIR: Path = ROOT / "data" / "processed" / "o4" / "l3_v2"
```

- [ ] **Step 5: Ejecutar el test y verlo pasar**

Run: `uv run pytest tests/test_ml_quantile.py::test_o4_l3v2_dir_exists tests/test_ml_quantile.py::test_o4_l3v1_dir_exists -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/tfg_aves/ml/_paths.py tests/test_ml_quantile.py
git commit -m "O4 L3: añade quantile-forest y la ruta l3_v2 para la extensión multi-familia"
```

---

### Task 2: Interfaz uniforme de predictor por eje (refactor XGBoost sin cambio de comportamiento)

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py` (clases nuevas + `fit_quantile_axis`, `_axis_quantiles_sorted`, `predict_quantiles`)
- Test: `tests/test_ml_quantile.py` (actualizar 3 tests al nuevo contrato)

- [ ] **Step 1: Actualizar los tests existentes al contrato `predict_raw`**

En `tests/test_ml_quantile.py`, reemplazar `test_fit_quantile_axis_shapes_and_order` por:
```python
def test_fit_quantile_axis_shapes_and_order():
    from tfg_aves.ml.quantile import QUANTILES, fit_quantile_axis
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=300)
    Xv = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    yv = (Xv["a"].to_numpy() * 2.0) + rng.normal(0, 0.1, size=80)
    axis = fit_quantile_axis(X, y, Xv, yv, family="xgb", seed=0)
    raw = axis.predict_raw(Xv)               # (n, 3) en orden QUANTILES
    assert raw.shape == (80, len(QUANTILES))
    # En media, el cuantil 90 (col 2) está por encima del 10 (col 0).
    assert float(np.mean(raw[:, 2] >= raw[:, 0])) > 0.9
```

Reemplazar `test_predict_quantiles_monotonic_and_crossings` por (envuelve los mocks en `_PerQuantileAxis`):
```python
def test_predict_quantiles_monotonic_and_crossings():
    from tfg_aves.ml.quantile import _PerQuantileAxis, predict_quantiles
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    # lat: cuantiles DESORDENADOS (1.0, 0.0, 0.5) -> cruce en las 3 filas.
    axis_lat = _PerQuantileAxis({0.10: _Const(1.0), 0.50: _Const(0.0), 0.90: _Const(0.5)})
    # lon: cuantiles ya ordenados (-0.5, 0.0, 0.5) -> sin cruces.
    axis_lon = _PerQuantileAxis({0.10: _Const(-0.5), 0.50: _Const(0.0), 0.90: _Const(0.5)})
    preds, crossings = predict_quantiles(axis_lat, axis_lon, X)
    assert (preds["dlat_p10"] <= preds["dlat_p50"]).all()
    assert (preds["dlat_p50"] <= preds["dlat_p90"]).all()
    assert (preds["dlon_p10"] <= preds["dlon_p50"]).all()
    assert (preds["dlon_p50"] <= preds["dlon_p90"]).all()
    assert crossings["lat"] == 3
    assert crossings["lon"] == 0
    assert len(preds) == 3
```

Reemplazar `test_fit_quantile_axis_handles_empty_val` por:
```python
def test_fit_quantile_axis_handles_empty_val():
    from tfg_aves.ml.quantile import fit_quantile_axis
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = X["a"].to_numpy() + rng.normal(0, 0.1, size=120)
    X_val = X.iloc[:0]                      # empty val
    y_val = np.empty(0, dtype=float)
    axis = fit_quantile_axis(X, y, X_val, y_val, family="xgb", seed=0)
    raw = axis.predict_raw(X)
    assert raw.shape == (120, 3)
```

- [ ] **Step 2: Ejecutar los tests y verlos fallar**

Run: `uv run pytest tests/test_ml_quantile.py -k "fit_quantile_axis or predict_quantiles_monotonic" -v`
Expected: FAIL (`_PerQuantileAxis` no existe / `fit_quantile_axis` no acepta `family` o no tiene `predict_raw`).

- [ ] **Step 3: Introducir las clases de eje y refactorizar `fit_quantile_axis`/`predict_quantiles`**

En `src/tfg_aves/ml/quantile.py`, **reemplazar** `fit_quantile_axis` (líneas 84-99), `_axis_quantiles_sorted` (102-111) y `predict_quantiles` (114-129) por:

```python
class _PerQuantileAxis:
    """Predictor de eje que envuelve un regresor independiente por cuantil
    (XGBoost o LightGBM). ``predict_raw`` apila los 3 cuantiles en orden
    QUANTILES → matriz (n, 3)."""

    def __init__(self, models: dict[float, object]) -> None:
        self.models = models

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [np.asarray(self.models[q].predict(X), dtype=np.float64) for q in QUANTILES]
        )


class _QRFAxis:
    """Predictor de eje basado en Quantile Regression Forest: un único bosque
    estima los tres cuantiles desde la distribución empírica de cada hoja
    (Meinshausen 2006). Los cuantiles son monótonos por construcción."""

    def __init__(self, forest: object) -> None:
        self.forest = forest

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(
            self.forest.predict(X, quantiles=list(QUANTILES)), dtype=np.float64
        )
        return out.reshape(len(X), len(QUANTILES))


def fit_quantile_axis(
    X_train: pd.DataFrame,
    y_train_axis: np.ndarray,
    X_val: pd.DataFrame,
    y_val_axis: np.ndarray,
    *,
    family: str = "xgb",
    seed: int = 0,
) -> _PerQuantileAxis | _QRFAxis:
    """Entrena el predictor de cuantiles de UN eje (Δlat o Δlon) para la
    familia indicada y lo devuelve tras una interfaz uniforme ``predict_raw``.

    - ``xgb``/``lgbm``: un regresor single-quantile por cuantil (pérdida
      pinball nativa), con early stopping sobre val.
    - ``rf``: un único Quantile Regression Forest (ignora val: no hay early
      stopping en bagging).
    """
    y_tr = np.asarray(y_train_axis, dtype=np.float64)
    if family in ("xgb", "lgbm"):
        cls = _XGBQuantileRegressor if family == "xgb" else _LGBMQuantileRegressor
        y_va = np.asarray(y_val_axis, dtype=np.float64)
        models = {
            q: cls(quantile=q, seed=seed).fit(X_train, y_tr, X_val, y_va)
            for q in QUANTILES
        }
        return _PerQuantileAxis(models)
    if family == "rf":
        from quantile_forest import RandomForestQuantileRegressor
        forest = RandomForestQuantileRegressor(
            n_estimators=300,
            min_samples_leaf=20,   # CRÍTICO en QRF: hojas con muestras suficientes
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        ).fit(X_train, y_tr)
        return _QRFAxis(forest)
    raise ValueError(f"familia de regresión desconocida: {family!r}")


def _axis_quantiles_sorted(
    axis: _PerQuantileAxis | _QRFAxis, X: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Devuelve (matriz (n,3) ordenada por fila, nº de filas con cruce)."""
    raw = np.asarray(axis.predict_raw(X), dtype=np.float64)
    ordered = (raw[:, 0] <= raw[:, 1]) & (raw[:, 1] <= raw[:, 2])
    n_crossings = int(np.sum(~ordered))
    sorted_q = np.sort(raw, axis=1)
    return sorted_q, n_crossings


def predict_quantiles(
    axis_lat: _PerQuantileAxis | _QRFAxis,
    axis_lon: _PerQuantileAxis | _QRFAxis,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Predice los 6 cuantiles, fuerza monotonía por eje (np.sort) y devuelve
    (DataFrame con dlat_p10/50/90, dlon_p10/50/90; dict de cruces ANTES de
    ordenar, para C5). Acepta cualquier predictor de eje con predict_raw."""
    lat_q, n_cross_lat = _axis_quantiles_sorted(axis_lat, X)
    lon_q, n_cross_lon = _axis_quantiles_sorted(axis_lon, X)
    out = pd.DataFrame({
        "dlat_p10": lat_q[:, 0], "dlat_p50": lat_q[:, 1], "dlat_p90": lat_q[:, 2],
        "dlon_p10": lon_q[:, 0], "dlon_p50": lon_q[:, 1], "dlon_p90": lon_q[:, 2],
    })
    return out, {"lat": n_cross_lat, "lon": n_cross_lon}
```

(`_XGBQuantileRegressor` y todo lo que sigue a `predict_quantiles` —
`point_to_cell`, `nearest_cells`, `pinball_loss`, `interval_coverage`,
`build_regression_predictions`— quedan **sin tocar**. `_LGBMQuantileRegressor`
se añade en la Task 3.)

- [ ] **Step 4: Ejecutar TODOS los tests de quantile y verlos pasar**

Run: `uv run pytest tests/test_ml_quantile.py -v`
Expected: todos PASS (incluidos los 3 actualizados; los demás no cambian).

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "O4 L3: interfaz uniforme de predictor por eje (predict_raw)"
```

---

### Task 3: Familia LightGBM (`objective='quantile'`)

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py` (añadir `_LGBMQuantileRegressor`)
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test de LightGBM**

Añadir a `tests/test_ml_quantile.py`:
```python
def test_fit_quantile_axis_lgbm_shapes_and_order():
    from tfg_aves.ml.quantile import _PerQuantileAxis, fit_quantile_axis
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (X["a"].to_numpy() * 2.0) + rng.normal(0, 0.2, size=400)
    Xv = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    yv = (Xv["a"].to_numpy() * 2.0) + rng.normal(0, 0.2, size=100)
    axis = fit_quantile_axis(X, y, Xv, yv, family="lgbm", seed=0)
    assert isinstance(axis, _PerQuantileAxis)
    raw = axis.predict_raw(Xv)
    assert raw.shape == (100, 3)
    assert float(np.mean(raw[:, 2] >= raw[:, 0])) > 0.9
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `uv run pytest tests/test_ml_quantile.py::test_fit_quantile_axis_lgbm_shapes_and_order -v`
Expected: FAIL con `NameError: name '_LGBMQuantileRegressor' is not defined`.

- [ ] **Step 3: Implementar `_LGBMQuantileRegressor`**

En `src/tfg_aves/ml/quantile.py`, tras la clase `_XGBQuantileRegressor` (antes de `_PerQuantileAxis`), añadir el import al inicio del fichero junto a los demás:
```python
import lightgbm as lgb
```
y la clase:
```python
class _LGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """Espejo de _XGBQuantileRegressor con la API de LightGBM
    (objective='quantile', alpha=q). Pérdida pinball nativa, un cuantil por
    instancia. Config conservadora análoga a §8.6 de O4 (G4 del spec).
    Early stopping sobre val con metric='quantile' si val no está vacío."""

    def __init__(self, quantile: float, seed: int = 0) -> None:
        self.quantile = quantile
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> _LGBMQuantileRegressor:
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        self._reg = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.quantile,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,      # necesario para que subsample<1 actúe en LGBM
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(50, verbose=False)] if has_val else None
        eval_set = [(X_val, y_val)] if has_val else None
        self._reg.fit(
            X, y, eval_set=eval_set, eval_metric="quantile", callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._reg.predict(X), dtype=np.float64)
```

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `uv run pytest tests/test_ml_quantile.py::test_fit_quantile_axis_lgbm_shapes_and_order -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
git commit -m "O4 L3: regresor de cuantiles LightGBM (objective='quantile')"
```

---

### Task 4: Familia Random Forest (QRF)

**Files:**
- Modify: `src/tfg_aves/ml/quantile.py` (ya implementado en Task 2 vía `_QRFAxis` + rama `family=='rf'`)
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test de QRF (formas + monotonía por construcción)**

Añadir a `tests/test_ml_quantile.py`:
```python
def test_fit_quantile_axis_rf_monotonic_by_construction():
    from tfg_aves.ml.quantile import _QRFAxis, fit_quantile_axis, predict_quantiles
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = (X["a"].to_numpy() * 2.0) + rng.normal(0, 0.2, size=400)
    Xv = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    yv = (Xv["a"].to_numpy() * 2.0) + rng.normal(0, 0.2, size=100)
    axis = fit_quantile_axis(X, y, Xv, yv, family="rf", seed=0)
    assert isinstance(axis, _QRFAxis)
    raw = axis.predict_raw(Xv)
    assert raw.shape == (100, 3)
    # QRF: cuantiles monótonos por construcción → CERO cruces antes de ordenar.
    _, crossings = predict_quantiles(axis, axis, Xv)
    assert crossings["lat"] == 0
    assert crossings["lon"] == 0
```

- [ ] **Step 2: Ejecutar y ver pasar (la implementación ya existe de la Task 2)**

Run: `uv run pytest tests/test_ml_quantile.py::test_fit_quantile_axis_rf_monotonic_by_construction -v`
Expected: PASS. (Si falla por la firma `predict(X, quantiles=...)` del paquete, revisar la versión instalada en la Task 1.)

- [ ] **Step 3: Ejecutar toda la suite de quantile + ruff**

Run:
```bash
uv run pytest tests/test_ml_quantile.py -v
uv run ruff check src/tfg_aves/ml/quantile.py tests/test_ml_quantile.py
```
Expected: todos PASS, ruff limpio.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ml_quantile.py
git commit -m "O4 L3: test de QRF (Random Forest) monótono por construcción"
```

---

### Task 5: `build_o4_l3` multi-familia (bucle familias × modos, columna `familia`, serialización `l3_v2`)

**Files:**
- Modify: `src/tfg_aves/ml/build_l3.py`
- Test: `tests/test_ml_build_l3.py`

- [ ] **Step 1: Actualizar los tests de integración al esquema multi-familia**

En `tests/test_ml_build_l3.py`, reemplazar `test_build_o4_l3_artifacts`, `test_build_o4_l3_individual_subset`, `test_build_o4_l3_no_leakage` e `test_build_o4_l3_idempotent` por:

```python
def test_build_o4_l3_artifacts(tmp_path):
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    result = build_o4_l3(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    # 8 modelos: xgb (pob+ind) + lgbm (pob) + rf (pob), 2 ejes cada combinación.
    for family in ("xgb", "lgbm", "rf"):
        for axis in ("dlat", "dlon"):
            assert (out_dir / f"model_{family}_poblacional_{axis}.pkl").exists()
    for axis in ("dlat", "dlon"):
        assert (out_dir / f"model_xgb_individual_{axis}.pkl").exists()
    assert (out_dir / "predictions_test.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"familia", "modo", "scope", "top1", "dist_centroide_km",
            "coverage_lat"}.issubset(metrics.columns)
    assert {"xgb", "lgbm", "rf"}.issubset(set(metrics["familia"]))
    assert "individual" in set(metrics["modo"])          # solo xgb
    assert "poblacional@A" in set(metrics["modo"])
    assert "persistencia" in set(metrics["modo"])
    # Cobertura global reportada (no NaN) para el poblacional de cada familia.
    for family in ("xgb", "lgbm", "rf"):
        glob = metrics[(metrics["familia"] == family)
                       & (metrics["modo"] == "poblacional")
                       & (metrics["scope"] == "global")]
        assert len(glob) == 1
        assert not np.isnan(glob["coverage_lat"].iloc[0])
    assert result.n_rows_test_ind > 0


def test_build_o4_l3_individual_subset(tmp_path):
    """El test individual (xgb) coincide con el subconjunto bird_id==A del
    poblacional xgb (mismo split temporal por ave)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    ind = preds[(preds["familia"] == "xgb") & (preds["modo"] == "individual")]
    pob_a = preds[(preds["familia"] == "xgb") & (preds["modo"] == "poblacional")
                  & (preds["bird_id"] == "A")]
    assert len(ind) == len(pob_a)
    assert set(ind["date_utc"]) == set(pob_a["date_utc"])


def test_build_o4_l3_rf_lgbm_poblacional_only(tmp_path):
    """RF y LightGBM solo se entrenan en modo poblacional (G3)."""
    from tfg_aves.ml.build_l3 import build_o4_l3
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    preds = pd.read_parquet(out_dir / "predictions_test.parquet")
    for family in ("lgbm", "rf"):
        assert set(preds[preds["familia"] == family]["modo"]) == {"poblacional"}
        assert not (out_dir / f"model_{family}_individual_dlat.pkl").exists()


def test_build_o4_l3_no_leakage(tmp_path):
    """feature_cols == las 10 causales, sin columnas de t+1, en las 3 familias."""
    import joblib

    from tfg_aves.ml.build_l3 import build_o4_l3
    from tfg_aves.ml.features import FEATURES_HMM, FEATURES_KINEMATIC
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l3_v2"
    build_o4_l3(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    expected = [*FEATURES_KINEMATIC, *FEATURES_HMM]
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon"}
    for family in ("xgb", "lgbm", "rf"):
        payload = joblib.load(out_dir / f"model_{family}_poblacional_dlat.pkl")
        assert payload["feature_cols"] == expected
        assert payload["familia"] == family
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
    ma = (pd.read_parquet(out_a / "metrics.parquet")
          .sort_values(["familia", "modo", "scope"]).reset_index(drop=True))
    mb = (pd.read_parquet(out_b / "metrics.parquet")
          .sort_values(["familia", "modo", "scope"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(ma, mb)
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `uv run pytest tests/test_ml_build_l3.py -v`
Expected: FAIL (la columna `familia` no existe, los `.pkl` tienen el nombre viejo).

- [ ] **Step 3: Reescribir `build_l3.py` para iterar familias × modos**

En `src/tfg_aves/ml/build_l3.py`:

(a) Cambiar el import de ruta (línea 11) a incluir la nueva:
```python
from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V2_DIR
```

(b) Tras `_FEATURES` y `_MODES` (líneas 36-37), añadir:
```python
_FAMILIES_DEFAULT = ("xgb", "lgbm", "rf")


def _modes_for(family: str) -> tuple[str, ...]:
    """XGBoost conserva poblacional + individual; RF/LGBM solo poblacional (G3)."""
    return ("poblacional", "individual") if family == "xgb" else ("poblacional",)
```

(c) Cambiar la firma de `_metric_rows` y `_baseline_rows` para aceptar `familia`
y añadirla a cada fila. En `_metric_rows` (línea 92), nueva firma y se inserta
`"familia": familia` en los dos `rows.append({...})`:
```python
def _metric_rows(
    preds: pd.DataFrame, y_move: np.ndarray, modo_label: str, familia: str,
    *, pinball_lat: float, pinball_lon: float, cov_lat: float, cov_lon: float,
) -> list[dict]:
    ...
    rows.append({
        "familia": familia,
        "modo": modo_label, "scope": scope, "n_obs": int(r["n_obs"]),
        ...  # resto igual
    })
    ...
    rows.append({
        "familia": familia,
        "modo": modo_label, "scope": "moves", "n_obs": mo["n_obs"],
        ...  # resto igual
    })
    return rows
```
En `_baseline_rows` (línea 133):
```python
def _baseline_rows(preds: pd.DataFrame, modo_label: str, familia: str) -> list[dict]:
    by = evaluate_by_state(preds)
    rows = []
    for _, r in by.iterrows():
        rows.append({
            "familia": familia,
            "modo": modo_label, "scope": r["state"], "n_obs": int(r["n_obs"]),
            ...  # resto igual
        })
    return rows
```

(d) En `build_o4_l3` (línea 149) cambiar la firma:
```python
def build_o4_l3(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L3V2_DIR,
    seed: int = 0,
    *,
    families: tuple[str, ...] = _FAMILIES_DEFAULT,
    individual_bird_id: str | None = None,
) -> BuildO4L3Result:
```

(e) Reemplazar el bucle de modos (líneas 189-257) por el bucle de familias ×
modos. Sustituir desde `for mode in _MODES:` hasta justo antes de
`# --- Corte poblacional@individual ...`:
```python
    xgb_pob_full: pd.DataFrame | None = None

    for family in families:
        for mode in _modes_for(family):
            train_m, val_m, test_m = splits_by_mode[mode]
            tgt_tr = derive_displacement_target(train_m)
            tgt_va = derive_displacement_target(val_m)

            axis_lat = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlat"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlat"].to_numpy(),
                family=family, seed=seed,
            )
            axis_lon = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlon"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlon"].to_numpy(),
                family=family, seed=seed,
            )

            qp_test, crossings = predict_quantiles(axis_lat, axis_lon, test_m[_FEATURES])
            key = f"{family}_{mode}"
            n_crossings[key] = crossings
            preds = build_regression_predictions(qp_test, test_m, cells)

            tgt_te = derive_displacement_target(test_m)
            pin_lat = float(np.mean([
                pinball_loss(tgt_te["y_dlat"].to_numpy(),
                             qp_test[f"dlat_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            pin_lon = float(np.mean([
                pinball_loss(tgt_te["y_dlon"].to_numpy(),
                             qp_test[f"dlon_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            cov_lat = interval_coverage(
                tgt_te["y_dlat"].to_numpy(),
                qp_test["dlat_p10"].to_numpy(), qp_test["dlat_p90"].to_numpy(),
            )
            cov_lon = interval_coverage(
                tgt_te["y_dlon"].to_numpy(),
                qp_test["dlon_p10"].to_numpy(), qp_test["dlon_p90"].to_numpy(),
            )
            coverage[key] = {"lat": cov_lat, "lon": cov_lon}

            metric_rows.extend(_metric_rows(
                preds, _y_move(test_m), mode, family,
                pinball_lat=pin_lat, pinball_lon=pin_lon,
                cov_lat=cov_lat, cov_lon=cov_lon,
            ))

            for axis_name, axis_model in (("dlat", axis_lat), ("dlon", axis_lon)):
                mp = out_dir / f"model_{family}_{mode}_{axis_name}.pkl"
                joblib.dump({
                    "model": axis_model, "feature_cols": _FEATURES,
                    "familia": family, "mode": mode, "axis": axis_name,
                    "individual_bird_id": (
                        individual_bird_id if mode == "individual" else None
                    ),
                }, mp)
                model_paths[f"{family}_{mode}_{axis_name}"] = mp

            preds_out = preds.copy()
            preds_out["familia"] = family
            preds_out["modo"] = mode
            preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
            preds_frames.append(preds_out)
            if family == "xgb" and mode == "poblacional":
                xgb_pob_full = preds.copy()
```

(f) Reemplazar el bloque del corte `@individual` (líneas 259-269) para usar
`xgb_pob_full` y etiquetar familia `"xgb"`:
```python
    # --- Corte xgb poblacional@individual: mismas filas del individual ---
    assert xgb_pob_full is not None
    pob_at_ind = xgb_pob_full[
        xgb_pob_full["bird_id"] == individual_bird_id
    ].reset_index(drop=True)
    if len(pob_at_ind) > 0:
        test_ind = splits_by_mode["individual"][2]
        metric_rows.extend(_metric_rows(
            pob_at_ind, _y_move(test_ind), f"poblacional@{individual_bird_id}", "xgb",
            pinball_lat=np.nan, pinball_lon=np.nan, cov_lat=np.nan, cov_lon=np.nan,
        ))
```

(g) Reemplazar el bloque de persistencia (líneas 271-277) para pasar
`familia="—"`:
```python
    persistence = compute_persistence_baseline(test_pob, cells=cells)
    metric_rows.extend(_baseline_rows(persistence, "persistencia", "—"))
    pers_ind = persistence[persistence["bird_id"] == individual_bird_id]
    if len(pers_ind) > 0:
        metric_rows.extend(_baseline_rows(
            pers_ind.reset_index(drop=True), f"persistencia@{individual_bird_id}", "—"))
```

(h) Borrar la variable obsoleta `preds_pob_full` (línea 179) y su asignación;
queda sustituida por `xgb_pob_full`.

- [ ] **Step 4: Ejecutar los tests de integración y verlos pasar**

Run: `uv run pytest tests/test_ml_build_l3.py -v`
Expected: todos PASS (incluido el nuevo `test_build_o4_l3_rf_lgbm_poblacional_only`).

- [ ] **Step 5: Suite completa de ml + ruff**

Run:
```bash
uv run pytest tests/test_ml_quantile.py tests/test_ml_build_l3.py -q
uv run ruff check src/tfg_aves/ml tests/test_ml_quantile.py tests/test_ml_build_l3.py
```
Expected: todos PASS, ruff limpio.

- [ ] **Step 6: Commit**

```bash
git add src/tfg_aves/ml/build_l3.py tests/test_ml_build_l3.py
git commit -m "O4 L3: build multi-familia (xgb+lgbm+rf), columna familia y salida l3_v2"
```

---

### Task 6: Notebook — extender C1/C2/C5 a las tres familias + tabla maestra `tab31`

**Files:**
- Modify: `notebooks/04l3_eda_o4l3.py` (reescritura completa, jupytext percent)
- Genera: artefactos `o4_fig25`, `o4_fig26`, `o4_fig29` (extendidos) y `o4_tab31` (nuevo)

> Tarea de generación de artefactos (no TDD). Verificación = ejecutar el
> notebook como script y comprobar que los ficheros se actualizan.

- [ ] **Step 1: Confirmar que `tab31` es el siguiente número libre**

Run: `grep -oE "o4_(fig|tab)[0-9]+" reports/INDEX.md | sort -u | tail -5`
Expected: el mayor número es 30 (`o4_tab30_comparativa-maestra-regimen`), así que 31 está libre para el nuevo artefacto. Si ya existiera un 31, usar el siguiente libre y ajustar `num=` abajo.

- [ ] **Step 2: Reescribir el notebook**

Sobrescribir `notebooks/04l3_eda_o4l3.py` con:

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
# # O4 · L3 — Regresión espacial con cuantiles (multi-familia)
#
# Ejecuta `build_o4_l3` con las tres familias del proposal (XGBoost, LightGBM y
# Random Forest). XGBoost conserva sus modos poblacional + individual 91916A;
# LightGBM y RF solo poblacional. Genera D1, C1..C5 (C1/C2/C5 extendidos a las
# tres familias) y la tabla maestra de familias (tab31).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V2_DIR, O4_OUT_DIR
from tfg_aves.ml.build_l3 import _prepare_poblacional_split, build_o4_l3
from tfg_aves.ml.quantile import INDIVIDUAL_BIRD_ID, derive_displacement_target
from tfg_aves.reporting import save_artifact

_FAMILIES = ["xgb", "lgbm", "rf"]
_FAM_LABEL = {"xgb": "XGBoost", "lgbm": "LightGBM", "rf": "Random Forest"}

result = build_o4_l3()
print("filas pob train/test:", result.n_rows_train_pob, result.n_rows_test_pob)
print("filas ind train/test:", result.n_rows_train_ind, result.n_rows_test_ind)
print("cobertura:", result.coverage)
print("cruces de cuantil:", result.n_crossings)

metrics = pd.read_parquet(O4_L3V2_DIR / "metrics.parquet")
preds = pd.read_parquet(O4_L3V2_DIR / "predictions_test.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")  # O4 causal (L3-v0)

# Splits reales para D1 (distribución del target y conteo de histórico por ave).
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
train_pob, _, _ = _prepare_poblacional_split(features_o3, cells, seed=0)

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
    fig=fig, table=hist, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C1 — Calibración de la incertidumbre (tres familias, poblacional)
# %%
cov_rows = []
for fam in _FAMILIES:
    sub = preds[(preds["familia"] == fam) & (preds["modo"] == "poblacional")]
    cov_rows.append({"familia": fam,
                     "cobertura_lat": float(sub["in_interval_lat"].mean()),
                     "cobertura_lon": float(sub["in_interval_lon"].mean())})
cov_df = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(cov_df)); w = 0.35
ax.bar(x - w / 2, cov_df["cobertura_lat"], w, label="Δlat")
ax.bar(x + w / 2, cov_df["cobertura_lon"], w, label="Δlon")
ax.axhline(0.80, color="red", ls="--", label="nominal 80%")
ax.set_xticks(x); ax.set_xticklabels([_FAM_LABEL[f] for f in cov_df["familia"]])
ax.set_ylabel("cobertura empírica de [p10, p90]"); ax.set_ylim(0, 1)
ax.set_title("Calibración del intervalo por familia (poblacional)"); ax.legend()
fig.tight_layout()
save_artifact(
    "calibration-coverage", objective="o4", num=25,
    decision="Cobertura empírica de [p10,p90] cerca del 80% nominal en las tres familias.",
    caption_es=(
        "Cobertura empírica del intervalo de predicción [p10, p90] frente al 80% "
        "nominal, por familia (XGBoost, LightGBM, Random Forest) y eje, en modo "
        "poblacional. Una cobertura próxima al 80% indica incertidumbre del "
        "desplazamiento bien calibrada; permite comparar la calibración del boosting "
        "(pinball) frente al QRF de Random Forest (cuantiles de hojas)."),
    fig=fig, table=cov_df, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C2 — Distribución de distancias (tres familias vs persistencia, poblacional)
# %%
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
v0_pers = preds_v0[preds_v0["modelo"] == "persistencia"]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 300, 60)
for fam in _FAMILIES:
    sub = preds[(preds["familia"] == fam) & (preds["modo"] == "poblacional")]
    ax.hist(sub["pred_dist_km"], bins=bins, density=True, histtype="step",
            linewidth=1.6, label=f"L3 {_FAM_LABEL[fam]}")
ax.hist(v0_pers["pred_dist_km"], bins=bins, alpha=0.35, density=True, label="persistencia")
ax.set_xlabel("distancia vía centroide (km)"); ax.set_ylabel("densidad")
ax.set_title("Distribución de distancias en test (poblacional)"); ax.legend()
fig.tight_layout()
dist_tbl = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["familia", "modo", "scope", "dist_centroide_km", "dist_nativa_km"]].reset_index(drop=True)
save_artifact(
    "distance-distribution", objective="o4", num=26,
    decision="Comparación de la distancia del error de las tres familias vs persistencia.",
    caption_es=(
        "Distribución de la distancia (vía centroide) entre la predicción puntual "
        "(p50) y la posición real en test (modo poblacional), para las tres familias "
        "de L3 y la persistencia trivial. Las diferencias, si existen, se concentran "
        "en los días de movimiento (cola de la distribución)."),
    fig=fig, table=dist_tbl, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C3 — Comparativas centrales XGBoost (A: ablación de target; B: per-individuo)
# %%
xgb_pob = (metrics["familia"] == "xgb") & (metrics["modo"] == "poblacional")
pers = metrics["modo"] == "persistencia"
tabla_a = metrics[xgb_pob | pers][
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
              "sobre 91916A (B, per-individuo), familia XGBoost."),
    caption_es=(
        "Comparativas centrales de L3 (XGBoost). Bloque A: efecto de reformular el "
        "target (categórico L3-v0 vs cuantil L3-v1) en modo poblacional, por estado "
        "HMM y en días de movimiento. Bloque B: hipótesis per-individuo, modelo "
        "individual de 91916A frente al poblacional sobre las mismas filas. Sin "
        "columna log-loss: L3 no produce distribución categórica, su veredicto es "
        "geométrico (distancia + top-1 mapeado)."),
    table=tabla_comparativa, overwrite=True,
)

# %% [markdown]
# ## C4 — Vectores de desplazamiento predichos vs reales (91916A, XGBoost)
# %%
ind = preds[(preds["familia"] == "xgb") & (preds["modo"] == "individual")].copy()
ind = ind.sort_values("date_utc").reset_index(drop=True)
sample = ind.iloc[:: max(1, len(ind) // 40)].copy()  # ~40 días para legibilidad
fig, ax = plt.subplots(figsize=(7, 7))
true_lat = sample["pred_lat"] - sample["dlat_p50"]  # = lat_t
true_lon = sample["pred_lon"] - sample["dlon_p50"]  # = lon_t
ax.quiver(true_lon, true_lat,
          sample["pred_lon"] - true_lon, sample["pred_lat"] - true_lat,
          angles="xy", scale_units="xy", scale=1, color="steelblue",
          width=0.004, label="predicho p50")
ax.errorbar(sample["pred_lon"], sample["pred_lat"],
            xerr=[(sample["dlon_p50"] - sample["dlon_p10"]).abs(),
                  (sample["dlon_p90"] - sample["dlon_p50"]).abs()],
            yerr=[(sample["dlat_p50"] - sample["dlat_p10"]).abs(),
                  (sample["dlat_p90"] - sample["dlat_p50"]).abs()],
            fmt="none", ecolor="orange", alpha=0.5, label="banda [p10,p90]")
ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
ax.set_title(f"Desplazamientos predichos de {ind_id} (muestra, XGBoost)")
ax.legend()
fig.tight_layout()
save_artifact(
    "vectores-desplazamiento-91916a", objective="o4", num=28,
    decision="Vectores de desplazamiento p50 del modelo individual con banda de incertidumbre.",
    caption_es=(
        f"Vectores de desplazamiento diario predichos (mediana p50) por el modelo "
        f"individual XGBoost de {ind_id} sobre una muestra de su test, con la banda "
        "de incertidumbre [p10, p90] por eje. Ilustra la salida geométrica e "
        "interpretable de la regresión de cuantiles, no disponible en el clasificador."),
    fig=fig, table=sample[["date_utc", "pred_lat", "pred_lon",
                           "dlat_p10", "dlat_p90", "dlon_p10", "dlon_p90"]],
    overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C5 — Incidencia de quantile crossing por familia (RF = 0 por construcción)
# %%
cross_rows = []
for key, d in result.n_crossings.items():       # key = "{familia}_{modo}"
    family, mode = key.rsplit("_", 1)
    n_test = result.n_rows_test_pob if mode == "poblacional" else result.n_rows_test_ind
    cross_rows.append({
        "familia_modo": f"{_FAM_LABEL[family]}·{mode}", "n_test": n_test,
        "cruces_lat": d["lat"], "cruces_lon": d["lon"],
        "pct_lat": 100 * d["lat"] / max(1, n_test),
        "pct_lon": 100 * d["lon"] / max(1, n_test),
    })
cross_df = pd.DataFrame(cross_rows)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(cross_df)); w = 0.35
ax.bar(x - w / 2, cross_df["pct_lat"], w, label="Δlat")
ax.bar(x + w / 2, cross_df["pct_lon"], w, label="Δlon")
ax.set_xticks(x); ax.set_xticklabels(cross_df["familia_modo"], rotation=30, ha="right")
ax.set_ylabel("% de filas con cruce (antes de ordenar)")
ax.set_title("Incidencia de quantile crossing por familia"); ax.legend()
fig.tight_layout()
save_artifact(
    "quantile-crossing", objective="o4", num=29,
    decision="Cruces de cuantil por familia: boosting (XGB/LGBM) > 0 corregidos; RF = 0.",
    caption_es=(
        "Porcentaje de filas donde los cuantiles predichos se cruzan (p10>p50 o "
        "p50>p90) antes de la corrección monótona post-hoc, por familia, modo y eje. "
        "XGBoost y LightGBM entrenan un modelo independiente por cuantil y pueden "
        "cruzarse (se corrige con ordenación); Random Forest (QRF) obtiene los tres "
        "cuantiles de la misma distribución de hojas y es monótono por construcción "
        "(cero cruces)."),
    fig=fig, table=cross_df, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## tab31 — Tabla maestra: comparativa de las tres familias (poblacional)
# %%
maestra = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["familia", "modo", "scope", "n_obs", "top1", "top3", "dist_centroide_km",
     "dist_nativa_km", "pinball_lat", "pinball_lon", "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
cruces_pob = {key.rsplit("_", 1)[0]: d["lat"] + d["lon"]
              for key, d in result.n_crossings.items() if key.endswith("poblacional")}
maestra["cruces_total"] = [
    cruces_pob.get(r["familia"]) if (r["scope"] == "global" and r["modo"] == "poblacional")
    else np.nan
    for _, r in maestra.iterrows()
]
maestra = maestra.sort_values(["scope", "familia", "modo"]).reset_index(drop=True)
print(maestra.to_string())
save_artifact(
    "comparativa-familias-l3", objective="o4", num=31,
    decision=("Comparativa de las tres familias supervisadas (XGBoost, LightGBM, "
              "Random Forest) en la regresión de cuantiles, modo poblacional."),
    caption_es=(
        "Comparativa maestra de las tres familias del proposal sobre la tarea de "
        "regresión de cuantiles del desplazamiento (modo poblacional, test completo), "
        "por régimen (global, estacionario, migración, días de movimiento). Métricas: "
        "top-1/top-3 mapeados, distancia vía centroide y nativa, pérdida pinball y "
        "cobertura [p10,p90] por eje, y nº total de cruces de cuantil corregidos "
        "(cero en Random Forest por construcción). Baseline: persistencia trivial. "
        "Cierra la comparativa de tres familias que O4 base hizo sobre clasificación, "
        "ahora sobre regresión."),
    table=maestra, overwrite=True,
)
print("Artefactos L3 multi-familia generados (D1, C1..C5, tab31).")
```

- [ ] **Step 3: Ejecutar el notebook como script y verificar artefactos**

Run:
```bash
uv run jupytext --to notebook --execute notebooks/04l3_eda_o4l3.py
git status --short reports/
```
Expected: el comando termina sin error e imprime "Artefactos L3 multi-familia generados". `git status` muestra como modificados/nuevos `reports/figures/o4_fig25*`, `o4_fig26*`, `o4_fig29*`, `reports/tables/o4_tab31*`, sus captions y `reports/INDEX.md`. Revisar visualmente que C1 muestra las tres familias y que la fila de Random Forest en `o4_tab31_*.csv` tiene `cruces_total = 0`.

- [ ] **Step 4: Sanity-check de coherencia biológica** ([[feedback-biological-coherence]])

Inspeccionar `reports/tables/o4_tab31_comparativa-familias-l3.csv`: ninguna familia debe tener una `dist_centroide_km` global absurda (p.ej. > 100 km en estacionario) ni una cobertura fuera de [0,5, 1,0]. Si LightGBM diverge (distancias enormes, cobertura ~0/1), es el riesgo R2 del spec: documentarlo y ajustar `learning_rate`/`num_leaves` (config, no rejilla) antes de continuar.

- [ ] **Step 5: Commit**

```bash
git add notebooks/04l3_eda_o4l3.py reports/figures/o4_fig25* reports/figures/o4_fig26* \
        reports/figures/o4_fig29* reports/tables/o4_tab31* reports/tables/o4_fig25* \
        reports/tables/o4_fig26* reports/tables/o4_fig29* reports/captions/ reports/INDEX.md
git commit -m "O4 L3: notebook multi-familia, C1/C2/C5 extendidos y tabla maestra tab31"
```

---

### Task 7: Notas de memoria, ai-log, verificación final y tag

**Files:**
- Modify: `reports/memoria/06_o4_ml.md` (subsección L3)
- Create: `reports/ai-log/00NN-o4l3-multifamilia.md`
- Modify: `CLAUDE.md` (estado de O4)

- [ ] **Step 1: Añadir notas de memoria de la comparativa de familias**

En `reports/memoria/06_o4_ml.md`, en la sección "L3 — Regresión con cuantiles",
añadir una subsección "Comparativa de familias" que documente: (1) las tres
familias y sus mecanismos (pinball para XGB/LGBM, QRF de hojas para RF); (2) la
revisión de la decisión F3 ("familia única"); (3) el hallazgo de los cruces
(RF = 0 por construcción); (4) los números de la tabla maestra `tab31`; (5) la
interpretación honesta según el resultado real ([[feedback-memoria-tone]]).
Tomar las cifras de `reports/tables/o4_tab31_comparativa-familias-l3.csv`.

- [ ] **Step 2: Crear la entrada de ai-log**

Determinar el siguiente número:
```bash
ls reports/ai-log/ | grep -oE "^[0-9]{4}" | sort | tail -1
```
Crear `reports/ai-log/00NN-o4l3-multifamilia.md` (NN = siguiente) siguiendo la
estructura de `reports/ai-log/README.md`. Tono: el autor decidió completar la
comparativa de tres familias, validó la elección de QRF vía paquete frente a
artesanal, y dirigió el alcance (poblacional). La IA propuso la interfaz
uniforme y ejecutó la implementación TDD. ([[feedback-ai-log-scope]]: es trabajo
sustantivo de O4 → sí se registra.)

- [ ] **Step 3: Verificación final completa**

Run:
```bash
uv run pytest -q
uv run ruff check src tests
```
Expected: toda la suite pasa (130 tests previos + los nuevos de quantile/build_l3),
ruff limpio. Si algún test no relacionado falla, investigar antes del tag.

- [ ] **Step 4: Actualizar el estado de O4 en `CLAUDE.md`**

En la sección "Estado actual" / "Esquema de líneas de O4", anotar que L3 se
extendió a las tres familias (`v0.4.5-o4l3-multifamilia`), modelo poblacional,
con la conclusión real de `tab31`. (CLAUDE.md no se versiona; es contexto local.)

- [ ] **Step 5: Commit de notas + tag de hito**

```bash
git add reports/memoria/06_o4_ml.md reports/ai-log/00*-o4l3-multifamilia.md
git commit -m "O4 L3: notas de memoria y ai-log de la comparativa de familias"
git tag v0.4.5-o4l3-multifamilia
git log --oneline -6
```

---

## Self-Review

**Spec coverage:**
- G1 (tres familias con cuantiles) → Tasks 2, 3, 4.
- G2 (RF vía `quantile-forest`) → Task 1 (dep) + Task 2 (`_QRFAxis`, rama `rf`) + Task 4 (test).
- G3 (RF/LGBM solo poblacional) → Task 5 (`_modes_for`) + `test_build_o4_l3_rf_lgbm_poblacional_only`.
- G4 (hiperparámetros por familia) → Task 2 (RF), Task 3 (LGBM); XGB sin cambios.
- G5 (monotonía: sort XGB/LGBM, RF por construcción) → Task 2 (`_axis_quantiles_sorted`) + Task 4 (test cruces=0).
- G6 (interfaz uniforme `predict_raw`) → Task 2.
- G7 (serialización por familia/modo/eje en `l3_v2`) → Task 1 (ruta) + Task 5 (dump + naming).
- G8 (columna `familia` en métricas/preds) → Task 5.
- G9 (criterio: pinball+cobertura primario) → Task 6 (tab31 incluye pinball+cobertura).
- C1/C2/C5 extendidos + tab31 → Task 6.
- Tests (§8.1 del spec) → Tasks 1-5.
- Dependencia (§6.3) → Task 1.
- Entregables (§11) → Tasks 6, 7.

**Placeholder scan:** los `00NN`/`NN` de la Task 7 son números de secuencia a
resolver con el comando dado (no placeholders de diseño); el resto del código es
completo. La subsección de memoria (Task 7 Step 1) describe contenido a redactar
con cifras reales del CSV, coherente con el modo híbrido del TFG.

**Type consistency:** `fit_quantile_axis(..., family=..., seed=...)` devuelve
`_PerQuantileAxis | _QRFAxis`, ambos con `predict_raw(X) -> (n,3)`;
`predict_quantiles(axis_lat, axis_lon, X)` consume esa interfaz y devuelve
`(DataFrame, dict)` (firma estable). `_metric_rows`/`_baseline_rows` reciben
`familia` y la emiten en cada fila. Las claves `n_crossings`/`coverage`/
`model_paths` pasan a `"{familia}_{modo}"`/`"{familia}_{modo}_{eje}"` de forma
consistente entre `build_l3.py` (Task 5) y el notebook (Task 6).

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-24-o4l3-multifamilia.md`.**
