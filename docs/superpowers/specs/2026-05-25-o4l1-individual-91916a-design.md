# Diseño — L1: sustituir el modelo "personalizado" por un modelo individual de 91916A y retirar la línea del viento

- **Fecha:** 2026-05-25
- **Objetivo del TFG:** O4 — línea base discretizada (L1). Reemplaza el
  modo "personalizado" (modelo global con `bird_id` como *feature*
  categórica) por un modelo **verdaderamente per-individuo** entrenado solo
  sobre el ave 91916A, replicando el patrón ya validado en L3. Además
  retira por completo la línea del viento (L1-v1), descartada por el autor.
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado el 2026-05-25, pendiente de implementación.

## 1. Resumen

El modo "personalizado" de L1 (`build_o4`, modo `include_bird_id=True`) no
es per-individuo: entrena **un único modelo global** sobre las 82 aves con
`bird_id` como columna categórica. El autor lo considera engañosamente
etiquetado. Este diseño lo sustituye por un modo **`individual`** que
entrena y evalúa **exclusivamente sobre el ave 91916A** (la de mayor
histórico, 2025 filas), siguiendo el patrón ya implementado en L3
(`build_l3.py`): filtrar los splits temporales a una sola ave antes de
entrenar, y comparar contra el corte **`poblacional@91916A`** (el modelo
poblacional evaluado sobre exactamente las mismas filas de test de 91916A)
para una comparación manzanas-con-manzanas.

En paralelo, se retira por completo la **línea del viento (L1-v1)**:
artefactos, paquete `meteo`, flag `with_wind`, notebook, spec/plan, tests,
figuras y dependencias. Es una decisión del autor (la ablación de viento
dio resultado mixto y no se conserva).

El cambio se **confina a L1** (`build_o4` + `notebooks/04_eda_o4.py`). O4
base sigue usando como canónico el modo **poblacional**; el modo individual
sustituye al personalizado como segundo brazo de la comparación.

## 2. Vocabulario

- **L1** — línea base de O4: clasificación multiclase de la próxima celda
  0,5° con RF/XGBoost/LightGBM (lo que internamente se llamaba "L1-v0").
  Artefactos en `data/processed/o4/`, figuras `o4_fig02–09`.
- **Modo `poblacional`** — un modelo entrenado sobre las 82 aves, sin
  `bird_id`. **Canónico.** No cambia.
- **Modo `individual`** (nuevo) — un modelo entrenado y evaluado solo sobre
  91916A, con las mismas 10 features causales que el poblacional (sin
  `bird_id`, que es constante). Sustituye a `personalizado`.
- **Corte `poblacional@91916A`** — predicciones del modelo poblacional
  restringidas a las filas de test de 91916A. Es el término de comparación
  justo frente al individual (patrón heredado de L3).
- **L1-v1 (viento)** — ablación de features de viento a 850 hPa. **Se
  elimina por completo en este diseño.**

## 3. Contexto y constraints

**Por qué `individual` no comparte el test global con `poblacional`.** El
poblacional conoce ~849 celdas (todas las rutas) y se evalúa sobre el test
de las 82 aves. El individual de 91916A solo conoce las celdas de su propia
ruta y solo puede evaluarse sobre el test de esa ave. Las métricas globales
de uno y otro **no son comparables**. La comparación válida es
`individual` vs `poblacional@91916A` sobre el mismo conjunto de filas, más
las baselines (persistencia, Markov) recalculadas sobre esas filas. Este es
exactamente el problema que L3 resolvió con el corte `@91916A`.

**Restricción dura — no romper L2.** La infraestructura `personalizado`
(`include_bird_id` en `features.py`, `_CategoricalEncoder` en `train.py`)
**la usa también L2** (`build_l2.py:43`, `_MODES=("personalizado",
"poblacional")`). Por tanto:
- NO se elimina `include_bird_id` ni `_CategoricalEncoder`.
- El cambio se aplica **solo en `build_o4`**; `build_l2` y `build_l3`
  quedan intactos.
- Consecuencia aceptada: queda una inconsistencia deliberada — L1 usa
  `{individual, poblacional}`, L2 mantiene `{personalizado, poblacional}`.
  Alinear L2 queda fuera del alcance de este diseño (posible trabajo
  posterior).

**Restricción — el corte `poblacional@91916A` reusa el patrón de L3.**
`build_l3.py` (líneas ~267-278) ya implementa el corte `poblacional@<ave>`.
Se replica la misma lógica en `build_o4`, no se reinventa.

**Feasibilidad.** 91916A tiene 2025 filas diarias válidas (rank 1 de
histórico); tras el split temporal 80/10/20 quedan datos suficientes para
RF y XGBoost. L3 ya entrenó un individual de 91916A con éxito (empató al
poblacional en regresión), lo que confirma viabilidad.

**Familias.** Individual = **RF + XGBoost** (decisión del autor; simetría
con el poblacional en las figuras). Poblacional mantiene `(rf, xgb, lgbm)`
sin cambios; LightGBM sigue descartado en el análisis por divergencia en
validación.

## 4. Decisiones fijadas

| # | Decisión | Valor |
|---|---|---|
| F1 | Modos de `build_o4` | `{individual, poblacional}` (sustituye `personalizado`) |
| F2 | Ave del modo individual | `91916A` (param `individual_bird_id="91916A"`, reutiliza `INDIVIDUAL_BIRD_ID` de `quantile.py`) |
| F3 | Features del individual | Las 10 causales (sin `bird_id`); idénticas al poblacional |
| F4 | Familias del individual | RF + XGBoost |
| F5 | Familias del poblacional | `(rf, xgb, lgbm)` sin cambios (lgbm descartado en figuras) |
| F6 | Comparación del individual | vs `poblacional@91916A` + persistencia + Markov, todo sobre las filas de test de 91916A |
| F7 | Split y barreras gap-aware | Sin cambios (80/10/20 temporal por ave, tres barreras heredadas) |
| F8 | Criterio ganador | log-loss (heredado de O2/O4) |
| F9 | L2/L3 | Intactos; `include_bird_id`/`_CategoricalEncoder` se conservan |
| F10 | Línea del viento (L1-v1) | Eliminación total (código, datos, figuras, tests, docs, deps) |

## 5. Pipeline e integración

### 5.1 `src/tfg_aves/ml/build.py` (núcleo del cambio)

- Nuevo parámetro `individual_bird_id: str = "91916A"` (importado de
  `ml.quantile.INDIVIDUAL_BIRD_ID` para una sola fuente de verdad).
- Sustituir la construcción de matrices por modo:
  - `poblacional`: `build_feature_matrix(kin, cells, include_bird_id=False)`
    (sin cambios).
  - `individual`: parte del split poblacional y **filtra a la ave** antes
    de entrenar:
    `tuple(d[d["bird_id"] == individual_bird_id].reset_index(drop=True)
    for d in (train_pob, val_pob, test_pob))`.
- Bucle de entrenamiento: `individual` → familias `("rf", "xgb")`;
  `poblacional` → `_FAMILIES`.
- Tras entrenar el poblacional, calcular el corte `poblacional@91916A`:
  filtrar las predicciones poblacionales a `bird_id == individual_bird_id`
  y emitir filas de métricas etiquetadas `poblacional@91916A` (patrón
  `build_l3.py`).
- Baselines persistencia y Markov: calcular **dos veces** — global (82
  aves, como ahora) y sobre las filas de test de 91916A (para la
  comparación del individual).
- Eliminar el flag `with_wind` y todas sus ramas (resolución de
  `output_dir`, carga de `wind_per_fix.parquet`, paso de `wind_df`,
  `families = ... if not with_wind else (...)`).

### 5.2 `src/tfg_aves/ml/features.py`

- Eliminar el parámetro `wind_df` de `build_feature_matrix` y la rama que
  lo usa, y la función `merge_wind_features` con su constante
  `_FEATURES_WIND`. (L2/L3 llaman a `build_feature_matrix` sin `wind_df`,
  verificado → seguro.)
- `include_bird_id` y el resto de la firma **se conservan** (L2).

### 5.3 `src/tfg_aves/ml/_paths.py`

- Eliminar `O4_L1V1_DIR`.

### 5.4 Eliminación del paquete de viento

- Borrar `src/tfg_aves/meteo/` completo (`__init__.py`, `_paths.py`,
  `wind.py`, `build_wind.py`).
- Borrar `data/raw/wind/` y `data/processed/wind_per_fix.parquet` (si
  existen; gitignored/regenerables).
- Quitar `netcdf4>=1.6` y `xarray>=2024.0` de `pyproject.toml` (sin otros
  consumidores) y actualizar `uv.lock` con `uv sync`.
- Revisar `tests/test_reporting.py` (aparece como importador de algo de
  `meteo`): si solo referencia `O4_L1V1_DIR` u otra ruta de viento,
  limpiarlo; si no, dejarlo.

## 6. Artefactos en `data/processed/o4/`

Regenerados por `build_o4()` (gitignored, regenerables):

- `model_individual_rf.pkl`, `model_individual_xgb.pkl` (nuevos, pequeños).
- `model_poblacional_{rf,xgb,lgbm}.pkl` (sin cambios estructurales).
- Desaparecen las referencias a `model_personalizado_*` (sus `.pkl` ya no
  estaban en disco; se eliminan del código que los esperaba).
- `predictions_test.parquet`: incluye una columna de modo con `individual`,
  `poblacional` y el corte `poblacional@91916A`.
- `metrics.parquet`: filas para `individual` (RF/XGB), `poblacional`
  (global, RF/XGB/LGBM), `poblacional@91916A`, y baselines persistencia y
  Markov tanto globales como sobre 91916A.
- Borrar el directorio `data/processed/o4/l1_v1/`.

## 7. Figuras (`notebooks/04_eda_o4.py`, números sin cambiar)

| Fig | Slug | Cambio |
|---|---|---|
| `o4_fig02` | `train-test-gap` | Gap train−test (overfit) de **individual vs poblacional@91916A** |
| `o4_fig03` | `learning-curves` | Añade la curva del individual |
| `o4_fig04` | `models-comparison` | **Global (82 aves)**: poblacional + baselines. El individual NO aparece aquí (no es global) |
| `o4_fig06` | `error-by-state-personalizado` → **`error-by-state-individual`** | Error por estado HMM del individual sobre 91916A |
| `o4_fig07` | `error-by-state-poblacional` | Sin cambios (global) |
| `o4_fig08` | `personalizado-vs-poblacional` → **`individual-vs-poblacional`** | Figura clave: individual vs poblacional@91916A + persistencia + Markov sobre las filas de 91916A |
| `o4_fig09` | `feature-importance-winners` | Poblacional XGB (global) + individual (91916A) |

- Renombrar slugs `o4_fig06` y `o4_fig08` implica `git rm` de los ficheros
  antiguos (`.png/.csv/.md`) y nuevos vía `save_artifact(overwrite=True)`.
- Borrar figuras `o4_fig10–16` (viento) y sus `.csv/.md`.

## 8. Decisiones sin figura (justificación para la memoria)

- **Por qué el individual no entra en `o4_fig04` (comparación global).**
  Sus clases (celdas) son un subconjunto del poblacional y su test es solo
  91916A; mezclarlo con métricas de 82 aves sería comparar conjuntos de
  evaluación distintos. La comparación honesta vive en `o4_fig08` sobre
  filas idénticas. Esto se documenta explícitamente en §L1 de la memoria.
- **Por qué se conserva `personalizado` en L2 pese a eliminarlo en L1.**
  El cambio responde a la crítica del autor sobre el etiquetado de L1; L2
  no se reabre en esta sesión. La inconsistencia se declara en la memoria.
- **Qué desaparece de la narrativa de L1.** Los hallazgos previos
  *"`bird_id` no memoriza"* (gap personalizado < poblacional) y *"el
  personalizado aporta ~1 pp"* eran propiedades del modelo-con-`bird_id` y
  se retiran de L1. Se sustituyen por la comparación per-individuo real,
  que añade evidencia al eje "global vs per-individuo" (junto con L3).

## 9. Criterios de éxito

- `build_o4()` produce `model_individual_{rf,xgb}.pkl` y filas
  `individual` + `poblacional@91916A` en `metrics.parquet`.
- El modo individual se entrena **solo** con filas de 91916A (test:
  `metrics` del individual cubren únicamente filas de esa ave).
- `notebooks/04_eda_o4.py` regenera `o4_fig02–09` con la semántica de §7,
  sin referencias a `personalizado`.
- Suite de tests verde tras retirar viento y añadir el test del modo
  individual; `ruff` limpio.
- Sin rastro de `meteo`, `with_wind`, `o4_fig10–16`, ni deps de viento.
- Coherencia biológica: las predicciones del individual sobre 91916A
  respetan la fenología de la ruta de esa ave (sanity-check en `o4_fig08`).

## 10. Alcance — qué NO se toca

- O4 base modo **poblacional** (canónico) y sus modelos.
- **L2** (`build_l2.py`) y **L3** (`build_l3.py`), incluido su
  `personalizado`/`individual` propios.
- **O5** (se construyó sobre L3 LightGBM poblacional; no depende del
  personalizado de L1).
- La infraestructura `include_bird_id` / `_CategoricalEncoder`.
- El target, el grid 0,5°, el split temporal y los hiperparámetros.
