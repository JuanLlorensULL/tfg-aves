# Diseño de L3 multi-familia (objetivo O4) — Regresión de cuantiles con RF, XGBoost y LightGBM

- **Fecha:** 2026-05-24
- **Objetivo del TFG:** O4 — extensión de la tercera línea de mejora (L3).
  L3 cerró (`v0.4.4-o4l3-regresion`) con la regresión de cuantiles del
  desplazamiento `(Δlat, Δlon)` implementada **solo con XGBoost**. Esta
  extensión replica la misma tarea con las **otras dos familias del
  proposal** — **Random Forest** y **LightGBM** — para producir la
  **comparativa de las tres familias** sobre el problema de regresión, en
  paralelo a la comparativa de tres familias que O4 base hizo sobre
  clasificación.
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado el 2026-05-24, pendiente de implementación.
  Arranca tras el cierre de L3 (`v0.4.4-o4l3-regresion`).
- **Tag esperado al cerrar:** `v0.4.5-o4l3-multifamilia`.

> **Restricción transversal heredada (rework causal de O4,
> `v0.4.2-o4-rework-causal`).** El feature set sigue siendo las **10
> features causales** (`lat, lon, sin_doy, cos_doy, step_in_km,
> sin_bearing_in, cos_bearing_in, cos_turning_in, state_b_causal,
> posterior_b_migracion_causal`). Toda feature predictiva debe ser
> calculable sin observar el futuro
> ([[feedback-causal-features-no-leakage]]). El target `(Δlat, Δlon)` usa
> la posición de `t+1`: es la etiqueta, no una feature — no hay fuga. Las
> tres familias usan **exactamente** este feature set y este target.

## 1. Resumen

L3-v1 (`v0.4.4`) demostró que la regresión de cuantiles del
desplazamiento devuelve la geometría al problema y añade incertidumbre
calibrada, pero lo hizo con **una sola familia (XGBoost)**. El proposal
del TFG comprometía tres familias supervisadas (**RF, XGBoost,
LightGBM**), y O4 base ya las comparó sobre **clasificación**. Esta
extensión cierra la simetría: comparar las **tres familias sobre la tarea
de regresión de cuantiles**, sobre el mismo target, el mismo feature set
causal y el mismo split temporal.

Las tres familias producen los tres cuantiles `{p10, p50, p90}` por eje
(`Δlat`, `Δlon`), pero por mecanismos distintos:

- **XGBoost** (ya implementado, sin cambios): `objective='reg:quantileerror'`
  con `quantile_alpha=q`. Minimiza la **pérdida pinball**. Un modelo por
  cuantil.
- **LightGBM** (nuevo): `objective='quantile'` con `alpha=q`. Minimiza la
  **pérdida pinball** igual que XGBoost. Un modelo por cuantil. Simétrico.
- **Random Forest** (nuevo): **NO** minimiza pinball. Sus árboles parten
  minimizando varianza (MSE); los cuantiles se obtienen *post-hoc* como
  cuantiles empíricos de las muestras de entrenamiento que caen en cada
  hoja — **Quantile Regression Forest** (QRF, Meinshausen 2006). UN bosque
  por eje predice los tres cuantiles a la vez.

Esta diferencia de mecanismo es en sí misma un hallazgo para la memoria
(§4) y es la razón por la que RF necesita una vía y una dependencia
distintas (G2).

La comparativa de las tres familias se hace sobre el modo **poblacional**
(terreno común), con las mismas métricas que L3-v1: **pinball loss** y
**cobertura** `[p10,p90]` (calidad y calibración de los cuantiles) como
criterio primario; **top-1/top-3 mapeados** y **distancia** (vía
centroide y nativa) como lectura puntual (la predicción puntual de las
tres es el `p50`, así que la comparación puntual es homogénea: `p50` vs
`p50` vs `p50`, sin el matiz mediana-vs-media). La persistencia trivial
sigue de baseline.

La narrativa para la memoria: *"sobre la regresión de cuantiles, las tres
familias supervisadas convergen en calidad y calibración; el boosting
(XGB/LGBM) ataca el problema por la pérdida pinball y el bagging (RF) por
la estructura de hojas, y la comparación honesta muestra [resultado]"*.

## 2. Qué cambia respecto a L3-v1

| Dimensión | L3-v1 (`v0.4.4`) | L3 multi-familia (esta extensión) |
|---|---|---|
| Familias | XGBoost | **XGBoost + LightGBM + Random Forest** |
| Cuantiles | `{p10,p50,p90}` × 2 ejes | Igual, para las tres familias |
| Modos | poblacional + individual (91916A) | XGB: poblacional + individual; **LGBM y RF: solo poblacional** (G3) |
| Target / features / split | causal, sin fuga | **idénticos, sin cambios** |
| Mecanismo de cuantiles | pinball (XGB) | pinball (XGB/LGBM) + QRF de hojas (RF) |
| Comparativa principal | categórico vs continuo (Mo1); per-individuo (D3) | **las tres familias entre sí (poblacional)** |

L3-v1 **no se borra ni se reescribe**: queda en el tag `v0.4.4` y en
`data/processed/o4/l3_v1/` (regenerable). Esta extensión escribe en un
directorio nuevo `l3_v2/` (G7) que contiene un **superconjunto**
autocontenido: las tres familias en poblacional + el individual de XGB.

## 3. Contexto y constraints

**Datos disponibles** (verificados al 2026-05-24):

- **Pipeline L3-v1 completo** en `data/processed/o4/l3_v1/` (XGBoost
  cuantil poblacional + individual). Sus números son la referencia de que
  la config conservadora entrena bien (cobertura ≈ 80 %, convergencia
  sana, sin divergencia).
- **Matriz de features causal** y `cells.parquet`: las mismas que consume
  `build_o4_l3` hoy. No se recalcula nada nuevo de O1/O2/O3.
- **Frecuencias del dataset (O4 causal, test):** persistencia top-1
  ≈ 0,775; ~88 % de las filas test son estacionarias; días de movimiento
  real ≈ 22,5 %. Consecuencia (heredada de L3-v1): el target `Δ` está
  concentrado en cero, el `p50` predirá desplazamientos pequeños y el
  top-1 mapeado global rondará la persistencia en **las tres familias**.
  El valor diferencial sigue estando en la **distancia de los días de
  movimiento** y en la **incertidumbre calibrada**.

**Restricciones heredadas (no se replantean):**

- Las 10 features causales, exactamente (F7 de L3-v1).
- Split temporal 80/10/20 por ave, gap-aware (F9 de L3-v1). Reutiliza
  `split_temporal_per_bird`.
- Sin tuning por rejilla: configuración conservadora fija **por familia**
  (G4), análoga a la directiva F8 de L3-v1 y a §8.6 de O4.
- Sin log-loss comparable: veredicto geométrico + pinball + cobertura
  (F6 de L3-v1).
- Monotonía de cuantiles garantizada (G5).

**Restricciones nuevas / revisadas en esta extensión:**

- **Revisa F3 de L3-v1** ("familia única XGBoost"). La razón original de
  F3 (QRF = sobreingeniería para una línea; LightGBM descartado por
  divergencia en clasificación) ya no aplica: el objetivo explícito es
  ahora la comparativa de las tres familias, y la regresión de 2 targets
  continuos es un régimen distinto del de la clasificación multiclase
  (~849 clases) donde LightGBM divergía (G1, §4).
- **RF vía el paquete `quantile-forest`** (G2): añade una dependencia.
- **RF y LightGBM solo en modo poblacional** (G3).

## 4. Decisiones de diseño fijadas

Las decisiones de L3-v1 (F1–F10) se mantienen salvo donde se indique. Las
decisiones **nuevas** de esta extensión se numeran G1–G9 para no colisionar.

### G1 — Las tres familias, todas con cuantiles `{p10, p50, p90}`

Se revisa **F3** de L3-v1. L3 deja de ser "familia única" y pasa a
comparar **XGBoost, LightGBM y Random Forest**, las tres del proposal,
las tres produciendo los tres cuantiles por eje. El mecanismo difiere:

| Familia | Parámetro clave | ¿Minimiza pinball al entrenar? | Modelos por eje |
|---|---|---|---|
| XGBoost | `objective='reg:quantileerror'`, `quantile_alpha=q` | **Sí** | 3 (uno por cuantil) |
| LightGBM | `objective='quantile'`, `alpha=q` | **Sí** | 3 (uno por cuantil) |
| Random Forest | QRF: cuantiles empíricos de las hojas | **No** (parte por varianza/MSE) | **1** (predice los 3) |

**Por qué LightGBM ya no se descarta.** En O4 base y L1/L2 LightGBM
divergía, pero allí la tarea era **clasificación multiclase** sobre ~849
clases. Aquí es **regresión de 2 targets continuos** con pérdida pinball:
régimen numérico completamente distinto, divergencia improbable. Se
vigila igualmente en validación (R2).

### G2 — Random Forest vía el paquete `quantile-forest`

`RandomForestRegressor` de scikit-learn **no** produce cuantiles. Se usa
`quantile_forest.RandomForestQuantileRegressor` (paquete `quantile-forest`
de Zillow, compatible con la API de sklearn, implementa Meinshausen 2006
con backend numba). Un **único bosque por eje** se entrena una vez y
predice cualquier cuantil con `predict(X, quantiles=[0.10, 0.50, 0.90])`.

Alternativa descartada: **QRF artesanal** (entrenar `RandomForestRegressor`
estándar, guardar las `y` de entrenamiento por hoja, calcular cuantiles
empíricos en predicción). Evita la dependencia pero añade ~40 LOC + tests
propios de correctitud y más memoria/latencia. Para una extensión acotada,
apoyarse en un paquete maduro y citable es más simple y defendible
([[feedback-keep-solutions-simple]]).

### G3 — RF y LightGBM solo en modo poblacional

XGBoost conserva sus dos modos (poblacional + individual 91916A) de
L3-v1. **RF y LightGBM se entrenan solo en poblacional.** Razones:

- La comparativa de las tres familias se hace sobre **terreno común**
  (poblacional); ese es el entregable que motiva la extensión.
- El modo individual ya cumplió su función en L3-v1: zanjó la decisión
  abierta global vs per-individuo (individual ≈ poblacional). Replicarlo
  para RF/LGBM solo añadiría cómputo y artefactos sin cambiar la
  conclusión ya tomada. El individual 91916A queda como **ablación
  exclusiva de XGBoost** (no se toca).

### G4 — Hiperparámetros: config conservadora fija por familia

Sin rejilla ni Optuna (coherente con F8 de L3-v1 y §8.6 de O4). Cada
familia recibe una configuración conservadora análoga, idéntica para los
tres cuantiles (solo varía el cuantil objetivo).

**XGBoost** — sin cambios (la de `quantile.py`).

**LightGBM** — espejo de la de XGBoost:

```python
LGBMRegressor(
    objective="quantile",
    alpha=q,                 # 0.10, 0.50 o 0.90
    n_estimators=1000,       # cap; early stopping decide el efectivo
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,           # < 2**6: la restricción binding (conservador)
    min_child_samples=20,    # análogo a min_child_weight=10 de XGB
    subsample=0.8,
    subsample_freq=1,        # necesario para que subsample<1 actúe en LGBM
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=seed,
    n_jobs=-1,
    verbose=-1,
)
# early_stopping_rounds=50 sobre X_val con metric="quantile".
```

**Random Forest (QRF)**:

```python
RandomForestQuantileRegressor(
    n_estimators=300,        # los bosques convergen con menos árboles
    min_samples_leaf=20,     # CRÍTICO en QRF (ver abajo)
    max_features=0.8,        # análogo a colsample_bytree=0.8
    random_state=seed,
    n_jobs=-1,
)
```

`min_samples_leaf=20` es la decisión específica de QRF: cada hoja necesita
muestras suficientes para estimar p10/p90; hojas diminutas (p.ej.
`min_samples_leaf=1`, default de RF) dan cuantiles degenerados y
sobreajustados. 20 es conservador y consistente con `min_child_samples`
de LGBM. **RF no usa early stopping** (no hay validación incremental); se
entrena solo sobre `train`, dejando `val` sin usar en RF (documentado).

### G5 — Monotonía de cuantiles

- **XGBoost y LightGBM** entrenan un modelo independiente por cuantil →
  pueden cruzarse (`p10 > p50` o `p50 > p90`). Se fuerza monotonía por
  fila y eje con `np.sort` (mecanismo de F5 ya implementado), y se mide la
  incidencia de cruces antes de ordenar (artefacto C5).
- **Random Forest (QRF)** obtiene los tres cuantiles de la **misma
  distribución empírica** de cada hoja → son monótonos por construcción,
  **cero cruces**. Es un punto a favor de RF que se reporta en C5 como
  hallazgo (no necesita la corrección post-hoc).

### G6 — Interfaz uniforme por eje (cambio de arquitectura mínimo)

`quantile.py` está hoy cableado a XGBoost (`predict_quantiles` recibe
`dict[float, _XGBQuantileRegressor]`). Se introduce una **interfaz
uniforme de predictor por eje** que expone `predict_raw(X) -> np.ndarray`
de forma `(n, 3)` con los cuantiles en orden `QUANTILES`:

- `_PerQuantileAxis`: envuelve el `dict` de 3 regresores single-quantile
  (XGBoost o LightGBM). `predict_raw` apila las 3 predicciones.
- `_QRFAxis`: envuelve el único `RandomForestQuantileRegressor`.
  `predict_raw` llama a `predict(X, quantiles=list(QUANTILES))`.

`fit_quantile_axis(family, X_train, y, X_val, y_val, *, seed)` despacha
por familia y devuelve la interfaz (RF ignora `X_val`/`y_val`).
`_axis_quantiles_sorted` y `predict_quantiles` pasan a operar sobre la
interfaz (`predict_raw`), de modo que `build_regression_predictions`,
`pinball_loss`, `interval_coverage` y **toda `evaluate.py` quedan sin
tocar**. Es el cambio mínimo que generaliza el código sin reescribir el
pipeline de métricas.

### G7 — Serialización uniforme por (familia, modo, eje); directorio `l3_v2/`

Se serializa **un `.pkl` por (familia, modo, eje)** que contiene la
interfaz de eje (con sus 3 regresores internos en XGB/LGBM, o el único QRF
en RF) más metadatos (`familia`, `mode`, `axis`, `feature_cols`,
`individual_bird_id`). Sustituye al esquema por-cuantil de L3-v1 (que
guardaba 6 `.pkl` por modo, uno por `eje×cuantil`). Es uniforme entre
familias y más simple.

Los artefactos van a un directorio **nuevo** `data/processed/o4/l3_v2/`
(`O4_L3V2_DIR`), preservando intacto `l3_v1/` (history del tag `v0.4.4`).
La disrupción del cambio de esquema es contenida: los `.pkl` son
gitignored y regenerables, y O5 aún no ha arrancado (no hay consumidores
del esquema viejo). Conteo de modelos en `l3_v2/`:

```
data/processed/o4/l3_v2/
    model_xgb_poblacional_dlat.pkl     model_xgb_poblacional_dlon.pkl
    model_xgb_individual_dlat.pkl      model_xgb_individual_dlon.pkl
    model_lgbm_poblacional_dlat.pkl    model_lgbm_poblacional_dlon.pkl
    model_rf_poblacional_dlat.pkl      model_rf_poblacional_dlon.pkl
    predictions_test.parquet           # columnas 'familia' y 'modo'
    metrics.parquet                    # dimensión 'familia'
```

8 `.pkl` (4 ejes XGB + 2 LGBM + 2 RF). Todo en `.gitignore`.

### G8 — Métricas y predicciones ganan dimensión `familia`

`metrics.parquet` y `predictions_test.parquet` añaden la columna
`familia` (`xgb`/`lgbm`/`rf`). El bucle de `build_o4_l3` pasa a iterar
**familias × modos** (RF/LGBM solo poblacional; XGB ambos). La
persistencia trivial es independiente de la familia: se computa **una
vez** y se etiqueta `familia="—"`. El corte poblacional@91916A se mantiene
solo para XGBoost (es donde existe el modo individual).

### G9 — Criterio de comparación de las tres familias

Primario: **pinball loss** (media de los 3 cuantiles por eje) y
**cobertura** `[p10,p90]` — calidad y calibración de los cuantiles, que es
lo que L3 aporta. Secundario: **top-1/top-3 mapeados** y **distancia** vía
centroide (comparable con L1/L2) y nativa. Las tres familias predicen el
`p50` como punto, así que la comparación puntual es homogénea. No hay
log-loss (F6 de L3-v1). El veredicto se ancla en pinball+cobertura, con la
distancia en moves como lectura geométrica.

## 5. Cobertura de causas raíz

Esta extensión **no ataca una causa raíz nueva**: replica el ataque de
L3-v1 a Mo1 (target categórico → continuo) con dos familias más. Su
aportación es **metodológica y comparativa** — completar la promesa de las
tres familias del proposal sobre la tarea de regresión, y caracterizar
cómo difieren boosting (pinball) y bagging (QRF) en calidad, calibración y
cruces de cuantiles. D3 (per-individuo) sigue cubierta solo por el
individual de XGBoost (G3).

## 6. Pipeline e integración

### 6.1 Módulos modificados (sin módulos nuevos)

```
src/tfg_aves/ml/
    quantile.py   # + _LGBMQuantileRegressor, _QRFAxis/_PerQuantileAxis,
                  #   fit_quantile_axis(family, ...) despacha por familia
    build_l3.py   # bucle familias × modos; columna 'familia'; serialización G7
    _paths.py     # + O4_L3V2_DIR = O4_OUT_DIR / "l3_v2"  (~1 línea)
```

`_XGBQuantileRegressor` se conserva tal cual; `_LGBMQuantileRegressor` es
su espejo con la API de LightGBM (mismo contrato `fit(X, y, X_val, y_val)`
+ `predict`). `evaluate.py`, `features.py`, `hmm_causal.py` **sin cambios**.

### 6.2 Firma de `build_o4_l3` (extendida, retrocompatible en intención)

```python
def build_o4_l3(
    features_path=FEATURES_O3_PARQUET,
    cells_path=CELLS_PARQUET,
    out_dir=O4_L3V2_DIR,
    seed=0,
    *,
    families=("xgb", "lgbm", "rf"),
    individual_bird_id=None,
) -> BuildO4L3Result:
    """Regresión de cuantiles multi-familia.
      - Ensamblaje causal poblacional idéntico a L3-v1 (sin tocar).
      - Para cada familia f en `families`:
          modos = (poblacional, individual) si f == 'xgb' else (poblacional,)
          para cada modo: fit_quantile_axis(f, ...) en dlat y dlon,
          predict_quantiles, build_regression_predictions, métricas
          (pinball + cobertura + por estado + moves), serialización G7.
      - Persistencia: una vez (familia='—'). Corte poblacional@91916A: solo xgb.
      - Escribe predictions_test.parquet y metrics.parquet con columna 'familia'.
    """
```

`BuildO4L3Result` gana las claves por familia donde aplique
(`n_crossings`, `coverage`, `model_paths` pasan a indexarse por
`familia/modo`).

### 6.3 Dependencias añadidas a `pyproject.toml`

- **`quantile-forest`** (`uv add quantile-forest`). Se verifica
  compatibilidad con Python 3.12.3, scikit-learn ≥ 1.5 y numpy del lock
  antes de fijarlo (R3). LightGBM (≥ 4.3) ya está en el stack.

## 7. Riesgos identificados

| # | Riesgo | Mitigación / acción |
|---|---|---|
| R1 | **Comparación mediana (XGB/LGBM p50) vs media implícita.** Las tres dan p50 (mediana condicional), así que la comparación puntual es homogénea; no hay riesgo de mezclar media y mediana. | Ninguna acción: las tres comparten `p50` como punto. Se documenta que el punto es la mediana en las tres. |
| R2 | **LightGBM diverge** (como en O4 clasificación). | Régimen distinto (regresión 2 targets vs multiclase 849). Se vigila `best_iteration`/cobertura en validación; si diverge, se reporta como hallazgo y se ajusta `learning_rate`/`num_leaves` (config, no rejilla). |
| R3 | **Incompatibilidad de `quantile-forest`** con el stack fijado (Python 3.12 / sklearn / numpy del lock). | Verificar con `uv add` + un import y un `fit/predict` mínimo antes de construir sobre él. Si falla, *fallback* a QRF artesanal (alternativa de G2). |
| R4 | **QRF con `min_samples_leaf` pequeño** da cuantiles degenerados. | `min_samples_leaf=20` fijado (G4); se inspecciona la cobertura de RF en C1 (debe caer en [0,70, 0,90]). |
| R5 | **Cambio de esquema de serialización** rompe el notebook/tests de L3-v1. | `l3_v2/` es nuevo; `l3_v1/` intacto. El notebook y los tests se actualizan al esquema G7 en esta misma extensión. |
| R6 | **Extender C1/C2/C5** cambia el contenido de artefactos ya citados en la memoria. | Decisión explícita del autor (extender C1/C2 a tres familias). Las notas de memoria de L3 se actualizan para reflejar las tres familias; el número de figura se conserva. |
| R7 | **Coste de cómputo** (3 familias × cuantiles × ejes). | Acotado: poblacional para LGBM/RF, sin rejilla. RF con 300 árboles y XGB/LGBM con early stopping. Tiempo comparable a un par de ejecuciones de L3-v1. |

## 8. Validación y artefactos

### 8.1 Tests automáticos

Ampliar `tests/test_ml_quantile.py` y `tests/test_ml_build_l3.py` (no se
rompe ningún test de L3-v1 que siga aplicando; los que asumen el esquema
viejo de 12 `.pkl` se actualizan al de G7):

- `test_fit_quantile_axis_lgbm_shapes`: LightGBM entrena 3 cuantiles por
  eje; `predict_raw` devuelve `(n, 3)`.
- `test_fit_quantile_axis_rf_shapes`: RF (QRF) entrena un bosque por eje;
  `predict_raw` devuelve `(n, 3)`.
- `test_rf_quantiles_monotonic_by_construction`: en QRF, `p10 ≤ p50 ≤ p90`
  en todas las filas **sin** ordenación post-hoc (cruces = 0).
- `test_lgbm_quantiles_sorted_after_predict`: con un mock que cruza,
  `predict_quantiles` los ordena y cuenta los cruces (como en XGB).
- `test_axis_interface_uniform`: las tres familias exponen el mismo
  contrato `predict_raw(X) -> (n, 3)`.
- `test_build_o4_l3_multifamilia_artifacts`: produce los 8 `.pkl`,
  `predictions_test.parquet` y `metrics.parquet` con columna `familia`
  conteniendo `{xgb, lgbm, rf}`.
- `test_build_o4_l3_no_leakage_all_families`: ninguna familia usa columnas
  de `t+1` como feature (las 10 causales; el target no entra como input).
- `test_build_o4_l3_rf_lgbm_poblacional_only`: no hay filas/modelos de
  modo individual para RF ni LGBM.
- `test_build_o4_l3_idempotent`: misma `seed` → mismos artefactos.

Criterio: todos los tests (los actuales que sigan aplicando + los nuevos)
pasan; ruff limpio.

### 8.2 Artefactos `save_artifact`

| ID | Tipo | Acción | Decisión / hallazgo documentado |
|---|---|---|---|
| **C1** (`o4_fig25_calibration-coverage`) | figura + tabla | **Extender** | Cobertura empírica de `[p10,p90]` vs 80 % nominal, **por familia** (xgb/lgbm/rf) y eje. Compara la calibración de las tres. |
| **C2** (`o4_fig26_distance-distribution`) | figura + tabla | **Extender** | Distribución de distancias del `p50` de **las tres familias** vs persistencia, global y por estado (poblacional). |
| **C5** (`o4_fig29_quantile-crossing`) | figura + tabla | **Extender** | Incidencia de cruces por familia y eje. Hallazgo: XGB/LGBM cruzan algo (corrección post-hoc); **RF = 0 por construcción**. |
| **NUEVO** (`o4_tab31_comparativa-familias-l3`) | tabla | **Crear** | **Tabla maestra de las tres familias** (poblacional) × régimen (global / estacionario / migración / moves): top-1, top-3, dist centroide, dist nativa, pinball lat/lon, cobertura lat/lon + baseline persistencia. **Es el entregable central de la extensión.** |

D1 (`fig24`), C3 (`tab27`, ablación de target / per-individuo) y C4
(`fig28`, vectores de 91916A) de L3-v1 **se conservan sin cambios** (son
XGB-céntricos y siguen siendo válidos).

### 8.3 Esquema de la tabla maestra (artefacto nuevo)

**Tres familias, modo poblacional, test completo.** Baseline persistencia
y, como contexto, el XGB categórico de O4 (L3-v0).

| familia | mecanismo | top-1 map | top-3 map | dist centroide km | dist nativa km | pinball lat | pinball lon | cobertura lat | cobertura lon | cruces |
|---|---|---|---|---|---|---|---|---|---|---|
| persistencia | baseline | 0,775 | 0,775 | 21,0 | — | — | — | — | — | — |
| XGBoost | pinball | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| LightGBM | pinball | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Random Forest | QRF (hojas) | ? | ? | ? | ? | ? | ? | ? | ? | **0** |

Su desglose por estado/moves-only es la evidencia primaria de la
subsección "L3 — comparativa de familias" del capítulo 6 de la memoria.

## 9. Métricas de éxito y criterio de aceptación

No hay umbral de "ganar a la persistencia" (heredado: el techo es
estructural). El éxito de la extensión es **producir la comparativa
honesta de las tres familias** con:

1. **Las tres familias entrenan sin divergencia** (best_iteration sano en
   XGB/LGBM; cobertura de RF en rango). LightGBM no diverge (R2).
2. **Cobertura `[p10,p90]` ∈ [0,70, 0,90]** en las tres (calibración).
3. **Pinball y distancia reportadas y comparables** entre las tres sobre
   el mismo test poblacional.
4. **Cruces de cuantiles caracterizados** (XGB/LGBM > 0 corregidos; RF = 0).

Interpretación honesta para la memoria ([[feedback-memoria-tone]]),
cualquier resultado es válido:

- **Las tres convergen** → "la elección de familia es secundaria en la
  regresión de cuantiles del desplazamiento; el cuello de botella es el
  problema (horizonte 1 día, target concentrado), no el algoritmo".
- **Una familia destaca** → se reporta cuál y en qué métrica (p.ej. RF
  mejor calibrado por su estimación no paramétrica de hojas, o boosting
  mejor en pinball por optimizar la pérdida directamente), con
  interpretación del mecanismo.

Toda conclusión pasa el sanity-check de coherencia biológica
([[feedback-biological-coherence]]): ninguna familia debe predecir saltos
imposibles para *Larus fuscus*.

## 10. Decisiones metodológicas sin figura (van a la memoria)

- **Por qué se reabre F3 (familia única).** El objetivo explícito de la
  extensión es la comparativa de las tres familias del proposal sobre
  regresión; la razón original de F3 (QRF sobreingeniería, LGBM divergente
  en clasificación) no aplica a este objetivo ni a este régimen numérico
  (G1).
- **Por qué RF llega a los cuantiles por una vía distinta.** RF parte por
  varianza (MSE), no por pinball; los cuantiles salen de la distribución
  empírica de las hojas (QRF, Meinshausen 2006). Es un contraste
  conceptual boosting-vs-bagging que enriquece la memoria (G1, G5).
- **Por qué el paquete `quantile-forest` y no QRF artesanal.** Maduro,
  citable, mínimo código; el artesanal sería sobreingeniería para una
  extensión acotada (G2, [[feedback-keep-solutions-simple]]).
- **Por qué RF y LGBM solo poblacional.** La comparativa de familias vive
  en terreno común; el per-individuo ya quedó zanjado en L3-v1 (G3).
- **Por qué `min_samples_leaf=20` en RF.** QRF necesita muestras
  suficientes por hoja para estimar p10/p90; hojas diminutas degeneran los
  cuantiles (G4).
- **Por qué RF no usa el split de validación.** RF no tiene early stopping;
  se entrena solo sobre `train`. El `val` solo lo usan XGB/LGBM (G4).
- **Por qué un `.pkl` por (familia, modo, eje) y no por cuantil.** Uniforme
  entre familias (RF guarda un solo bosque por eje); más simple (G7).
- **Por qué la comparación puntual es homogénea.** Las tres familias
  predicen el `p50` (mediana condicional) como punto; no se mezcla con la
  media (R1).

## 11. Entregables al cerrar la extensión

Al cerrar (tag esperado `v0.4.5-o4l3-multifamilia`):

- `quantile.py`, `build_l3.py`, `_paths.py` modificados y commiteados.
- Dependencia `quantile-forest` en `pyproject.toml` + `uv.lock`.
- Tests nuevos/actualizados en `tests/test_ml_quantile.py` y
  `tests/test_ml_build_l3.py`. Todos pasando; ruff limpio.
- Notebook `notebooks/04l3_eda_o4l3.py` actualizado: ejecuta el build
  multi-familia, extiende C1/C2/C5 y crea la tabla maestra `tab31`.
- Artefactos: C1/C2/C5 extendidos + `o4_tab31_comparativa-familias-l3`
  nuevos en `reports/{figures,tables,captions}/` y `reports/INDEX.md`.
- Notas de memoria en `reports/memoria/06_o4_ml.md` (subsección "L3 —
  comparativa de familias").
- Entrada en `reports/ai-log/` (`00NN-o4l3-multifamilia.md`) según scope.
- Commit(s) en castellano + tag `v0.4.5-o4l3-multifamilia`.

## 12. Conexiones con el resto del TFG

- **L3-v1** ([spec](2026-05-24-o4l3-regresion-design.md)): esta extensión
  reutiliza su target, features, split, métricas y maquinaria; revisa solo
  su F3 (familia única) y su esquema de serialización.
- **O4 base**: cierra la simetría con la comparativa de tres familias que
  O4 hizo sobre clasificación. Ancla en [[project-o4-outcome]] y
  [[project-o4-improvement-lines]].
- **O3 (HMM)**: las tres familias consumen `state_b_causal` y
  `posterior_b_migracion_causal` y desglosan por estado. [[project-o3-outcome]].
- **O5**: la banda `[p10,p90]` (de cualquiera de las tres, la mejor
  calibrada según C1) sigue siendo el candidato a la capa de incertidumbre
  del mapa interactivo.

---

Conecta con [[project-o4-improvement-lines]], [[project-o4-outcome]],
[[project-o4-causal-rework]], [[feedback-causal-features-no-leakage]]
(feature set causal obligatorio en las tres familias),
[[feedback-memoria-tone]] (tono investigador al narrar una posible
convergencia de familias), [[feedback-decisions-must-be-justified]] (cada
decisión G* y sin figura va a la memoria), [[feedback-methodological-justifications]]
y [[feedback-keep-solutions-simple]] (paquete maduro vs QRF artesanal, sin
tuning, sin modos redundantes).
