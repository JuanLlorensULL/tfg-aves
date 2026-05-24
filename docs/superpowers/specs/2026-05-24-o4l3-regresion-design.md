# Diseño de L3 (objetivo O4) — Regresión espacial con cuantiles

- **Fecha:** 2026-05-24
- **Objetivo del TFG:** O4 — tercera y última línea de mejora (L3) del
  pipeline supervisado base. Sustituye el **target categórico** (celda
  0,5°) del clasificador monolítico de O4 por un **target continuo**
  (desplazamiento `(Δlat, Δlon)` en grados) modelado con **regresión de
  cuantiles**, para atacar la causa estructural **Mo1** ("el target
  categórico desperdicia la geometría del problema") diagnosticada en
  `reports/memoria/06_o4_ml.md` §"Análisis estructural del techo de
  rendimiento". Además, prueba de forma directa la **decisión abierta del
  TFG** (modelo global vs por-individuo) mediante un modelo dedicado al
  ave con más histórico (**91916A**), atacando de paso la causa **D3**.
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado el 2026-05-24, pendiente de implementación.
  Arranca tras el cierre de L2 (`v0.4.3-o4l2-dos-etapas`).

> **Restricción transversal heredada (rework causal de O4,
> `v0.4.2-o4-rework-causal`).** El feature set de L3 son las **10
> features causales** (`lat, lon, sin_doy, cos_doy, step_in_km,
> sin_bearing_in, cos_bearing_in, cos_turning_in, state_b_causal,
> posterior_b_migracion_causal`). Cinemática **entrante** (sin fuga, con
> inercia real) y estado de **HMM causal filtrado**. Toda feature
> predictiva debe ser calculable sin observar el futuro
> ([[feedback-causal-features-no-leakage]]). El target `(Δlat, Δlon)`
> usa la posición de `t+1`: es la etiqueta, no una feature — no hay fuga.

## 1. Resumen

O4 base, L1 y L2 plantean el problema como **clasificación** sobre 849
celdas activas del grid 0,5°. Esa discretización tira a la basura la
estructura geométrica: dos celdas adyacentes son tan "distintas" para la
log-loss como dos celdas en continentes opuestos, y el modelo no puede
expresar "el ave se desplaza ~30 km al noreste" salvo eligiendo una
celda concreta.

L3 reformula el problema como **regresión del desplazamiento continuo**:

- **Target:** `y_dlat = lat_{t+1} − lat_t`, `y_dlon = lon_{t+1} − lon_t`
  (grados). Cartesianas, no rumbo+distancia, para evitar la
  discontinuidad angular en ±180°.
- **Modelo:** regresión de cuantiles con XGBoost
  (`objective='reg:quantileerror'`), tres cuantiles `{p10, p50, p90}`
  por eje. La predicción puntual es el `p50`; el par `[p10, p90]`
  cuantifica la **incertidumbre** del desplazamiento — algo que el
  clasificador no entrega de forma geométricamente interpretable.
- **Alcance: monolítico sobre todos los días.** Cada regresor se entrena
  sobre TODAS las filas candidatas (no sólo las de movimiento). En días
  estacionarios el target es `Δ ≈ 0`; la calidad en días de movimiento
  real se aísla con la vista "moves only".

L3 se entrena en **dos modos** (ver F2): un modelo **poblacional**
(todas las aves, sin `bird_id`) y un modelo **individual** dedicado al
ave con más histórico (**91916A**, 2025 filas candidatas). El modo
"personalizado" de O4 (`bird_id` como feature de un modelo global) se
**descarta** y se sustituye por este modelo individual, que es una
prueba más limpia de la hipótesis per-individuo.

L3 **no toca features, ni split, ni la base de hiperparámetros**
respecto a O4 causal. Las dos variables manipuladas son la **naturaleza
del target** (categórico → continuo, ataca Mo1) y el **alcance del
modelo** (poblacional vs un único individuo, ataca D3). Cada una se
evalúa en su propia comparativa aislada (§8.3).

Tras la implementación, el sistema produce artefactos comparables sobre
**el mismo test split temporal por ave** que el resto de O4:

- **L3-v0** (baseline): el O4 **monolítico causal poblacional** (tag
  `v0.4.2-o4-rework-causal`), familia **XGBoost**, regenerado con
  `build_o4()`. Es el contraste categórico directo del modo poblacional.
  (NO el O4 con fuga `v0.4-o4-completo`, NI la arquitectura de dos etapas
  de L2.)
- **L3-v1**: los regresores de cuantiles (poblacional + individual).
  Genera nuevos modelos `.pkl`, `predictions_test.parquet` y
  `metrics.parquet` bajo `data/processed/o4/l3_v1/`.

La narrativa fuerte para la memoria: *"tratar la predicción como
regresión continua devuelve la geometría al problema y añade
incertidumbre calibrada; y un modelo dedicado al ave con más datos
permite, por fin, contrastar de frente el dilema global vs per-individuo
abierto desde O3"*.

### 1.1 Por qué monolítico y no "moves only" (decisión del autor)

L2 dejó al descubierto que el problema duro es el **destino de las aves
que sí se mueven**: su `clf_dest` cayó a top-1 0,037 sobre `y_move = 1`.
Una tentación era construir L3 como un regresor entrenado sólo sobre
filas de movimiento (espejo de la etapa 2B de L2). Se descartó porque:

- Pierde la ablación que da sentido a L3 — "categórico vs continuo sobre
  el mismo dataset". Un regresor moves-only ya no es comparable con el
  O4 monolítico, sino con la etapa 2B de L2 (otra línea).
- Reintroduciría la dependencia de un `clf_move` externo para los días
  estacionarios, mezclando la contribución de L3 (target continuo) con
  la de L2 (arquitectura en dos etapas).

El regresor monolítico se evalúa **igualmente** sobre el subconjunto
`y_move = 1` (vista "moves only", §8) para responder a la pregunta que
L2 dejó abierta: *¿la geometría continua rescata el destino que el
clasificador no veía?* — pero el modelo entrenado es uno solo, sobre
todos los días.

## 2. Vocabulario

- **L3-v0** — estado base sin regresión. Equivale al O4 monolítico
  **causal poblacional** (`v0.4.2-o4-rework-causal`), familia XGBoost. No
  requiere implementación, sólo regeneración de sus artefactos con
  `build_o4()` y referenciación como punto de comparación.
- **L3-v1** — los regresores de cuantiles del desplazamiento. Es lo que
  se implementa en este spec.
- **Modo poblacional** — un único modelo entrenado sobre las 82 aves, sin
  `bird_id`. Modelo principal de L3, comparable con L3-v0, L1/L2 y
  baselines sobre el test completo.
- **Modo individual (91916A)** — un modelo entrenado SÓLO con las filas
  de 91916A (el ave con más histórico). Prueba la hipótesis per-individuo
  sobre el mejor candidato posible.
- **Target continuo** — `(Δlat, Δlon)` en grados, derivado de
  `lat_t_next − lat` y `lon_t_next − lon`.
- **Cuantiles `{p10, p50, p90}`** — los tres niveles modelados por eje.
  `p50` es la predicción puntual; `[p10, p90]` el intervalo nominal 80 %.
- **Predicción puntual** — `(lat_t + Δlat_p50, lon_t + Δlon_p50)`.
- **Celda mapeada** — celda 0,5° que contiene la predicción puntual
  (`assign_cell`). Puente para comparar con la clasificación.
- **Distancia nativa** — haversine entre la predicción puntual y la
  posición verdadera de `t+1`. Métrica natural de L3.
- **Distancia vía centroide** — haversine entre el **centroide de la
  celda mapeada** y la posición verdadera. Directamente comparable con
  `dist_median_km` de L1/L2 (que también parten de centroides).
- **Cobertura** — fracción de filas test donde el desplazamiento
  verdadero cae dentro de `[p10, p90]` (por eje). Mide la calibración de
  la incertidumbre; el ideal es ≈ 0,80.
- **Pinball loss** — pérdida de cuantil estándar; calidad de los
  cuantiles. Por eje y promedio.
- **Subset "moves only"** — restricción del test a filas donde
  verdaderamente `y_move = 1` (`cell_id_t_next ≠ cell_id_t`).
- **Comparativa de target** (poblacional) — L3-v0 vs L3-v1 sobre el test
  completo; cuantifica el efecto de reformular el target (Mo1).
- **Comparativa per-individuo** (91916A) — modelo individual vs modelo
  poblacional, ambos sobre el mismo test de 91916A; cuantifica el efecto
  del alcance del modelo (D3).

## 3. Contexto y constraints

**Datos disponibles** (verificados al 2026-05-24):

- **Pipeline O4 causal completo** en `data/processed/o4/` con los 6
  modelos `.pkl`, `predictions_test.parquet` y `metrics.parquet`. El
  baseline L3-v0 son las filas **XGBoost poblacional** de
  `metrics.parquet`.
- **Matriz de features causal** (`build_feature_matrix` +
  `compute_causal_kinematics`) contiene `lat`, `lon`, `lat_t_next`,
  `lon_t_next` (posición de `t+1`, ya filtrada a pares consecutivos
  válidos), `cell_id_t` y `cell_id_t_next`. El target continuo se deriva
  de estas columnas sin recálculo geométrico nuevo.
- **Distribución de histórico por ave (verificada sobre la matriz
  candidata poblacional, 20 188 filas, 82 aves):** 91916A es el ave con
  más filas con diferencia (**2025**, rank 1), seguida de 91752A (1430)
  y 91823A (1343). Esto justifica la selección de 91916A para el modo
  individual (F2). Su split temporal individual: **train 1458** (370 de
  movimiento) / **val 162** (41) / **test 405** (126) — suficiente para
  6 regresores de cuantil con early stopping.
- **Feature set causal (10)**: idéntico a O4 causal y L2. El estado HMM
  causal (`state_b_causal`, `posterior_b_migracion_causal`) se adjunta
  tras el split vía `fit_causal_hmm` + `decode_causal_states`, igual que
  en `build_o4`. El modo individual **reutiliza el estado del HMM global**
  (ver F2), no reentrena un HMM por ave.
- **`cells.parquet`** (O2) con `cell_id`, `lat_c`, `lon_c` (centroides)
  para el mapeo punto→celda y la distancia vía centroide.

**Frecuencias relevantes del dataset (O4 causal, test):**

- Persistencia trivial top-1 = **0,775**; ~88 % de las filas test son
  estacionarias por estado HMM. Los **días de movimiento real** son
  **22,5 %** del test (908/4037).
- Consecuencia esperada: el target `Δ` está fuertemente concentrado
  cerca de cero. El `p50` predirá desplazamientos pequeños en la mayoría
  de filas → la celda mapeada coincidirá con `cell_t` → top-1 mapeado
  ≈ persistencia. El valor de L3 NO está en el top-1 global sino en la
  distancia de los días de movimiento y en la incertidumbre calibrada.

**Restricciones heredadas de O4 causal (no se replantean en L3):**

- **Features**: las 10 causales, exactamente. No se reabre F7 de O4
  (`veg_*`, `daylight_hours` entran sólo en la emisión del HMM causal).
- **Split temporal 80/10/20 por ave**, gap-aware (cuatro barreras de
  §8.11 del spec O4). Reutiliza `split_temporal_per_bird`.
- **Base de hiperparámetros**: la configuración conservadora §8.6 de O4,
  adaptada a regresión (F8). Sin tuning nuevo.

**Restricciones nuevas introducidas en L3:**

- **Modos poblacional + individual (91916A); sin modo personalizado.**
  Decisión F2.
- **Familia única: XGBoost.** RF de scikit-learn no soporta regresión de
  cuantiles nativa y LightGBM sigue descartado. L3 prueba el *target* y
  el *alcance*, no la familia (F3).
- **Cuantiles monótonos garantizados** por ordenación post-hoc (F5).
- **Sin log-loss comparable.** L3 se juzga en terreno geométrico
  (distancia) + top-1 mapeado (F6, §9).

## 4. Decisiones de diseño fijadas

### F1 — Target continuo `(Δlat, Δlon)`, cartesiano

El desplazamiento se modela en grados de latitud/longitud, no como
rumbo+distancia. Razones:

- **Sin discontinuidad angular.** El rumbo salta de +180° a −180° en el
  meridiano de inversión; un regresor con pinball sobre el ángulo se
  rompe ahí. `(Δlat, Δlon)` es continuo en todo el dominio.
- **Aditividad trivial.** La predicción puntual se reconstruye como
  `(lat_t + Δlat, lon_t + Δlon)` sin trigonometría.
- **Coherencia con las features.** `sin_bearing_in`/`cos_bearing_in` ya
  codifican la dirección entrante de forma continua; el target hace lo
  propio en la salida.

Limitación asumida (§10): un grado de longitud no equivale a un grado de
latitud en km salvo en el ecuador. La latitud del dataset (~0°–60°N) hace
la distorsión moderada; la métrica de evaluación (haversine) la corrige
íntegramente, sólo el **entrenamiento** opera en grados. No se compensa
con pesos por latitud (YAGNI).

### F2 — Modos poblacional + individual (91916A); sin modo personalizado

L3 se entrena en **dos modos**, ambos XGBoost cuantil:

- **Poblacional** — un modelo sobre las 82 aves, **sin `bird_id`**.
  Modelo principal, contraste directo de L3-v0 (categórico) y baselines
  sobre el test completo. Ataca **Mo1**.
- **Individual (91916A)** — un modelo entrenado **sólo con las filas de
  91916A**, el ave con más histórico (2025 filas, rank 1 verificado en
  §3). Sin `bird_id` (es constante). Ataca **D3** (rutas individuales).

**Se descarta el modo "personalizado" de O4** (un modelo global con
`bird_id` como feature categórica). Razones:

- En O4 causal aportaba sólo +1 pp de top-1 y, tras eliminar la fuga,
  `bird_id` dejó de "memorizar" (gap train-test casi igual al
  poblacional). Su valor marginal es bajo.
- Un modelo **genuinamente per-individuo** es una prueba más limpia y
  honesta de la **decisión abierta del TFG** (global vs por-individuo vs
  muestra representativa, registrada en `CLAUDE.md` desde O3): en lugar
  de diluir `bird_id` entre 82 aves, dedica toda la capacidad del modelo
  al individuo con datos suficientes para sostenerlo.

**Comparación per-individuo justa.** El modelo individual se evalúa sobre
el **test de 91916A** (sus 405 filas). El modelo poblacional se evalúa
también sobre **ese mismo subconjunto** (filtrando sus predicciones a
`bird_id == 91916A`). Como `split_temporal_per_bird` parte cada ave igual,
las 405 filas test de 91916A son **idénticas** en ambos modelos → el
contraste individual vs poblacional es directo, sobre el mismo holdout, y
sin fuga (ambos modelos sólo vieron pasado; el poblacional vio además el
pasado de las otras 81 aves).

**El HMM causal NO se reentrena por ave.** El modo individual reutiliza
las features `state_b_causal`/`posterior_b_migracion_causal` decodificadas
por el **HMM global** (que ya respeta el corte temporal de 91916A vía
`cutoff_by_bird`). Así la única variable del contraste per-individuo es el
**alcance del regresor supervisado**, no el del HMM. Ablación limpia.

### F3 — Familia única XGBoost; ni RF ni LightGBM

L3 usa **sólo XGBoost** con `objective='reg:quantileerror'`. RF de
scikit-learn no produce cuantiles nativos (añadir *quantile regression
forests* sería sobreingeniería para una línea); LightGBM soporta
`objective='quantile'` pero fue descartado en O4 por divergencia, y
reabrirlo mezclaría contribuciones. L3 manipula el target y el alcance,
no la familia. La consecuencia — que el mejor modelo absoluto de O4 (RF
personalizado) no tenga contrapartida directa — se documenta: la
comparación honesta es **XGBoost categórico (L3-v0) vs XGBoost cuantil
(L3-v1)**, con el RF de O4 como mera referencia de contexto.

### F4 — Tres cuantiles `{p10, p50, p90}`, un regresor por (eje × cuantil)

Por modo se entrenan **6 regresores**: `{p10, p50, p90} × {Δlat, Δlon}`.
Con 2 modos, **12 modelos** en total.

- `p50` da la predicción puntual (mediana condicional, robusta a las
  colas de migración).
- `{p10, p90}` definen un intervalo nominal al 80 %, interpretable, sin
  multiplicar modelos. No se añaden p05/p95 (YAGNI).
- Un regresor independiente por cuantil (en lugar de un multi-cuantil)
  mantiene el código simple y el control de la monotonía explícito (F5).

### F5 — Monotonía de cuantiles forzada post-hoc

Los regresores se entrenan independientemente, así que nada garantiza
`p10 ≤ p50 ≤ p90` fila a fila (*quantile crossing*). Tras predecir se
ordenan los tres valores por fila y eje (`np.sort`). La incidencia de
cruces antes de la corrección se mide y reporta (C5): si es alta, es
señal de cuantiles mal estimados y se discute en la memoria.

### F6 — Sin log-loss derivado; veredicto geométrico

Se descarta derivar un log-loss asumiendo gaussiana bivariada a partir de
los cuantiles e integrando sobre las 849 celdas: introduce un supuesto
frágil (gaussianidad, independencia lat/lon) de sesgo no neutral y añade
complejidad para recuperar una métrica que L3 no necesita para su tesis
geométrica. L3 se juzga con **distancia** (nativa y vía centroide),
**top-1/top-3 mapeados**, **pinball loss** y **cobertura**. La ausencia
de log-loss directamente comparable con el criterio ganador F8 de O4 se
declara como **límite metodológico explícito** (§9, §10), coherente con
[[feedback-keep-solutions-simple]].

### F7 — Features causales idénticas a O4, sin excepción

Las 10 features causales son las únicas que ven los regresores. En el
modo poblacional no se incluye `bird_id`; en el individual tampoco (es
constante). Se reutiliza el ensamblaje de `build_o4` (cinemática causal +
HMM causal adjuntado tras el split).

### F8 — Hiperparámetros: config conservadora §8.6 adaptada, validada empíricamente

**Directiva del autor (2026-05-24):** sin tuning específico; una
configuración básica que funcione bien. Se adopta la config conservadora
§8.6 de O4 trasladada a la API de regresión, **idéntica para los dos
modos y los tres cuantiles** (sólo varía `quantile_alpha`). Mantenerla
idéntica entre modos es lo correcto para la ablación: la única variable
del contraste per-individuo es el **alcance de los datos**, no los
hiperparámetros.

```python
xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=q,            # 0.10, 0.50 o 0.90
    n_estimators=1000,           # cap; early stopping decide el efectivo
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=seed,
    n_jobs=-1,
    early_stopping_rounds=50,
)
```

Early stopping sobre `X_val` con la métrica por defecto del objetivo
(pinball). Sin rejilla ni Optuna.

**Validación empírica (sanity check 2026-05-24, preliminar — sólo las 8
features cinemáticas, sin estado HMM; confirma que la config entrena bien,
no son los números finales):**

- **Convergencia sana, sin divergencia** (a diferencia de LightGBM en O4):
  `best_iteration` ∈ [50, 380] en poblacional y [63, 257] en individual;
  el cap de 1000 nunca se alcanza y early stopping dispara correctamente.
- **Cobertura del intervalo [p10,p90] ≈ 80 % nominal**: poblacional
  lat 0,83 / lon 0,85; individual 91916A lat 0,79 / lon 0,76. La
  incertidumbre sale calibrada sin tuning.
- **Distancia coherente**: en días de movimiento, el modelo individual
  baja a 23,7 km (vs 29,9 km persistencia); el poblacional 29,7 km
  (vs 30,3 km). Modesto en global (target concentrado en cero), con señal
  en moves, especialmente per-individuo.

Esta evidencia respalda fijar la config sin más tuning (la curva de
mejora marginal no compensa, consistente con la decisión post-cierre de
O4). xgboost 3.2.0 soporta `reg:quantileerror` nativo.

### F9 — Split temporal idéntico a O4 base

Sin cambios. Mismo 80/10/20 por ave con las cuatro barreras gap-aware.
Para el modo individual, el split se aplica **sobre las filas de 91916A**
(su propio 80/10/20: 1458/162/405). Garantiza comparabilidad total entre
modos y con L3-v0 sobre el mismo test.

### F10 — Target derivado de columnas existentes, sin fuga

`y_dlat = lat_t_next − lat`, `y_dlon = lon_t_next − lon`. Ambas columnas
ya existen en la matriz causal, filtradas a pares consecutivos válidos.
Definen la etiqueta, no son features de entrada: **no hay fuga**. Las
filas sin `t+1` consecutivo válido ya están excluidas de la matriz.

## 5. Cobertura de causas raíz

Causas raíz diagnosticadas en `reports/memoria/06_o4_ml.md`:

| Causa | Cobertura L3 | Comentario |
|---|---|---|
| D1 self-loops dominantes (73 %) | — | Cubierto por L2. |
| D2 snapshot diario (1 punto/día) | — | Fuera de scope (trabajo futuro: secuencia). |
| **D3 rutas individuales heterogéneas** | **Directa** | El modo individual 91916A prueba la hipótesis per-individuo sobre el ave con más datos, contra el poblacional, en el mismo test. |
| D4 sin viento | — | Cubierto por L1. |
| D5 sin destino | — | Fuera de scope (trabajo futuro). |
| **Mo1 target categórico** | **Directa** | Núcleo del diseño: target continuo `(Δlat, Δlon)` con cuantiles. |
| Mo2 horizonte 1 día | — | Fuera de scope. |
| Mo3 sin historia multi-día | — | Fuera de scope. |

L3 ataca **dos** causas (Mo1 y D3) pero cada una en su **comparativa
aislada** (§8.3): la de target sobre el modo poblacional, la
per-individuo sobre 91916A. Así ninguna mejora se atribuye con
ambigüedad.

## 6. Pipeline e integración

### 6.1 Módulos nuevos

```
src/tfg_aves/ml/
    quantile.py      # regresión de cuantiles + puente a celda + métricas (~190 LOC)
    build_l3.py      # orquestador build_o4_l3() (~210 LOC)
```

**`quantile.py`** (funciones puras + un wrapper fino):

```python
QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
INDIVIDUAL_BIRD_ID: str = "91916A"   # ave con más histórico (rank 1, 2025 filas)

def derive_displacement_target(matrix: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con y_dlat, y_dlon en grados.
    y_dlat = lat_t_next − lat ; y_dlon = lon_t_next − lon.
    No tiene fuga: ambas definen la etiqueta, no son features."""

class _XGBQuantileRegressor(BaseEstimator, RegressorMixin):
    """Wrapper sobre XGBRegressor(objective='reg:quantileerror') con
    early stopping sobre val. Un cuantil por instancia. Config §8.6 (F8)."""
    def __init__(self, quantile: float, seed: int = 0): ...
    def fit(self, X, y, X_val=None, y_val=None): ...
    def predict(self, X) -> np.ndarray: ...

def fit_quantile_axis(
    X_train, y_train_axis, X_val, y_val_axis, *, seed,
) -> dict[float, _XGBQuantileRegressor]:
    """Entrena los 3 regresores {p10,p50,p90} para UN eje."""

def predict_quantiles(
    models_lat, models_lon, X,
) -> pd.DataFrame:
    """Predice los 6 cuantiles, fuerza monotonía por eje (np.sort) y
    devuelve dlat_p10/50/90, dlon_p10/50/90 + n_crossings_lat/lon
    (incidencia de cruces ANTES de ordenar, para C5)."""

def point_to_cell(lat_pred, lon_pred, cells, cell_deg=0.5):
    """Mapea cada punto a (cell_id, dist_via_centroide). Fuera del grid
    activo → NaN (se cuenta y reporta)."""

def nearest_cells(lat_pred, lon_pred, cells, k=3) -> list[str]:
    """k celdas activas más cercanas al punto. Define el top-k 'por
    proximidad geográfica' de un regresor."""

def pinball_loss(y_true, y_pred_q, q) -> float: ...
def interval_coverage(y_true, y_p10, y_p90) -> float:
    """Fracción de y_true dentro de [p10,p90]. Ideal ≈ 0.80."""

def build_regression_predictions(quantile_preds, meta, cells) -> pd.DataFrame:
    """Ensambla predicciones con el MISMO esquema que la clasificación
    (true_cell, pred_cell_top1, pred_cell_topk, pred_dist_km [=vía
    centroide], state_b_causal) MÁS columnas de regresión (pred_lat,
    pred_lon, dist_native_km, dlat_p*/dlon_p*, in_interval_lat,
    in_interval_lon). Así reutiliza evaluate.py sin cambios."""
```

**`build_l3.py`** (orquestador):

```python
def build_o4_l3(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L3V1_DIR,
    seed: int = 0,
) -> BuildO4L3Result:
    """Orquesta L3-v1 sobre el feature set CAUSAL.

    Ensamblaje idéntico a build_o4 (poblacional, include_bird_id=False):
      0a. kin = compute_causal_kinematics(features_o3)
      0b. matriz = build_feature_matrix(kin, cells, include_bird_id=False)
      0c. split_temporal_per_bird → train/val/test
      0d. adjuntar state_b_causal y posterior_b_migracion_causal vía
          fit_causal_hmm + decode_causal_states (HMM GLOBAL, una vez).
      0e. target = derive_displacement_target(matriz) sobre cada split.

    Modo POBLACIONAL: entrena sobre train/val completos; evalúa en test.
    Modo INDIVIDUAL: filtra train/val/test a bird_id == INDIVIDUAL_BIRD_ID
      (reusa el MISMO estado HMM y el MISMO split, sólo subconjunta filas);
      entrena sobre el train de 91916A; evalúa en su test.

    Por cada modo m:
      1. models_lat = fit_quantile_axis(X_train, y_dlat, X_val, ...).
      2. models_lon = fit_quantile_axis(X_train, y_dlon, X_val, ...).
      3. quantile_preds = predict_quantiles(models_lat, models_lon, X_test).
      4. preds = build_regression_predictions(quantile_preds, meta_test, cells).
      5. Persiste los 6 model_*.pkl del modo.

    Tras ambos modos: predictions_test.parquet (concatena con columna
    'modo'), metrics.parquet (global + por estado + moves-only + pinball +
    cobertura, por modo, MÁS el corte poblacional sobre 91916A para la
    comparativa per-individuo). Referencia el baseline L3-v0 (XGBoost
    poblacional de build_o4) sin recalcularlo."""
```

### 6.2 Cambios en módulos existentes

**Mínimos.** L3 reutiliza sin modificación `features.py`
(`compute_causal_kinematics`, `build_feature_matrix`,
`split_temporal_per_bird`), `hmm_causal.py` (`fit_causal_hmm`,
`decode_causal_states`) y `evaluate.py` (`top_k_accuracy`,
`dist_median_km`, `evaluate_by_state`, `evaluate_moves_only`,
`compare_models` — operan sobre el esquema que
`build_regression_predictions` reproduce; **sin cambios**).

**Único cambio en `_paths.py`:** añadir
`O4_L3V1_DIR = O4_OUT_DIR / "l3_v1"`. ~1 línea. (No se reutiliza el
`_CategoricalEncoder` de `train.py` porque ningún modo usa `bird_id`.)

### 6.3 Estructura de outputs

```
data/processed/
    o4/                                      # L3-v0 (existente, no se toca)
        model_*.pkl                           # XGBoost poblacional es el baseline
        predictions_test.parquet
        metrics.parquet
    o4/l3_v1/                                 # L3-v1 (nuevo)
        model_poblacional_dlat_p10.pkl        # 6 modelos poblacionales
        model_poblacional_dlat_p50.pkl
        model_poblacional_dlat_p90.pkl
        model_poblacional_dlon_p10.pkl
        model_poblacional_dlon_p50.pkl
        model_poblacional_dlon_p90.pkl
        model_individual_91916A_dlat_p10.pkl  # 6 modelos individuales
        ... (5 análogos del individual)
        predictions_test.parquet              # ambos modos, columna 'modo'
        metrics.parquet
```

Todos los `.pkl`/`.parquet` de `l3_v1/` van a `.gitignore`.

### 6.4 Dependencias añadidas a `pyproject.toml`

Ninguna. `xgboost.XGBRegressor` (3.2.0) ya está instalado por O4 base.

## 7. Riesgos identificados

| # | Riesgo | Mitigación / acción |
|---|---|---|
| R1 | **Sesgo de métrica nativa hacia L3.** La distancia nativa (punto exacto p50) parte con ventaja sobre la dist vía centroide de la clasificación (cuantizada a centroides). | Reportar SIEMPRE la **distancia vía centroide** junto a la nativa, y el **top-1 mapeado**, para comparación honesta con L1/L2. El veredicto se ancla en las métricas mapeadas (§9). |
| R2 | **Ruido de discretización en bordes de celda.** Un p50 cerca de la frontera mapea a la celda contigua y falla el top-1 aunque sea geométricamente bueno. | Reportar top-3 por proximidad y la dist vía centroide (continua). Se discute como límite del puente categórico. |
| R3 | **Quantile crossing** (`p10 > p50` o `p50 > p90`). | Ordenación post-hoc (F5) + medición de la incidencia (C5). |
| R4 | **Target concentrado en cero** (88 % estacionario) puede llevar a un regresor que predice Δ≈0 siempre (colapso a persistencia). | Esperado y honesto: la vista "moves only" (§8) revela si aporta en movimiento. El sanity check (F8) ya muestra señal en moves, especialmente per-individuo. |
| R5 | **Distorsión grado-lon vs grado-lat** en el entrenamiento. | El error se mide en haversine (km), que la corrige. Sólo el entrenamiento opera en grados (decisión sin figura, §10). No se compensa con pesos (YAGNI). |
| R6 | **Sobreajuste del modelo individual** (1458 filas train, val de sólo 162). | Mismos hiperparámetros conservadores que poblacional (min_child_weight=10, subsample/colsample 0,8) + early stopping. Si el gap train-test del individual se dispara, se reporta como limitación honesta. El sanity check mostró convergencia sana (best_iter 63–257). |
| R7 | **Doble dependencia de O3** (features + desglose por `state_b_causal`). | Aceptable: O3 validado triplemente (rework v0.3.1); el HMM causal reproduce sus parámetros. Mismo riesgo asumido en L2. |
| R8 | **Coordinación con L2/L1 en `src/tfg_aves/ml/`.** | L3 arranca tras el cierre de L2. Módulos nuevos (`quantile.py`, `build_l3.py`) no colisionan; el único toque compartido (`_paths.py`, una línea) es trivial. Si hace falta paralelismo, git worktree ([[feedback-no-amend-during-parallel-work]]). |

## 8. Validación y artefactos

### 8.1 Tests automáticos

En `tests/test_ml_quantile.py` (nuevo):

- `test_derive_displacement_target`: para posiciones conocidas,
  `y_dlat`/`y_dlon` son las diferencias exactas.
- `test_predict_quantiles_monotonic`: tras `predict_quantiles`, en TODAS
  las filas/ejes `p10 ≤ p50 ≤ p90` (verificado con un mock que produce
  cuantiles desordenados).
- `test_predict_quantiles_counts_crossings`: con un mock que cruza en N
  filas, `n_crossings_*` reporta N.
- `test_point_to_cell_known`: un punto dentro de una celda activa mapea a
  esa celda con dist vía centroide correcta; fuera del grid → NaN.
- `test_nearest_cells_orders_by_distance`: las k celdas están ordenadas
  por proximidad al punto.
- `test_pinball_loss_known`: valor analítico (asimetría q=0,9 penaliza
  más la sub-predicción).
- `test_interval_coverage`: 8 de 10 dentro de `[p10,p90]` → 0,8.
- `test_build_regression_predictions_schema`: el DataFrame producido
  tiene las columnas que `evaluate_by_state`/`evaluate_moves_only`
  esperan, y éstas corren sin error sobre él.

En `tests/test_ml_build_l3.py` (nuevo, integración):

- `test_build_o4_l3_artifacts`: produce los 12 `.pkl` (6 poblacional +
  6 individual), `predictions_test.parquet` y `metrics.parquet`.
- `test_build_o4_l3_individual_subset`: el modo individual entrena y
  evalúa SÓLO sobre filas de 91916A; el test del modo individual coincide
  exactamente con el subconjunto `bird_id==91916A` del test poblacional.
- `test_build_o4_l3_no_leakage`: ninguna feature usa información de `t+1`;
  el target sí, pero no aparece entre las columnas de entrada.
- `test_build_o4_l3_idempotent`: dos ejecuciones con el mismo `seed`
  producen idénticos artefactos.

Criterio: todos los tests existentes (146 actuales) más los nuevos deben
pasar; ruff limpio.

### 8.2 Artefactos `save_artifact` planificados

| ID | Tipo | Decisión / hallazgo documentado |
|---|---|---|
| L3-v1-D1 | figura + tabla | (a) Distribución del target `Δ` (lat/lon) mostrando concentración en cero y colas de migración; justifica `{p10,p50,p90}` y la pinball. (b) **Tabla de histórico por ave** (top-N filas candidatas) que justifica la selección de 91916A para el modo individual (F2). |
| **L3-v1-C1** | **figura + tabla** | **Calibración de la incertidumbre**: cobertura empírica de `[p10,p90]` vs 80 % nominal, por modo y por estado HMM. Valor diferencial de L3. |
| **L3-v1-C2** | **figura + tabla** | **Distribución de distancias**: L3-v1 vs persistencia vs L3-v0 (categórico), global y por estado, para el modo poblacional (ablación de target). |
| **L3-v1-C3** | **tabla** | **Comparativas centrales** (esquema §8.3): (A) target poblacional L3-v0 vs L3-v1; (B) per-individuo 91916A: individual vs poblacional vs persistencia. |
| L3-v1-C4 | figura | Mapa de **vectores de desplazamiento** predichos (p50) vs reales de **91916A** (eco de O3-C7, cartopy), con la banda `[p10,p90]` como abanico de incertidumbre en una muestra de días. Usa el modelo individual. |
| L3-v1-C5 | figura + tabla | Incidencia de **quantile crossing** antes de la corrección monótona, por eje y modo (R3). |

### 8.3 Esquema exacto de las comparativas (L3-v1-C3)

**Tabla A — ablación de target (test completo, modo poblacional).**
Baseline L3-v0 = O4 monolítico **causal poblacional** XGBoost.

| modelo | versión | top-1 map | top-3 map | dist centroide km | dist nativa km | pinball | cobertura |
|---|---|---|---|---|---|---|---|
| persistencia | baseline | 0,775 | 0,775 | 21,0 | — | — | — |
| markov(1) | baseline | 0,548 | 0,632 | 24,9 | — | — | — |
| XGB categórico | L3-v0 | 0,581 | 0,701 | 24,3 | — | — | — |
| XGB cuantil | L3-v1 | ? | ? | ? | ? | ? | ? |
| RF categórico (contexto) | O4 | 0,580 | 0,735 | 24,0 | — | — | — |

**Tabla B — hipótesis per-individuo (test de 91916A, 405 filas).**

| modelo | alcance | top-1 map | dist centroide km | dist nativa moves km | cobertura |
|---|---|---|---|---|---|
| persistencia | — | ? | ? | — | — |
| XGB cuantil | poblacional @91916A | ? | ? | ? | ? |
| XGB cuantil | individual 91916A | ? | ? | ? | ? |

> **Sin columna log-loss.** L3 no produce una distribución categórica
> sobre las 849 celdas, así que el log-loss (criterio ganador F8 de O4)
> no es computable de forma honesta. La comparativa con L1/L2 se hace en
> top-1 mapeado y distancia vía centroide. Límite metodológico declarado
> (F6, §10).

Estas tablas y su desglose por estado/moves-only son la evidencia
primaria del capítulo 6 de la memoria, sección "L3 — Regresión con
cuantiles".

## 9. Métricas de éxito y criterio de aceptación

L3 NO se juzga por log-loss (F6). Cuatro criterios independientes (los
dos primeros sobre el modo poblacional, el tercero per-individuo, el
cuarto diagnóstico):

1. **Target — distancia de migración (poblacional, vía centroide).**
   L3-v1 mejora a L3-v0 si reduce la **dist mediana en días `y_move = 1`**
   y/o en `state_b_causal = 1` en ≥ 20 km absolutos. Ganancia esperada
   honesta: −50 a −80 km en migración.
2. **Target — top-1 mapeado global (poblacional).** L3-v1 se mantiene en
   `[L3-v0 − 1 pp, L3-v0 + 3 pp]`. No se espera mejora grande (target
   concentrado en cero); basta con que **no degrade** apreciablemente.
3. **Per-individuo — individual vs poblacional sobre 91916A.** El modelo
   individual mejora si reduce la dist mediana en moves de 91916A frente
   al poblacional evaluado en las mismas 405 filas. El sanity check ya
   apunta a −6 km; el criterio se cumple si la mejora persiste con el set
   completo de features y es ≥ 0 (no empeora) en top-1 mapeado.
4. **Diagnóstica — calibración (C1).** La cobertura empírica de
   `[p10,p90]` cae en `[0,70, 0,90]` (idealmente ≈ 0,80) en ambos modos.
   El sanity check ya la sitúa en 0,76–0,85.

Interpretación honesta del resultado para la memoria
([[feedback-memoria-tone]]):

- **1 o 3 cumplidos** → L3 aporta: "la regresión continua / el modelo
  dedicado mejora la precisión geométrica en los días de movimiento real,
  donde el target categórico era más grosero".
- **1/3 no cumplidos pero 4 sí** → "la regresión no mejora la distancia
  pero entrega incertidumbre calibrada, contribución interpretativa que
  el clasificador no ofrece".
- **3 cumplido y 1 no** → hallazgo sobre la decisión abierta del TFG:
  "un modelo dedicado al individuo con más datos supera al global en ese
  individuo, pero el enfoque per-individuo no es generalizable a las 82
  aves (la mayoría no tiene histórico suficiente)".
- **Ninguno / colapso a persistencia (R4)** → hallazgo estructural:
  "reformular el target y dedicar el modelo a un individuo no rompe el
  techo; el cuello de botella es D1 (self-loops) y el horizonte de un día
  (Mo2). Refuerza el trabajo futuro de modelos de secuencia".

Cualquier resultado es válido para el TFG. La aportación de L3 a la
memoria es la **diversidad metodológica** (clasificación categórica vs
arquitectura por etapas vs regresión continua), el cierre de la
**decisión abierta global vs per-individuo**, y la **honestidad
geométrica + incertidumbre**, no necesariamente un salto de rendimiento.

## 10. Decisiones metodológicas sin figura (van a la memoria como "decisiones documentadas en el spec")

- **Por qué `(Δlat, Δlon)` cartesianas y no rumbo+distancia.** El rumbo
  tiene una discontinuidad en ±180° que rompe la pérdida basada en el
  ángulo; las cartesianas son continuas y aditivas (F1).
- **Por qué se entrena en grados pese a que un grado de lon ≠ lat en km.**
  El error se evalúa en haversine, que corrige la distorsión; compensar
  con pesos por latitud sería sobreingeniería (R5, F1).
- **Por qué se descarta el modo personalizado y se añade el individual.**
  `bird_id` aportaba +1 pp y dejó de memorizar tras quitar la fuga; un
  modelo per-individuo dedicado al ave con más datos prueba mejor la
  decisión abierta global vs per-individuo del TFG (F2).
- **Por qué 91916A.** Es el ave con más filas candidatas (2025, rank 1
  verificado; la siguiente tiene 1430), única con histórico claramente
  suficiente para 6 regresores con early stopping. La tabla de histórico
  por ave es la justificación (artefacto D1).
- **Por qué el HMM no se reentrena por ave.** Para que la única variable
  del contraste per-individuo sea el alcance del regresor supervisado, no
  el del HMM (F2).
- **Por qué sólo XGBoost.** RF de sklearn no da cuantiles nativos;
  LightGBM fue descartado en O4. L3 prueba target y alcance, no familia
  (F3).
- **Por qué tres cuantiles y no cinco.** `{p10,p50,p90}` dan punto +
  intervalo al 80 % interpretable; p05/p95 duplicarían modelos para una
  ganancia marginal (F4, YAGNI).
- **Por qué hiperparámetros idénticos entre modos y sin tuning.** Directiva
  del autor: básico que funcione. La config §8.6 valida empíricamente
  (cobertura ≈ 80 %, convergencia sana); mantenerla idéntica entre modos
  aísla el alcance como única variable del contraste per-individuo (F8).
- **Por qué se descarta el log-loss derivado por gaussiana.** Supuesto
  frágil (gaussianidad bivariada, independencia lat/lon) de sesgo no
  neutral; complejidad que L3 no necesita para su tesis geométrica (F6).
- **Por qué se reporta distancia nativa Y vía centroide.** La nativa es la
  métrica natural de L3 pero parte con ventaja; la vía centroide es la
  comparación honesta con L1/L2 (R1, §9).
- **Por qué monolítico y no moves-only.** Mantiene la ablación
  "categórico vs continuo sobre el mismo dataset" y evita reintroducir un
  `clf_move` (contribución de L2). La vista moves-only se evalúa igual,
  sin entrenar modelo aparte (§1.1).

## 11. Entregables al cerrar L3

Al cerrar L3-v1 (tag esperado `v0.4.4-o4l3-regresion`), el repositorio
debe contener:

- `src/tfg_aves/ml/quantile.py` y `src/tfg_aves/ml/build_l3.py`
  implementados y commiteados.
- Línea `O4_L3V1_DIR` añadida a `src/tfg_aves/ml/_paths.py`.
- Tests nuevos en `tests/test_ml_quantile.py` y
  `tests/test_ml_build_l3.py`. Todos pasando; ruff limpio.
- Notebook `notebooks/04l3_eda_o4l3.py` (jupytext percent) que regenera
  figuras y tablas vía `save_artifact`.
- 6 artefactos `L3-v1-{D1, C1..C5}` en `reports/figures/`,
  `reports/tables/`, `reports/captions/` y `reports/INDEX.md`.
- Notas de memoria en `reports/memoria/06_o4_ml.md` (sección "L3 —
  Regresión con cuantiles").
- Entrada en `reports/ai-log/` (`00NN-o4l3-regresion.md`) según política
  de scope.
- Commit(s) en castellano + tag `v0.4.4-o4l3-regresion`.

## 12. Trabajo futuro (fuera del scope de L3)

- **L3 × L1**: regresor de cuantiles sobre el feature set ampliado con
  viento. Sólo si ambos cierres lo justifican.
- **L3 × L2**: regresor de cuantiles condicionado por `clf_move`. Combina
  target continuo + arquitectura por etapas.
- **Modelos individuales para más aves**: extender el modo individual a
  las aves con suficiente histórico (top-5..10) y comparar la curva
  "datos por ave vs ganancia per-individuo". Fuera de scope por coste; se
  apunta como continuación natural de la decisión global vs per-individuo.
- **Regresión multi-salida con covarianza lat/lon** (no sólo cuantiles
  marginales por eje). Mayor fidelidad geométrica a costa de complejidad;
  YAGNI para el TFG.
- **Pérdida en km (haversine diferenciable)** en lugar de pinball en
  grados: corregiría la distorsión grado-longitud en el entrenamiento.

## 13. Conexiones con el resto del TFG

- **O3 (HMM)**: L3 consume `state_b_causal` y
  `posterior_b_migracion_causal` (HMM global) como features y desglosa
  las métricas por `state_b_causal`. Ancla en [[project-o3-outcome]].
- **O4 causal**: L3 reutiliza features (las 10 causales), ensamblaje,
  split, base de hiperparámetros y la maquinaria de `evaluate.py`. El
  baseline L3-v0 es el XGBoost poblacional de `build_o4()`.
- **Decisión abierta global vs per-individuo** (`CLAUDE.md`, desde O3):
  el modo individual 91916A la cierra empíricamente sobre el mejor
  candidato posible.
- **L1 y L2**: las tres líneas son ortogonales por diseño (D4, D1,
  Mo1+D3). Las combinaciones quedan como trabajo futuro condicional
  (§12). L3 NO hereda la arquitectura de dos etapas de L2 (decisión del
  autor: monolítico para aislar el target).
- **O5 (mapas y evaluación del error)**: L3 aporta a O5 una **banda de
  incertidumbre geográfica** (`[p10,p90]` por eje) dibujable como
  abanico/elipse sobre el mapa, complementando los heatmaps categóricos
  de L1/L2. Si la calibración (C1) es buena, candidato fuerte para la
  capa de incertidumbre de la UI interactiva.

---

Conecta con [[project-o4-improvement-lines]] (L3 es la tercera de las
tres líneas), [[project-o4-outcome]] (estado de cierre de O4 base, base
de comparación), [[project-o4-causal-rework]] y
[[feedback-causal-features-no-leakage]] (feature set causal obligatorio),
[[feedback-memoria-tone]] (tono investigador al narrar un resultado
probablemente modesto), [[feedback-decisions-must-be-justified]] (cada
decisión sin figura va a §10 y a la memoria; la selección de 91916A se
justifica con la tabla de histórico),
[[feedback-methodological-justifications]] (los riesgos de §7 van a la
memoria como decisiones razonadas) y [[feedback-keep-solutions-simple]]
(descarte del log-loss gaussiano, de los pesos por latitud y del tuning).
