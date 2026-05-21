# Diseño de O3 — Detección de comportamiento con HMM

- **Fecha:** 2026-05-22
- **Objetivo del TFG:** O3 (50 h) — detección de comportamiento
  (estacionario vs migración) con HMM
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado, pendiente de implementación

## 1. Resumen

O3 entrena **dos HMMs globales gaussianos de 2 estados** sobre la
población de 82 *Larus fuscus*, diseñados como ablación comparativa
para detectar el régimen comportamental de cada (ave, día) en
`{estacionario, migración}`:

- **Modelo A**: observaciones puramente cinemáticas —
  `log_displacement_km`, `abs_turning_angle_rad`. Refleja la postura
  metodológica "el HMM detecta comportamiento a partir del movimiento;
  el contexto temporal y ambiental se reserva para validación externa".
- **Modelo B**: añade contexto ambiental y temporal a las features
  cinemáticas — `veg_low`, `veg_high`, `daylight_hours`. Refleja la
  recomendación de la tutora "incluir todo el contexto disponible
  como observación".

El propósito de mantener ambos modelos es **comparar empíricamente** qué
configuración produce estados más coherentes con la biología conocida
de la especie, defendiendo la elección final con evidencia en lugar de
con preferencia metodológica.

Ambos HMMs se entrenan con `hmmlearn 0.3.3` (`GaussianHMM`,
`covariance_type='diag'`, 10 restarts con inicialización k-means).
Lee `data/processed/daily.parquet` (de O1) y la columna
`source_event_id` para anexar `veg_low`/`veg_high` del CSV crudo
(`data/raw/migration_original.csv`). Escribe en `data/processed/o3/`
un parquet enriquecido `features.parquet` con una fila por (ave, día)
que contiene features crudos, estados Viterbi de A y B, y
probabilidades posteriores — diseñado para que O4 elija libremente qué
columnas consumir como features de su modelo supervisado.

La evaluación combina tres criterios complementarios: **log-likelihood
en holdout** (20 % de aves apartadas, estratificadas por días válidos),
**coherencia biológica** de los estados con la fenología conocida de
*Larus fuscus* (migración concentrada en mar-may y ago-oct), y
**acuerdo entre A y B** sobre el conjunto de (ave, día) válidos, con
análisis cualitativo de los desacuerdos.

## 2. Contexto y constraints

- **Entregable de O1 ya disponible:** `daily.parquet` con una fila por
  `(bird_id, date_utc)` para 82 aves entre 2009-05-25 y 2015-08-27.
  Posiciones a 08:00 UTC ± 60 min, huecos explícitos
  (`is_valid=False, lat=NaN, lon=NaN`), columna `source_event_id` que
  permite re-localizar la fila correspondiente en el CSV crudo.
- **Aprendizaje crítico de O2 (commit `00061f1`):** un modelo global
  evaluado por LOBO castiga sistemáticamente porque cada ave tiene
  rutas individuales. En O3, donde las features son **cinemáticas
  (movimiento) y no de localización**, ese problema es menor, pero se
  evita igualmente sustituyendo LOBO por un holdout 80/20 estratificado
  (ver decisión 8.8).
- **Columnas del CSV crudo necesarias para Modelo B:**
  `ECMWF Interim Full Daily Invariant Low Vegetation Cover` y
  `ECMWF Interim Full Daily Invariant High Vegetation Cover`. Fueron
  descartadas explícitamente en O1; se re-anexan en O3 mediante un
  join por `event_id`.
- **Política de convenciones del proyecto** (CLAUDE.md): toda
  decisión no trivial requiere figura/tabla + caption en castellano
  vía `save_artifact()`; commits en castellano sin trailer de IA;
  código con identificadores en inglés; tests con pytest sobre
  datasets sintéticos. **Específico para O3**: las decisiones
  metodológicas sin figura justificativa se recogen en la sección 8
  con razonamiento completo, ya que la memoria final las citará
  literalmente.
- **Patrón heredado de O1/O2:** módulos puros + orquestador
  `build_oN` + notebook EDA jupytext + tests. Validado por el autor
  tras O1/O2.

## 3. Decisiones de diseño fijadas

Decididas en el brainstorming previo, sin EDA:

| # | Decisión | Valor |
|---|---|---|
| F1 | Estados ocultos | **2 estados: estacionario y migración** |
| F2 | Estrategia comparativa | **Dos modelos como ablación**: A (cinemático) y B (cinemático + contexto) |
| F3 | Alcance del modelo | **Global**: un HMM por modelo, las 82 aves comparten parámetros. Viterbi se ejecuta por ave para asignar estados |
| F4 | Tipo de HMM | **`GaussianHMM`** de `hmmlearn 0.3.3`, `covariance_type='diag'` |
| F5 | Inicialización + restarts | **K-means con `k=2`** para inicializar `means_init` y varianza diagonal; **10 restarts** con seeds distintos, se retiene el modelo con mayor log-likelihood en train |
| F6 | Estandarización | **`StandardScaler` ajustado en train, aplicado a holdout** con las mismas medias y desviaciones |
| F7 | Split train/holdout | **80/20 por ave**, estratificado por número de días válidos por ave; seed fijo (`random_state=0`) |
| F8 | Validación | **Triángulo**: log-likelihood en holdout + coherencia biológica (mes, latitud, fotoperiodo, vegetación) + acuerdo A-B con análisis de desacuerdos |
| F9 | Features Modelo A | `log_displacement_km` y `abs_turning_angle_rad` (2 features) |
| F10 | Features Modelo B | features de A + `veg_low` + `veg_high` + `daylight_hours` (5 features) |
| F11 | Tratamiento de huecos | Sólo se calcula observación para días con **3 días consecutivos válidos** (t-1, t, t+1). Tramos consecutivos se pasan a `hmmlearn` vía `lengths` |
| F12 | Re-etiquetado de estados | Convención post-hoc: el estado con menor `μ[log_displacement_km]` recibe etiqueta `estacionario` (0); el otro `migración` (1) |
| F13 | Entregable principal | **Parquet enriquecido completo**: una fila por (bird_id, date_utc) con features crudos + estado Viterbi de A y B + posteriores de A y B + flag de validez |

Decisiones que se materializan en `reports/INDEX.md` con
`decision=...` y se resuelven en el EDA con figura:

| # | Parámetro | Slug del artefacto justificativo |
|---|---|---|
| D1 | `n_components=2` (número de estados ocultos) | `nstates-aic-bic-sweep` |

Las restantes decisiones metodológicas (covarianza diagonal, número
exacto de restarts, no combinar vegetación, etc.) **no** generan
artefacto sino que se documentan textualmente en la sección 8.

## 4. Arquitectura

Pipeline en cuatro fases ejecutadas desde el notebook
`notebooks/03_eda_o3.py` que importa funciones de
`src/tfg_aves/hmm/`:

```
data/processed/daily.parquet         data/raw/migration_original.csv
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                  (1) features.py    — calcula log_dist, turning_angle,
                             │           veg_low, veg_high, daylight_hours
                             ▼
                  (2) fit.py         — HMM A (2 features) + HMM B (5 features)
                             │           k-means init + EM, 10 restarts
                             ▼
                  (3) evaluate.py    — held-out LL + coherencia biológica
                             │           + acuerdo A-B
                             ▼
                  (4) build.py       — orquestar y materializar
                             │
                             ▼
data/processed/o3/
  features.parquet           — entregable principal (filas (bird, day))
  models_a_b.pkl             — HMMs entrenados (joblib pickle)
  metrics.parquet            — log-likelihood + acuerdo + métricas biológicas
```

Tras O3, regenerar los outputs se hace con:

```python
from tfg_aves.hmm import build_o3
build_o3(
    holdout_frac=0.20,
    n_restarts=10,
    random_state=0,
)
```

Las funciones de los submódulos son puras (entrada DataFrame/array →
salida DataFrame/array). Sólo `build.py` escribe a disco.

## 5. Módulos

### 5.0 Constantes del paquete

`src/tfg_aves/hmm/_paths.py` expone:

```python
DAILY_PARQUET = ROOT / "data" / "processed" / "daily.parquet"
RAW_CSV       = ROOT / "data" / "raw"       / "migration_original.csv"
O3_OUT_DIR    = ROOT / "data" / "processed" / "o3"
```

Las funciones aceptan estas rutas como argumentos para mantenerse
parametrizables (tests las sobrescriben con `tmp_path`).

### 5.1 `src/tfg_aves/hmm/features.py`

```python
def bearing_rad(lat1, lon1, lat2, lon2) -> float | np.ndarray
def daylight_hours(lat: float, day_of_year: int) -> float
def load_vegetation_from_raw(raw_csv: Path, event_ids: pd.Series) -> pd.DataFrame
def compute_observation_features(
    df_daily: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
) -> pd.DataFrame
```

- `bearing_rad`: ángulo inicial (rumbo) del trayecto recto de `(lat1,
  lon1)` a `(lat2, lon2)`, en radianes `[-π, π]`. Vectorizado.
- `daylight_hours`: horas de luz al mediodía local, dada latitud y
  día del año (1-366). Usa la fórmula astronómica
  `acos(-tan(lat) tan(δ)) · 24 / π` con declinación solar aproximada
  `δ = 0.4093 sin(2π (J − 81) / 365)`. Clip a `[0, 24]` para latitudes
  polares.
- `load_vegetation_from_raw`: lee `migration_original.csv`, selecciona
  las columnas `ECMWF Interim Full Daily Invariant Low Vegetation
  Cover` y `... High Vegetation Cover`, indexa por `event-id` y
  devuelve un DataFrame con `event_id, veg_low, veg_high` filtrado a
  los `event_ids` solicitados.
- `compute_observation_features`: el corazón del módulo. Para cada
  (bird_id, date_utc) en `df_daily`, calcula `log_displacement_km`
  (de t a t+1), `abs_turning_angle_rad` (entre rumbos de
  t-1→t y t→t+1), `daylight_hours` (al mediodía local de t).
  Anexa `veg_low` y `veg_high` mediante join por
  `source_event_id`. Marca con `is_observation_valid=False` los días
  sin triplete consecutivo. Devuelve DataFrame con esquema 6.1.

### 5.2 `src/tfg_aves/hmm/fit.py`

```python
def stratified_holdout_split(
    df_features: pd.DataFrame, holdout_frac: float, random_state: int
) -> tuple[list[str], list[str]]    # (train_bird_ids, holdout_bird_ids)

def build_sequences(
    df_features: pd.DataFrame, bird_ids: list[str], feature_cols: list[str]
) -> tuple[np.ndarray, list[int]]    # (X, lengths)

def fit_hmm_with_restarts(
    X_train: np.ndarray,
    lengths_train: list[int],
    n_components: int = 2,
    n_restarts: int = 10,
    random_state: int = 0,
) -> tuple[GaussianHMM, StandardScaler, float, list[float]]
    # (best_model, fitted_scaler, best_ll, all_lls)

def relabel_states(
    hmm: GaussianHMM, feature_cols: list[str]
) -> dict[int, str]    # mapping {0: "estacionario", 1: "migración"} en el orden del HMM
```

- `stratified_holdout_split`: distribuye las aves en train (80 %) y
  holdout (20 %) usando estratificación por número de días válidos
  (quintiles). Garantiza que aves "ricas" y "pobres" estén
  representadas en ambos conjuntos.
- `build_sequences`: dado un subconjunto de aves y la lista de
  columnas de features, construye la matriz `X` concatenando todos
  los tramos consecutivos válidos de cada ave y el vector `lengths`
  que `hmmlearn` necesita para tratar cada tramo como secuencia
  independiente.
- `fit_hmm_with_restarts`: el corazón del entrenamiento. Internamente:
  (a) ajusta un `StandardScaler` sobre `X_train` y lo aplica a la
  misma matriz (transforma in-place a versión estandarizada); (b)
  ejecuta `n_restarts` veces el ciclo k-means + EM con seeds
  distintos sobre `X_train` estandarizado, inicializando cada HMM con
  `means_init` de k-means y `covars_init` diagonal con la varianza
  intra-cluster; (c) retorna el modelo con la mayor log-likelihood en
  train **junto con el scaler ajustado** (necesario para escalar el
  holdout y la inferencia posterior). El scaler se serializa en
  `models_a_b.pkl` junto al modelo.
- `relabel_states`: aplica la convención de F12 sobre las medias
  desescaladas. Útil para mapear índices internos del HMM a
  etiquetas semánticas.

### 5.3 `src/tfg_aves/hmm/evaluate.py`

```python
def log_likelihood_per_obs(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    X_raw: np.ndarray,
    lengths: list[int],
) -> float

def viterbi_per_bird(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    label_map: dict[int, str],
) -> pd.DataFrame    # añade state_<m> y posteriores

def biological_coherence_table(
    df_features: pd.DataFrame, state_col: str
) -> pd.DataFrame    # estado × mes, estado × latitud_quartile, etc.

def ab_agreement(df_features: pd.DataFrame) -> dict
```

- `log_likelihood_per_obs`: aplica el `scaler` a `X_raw`, calcula
  `hmm.score(X_scaled, lengths) / len(X_raw)`. El scaler es el
  ajustado en train (no se re-ajusta sobre holdout — ver 8.2).
- `viterbi_per_bird`: para cada ave en `df_features`, extrae los
  tramos consecutivos válidos, aplica el `scaler` ajustado en train,
  ejecuta Viterbi (`hmm.predict`) y `predict_proba`, mapea los
  estados a etiquetas semánticas usando `label_map` y añade las
  columnas `state_<m>`, `posterior_<m>_estacionario`,
  `posterior_<m>_migracion` al DataFrame. Filas con
  `is_observation_valid=False` reciben `NaN`.
- `biological_coherence_table`: tabula la distribución del estado
  asignado vs variables externas (mes, latitud, fotoperiodo,
  vegetación). Sirve de input para el artefacto C3.
- `ab_agreement`: calcula la matriz de confusión 2×2 entre `state_a`
  y `state_b`, devuelve dict con `pct_agreement`,
  `pct_b_adds_migration`, `pct_b_adds_stationary` y un sub-DataFrame
  con los desacuerdos para análisis.

### 5.4 `src/tfg_aves/hmm/build.py`

```python
@dataclass
class BuildO3Result:
    features_path: Path
    models_path: Path
    metrics_path: Path
    n_birds_train: int
    n_birds_holdout: int
    n_observations: int
    ll_per_obs_a: float
    ll_per_obs_b: float
    pct_agreement_ab: float

def build_o3(
    *,
    holdout_frac: float = 0.20,
    n_restarts: int = 10,
    random_state: int = 0,
    daily_path: Path = DAILY_PARQUET,
    raw_csv: Path = RAW_CSV,
    out_dir: Path = O3_OUT_DIR,
) -> BuildO3Result
```

- Compone `compute_observation_features → stratified_holdout_split →
  build_sequences (train) → fit_hmm_with_restarts (A) →
  fit_hmm_with_restarts (B) → viterbi_per_bird (A, B) →
  log_likelihood_per_obs (en holdout) → ab_agreement → escribir`.
- Escribe los 3 ficheros de `data/processed/o3/`.
- Devuelve dataclass con paths + métricas agregadas.

## 6. Esquema de datos

### 6.1 `data/processed/o3/features.parquet` (entregable principal)

| Columna | Tipo | Nullable | Descripción |
|---|---|---|---|
| `bird_id` | string | no | identificador del individuo |
| `date_utc` | date32 | no | día calendario UTC |
| `lat`, `lon` | float64 | sí | posición a 08:00 UTC del día *t* (de daily.parquet) |
| `log_displacement_km` | float64 | sí | `log1p(haversine_km(t, t+1))` |
| `abs_turning_angle_rad` | float64 | sí | `\|bearing(t,t+1) − bearing(t-1,t)\|` normalizado a `[0, π]` |
| `daylight_hours` | float64 | sí | horas de luz al mediodía local del día *t* |
| `veg_low` | float64 | sí | cobertura ECMWF de vegetación baja en el fix de referencia |
| `veg_high` | float64 | sí | cobertura ECMWF de vegetación alta en el fix de referencia |
| `state_a` | int8 | sí | estado Viterbi del Modelo A (0=estacionario, 1=migración) |
| `state_b` | int8 | sí | estado Viterbi del Modelo B (0=estacionario, 1=migración) |
| `posterior_a_estacionario` | float64 | sí | `P(state_a=0 \| obs, modelo A)` |
| `posterior_a_migracion` | float64 | sí | `P(state_a=1 \| obs, modelo A)` |
| `posterior_b_estacionario` | float64 | sí | análogo modelo B |
| `posterior_b_migracion` | float64 | sí | análogo modelo B |
| `is_observation_valid` | bool | no | `True` si hay 3 días consecutivos válidos en t-1, t, t+1 |
| `in_holdout` | bool | no | `True` si el `bird_id` está en el 20 % apartado |

Filas con `is_observation_valid=False` tienen features, estados y
posteriores a `NaN` (siguen apareciendo en el parquet para que O4
sepa qué días no tienen estado HMM disponible).

### 6.2 `data/processed/o3/models_a_b.pkl`

Diccionario serializado con `joblib`:

```python
{
    "model_a": <GaussianHMM>,
    "model_b": <GaussianHMM>,
    "scaler_a": <StandardScaler>,
    "scaler_b": <StandardScaler>,
    "feature_cols_a": ["log_displacement_km", "abs_turning_angle_rad"],
    "feature_cols_b": ["log_displacement_km", "abs_turning_angle_rad",
                       "veg_low", "veg_high", "daylight_hours"],
    "label_map_a": {0: "estacionario", 1: "migración"},
    "label_map_b": {0: "estacionario", 1: "migración"},
    "train_bird_ids": [...],
    "holdout_bird_ids": [...],
    "random_state": 0,
}
```

### 6.3 `data/processed/o3/metrics.parquet`

Una fila por (modelo, scope, métrica) en formato long:

| Columna | Tipo | Descripción |
|---|---|---|
| `model` | string | `"a"` o `"b"` |
| `scope` | string | `"train"`, `"holdout"`, `"both"` |
| `metric` | string | nombre de la métrica |
| `value` | float64 | valor |

Métricas incluidas: `ll_per_obs`, `pct_state_estacionario`,
`pct_state_migracion`, `n_observations_used`. Para A-B: `pct_agreement`,
`pct_b_adds_migration`, `pct_b_adds_stationary` con `scope="both"` y
`model="agreement"`.

## 7. Decisiones justificadas (artefactos)

Convención `save_artifact()` con `objective="o3"`, captions en
castellano.

### 7.1 Decisión con figura justificativa

| Slug | Decisión | Forma del artefacto |
|---|---|---|
| `nstates-aic-bic-sweep` (D1) | `n_components` fijado en 2 | Barras de AIC y BIC para HMMs entrenados con `n_components ∈ {2, 3, 4}` sobre las features de Modelo A en el conjunto de entrenamiento. Caption explica que se mantiene n=2 por (a) alineación con el proposal (estacionario + migración), (b) interpretabilidad y (c) que AIC/BIC no muestran mejora sustancial al añadir estados, o si la muestran, los estados extra no admiten etiquetado biológico claro. La figura **respalda** la decisión a priori, no la **decide** |

### 7.2 Caracterización (sin asociar a decisión)

| Slug | Contenido |
|---|---|
| `features-by-state-a` (C1) | Histograma o boxplot de cada feature de Modelo A condicionado al estado Viterbi (estacionario / migración). Visualmente: el estado *estacionario* concentra masa en `log_displacement_km` bajo y `abs_turning_angle_rad` alto/aleatorio; el *migración* lo contrario. Caption interpreta la separación detectada |
| `features-by-state-b` (C2) | Análogo para Modelo B: histograma de las 5 features condicionado al estado. Permite ver cómo `veg_low`, `veg_high` y `daylight_hours` enriquecen (o no) la separación que A ya tenía con sólo movimiento |
| `state-vs-biology` (C3) | Coherencia biológica de los estados. 3 paneles por modelo (Modelo A arriba, Modelo B abajo): (a) distribución del estado *migración* por mes; (b) por latitud media diaria; (c) — sólo Modelo A — por fotoperiodo. **Artefacto principal para defender la elección de features ante la tutora** |
| `ab-agreement` (C4) | Matriz de confusión 2×2 entre `state_a` y `state_b`, más dos histogramas sobre los desacuerdos: features marginales en los días "A dice estacionario, B dice migración" y viceversa. Caption interpreta dónde difieren los modelos |
| `per-bird-state-proportions` (C5) | Para cada ave, proporción de días asignados a cada estado por cada modelo (boxplot por modelo + scatter por ave). Conecta con el hallazgo de O2 sobre heterogeneidad individual y prepara terreno para O5 |

**Total esperado en `INDEX.md`:** 1 decisión + 5 caracterizaciones = 6
entradas O3.

### 7.3 Decisiones triviales sin figura

Mencionadas en notas de memoria pero sin artefacto propio:

- `random_state=0` (reproducibilidad).
- `tol=1e-4`, `n_iter=200` (defaults razonables, sin barrido).
- Holdout 80/20 (proporción estándar; menor daría holdout demasiado
  pequeño para una validación robusta).

## 8. Decisiones metodológicas justificadas (sin figura)

Esta sección recoge las decisiones de diseño **sin figura** que
requieren razonamiento explícito para defender en la memoria. Cada
entrada tiene formato uniforme: **Decisión / Por qué / Alternativa
descartada / Implicación para la memoria**.

### Bloque A — Sobre las features

**8.1 No combinar `veg_low` y `veg_high`**

- **Decisión**: usar ambas columnas como features separadas del Modelo
  B (cinco features en total, no cuatro).
- **Por qué**: `veg_low` (cobertura de vegetación baja: pastos,
  cultivos, matorral) y `veg_high` (vegetación alta: bosques)
  representan biomas físicamente distintos. Una celda con `(0.8,
  0.0)` (estepa) y una con `(0.0, 0.8)` (bosque cerrado) tienen
  biomas diferentes pero misma suma. Combinarlas por suma, ratio o
  promedio impondría una asunción sobre cómo se relacionan los dos
  estratos en el comportamiento del ave; esa asunción no la respaldan
  los datos. Dejarlas separadas permite que la gaussiana bivariada
  de cada estado del HMM descubra qué combinación importa via su
  matriz de covarianza diagonal.
- **Alternativa descartada**: combinarlas como `veg_low + veg_high`.
  Conllevaría perder información biológica y colapsar tres biomas
  distintos (suelo desnudo, prado, bosque) en valores numéricos no
  distinguibles.
- **Implicación para la memoria**: justifica que el espacio de
  features de B sea 5-dimensional. Se menciona en el apartado de
  preparación de datos del capítulo 5.

**8.2 Estandarización con `StandardScaler` ajustada en train y aplicada a holdout**

- **Decisión**: aplicar `StandardScaler` (resta media, divide por
  desviación típica) a las features antes del HMM. El scaler se
  ajusta exclusivamente sobre el conjunto de entrenamiento y se
  aplica con las mismas estadísticas al holdout.
- **Por qué**: la emisión gaussiana del HMM es sensible a la escala
  de las features. `daylight_hours` (rango 8-18) tendría peso
  desproporcionado sobre `log_displacement_km` (rango 0-6) si no
  estandarizamos: la inicialización por k-means agruparía puntos por
  horas de luz en lugar de por comportamiento, y la convergencia de
  EM sería más lenta y menos estable. Estandarizando, todas las
  features contribuyen por igual a la geometría del espacio de
  observaciones.
- **Alternativa descartada**: no escalar. Forzaría al HMM a "absorber"
  la diferencia de escala via la varianza diagonal aprendida, pero
  el sesgo en la inicialización (k-means se ejecuta antes de EM)
  sería difícil de corregir.
- **Por qué fit en train y no en train+holdout combinados**: si el
  scaler usa estadísticas del holdout, hay *data leakage*:
  información del conjunto de evaluación contamina el preprocesamiento.
  La métrica de log-likelihood en holdout dejaría de ser honesta.
- **Implicación para la memoria**: mencionar como un paso estándar
  de higiene de pipeline, con una frase justificativa sobre la
  escala desigual de features.

### Bloque B — Sobre el modelo y su entrenamiento

**8.3 Matriz de covarianza diagonal**

- **Decisión**: `covariance_type='diag'` en `GaussianHMM`. Cada
  estado tiene su propia varianza por feature, sin modelar
  correlaciones entre features dentro de un estado.
- **Por qué**: (1) Es la práctica estándar en HMMs para análisis de
  movimiento animal (Patterson et al. 2017; biblioteca `moveHMM` en
  R). (2) Las features fueron diseñadas para ser semánticamente
  ortogonales (magnitud, direccionalidad, contexto temporal,
  contexto ambiental); las correlaciones residuales dentro de un
  estado son débiles. (3) Robustez estadística: en Modelo B la
  diferencia entre `'diag'` (10 parámetros de covarianza por estado)
  y `'full'` (15) parece pequeña pero `'full'` introduce 10
  correlaciones extra que pueden ser ruido más que señal. (4)
  Interpretabilidad: una varianza por feature es directamente
  legible; una matriz completa requiere análisis de eigenvalores
  para interpretar.
- **Alternativa descartada**: `'full'`. Daría más flexibilidad pero
  peor estabilidad numérica y un modelo más difícil de explicar en
  la memoria.
- **Implicación para la memoria**: una frase sobre la elección
  estándar en literatura. Citar Patterson et al.

**8.4 Inicialización con k-means**

- **Decisión**: antes del EM, ejecutar k-means con `k=2` sobre las
  features estandarizadas. Usar los centroides resultantes como
  medias iniciales (`means_init`) y la varianza intra-cluster como
  varianza inicial.
- **Por qué**: EM converge a un óptimo local de la log-likelihood.
  Sin una inicialización razonable (con medias aleatorias uniformes
  sobre el espacio de features), EM puede caer en soluciones
  degeneradas (un estado modela todo, el otro queda vacío) o en
  óptimos muy subóptimos. K-means es un pre-clustering rápido y
  robusto que da a EM un punto de partida que ya separa los datos
  en dos grupos sensibles.
- **Alternativa descartada**: inicialización aleatoria. Requeriría
  muchos más restarts (20-50) para encontrar el óptimo con
  confianza, y los restarts serían más variables.
- **Implicación para la memoria**: una frase mencionando que se usa
  k-means como semilla.

**8.5 Diez restarts de EM**

- **Decisión**: repetir el procedimiento de entrenamiento (k-means
  init + EM) 10 veces con seeds distintos en el k-means, y reportar
  el modelo con mayor log-likelihood en train.
- **Por qué**: incluso con k-means init, EM puede converger a
  óptimos locales ligeramente distintos según la inicialización.
  Repetir múltiples veces y quedarse con el mejor es la práctica
  estándar en HMMs. Diez es un número defensivo (suficiente margen
  para descartar trayectorias patológicas) y computacionalmente
  asequible (~2 min total para ambos modelos sobre 21 000
  observaciones).
- **Alternativa considerada**: 5 restarts. Estadísticamente
  suficiente para este dataset pero con menos margen. La diferencia
  de tiempo entre 5 y 10 (~1 min vs ~2 min) no es relevante;
  preferimos 10 por ser más defensivo ante el tribunal.
- **Implicación para la memoria**: mencionar el número de restarts y
  citar la práctica estándar.

**8.6 Re-etiquetado de estados con la convención "menor `log_displacement` = estacionario"**

- **Decisión**: después de entrenar, los estados `0` y `1` se
  reasignan a `estacionario` y `migración` según una regla
  determinista: el estado con menor media en `log_displacement_km`
  recibe la etiqueta `estacionario`. Aplicado por igual en Modelo A
  y Modelo B.
- **Por qué**: `hmmlearn` no sabe qué significan los estados (no es
  supervisado). Los entrena como `0` y `1` arbitrariamente y el
  orden depende del seed. Sin re-etiquetado, las columnas `state_a`
  y `state_b` del entregable serían inconsistentes entre runs. La
  regla "menor desplazamiento = estacionario" es biológicamente
  correcta por definición y **independiente del estado del modelo**
  (no se hace look-ahead).
- **Alternativa descartada**: re-etiquetar manualmente tras
  inspección visual. Frágil, irreproducible, sesgo del autor.
- **Implicación para la memoria**: una frase explicando la
  convención.

**8.7 Número de estados fijado en 2 a priori (respaldado por D1)**

- **Decisión**: `n_components=2` tanto en Modelo A como en B.
- **Por qué**: el proposal del TFG especifica detección binaria
  (estacionario vs migración). N=2 alinea el modelo con la pregunta
  de investigación y maximiza interpretabilidad.
- **Respaldo con datos**: el artefacto D1 (`nstates-aic-bic-sweep`)
  muestra que añadir un tercer o cuarto estado no produce una mejora
  sustancial de AIC/BIC sobre Modelo A, lo que confirma que 2 es
  una elección razonable y no una restricción artificial.
- **Alternativa descartada**: dejar que AIC/BIC elijan N entre 2-6.
  Daría un modelo posiblemente más predictivo pero menos
  interpretable; los estados extra serían de difícil etiquetado
  biológico y descalibrarían la narrativa del proposal.
- **Implicación para la memoria**: presentar D1 como respaldo de la
  elección. La decisión es a priori por interpretabilidad y la
  figura **valida** que es razonable, no es lo que la fija.

### Bloque C — Sobre la evaluación

**8.8 Holdout 80/20 estratificado por días válidos, en lugar de LOBO**

- **Decisión**: el conjunto de evaluación es un 20 % de aves
  apartadas, no un esquema leave-one-bird-out (82 folds). El split
  se estratifica por el número total de días válidos por ave para
  que el holdout contenga aves de tracking rico y pobre en
  proporciones similares al train.
- **Por qué**: (1) **Aprendizaje de O2**: LOBO sobre un modelo
  global castiga sistemáticamente porque cada ave tiene su propia
  distribución espacial de rutas, lo que hace que el 43 % de las
  predicciones LOBO en O2 cayeran en celdas no observadas en
  training. Para O3, donde el HMM opera sobre features de movimiento
  (no localización), ese problema es menor pero **no nulo**: aves
  con patrones de movimiento idiosincráticos podrían sesgar la
  evaluación LOBO. (2) **Costo**: LOBO en O3 implicaría 82
  entrenamientos × 2 modelos × 10 restarts = 1640 fits HMM.
  Excesivo para el aporte marginal sobre una validación honesta.
  (3) **Estratificación**: garantiza que el holdout no esté sesgado
  hacia un subgrupo de aves.
- **Alternativa descartada**: LOBO completo. Más riguroso
  estadísticamente pero impracticable y, en este caso, no aporta
  sobre el aprendizaje de O2.
- **Implicación para la memoria**: justificar el cambio de protocolo
  respecto a O2 con una frase: "LOBO se mostró inadecuado en O2 por
  las rutas individuales; en O3, donde las features son cinemáticas
  y comparables entre individuos, basta un holdout estratificado por
  ave."

**8.9 Log-likelihood reportada por observación, no total**

- **Decisión**: la métrica de ajuste reportada es
  `log_likelihood_total / n_observaciones_holdout`, en lugar de la
  log-likelihood total.
- **Por qué**: la log-likelihood total depende del tamaño del
  dataset. Un dataset de 21 000 observaciones tendrá log-likelihood
  total ~10 veces más negativa que uno de 2 100 observaciones,
  aunque el modelo sea idéntico. Reportar por observación da una
  métrica comparable entre datasets, configuraciones o
  reproducciones.
- **Alternativa descartada**: log-likelihood total. Engañosa cuando
  se compara entre datasets.
- **Implicación para la memoria**: trivial pero hay que mencionarlo
  en la sección de resultados.

**8.10 No comparar log-likelihood directamente entre Modelo A y Modelo B**

- **Decisión**: la log-likelihood se reporta para cada modelo en su
  propia escala, no como métrica de comparación.
- **Por qué**: la log-likelihood mide `log P(observaciones |
  modelo)`. Cuando los modelos operan sobre **espacios de
  observación distintos** (A tiene 2 features, B tiene 5), las
  observaciones no son las mismas matemáticamente — no es justo
  comparar "qué tan probable es lo que yo veo" con "qué tan probable
  es lo que tú ves". Es como comparar el precio de un coche con el
  precio de una bici: ambos en euros, pero "el coche es más caro"
  no significa "el coche es mejor".
- **Alternativa descartada**: ajustar las LL con criterios de
  información (AIC, BIC) y comparar. Posible en principio pero
  requiere asumir que las features de B son una "extensión" de las
  de A, lo que es discutible.
- **Implicación para la memoria**: explicar en la sección de
  evaluación que A y B se evalúan **por separado** (LL en su escala)
  y se **comparan** con criterios independientes (coherencia
  biológica + acuerdo).

**8.11 Coherencia biológica como criterio sin ground truth**

- **Decisión**: la validación principal de los estados detectados
  es su **coherencia con conocimiento biológico previo** sobre
  *Larus fuscus* (concentración estacional de la migración,
  latitudes esperadas, en Modelo A también correlación con
  fotoperiodo y vegetación), no contra etiquetas verdaderas.
- **Por qué**: no existe ground truth diario sobre el comportamiento
  de las 82 aves. Nadie ha etiquetado manualmente 21 000 días de
  tracking. Pero **sí existe conocimiento poblacional** sobre la
  fenología de la especie (Wikelski et al. 2015 y bibliografía
  sobre migración de gaviotas), que actúa como criterio externo
  independiente del modelo.
- **Limitaciones reconocidas**: este criterio no detecta errores
  idiosincrásicos (un ave concreta puede tener una fenología atípica
  que el HMM acierte y la biología "promedio" no contemple). Es un
  criterio poblacional, no individual.
- **Alternativa descartada**: validar contra etiquetas humanas
  anotadas en una sub-muestra (e.g., 50 días). Demasiado costoso
  para el alcance del TFG y depende de la pericia del anotador.
- **Implicación para la memoria**: presentarlo como **el** criterio
  principal de validación, con la precaución metodológica explícita
  ("validación poblacional, no individual").

**8.12 Acuerdo A-B reportado como porcentaje crudo, no como Cohen's kappa**

- **Decisión**: el acuerdo entre Modelo A y Modelo B sobre cada
  (ave, día) se reporta como `% de coincidencia = n_iguales /
  n_total`, no como Cohen's kappa u otra métrica corregida por
  azar.
- **Por qué**: Cohen's kappa corrige por la "tasa de acuerdo
  esperada por azar", lo que es útil cuando los clasificadores son
  independientes y se evalúan contra ground truth. En nuestro caso,
  A y B **no son independientes** (comparten las dos primeras
  features) y **no hay ground truth**. El porcentaje crudo es
  directamente interpretable ("coinciden en el 87 % de los días")
  y suficiente para el análisis cualitativo de los desacuerdos.
- **Alternativa descartada**: Cohen's kappa. Más sofisticado, menos
  interpretable, no necesario para el propósito de este capítulo.
- **Implicación para la memoria**: reportar el porcentaje crudo. Si
  se quiere ser exhaustivo, añadir kappa como nota al pie.

## 9. Tests

Patrón heredado de O1/O2. Un fichero por módulo en
`tests/test_hmm_<nombre>.py`. Datasets sintéticos pequeños
deterministas; el `daily.parquet` real no entra en los tests excepto
en el smoke de integración con un subset.

### 9.1 `tests/test_hmm_features.py` (~6 tests)

- **Turning angle, caso conocido recto**: secuencia de 3 puntos en
  línea recta (lat 50, 50.5, 51, lon 0), `abs_turning_angle ≈ 0`.
- **Turning angle, giro 180°**: secuencia que va norte-norte-sur
  (lat 50, 50.5, 50), `abs_turning_angle ≈ π`.
- **Daylight hours en equinoccio**: para lat=0 y día 80 (equinoccio
  de primavera aprox.), `daylight ≈ 12h`. Para lat=60 día 172
  (solsticio de verano), `daylight ≈ 18-19h`.
- **Tramos de días consecutivos**: dado un DataFrame con un hueco
  (`is_valid=False`) en medio, las features sólo se calculan para
  los días con triplete consecutivo. Endpoints del tramo tienen
  `is_observation_valid=False`.
- **Join con CSV crudo para vegetación**: dado un mini-CSV crudo y
  un mini-daily.parquet sintético, el join por `event_id` produce
  los valores correctos de `veg_low` y `veg_high`.
- **Días con `source_event_id` ausente en raw**: si la fila de
  daily tiene `source_event_id` que no existe en el CSV (caso
  patológico), `veg_low` y `veg_high` quedan a `NaN` y
  `is_observation_valid=False`.

### 9.2 `tests/test_hmm_fit.py` (~5 tests)

- **Convergencia sobre datos sintéticos**: genero observaciones de
  dos gaussianas conocidas (media `(0,0)` y `(3,3)`) con transiciones
  simples. El HMM ajusta medias cercanas a `(0,0)` y `(3,3)`.
- **Múltiples restarts no decrecen la log-likelihood**: la mejor LL
  de 5 restarts es ≥ la LL de 1 restart con el mismo seed inicial.
- **K-means init produce medias razonables**: las medias iniciales
  del HMM (antes de EM) coinciden con los centroides de k-means
  sobre las mismas features.
- **Re-etiquetado de estados**: tras entrenar, el estado con menor
  `μ[log_displacement_km]` tiene etiqueta 0 (estacionario). Test
  con datos sintéticos donde sabemos cuál debería ser estacionario.
- **Holdout split estratificado**: el split por ave reparte aves
  "ricas" (>200 días válidos) y "pobres" (<100 días) de forma
  proporcional entre train y holdout.

### 9.3 `tests/test_hmm_evaluate.py` (~4 tests)

- **Log-likelihood per observation**: dado un HMM entrenado y un
  holdout sintético, `hmm.score(X) / len(X)` devuelve un escalar
  finito (no `-inf` ni `nan`).
- **Acuerdo A-B en caso degenerado**: si ambos modelos asignan el
  mismo estado a todas las observaciones, la tasa de acuerdo es
  100 %.
- **Acuerdo A-B con desacuerdos sintéticos**: si A y B asignan
  estados opuestos para una mitad de las observaciones, el acuerdo
  es 50 %.
- **Coherencia biológica con datos artificiales**: dado un dataset
  sintético donde los estados de migración están concentrados en
  marzo y octubre, la métrica de coherencia detecta esa
  concentración (chi² o equivalente).

### 9.4 `tests/test_hmm_build.py` (~2 tests integración)

- **`build_o3(...)` con dataset sintético**: 3 aves × 60 días
  sintéticos con un par de tramos con huecos. Verifica que existen
  los 3 ficheros esperados y que `features.parquet` tiene las
  columnas del esquema 6.1.
- **Esquemas se releen correctamente**: `pyarrow` puede leer
  `features.parquet` sin errores, todas las columnas presentes,
  tipos correctos.

**Total: ~17 tests nuevos.** Sumados a los 60 actuales → 77 tests al
cerrar O3.

### 9.5 Lo que NO se testea

- Los valores concretos de las features sobre el dataset real (eso
  lo valida el notebook EDA con figuras, no aserciones de código).
- `save_artifact()`: ya cubierto en `tests/test_reporting.py`.
- `hmmlearn` internals: confiamos en la librería.
- Convergencia con seeds específicos: EM puede ser sensible a
  versiones; no fijar números mágicos en tests.

## 10. Riesgos identificados

- **Convergencia degenerada del HMM**: en runs concretos, EM puede
  asignar la mayoría de las observaciones a un solo estado. **Mitigación**:
  10 restarts y selección por mayor LL en train. Si todos los restarts
  caen en soluciones degeneradas, hay un problema más profundo en las
  features.
- **Coherencia biológica baja en Modelo B**: si las features de
  contexto (veg, daylight) dominan el aprendizaje del HMM, los
  estados podrían codificar "verano vs invierno" o "Europa vs África"
  en lugar de "estacionario vs migración". **Mitigación**: el
  artefacto C3 lo hace visible; si patológico, follow-up sobre la
  estandarización o el peso relativo de las features.
- **Desacuerdo masivo entre A y B**: si A y B coinciden en <50 %,
  significa que "migración" significa cosas distintas en ambos
  modelos. **Mitigación**: el artefacto C4 caracteriza los
  desacuerdos para diagnóstico.
- **Aves con todos sus días sin triplete válido**: improbable
  (mediana de racha 12 días, p90 142) pero posible para aves con
  tracking muy fragmentado. **Mitigación**: estas aves simplemente
  no contribuyen observaciones al modelo ni reciben Viterbi; aparecen
  en `features.parquet` con todo a `NaN` y `is_observation_valid=False`.
- **Aves con todos sus días asignados a un solo estado**: esperable
  para aves residentes puras. Aparece en el artefacto C5 como
  heterogeneidad poblacional.

## 11. Cierre de O3

### 11.1 Granularidad de commits

Todos en castellano, sin trailer Co-Authored-By:

1. Esqueleto de `tfg_aves.hmm`: features, fit, evaluate, build
2. Tests unitarios de features
3. Implementar features (turning_angle, daylight, join veg)
4. Tests unitarios de fit (k-means init, restarts, label switching)
5. Implementar fit + restarts
6. Tests unitarios de evaluate
7. Implementar evaluate
8. Orquestación `build_o3` + tests de integración
9. Notebook EDA O3 + D1 (nstates AIC/BIC sweep)
10. Ejecutar `build_o3` con holdout final y materializar artefactos
    O3 + memoria + ai-log + tag

Granularidad orientativa. Si subagent-driven-development consolida
naturalmente algunos pasos, se permite.

### 11.2 Artefactos generados

- `reports/figures/o3_fig*.png`, `reports/tables/o3_tab*.csv`,
  `reports/captions/o3_*.md` — versionados.
- `reports/INDEX.md` — actualizado por `save_artifact()` a 6 entradas
  O3.
- `reports/ai-log/0008-o3-hmm-comportamiento.md` — una entrada para
  toda la fase, no por commit (gitignored).
- `reports/memoria/05_o3_hmm.md` — notas estructuradas siguiendo
  `reports/memoria/_plantilla.md`. Incluye apartado "Justificación
  de decisiones metodológicas" replicando la sección 8 en formato
  citable para el capítulo 5 LaTeX. Sin prosa final.

### 11.3 Tag de hito

`v0.3-o3-completo` apuntando al último commit cuando se cumplan los
criterios de aceptación.

### 11.4 Criterios de aceptación

- `uv run pytest -q` verde (60 previos + ~17 nuevos).
- `uv run ruff check src tests` verde.
- `data/processed/o3/` regenerable desde `daily.parquet` y el CSV
  crudo en una sola llamada a `build_o3(...)`.
- 6 entradas O3 en `reports/INDEX.md` con caption en castellano.
- Notas O3 en `reports/memoria/05_o3_hmm.md`, incluyendo apartado
  "Justificación de decisiones metodológicas" derivado de la sección
  8 del spec.
- Entrada `0008-*` en `reports/ai-log/`.
- Tag `v0.3-o3-completo` creado.
- **No hay criterio de aceptación cuantitativo duro** (a diferencia
  de O2): el modelo es no supervisado, no hay un umbral mínimo de
  log-likelihood ni de acuerdo A-B. La aceptación es **cualitativa**:
  el artefacto C3 muestra coherencia biológica visible para al
  menos uno de los dos modelos. Si ambos fallan, hay un problema
  más profundo en las features y se diagnostica como follow-up.

## 12. Fuera de alcance

Explícitamente fuera de O3 (van en objetivos posteriores):

- Modelo predictivo (Markov, ML supervisado): O2 y O4 respectivamente.
- HMM per-individuo: descartado en F3.
- HMMs con más de 2 estados como modelo principal: el artefacto D1
  los explora como respaldo de la decisión, pero la decisión es n=2.
- Features cinemáticas de orden superior (aceleración, jerk): el
  signal/ruido de 1 fix/día no las hace defendibles.
- Comparación cuantitativa de O3 con O2 o O4: O5 (mapas + análisis
  del error) consume los outputs de los tres y los compara
  cuantitativamente.
- Validación contra ground truth anotada manualmente: descartada en
  8.11 (fuera de presupuesto y alcance).
- Distribución de emisión no-gaussiana (e.g., von Mises para
  ángulos, beta para vegetación): explorable como follow-up si la
  gaussiana sobre `abs_turning_angle_rad` muestra patologías
  visibles.
