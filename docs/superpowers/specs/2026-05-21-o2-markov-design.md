# Diseño de O2 — Predicción con cadenas de Markov visibles

- **Fecha:** 2026-05-21
- **Objetivo del TFG:** O2 (40 h) — predicción con cadenas de Markov
  visibles
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado, pendiente de implementación

## 1. Resumen

O2 entrena **12 matrices de transición mensuales** (Markov de primer
orden, paso 1 día) sobre el espacio discretizado de posiciones diarias
de *Larus fuscus*, y construye un predictor evaluado por
leave-one-bird-out (LOBO). El rol del modelo en el TFG es el de
**baseline interpretable** que los modelos de O4 (RF / XGBoost /
LightGBM) deberán batir. Lee
`data/processed/daily.parquet` (entregable de O1, 82 aves, 21 823 días
válidos) y escribe en `data/processed/o2/` cuatro ficheros:
`cells.parquet`, `transitions_counts.npz`, `transition_matrices.npz` y
`predictions_lobo.parquet` + `metrics.parquet`. Las matrices ajustadas
con todas las aves se usan para visualización e interpretación; las
predicciones almacenadas en LOBO alimentan O5 (mapas y análisis del
error) y proporcionan el punto de comparación cuantitativa para O4.

## 2. Contexto y constraints

- **Entregable de O1 ya disponible:** `daily.parquet` con una fila por
  `(bird_id, date_utc)` para 82 aves entre 2009-05-25 y 2015-08-27.
  Posiciones a 08:00 UTC ± 60 min. Huecos explícitos
  (`is_valid=False, lat=NaN, lon=NaN`).
- **Sesgo temporal del dataset:** la actividad cae de 82 aves en 2009
  a ~10 en 2011 y prácticamente una sola en 2014-2015. Esto **descarta**
  estrategias de validación basadas en split por año.
- **Paso de la cadena:** 1 día calendario UTC. Decisión heredada de O1
  (no se re-abre).
- **Política de convenciones del proyecto** (CLAUDE.md): toda decisión
  no trivial requiere figura/tabla + caption en castellano vía
  `save_artifact()`; commits en castellano sin trailer de IA; código
  con identificadores en inglés; tests con pytest sobre datasets
  sintéticos.
- **Patrón heredado de O1:** módulos puros + orquestador `build_oN` +
  notebook EDA jupytext + tests. El usuario lo validó tras O1
  (`feedback_subagent_driven_validated`).

## 3. Decisiones de diseño fijadas

Decididas en el brainstorming previo, sin EDA:

| # | Decisión | Valor |
|---|---|---|
| F1 | Rol de O2 | Baseline interpretable que ML deberá batir |
| F2 | Alcance del modelo | **Global** — una matriz por unidad temporal, 82 aves agregadas. Per-individual queda como ablación de evaluación, no de modelado. |
| F3 | Granularidad temporal | **12 matrices mensuales** (una por mes calendario). |
| F4 | Discretización del espacio | **Grid uniforme lat/lon**. Tamaño de celda candidato `cell_deg = 0.5°`, decisión final justificada por trade-off entre `{0.25°, 0.5°, 1°, 2°}`. |
| F5 | Tratamiento de huecos | **Saltar transición**: sólo se cuenta el par (t→t+1) cuando ambos días son válidos y consecutivos en calendario (`date_{t+1} − date_t = 1 día`). |
| F6 | Suavizado | **Laplace add-α con α = 1** fijo. Sin barrido. |
| F7 | Validación | **Leave-one-bird-out (LOBO)**, 82 folds. Para cada ave: matrices entrenadas con las otras 81, predicciones almacenadas para todos sus pares válidos. |
| F8 | Métricas | Top-1 accuracy, distancia mediana (km), log-loss, + baseline persistencia (`mañana = hoy`). |
| F9 | Centroide de celda | Centro geométrico `(lat_c, lon_c) = ((i+0.5)·cell_deg, (j+0.5)·cell_deg)`. Distancia entre celdas: **Haversine** (km). |
| F10 | Mes de la transición | Mes del **origen** `t`. Una transición que cruza fin de mes pertenece al mes de `t`. |
| F11 | Origen del grid | Fijo en `(lat=0, lon=0)`, no flotante por bounding box. Reproducible y comparable entre runs. |
| F12 | Estructura de la pipeline | Paquete `tfg_aves.markov` con módulos `discretize`, `transition`, `smooth`, `predict`, `evaluate`, `build`. Notebook EDA en `notebooks/02_eda_o2.py`. |

Decisiones que se materializan en `reports/INDEX.md` con `decision=...`
y se resuelven en el EDA con figura:

| # | Parámetro | Slug del artefacto justificativo |
|---|---|---|
| D1 | `cell_deg` (tamaño de celda final) | `grid-size-tradeoff` |

α se fija a 1 a priori (decisión F6). Si el inspeccionar las matrices
ajustadas muestra patologías (filas uniformes en celdas con datos), se
documenta como follow-up; no es punto de decisión por figura.

## 4. Arquitectura

Pipeline en seis fases ejecutadas desde el notebook
`notebooks/02_eda_o2.py` que importa funciones de
`src/tfg_aves/markov/`:

```
data/processed/daily.parquet         (entregable de O1)
              │
              ▼
     (1) discretize.py    — (lat, lon) → cell_id
              │
              ▼
     (2) transition.py    — pares válidos (cell_t, cell_{t+1}, month)
              │
              ▼
     (3) smooth.py        — counts + Laplace α → 12 matrices estocásticas
              │
              ▼
     (4) predict.py       — origen + mes → distribución sobre celdas
              │
              ▼
     (5) evaluate.py      — LOBO 82 folds + baseline persistencia
              │
              ▼
     (6) build.py         — orquestar y materializar
              │
              ▼
data/processed/o2/
  cells.parquet
  transitions_counts.npz
  transition_matrices.npz
  predictions_lobo.parquet
  metrics.parquet
```

Tras O2, regenerar los outputs se hace con:

```python
from tfg_aves.markov import build_o2
build_o2(
    cell_deg=<D1>,
    alpha=1.0,
)
```

Las funciones de los submódulos son puras (entrada DataFrame/array →
salida DataFrame/array). Sólo `build.py` escribe a disco.

## 5. Módulos

### 5.0 Constantes del paquete

`src/tfg_aves/markov/__init__.py` expone:

```python
DAILY_PARQUET = ROOT / "data" / "processed" / "daily.parquet"
O2_OUT_DIR    = ROOT / "data" / "processed" / "o2"
```

Las funciones aceptan estas rutas como argumentos para mantenerse
parametrizables (tests las sobrescriben con `tmp_path`).

### 5.1 `src/tfg_aves/markov/discretize.py`

```python
def assign_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]
def cell_centroid(cell_id: tuple[int, int], cell_deg: float) -> tuple[float, float]
def discretize_dataframe(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame
def haversine_km(lat1, lon1, lat2, lon2) -> float
```

- `assign_cell`: `(floor(lat/cell_deg), floor(lon/cell_deg))`.
- `cell_centroid`: inverso, devuelve el centro geométrico.
- `discretize_dataframe`: añade columnas `cell_lat_idx`, `cell_lon_idx`,
  `cell_id` (string `"i_j"`) al DataFrame. Filas con `is_valid=False`
  reciben `cell_id=None`.
- `haversine_km`: vectorizada sobre arrays numpy.

### 5.2 `src/tfg_aves/markov/transition.py`

```python
def build_transitions(df_daily_discretized: pd.DataFrame) -> pd.DataFrame
def build_counts(
    df_transitions: pd.DataFrame,
    cells: list[str],
) -> np.ndarray  # shape (12, n_cells, n_cells)
```

- `build_transitions`: ordena por `(bird_id, date_utc)`, calcula la
  diferencia de fechas entre filas consecutivas del mismo ave, emite
  un par sólo si:
  - `df.shift(1).bird_id == df.bird_id`
  - `df.shift(1).is_valid and df.is_valid`
  - `(df.date_utc − df.shift(1).date_utc) == 1 día`

  Devuelve DataFrame con columnas
  `bird_id, date_t, month_int, cell_from, cell_to, lat_from, lon_from, lat_to, lon_to`.

- `build_counts`: dado el conjunto ordenado de celdas activas, produce
  un tensor de counts denso `(12, n_cells, n_cells)` por agregación
  con `numpy.add.at`.

### 5.3 `src/tfg_aves/markov/smooth.py`

```python
def laplace_smooth(counts: np.ndarray, alpha: float) -> np.ndarray
def marginal_distribution(counts: np.ndarray) -> np.ndarray
```

- `laplace_smooth`: aplica
  `P[m,i,j] = (counts[m,i,j] + α) / (sum_j counts[m,i,j] + α · n_cells)`
  por mes, garantizando que cada fila suma 1. Requiere `α > 0`
  (lanza `ValueError` si no). Devuelve `P` mismo shape que `counts`,
  dtype float64.
- `marginal_distribution`: para cada mes devuelve la marginal
  `π_m[j] = counts[m,:,j].sum() / counts[m].sum()`. Shape
  `(12, n_cells)`. Usada como fallback en `predict_distribution`
  cuando el `cell_from` no está en el espacio de celdas del modelo
  (caso LOBO con celda nueva). Si un mes tiene 0 counts totales
  (improbable pero posible para datasets sintéticos), se devuelve
  una uniforme `1/n_cells`.

### 5.4 `src/tfg_aves/markov/predict.py`

```python
def predict_distribution(
    P: np.ndarray,                 # (12, n_cells, n_cells)
    cell_from: str,
    month_int: int,
    cells: list[str],
    marginal: np.ndarray | None = None,  # fallback (12, n_cells)
) -> np.ndarray                   # vector (n_cells,)

def topk_from_distribution(
    distribution: np.ndarray,
    k: int,
    cells: list[str],
) -> list[str]

def prediction_distance_km(
    cell_pred: str,
    lat_real: float,
    lon_real: float,
    cell_deg: float,
) -> float
```

- `predict_distribution`: devuelve la fila `P[month_int][index_of(cell_from)]`.
  Si `cell_from` no está en `cells` (caso LOBO con celda nueva) y
  `marginal` se ha pasado, usa `marginal[month_int]` como distribución
  de salida. Si `marginal is None` y `cell_from` no está en `cells`,
  lanza `KeyError`; el llamador (LOBO) es responsable de pasar la
  marginal cuando contemple este caso.
- `topk_from_distribution`: argsort estable, devuelve los `k` cell_ids
  con mayor probabilidad.
- `prediction_distance_km`: Haversine entre centroide del `cell_pred`
  y la posición real `(lat_real, lon_real)`.

### 5.5 `src/tfg_aves/markov/evaluate.py`

```python
def lobo_predictions(
    df_transitions: pd.DataFrame,
    cells: list[str],
    cell_deg: float,
    alpha: float,
) -> pd.DataFrame

def persistence_predictions(
    df_transitions: pd.DataFrame,
    cell_deg: float,
) -> pd.DataFrame

def aggregate_metrics(
    predictions: pd.DataFrame,    # long format con columna `model`
) -> pd.DataFrame
```

- `lobo_predictions`: itera sobre `bird_id` únicos. Para cada ave:
  - Filtra `df_transitions` a `bird_id ≠ held_out` (training).
  - Recalcula `counts` con `build_counts`, `P` con `laplace_smooth`,
    y `marginal` con `marginal_distribution` (todo a partir del
    training del fold).
  - Para cada par del ave held-out: predice distribución, top-1, top-3,
    prob asignada al destino real, distancia al destino real.
  - Si el ave held-out tiene 0 pares válidos, se salta sin error.
  - Si su `cell_from` no está en `cells` del fold, `predict_distribution`
    cae a la marginal.
  - Si `cell_real` no está en `cells` del fold, `prob_assigned_real`
    se fija a `α / (counts_row_sum + α · n_cells)` (la probabilidad
    que Laplace asignaría a una celda no observada) — para que
    log-loss siga teniendo valor finito.

  Devuelve DataFrame long con `model="markov"`.

- `persistence_predictions`: para cada par, `cell_pred_top1 = cell_from`.
  Probabilidades: `1 − ε · (n_cells − 1)` al destino predicho,
  `ε = 1e-9` al resto. Esto garantiza distribución estocástica con
  log-loss finito tanto si `cell_real == cell_from` como si no. Mismo
  formato de salida, `model="persistence"`.

- `aggregate_metrics`: agrupa las predicciones en tres niveles y
  devuelve un DataFrame único en formato long con columna `scope`:
  `"bird_month"` (por `bird_id × month_int × model`), `"month"`
  (por `month_int × model`) y `"global"` (por `model`). Calcula
  `n_predictions`, `top1_acc`, `top3_acc`, `dist_km_median`,
  `dist_km_p90`, `log_loss`. Log-loss con clipping a `ε=1e-9` antes
  de tomar el logaritmo.

### 5.6 `src/tfg_aves/markov/build.py`

```python
@dataclass
class BuildO2Result:
    cells_path: Path
    counts_path: Path
    matrices_path: Path
    predictions_path: Path
    metrics_path: Path
    n_cells: int
    n_transitions: int
    summary: dict   # claves: top1_acc_markov, top1_acc_persistence,
                    # dist_km_median_markov, dist_km_median_persistence,
                    # log_loss_markov, log_loss_persistence, n_predictions

def build_o2(
    *,
    cell_deg: float,
    alpha: float = 1.0,
    do_lobo: bool = True,
    daily_path: Path = DAILY_PARQUET,
    out_dir: Path = O2_OUT_DIR,
    seed: int = 0,
) -> BuildO2Result
```

- Compone `load(daily) → discretize → build_transitions → build_counts
  → laplace_smooth → evaluate (LOBO + persistence) → escribir`.
- Escribe los 5 ficheros de `data/processed/o2/`.
- Devuelve dataclass con paths + métricas agregadas globales que el
  notebook usa para una tabla resumen en el INDEX.

Principios que vertebran el diseño:

- Umbrales (`cell_deg`, `alpha`) como argumentos, nunca constantes
  globales.
- Funciones puras devolviendo DataFrames/arrays; sólo `build.py`
  escribe a disco.
- Reutilización: `build_transitions` se llama una sola vez; LOBO
  filtra sobre el resultado en lugar de regenerar desde `daily.parquet`.

## 6. Esquema de datos

### 6.1 `data/processed/o2/cells.parquet`

Diccionario de celdas activas (todas las que tienen ≥1 fix válido en
`daily.parquet`).

| Columna | Tipo | Descripción |
|---|---|---|
| `cell_id` | string | `"i_j"` con `i = cell_lat_idx`, `j = cell_lon_idx` |
| `cell_lat_idx` | int32 | índice de fila del grid |
| `cell_lon_idx` | int32 | índice de columna del grid |
| `lat_c` | float64 | centroide latitud |
| `lon_c` | float64 | centroide longitud |
| `n_obs_total` | int64 | nº de filas diarias válidas en esa celda agregado sobre todas las aves |

### 6.2 `data/processed/o2/transitions_counts.npz`

Archivo numpy con dos arrays:
- `counts` — shape `(12, n_cells, n_cells)`, dtype `int32`.
- `cells` — array de strings con el orden de celdas (consistente con
  `cells.parquet`).

### 6.3 `data/processed/o2/transition_matrices.npz`

Mismo shape que `transitions_counts.npz` pero suavizado con Laplace
α=1, dtype `float64`. Las filas suman 1 modulo error de coma flotante.

### 6.4 `data/processed/o2/predictions_lobo.parquet`

Long-format: una fila por (par válido del ave, modelo). Para 82 aves
con ~265 pares válidos en mediana, ~20-25 k filas por modelo → ~40-50 k
filas totales.

| Columna | Tipo | Descripción |
|---|---|---|
| `bird_id` | string | ave (held-out en su fold para markov) |
| `date_t` | date32 | fecha del origen |
| `month_int` | int8 | 1-12 |
| `lat_t`, `lon_t` | float64 | posición real en *t* |
| `cell_t` | string | celda discretizada de *t* |
| `lat_real`, `lon_real` | float64 | posición real en *t+1* |
| `cell_real` | string | celda real en *t+1* |
| `cell_pred_top1` | string | celda predicha (argmax) |
| `cell_pred_top3` | string | 3 celdas más probables, separadas por `,` |
| `prob_top1` | float64 | probabilidad de la top-1 |
| `prob_assigned_real` | float64 | prob que el modelo dio a `cell_real` |
| `dist_km` | float64 | Haversine entre centroide(top1) y `(lat_real, lon_real)` |
| `is_top1_hit` | bool | `cell_pred_top1 == cell_real` |
| `is_top3_hit` | bool | `cell_real ∈ top3` |
| `model` | string | `"markov"` o `"persistence"` |

### 6.5 `data/processed/o2/metrics.parquet`

Métricas agregadas a tres niveles, todos en el mismo fichero
distinguidos por una columna `scope`:

| Columna | Tipo | Descripción |
|---|---|---|
| `scope` | string | `"bird_month"`, `"month"`, `"global"` |
| `bird_id` | string \| `null` | sólo para `scope="bird_month"` |
| `month_int` | int8 \| `null` | sólo para `bird_month` y `month` |
| `model` | string | `"markov"` / `"persistence"` |
| `n_predictions` | int64 | |
| `top1_acc` | float64 | |
| `top3_acc` | float64 | |
| `dist_km_median` | float64 | |
| `dist_km_p90` | float64 | |
| `log_loss` | float64 | |

## 7. Decisiones justificadas (artefactos)

Convención `save_artifact()` con `objective="o2"`, captions en
castellano.

### 7.1 Decisiones que fijan parámetros del pipeline

| Slug | Decisión | Forma del artefacto |
|---|---|---|
| `grid-size-tradeoff` (D1) | `cell_deg` (tamaño de celda final) | 3 paneles para candidatos `{0.25°, 0.5°, 1°, 2°}`: (A) histograma del desplazamiento diario observado con líneas verticales en el tamaño físico de cada celda; (B) tabla con `n_celdas_ocupadas`, `mediana_transiciones_por_celda_mes`, `% pares observados`, `% self-loops`; (C) 4 mini-mapas con los grids superpuestos sobre la nube de fixes |

### 7.2 Caracterización del modelo y de los resultados

| Slug | Contenido |
|---|---|
| `transition-matrix-example` (C1) | Heatmap de una matriz mensual representativa (e.g., octubre, mes con mayor n según `o1_tab10_monthly-seasonal-coverage`). Caption explica diagonal (self-loops) y off-diagonal |
| `monthly-matrices-overview` (C2) | 12 mini-heatmaps en grid 3×4. Lectura cualitativa de la estacionalidad |
| `accuracy-vs-baseline` (C3) | Barras comparando markov vs persistence en top-1, dist mediana, log-loss. Desglose por mes |
| `per-bird-performance` (C4) | Distribución por ave (boxplot/strip) de top-1 acc y dist mediana km. Conexión con O5 |
| `summary` (C5) | Tabla resumen final: una fila por mes con métricas markov y delta vs persistence |

**Total esperado en `INDEX.md`:** 1 decisión + 5 artefactos de
caracterización = 6 entradas O2.

### 7.3 Decisiones triviales sin figura

Mencionadas en notas de memoria pero sin artefacto propio:

- α = 1 (justificable en una frase: valor add-one estándar).
- Origen del grid en (0, 0) (reproducibilidad).
- Saltar transición ante hueco (semántica de Markov(1) a paso 1 día).
- Mes de la transición = mes del origen.

## 8. Tests

Patrón heredado de O1. Un fichero por módulo en
`tests/test_markov_<nombre>.py`. Datasets sintéticos pequeños y
deterministas; el `daily.parquet` real no entra en los tests excepto
en el smoke test de integración (que puede usar un subset).

### 8.1 `tests/test_markov_discretize.py` (5 tests)

- `assign_cell` para puntos conocidos en distintos cuadrantes.
- `assign_cell` monotonía: lat creciente → idx no decreciente.
- `assign_cell` borde: punto exactamente sobre línea de celda cae en
  la celda superior (consistencia con `floor`).
- `cell_centroid` inverso de `assign_cell` modulo `cell_deg/2`.
- `haversine_km` para puntos conocidos: París ↔ Madrid ≈ 1054 km.

### 8.2 `tests/test_markov_transition.py` (6 tests)

- Par consecutivo válido se emite con `month_int` del origen.
- Hueco entre días rompe el par: 1-ene, 2-ene inválido, 3-ene válido →
  no se emite ningún par para este ave.
- Cambio de ave no emite par entre la última fila de un bird y la
  primera del siguiente.
- Cambio de mes: par 31-ene → 1-feb pertenece a mes 1 (origen).
- `build_counts` con DataFrame sintético conocido produce los counts
  esperados (matriz 3×3 con valores fijos).
- `build_counts` para mes sin transiciones produce matriz de ceros.

### 8.3 `tests/test_markov_smooth.py` (5 tests)

- Filas suman 1 modulo `1e-12`.
- Sin entradas en cero ni negativas.
- `laplace_smooth(counts, alpha=0)` lanza `ValueError`.
- α grande tiende a uniforme: con α = 1e6 todas las entradas ≈ 1/n_cells.
- `marginal_distribution` para mes con counts totales 0 devuelve
  uniforme `1/n_cells`.

### 8.4 `tests/test_markov_predict.py` (5 tests)

- `predict_distribution` para celda conocida devuelve la fila esperada.
- `predict_distribution` para celda no vista en `cells` usa marginal
  (cuando se pasa).
- `topk_from_distribution` devuelve k celdas más probables, ordenadas
  por probabilidad descendente.
- `prediction_distance_km` con `cell_pred` = celda real y posición
  igual al centroide da 0 km.
- `prediction_distance_km` con celda predicha lejos da la distancia
  Haversine esperada.

### 8.5 `tests/test_markov_evaluate.py` (5 tests)

- `lobo_predictions` con dataset sintético de 3 aves × 10 días
  válidos produce predictions con shape esperada (~3 × 9 = 27 filas).
- `lobo_predictions` ave con 0 pares válidos no rompe el bucle.
- `persistence_predictions` da `cell_pred_top1 == cell_from` siempre.
- `aggregate_metrics`: `top1_acc` agregado coincide con la media de
  `is_top1_hit` por grupo.
- `aggregate_metrics`: log-loss para predicciones perfectas (prob = 1
  al destino real) → log-loss ≈ 0.

### 8.6 `tests/test_markov_build.py` (2 tests de integración)

- `build_o2(...)` con `daily.parquet` sintético en `tmp_path`
  (3 aves × 60 días) verifica que existen los 5 ficheros esperados.
- Esquemas: `predictions_lobo.parquet` y `metrics.parquet` se releen
  con `pyarrow` y cumplen el esquema de la sección 6.

**Total:** 28 tests nuevos (5+6+5+5+5+2). Sumados a los 29 actuales →
57 tests al cerrar O2.

### 8.7 Lo que NO se testea

- El valor concreto de `cell_deg` final (D1): decisión humana
  justificada con figura, no aserción de código.
- `save_artifact()`: ya cubierto por `tests/test_reporting.py`.
- El dataset real completo: validado por la ejecución del notebook y
  la revisión visual de las figuras.

## 9. Riesgos identificados

- **Sparsity a 0,5°.** Con 21 823 días válidos repartidos en 12 meses,
  cada matriz mensual tiene ~1 800 transiciones. Si la decisión final
  D1 es 0,5° y resultan ~500-1000 celdas activas, el % de pares
  (origen, destino) observados será bajo. **Mitigación:** el trade-off
  de D1 lo cuantifica y permite refinar a 1° si la sparsity es
  patológica. El caption documenta el ratio.
- **Heterogeneidad temporal del dataset (sesgo 2009).** Las matrices
  mezclan años, asumiendo estacionalidad estable. **Mitigación:**
  supuesto explicitado en el caption del artefacto C2.
- **Filas casi-uniformes por Laplace.** En celdas con pocos counts,
  α = 1 puede dominar. **Mitigación:** el artefacto C1 lo hace visible;
  si patológico, follow-up con α menor (decisión documentada como
  follow-up, no como D2).
- **Ave held-out con todos sus pares cruzando huecos.** Improbable
  (mediana de racha 12 días, p90 142) pero posible. **Mitigación:**
  `lobo_predictions` salta el ave sin error.
- **Celda nueva en el ave held-out.** El ave sale en una celda no
  vista en el training. **Mitigación:** fallback a marginal del mes;
  la métrica de distancia penaliza la predicción como debe.

## 10. Cierre de O2

### 10.1 Granularidad de commits

Todos en castellano, sin trailer Co-Authored-By:

1. `Esqueleto de tfg_aves.markov: discretize, transition, smooth, predict, evaluate, build`
2. `Tests unitarios de discretize y haversine`
3. `Implementar discretize con tests`
4. `Tests unitarios de transition y counts`
5. `Implementar transition + counts con tests`
6. `Tests unitarios de smooth`
7. `Implementar smooth con tests`
8. `Tests unitarios de predict`
9. `Implementar predict con tests`
10. `Tests unitarios de evaluate (LOBO + persistencia)`
11. `Implementar evaluate con tests`
12. `Notebook EDA: trade-off de tamaño de celda, matrices mensuales`
13. `Orquestación build_o2 + test de integración`
14. `Ejecutar build_o2 con cell_deg final y materializar outputs`

Granularidad orientativa. Si subagent-driven-development consolida
naturalmente algunos pasos, se permite.

### 10.2 Artefactos generados

- `reports/figures/o2_fig*.png`, `reports/tables/o2_tab*.csv`,
  `reports/captions/o2_*.md` — versionados.
- `reports/INDEX.md` — actualizado por `save_artifact()` a 6 entradas
  O2.
- `reports/ai-log/0007-o2-markov-mensual.md` — una entrada para toda la
  fase, no por commit. Tono según la política (autor decide; IA propone
  y ejecuta mecánica).
- `reports/memoria/04_o2_markov.md` — notas estructuradas siguiendo
  `reports/memoria/_plantilla.md`. Sin prosa final.

### 10.3 Tag de hito

`v0.2-o2-completo` apuntando al último commit cuando se cumplan los
criterios de aceptación.

### 10.4 Criterios de aceptación

- `uv run pytest -q` verde (29 previos + ~27 nuevos).
- `uv run ruff check src tests` verde.
- `data/processed/o2/` regenerable desde `daily.parquet` en una sola
  llamada a `build_o2(...)`.
- 6 entradas O2 en `reports/INDEX.md` con caption en castellano.
- Notas O2 en `reports/memoria/04_o2_markov.md`.
- Entrada `0007-*` en `reports/ai-log/`.
- Tag `v0.2-o2-completo` creado.
- Markov bate persistencia en al menos top-1 accuracy global. Si no,
  hay un bug; no se cierra el tag hasta diagnosticar.

## 11. Fuera de alcance

Explícitamente fuera de O2 (van en objetivos posteriores):

- Estados ocultos / detección de comportamiento (forraje, descanso,
  vuelo): O3.
- Features derivadas para ML (velocidad, rumbo, distancia diaria,
  contexto ambiental): O4.
- Mapas interactivos folium/plotly: O5.
- Comparación cuantitativa Markov vs ML: O5 (consume
  `predictions_lobo.parquet` y los equivalentes de O3, O4).
- Higher-order Markov (paso > 1 día, contexto multi-día): no
  considerado en el TFG.
- Per-individual transition matrices como modelo: descartado en F2.
- Validación cruzada de α: descartada en F6 (decisión a priori).
