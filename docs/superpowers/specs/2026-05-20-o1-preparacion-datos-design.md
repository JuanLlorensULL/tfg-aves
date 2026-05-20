# Diseño de O1 — Preparación de datos GPS

- **Fecha:** 2026-05-20
- **Objetivo del TFG:** O1 (48 h) — preparación de datos GPS
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado, pendiente de implementación

## 1. Resumen

O1 transforma el CSV crudo de Movebank
(`data/raw/migration_original.csv`, 89 867 fixes de 126 individuos de
*Larus fuscus*) en una **tabla diaria por individuo** lista para
alimentar O2 (cadena de Markov día-a-día). El entregable principal es
`data/processed/daily.parquet`: una fila por `(bird_id, date_utc)`
con la posición del ave a una hora UTC de referencia fijada por el
EDA, con huecos explícitos para los días sin fix válido. Se materializa
también `data/processed/fixes_clean.parquet` como tabla de auditoría
para O5.

## 2. Contexto y constraints

- **Decisión del autor que vertebra el diseño:** la predicción es
  diaria. Cada fila del entregable representa la posición del ave un
  día concreto a una hora fija. La granularidad intra-día se descarta
  del modelado.
- **Implicación:** la cadena de Markov de O2 operará sobre transiciones
  día → día (paso 24 h), no fix → fix (paso ~1 h). Por tanto la noción
  de "segmento denso intra-día" desaparece. El análogo a esa escala
  ("racha de días consecutivos válidos") es una *vista* derivable de
  `daily.parquet` con un `groupby` y la deriva O2, no O1.
- **Frecuencia nativa del dataset:** Movebank entrega ~1 fix cada
  30-60 min cuando el GPS estaba activo. Tras O1 la frecuencia efectiva
  para el modelado es 24 h.
- **Política de convenciones del proyecto** (CLAUDE.md): toda decisión
  no trivial requiere figura/tabla + caption en castellano vía
  `save_artifact()`; commits en castellano sin trailer de IA; código
  con identificadores en inglés.

## 3. Decisiones de diseño fijadas

Las que ya están decididas antes de tocar código (sin EDA):

| # | Decisión | Valor |
|---|---|---|
| F1 | Entregable principal | Una única tabla diaria, no tabla de fixes densos |
| F2 | Definición de "día" | Día calendario UTC |
| F3 | Zona horaria | UTC fijo (Movebank entrega en UTC) |
| F4 | Selección del fix diario | Más cercano a hora de referencia, dentro de tolerancia |
| F5 | Hora de referencia | Elegida por el EDA (maximiza cobertura) |
| F6 | Representación de huecos | Explícita: una fila por día calendario, `is_valid=False` cuando no hay fix |
| F7 | Outliers GPS | Flags Movebank + coords inválidas + duplicados + velocidad imposible |
| F8 | Columnas conservadas | `event_id`, `timestamp`, `lon`, `lat`, `bird_id`, `sensor_type`, `visible`, `manually_marked_outlier`. Se descartan las covariables ECMWF/NCEP |
| F9 | Estructura de la pipeline | Notebook único con figuras + módulos consolidados en `src/tfg_aves/data/` |

Las que decide el EDA con figura (se materializan en
`reports/INDEX.md` con `decision=...`):

| # | Parámetro | Slug del artefacto justificativo |
|---|---|---|
| D1 | `max_speed_kmh` (umbral de velocidad para outliers) | `speed-distribution` |
| D2 | `reference_hour_utc` (hora UTC para el fix diario) | `hourly-coverage` |
| D3 | `tolerance_min` (ventana alrededor de la hora de referencia) | `tolerance-tradeoff` |
| D4 | `min_valid_days` (mínimo de días válidos por ave) | `valid-days-per-bird` |

## 4. Arquitectura

Pipeline en cuatro fases ejecutadas desde
`notebooks/01_eda_o1.ipynb` que importa funciones de
`src/tfg_aves/data/`:

```
data/raw/migration_original.csv
              │
              ▼
     (1) load.py       — leer y normalizar
              │
              ▼
     (2) clean.py      — descartar outliers
              │
              ▼
     (3) daily.py      — colapsar a una fila por día
              │
              ▼
     (4) build.py      — orquestar y materializar
              │
              ▼
data/processed/daily.parquet       (entregable principal)
data/processed/fixes_clean.parquet (auditoría para O5)
```

Tras O1, regenerar los parquets se hace con:

```python
from tfg_aves.data import build_o1
build_o1(
    max_speed_kmh=<D1>,
    reference_hour_utc=<D2>,
    tolerance_min=<D3>,
    min_valid_days=<D4>,
)
```

## 5. Módulos

### 5.0 Constantes del paquete

`src/tfg_aves/data/__init__.py` expone tres rutas estables:

```python
RAW_CSV   = ROOT / "data" / "raw"       / "migration_original.csv"
INTERIM   = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
```

`ROOT` se resuelve relativo al fichero `__init__.py`. Las funciones de
los submódulos aceptan estas rutas como argumentos por defecto para
mantenerse parametrizables (tests las sobrescriben con `tmp_path`).

### 5.1 `src/tfg_aves/data/load.py`

```python
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame
```

- Lee el CSV con `pyarrow`.
- Renombra a snake_case: `location-long` → `lon`,
  `location-lat` → `lat`, `individual-local-identifier` → `bird_id`,
  `manually-marked-outlier` → `manually_marked_outlier`,
  `sensor-type` → `sensor_type`, `event-id` → `event_id`.
- Tipa `timestamp` como `datetime64[ns, UTC]`.
- Ordena por `(bird_id, timestamp)`.
- Descarta columnas ambientales ECMWF/NCEP.
- Devuelve el dataset completo sin filtros.

### 5.2 `src/tfg_aves/data/clean.py`

Tres funciones puras encadenables; cada una devuelve
`(DataFrame, dict_de_descartes)` para trazabilidad:

```python
def drop_movebank_flags(df) -> tuple[pd.DataFrame, dict]
def drop_invalid_coords_and_dupes(df) -> tuple[pd.DataFrame, dict]
def drop_speed_outliers(df, max_speed_kmh: float) -> tuple[pd.DataFrame, dict]
```

- `drop_movebank_flags`: descarta filas con `visible=False` o
  `manually_marked_outlier=True`.
- `drop_invalid_coords_and_dupes`: descarta filas con
  `lat ∉ [-90, 90]` o `lon ∉ [-180, 180]`, y deduplica por
  `(bird_id, timestamp)` conservando la primera ocurrencia.
- `drop_speed_outliers`: calcula la velocidad haversine entre fixes
  consecutivos del mismo `bird_id` y descarta fixes con
  `v > max_speed_kmh`. Iterativo simple — descartar uno y recomputar —
  porque un fix outlier "envenena" su salto de entrada y el de salida y
  no queremos descartar al inocente. Cap a un máximo de iteraciones
  para evitar bucles patológicos. El `max_speed_kmh` lo fija el EDA;
  el módulo no asume valor.

### 5.3 `src/tfg_aves/data/daily.py`

```python
def coverage_by_hour(df, tolerance_min: float) -> pd.DataFrame
def pick_reference_hour(df, tolerance_min: float) -> tuple[int, pd.DataFrame]
def build_daily(
    df,
    reference_hour_utc: int,
    tolerance_min: float,
) -> pd.DataFrame
def filter_birds_by_validity(df_daily, min_valid_days: int) -> pd.DataFrame
```

- `coverage_by_hour`: para cada `h ∈ {0..23}`, calcula el porcentaje
  de pares `(bird_id, date_utc)` con al menos un fix en
  `[h − tolerance, h + tolerance]`. Devuelve tabla `hour × coverage`.
- `pick_reference_hour`: aplica `coverage_by_hour` y devuelve la hora
  `argmax(coverage)` junto con la tabla completa, que se guarda como
  figura D2.
- `build_daily`: el corazón del módulo. Para cada `bird_id`, expande
  el rango calendario `[first_date, last_date]` día a día; para cada
  día selecciona el fix más cercano a `reference_hour_utc` dentro de
  `±tolerance_min`; deja `lat`/`lon` a `NaN` si no hay ninguno.
  Devuelve la tabla con huecos explícitos.
- `filter_birds_by_validity`: descarta individuos cuyo
  `count(is_valid) < min_valid_days`. Umbral fijado por el EDA.

### 5.4 `src/tfg_aves/data/build.py`

```python
def build_o1(
    *,
    max_speed_kmh: float,
    reference_hour_utc: int,
    tolerance_min: float,
    min_valid_days: int,
    out_dir: Path = PROCESSED,
) -> dict
```

- Compone `load → clean → build_daily → filter_birds_by_validity`.
- Escribe `data/processed/daily.parquet` y
  `data/processed/fixes_clean.parquet`.
- Devuelve un dict de métricas con claves fijas:
  `n_initial`, `discarded_movebank_flags`, `discarded_invalid_coords`,
  `discarded_duplicates`, `discarded_speed`, `n_fixes_clean`,
  `n_birds_initial`, `n_birds_kept`, `n_daily_rows`, `n_valid_rows`.
  Lo usa el notebook para una tabla resumen en el INDEX.

Principios que vertebran el diseño:

- Umbrales como argumentos, nunca constantes globales.
- Funciones puras devolviendo DataFrames; sólo `build.py` escribe a
  disco.
- Trazabilidad de descartes: cada paso emite un dict con conteos.

## 6. Esquema de datos

### 6.1 `data/processed/daily.parquet` (entregable principal)

| Columna | Tipo | Nullable | Descripción |
|---|---|---|---|
| `bird_id` | string | no | identificador del individuo |
| `date_utc` | date32 | no | día calendario UTC |
| `lat` | float64 | sí | latitud del fix elegido |
| `lon` | float64 | sí | longitud del fix elegido |
| `is_valid` | bool | no | `True` ⇔ `lat` y `lon` no NaN |
| `source_event_id` | int64 | sí | `event-id` original del fix elegido |
| `delta_minutes` | float64 | sí | distancia en minutos del fix elegido a la hora de referencia |

Granularidad: una fila por `(bird_id, date_utc)` en el rango
`[first_valid_date, last_valid_date]` de cada ave superviviente al
filtro `min_valid_days`. Huecos explícitos.

### 6.2 `data/processed/fixes_clean.parquet` (auditoría)

Fixes post-cleaning sin colapsar a día — mismas columnas que la salida
de `load.py` pero filtradas. Consumidor previsto: O5 (mapas detallados,
análisis del error).

## 7. Decisiones justificadas (artefactos)

O1 genera dos tipos de artefactos en `reports/`. Todos vía
`save_artifact()` con `objective="o1"` y caption en castellano.

### 7.1 Decisiones que fijan parámetros del pipeline

| Slug | Decisión | Forma del artefacto |
|---|---|---|
| `speed-distribution` (D1) | `max_speed_kmh` | Histograma de velocidades entre fixes consecutivos + percentiles 95 / 99 / 99.9 + tabla con conteo de descartes por umbral candidato |
| `hourly-coverage` (D2) | `reference_hour_utc` | Curva de cobertura `% días-individuo con fix ∈ [h ± tol]` para `h ∈ {0..23}`, comparando tolerancias ±60 / ±120 / ±180 min |
| `tolerance-tradeoff` (D3) | `tolerance_min` | Tabla `tolerance × cobertura × delta_medio_min` a la hora elegida en D2 |
| `valid-days-per-bird` (D4) | `min_valid_days` | Histograma de días válidos por ave + tabla del corte (aves conservadas vs descartadas por umbral) |

### 7.2 Caracterización del dataset (contexto, sin `decision`)

| Slug | Contenido |
|---|---|
| `dataset-overview` (C1) | Tabla resumen: `n_fixes`, `n_birds`, rango temporal, fixes/ave (mediana, p10, p90) |
| `fix-interval-distribution` (C2) | Histograma de Δt entre fixes consecutivos del mismo individuo |
| `discard-breakdown` (C3) | Tabla de descartes acumulados: flags Movebank, coords inválidas, duplicados, velocidad — número y % sobre el total inicial |
| `spatial-overview` (C4) | Mapa con los fixes limpios — contexto geográfico |
| `streak-length-distribution` (C5) | Histograma de longitud de rachas de días consecutivos válidos en `daily.parquet` |

**Total esperado en `INDEX.md`:** 4 decisiones + 5 artefactos de
contexto = 9 entradas O1.

### 7.3 Decisiones triviales sin figura

Se mencionan en las notas de memoria pero no generan artefacto:

- Descarte de columnas ambientales ECMWF/NCEP (política previa).
- Filtros `visible=false`, `manually_marked_outlier=true`, coords
  fuera de rango, duplicados — descartes de cajón. Su conteo aparece
  agregado en C3.

## 8. Tests

Convención: un fichero por módulo en `tests/test_data_<nombre>.py`,
fixtures vivos en `tests/conftest.py`. Datasets sintéticos pequeños y
deterministas; el dataset real (~90 k filas) no entra en los tests,
sólo en el notebook.

### 8.1 `tests/test_data_load.py` (1 test)

- Lee un mini-CSV (≤10 filas, mismas columnas que Movebank) y
  verifica: nombres en snake_case, `timestamp` tipado UTC, orden
  `(bird_id, timestamp)`, columnas ambientales ausentes.

### 8.2 `tests/test_data_clean.py` (3 tests)

- `drop_movebank_flags`: dataset con 4 filas (1 `visible=False`, 1
  `manually_marked_outlier=True`, 2 limpias) → quedan 2 y el dict
  reporta los descartes correctos.
- `drop_invalid_coords_and_dupes`: filas con `lat=95`, `lon=181` y un
  duplicado por `(bird_id, timestamp)` → sobreviven las correctas y
  el dict reporta cada causa.
- `drop_speed_outliers`: secuencia de 5 fixes con uno a 1000 km del
  resto → ese fix se descarta y el resto sobrevive. Caso adicional:
  sin outliers → dataset intacto y 0 descartes.

### 8.3 `tests/test_data_daily.py` (6 tests)

- `coverage_by_hour`: fixes concentrados en `h=12` → la fila `h=12`
  tiene cobertura máxima.
- `pick_reference_hour`: dataset construido para que `h=14` sea
  óptima → la función devuelve `14`.
- `build_daily — selección correcta`: ave con fixes a las 11:30,
  12:10 y 13:00 y `reference_hour_utc=12` → se elige el de 12:10;
  `source_event_id` y `delta_minutes=10` cuadran.
- `build_daily — huecos explícitos`: ave con fixes los días 1 y 3 pero
  no el 2 → la tabla tiene 3 filas, la del día 2 con `is_valid=False`
  y lat/lon NaN.
- `build_daily — tolerancia`: fix a 200 min de la hora de referencia
  con `tolerance_min=120` → fila inválida. Con `tolerance_min=300` →
  fila válida.
- `filter_birds_by_validity`: dos aves, una con 3 días válidos y otra
  con 1 día; `min_valid_days=2` → sobrevive sólo la primera.

### 8.4 `tests/test_data_build.py` (1 test de integración)

- `build_o1(...)` con mini-CSV en `tmp_path`: verifica que existen
  `daily.parquet` y `fixes_clean.parquet`, el dict de métricas tiene
  las claves esperadas, y `daily.parquet` se relee sin errores
  respetando el esquema de la sección 6.

**Total:** 11 tests nuevos en `tests/` (sobre los 16 actuales): 1 de
`load`, 3 de `clean`, 6 de `daily`, 1 de integración.

### 8.5 Lo que NO se testea

- Los valores concretos de los umbrales (D1–D4): son decisiones
  humanas justificadas con figura, no aserciones de código.
- `save_artifact()`: ya cubierto por `tests/test_reporting.py`.
- El dataset real: validado por la ejecución del notebook y la
  revisión visual de las figuras.

## 9. Cierre de O1

### 9.1 Granularidad de commits

Todos en castellano, sin trailer Co-Authored-By:

1. `Esqueleto de tfg_aves.data: load, clean, daily, build`
2. `Tests unitarios de carga y limpieza`
3. `Implementar load + clean con tests`
4. `Tests unitarios de resample diario`
5. `Implementar resample diario con tests`
6. `Notebook EDA: cobertura, velocidades, días por individuo`
7. `Orquestación build_o1 + test de integración`
8. `Ejecutar build_o1 con umbrales finales y materializar parquets`

### 9.2 Artefactos generados

- `reports/figures/o1_fig*.png`, `reports/tables/o1_tab*.csv`,
  `reports/captions/o1_*.md` — versionados.
- `reports/INDEX.md` — actualizado por `save_artifact()` a 9 entradas
  O1.
- `reports/ai-log/0006-o1-eda-y-pipeline-datos.md` — una entrada para
  toda la fase, no por commit. Tono según la política
  (autor decide; IA propone y ejecuta mecánica).
- `reports/memoria/03_o1_datos.md` — notas estructuradas siguiendo
  `reports/memoria/_plantilla.md`. Sin prosa final.

### 9.3 Tag de hito

`v0.1-o1-completo` apuntando al último commit cuando se cumplan los
criterios de aceptación.

### 9.4 Criterios de aceptación

- `uv run pytest -q` verde (16 previos + 11 nuevos).
- `uv run ruff check src tests` verde.
- `data/processed/daily.parquet` y `fixes_clean.parquet` regenerables
  desde el CSV crudo en una sola llamada a `build_o1(...)`.
- 9 entradas O1 en `reports/INDEX.md` con caption en castellano.
- Notas O1 en `reports/memoria/03_o1_datos.md`.
- Entrada `0006-*` en `reports/ai-log/`.
- Tag `v0.1-o1-completo` creado.

## 10. Fuera de alcance

Explícitamente fuera de O1 (van en objetivos posteriores):

- Cualquier feature derivada (velocidad, rumbo, distancia diaria) —
  las calcula O2/O3 a partir de `daily.parquet`.
- Discretización del espacio en celdas / clusters — O2.
- Construcción de "rachas de días consecutivos válidos" como tabla
  materializada — O2 la deriva si la necesita.
- Decisión global vs. por-individuo vs. muestra representativa — se
  resuelve antes de O3/O4 (marcado como decisión abierta en
  CLAUDE.md).
- Imputación de huecos diarios — O2 decide si los rellena, los
  ignora o los modela explícitamente.
