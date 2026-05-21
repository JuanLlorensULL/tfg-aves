# Capítulo 4 — O2: Predicción con cadenas de Markov visibles

> **Estado:** notas
> **Última actualización:** 2026-05-22

## Resumen ejecutivo

Se entrena un modelo de Markov de primer orden con 12 matrices de
transición mensuales sobre el dataset diario de posiciones de
*Larus fuscus* (82 aves, 21 823 días válidos), usando el espacio
discretizado en celdas de 0,5° × 0,5°. El modelo se evalúa con
leave-one-bird-out (LOBO) frente a un baseline de persistencia. El
resultado principal es que Markov **gana en log-loss** (modelo mejor
calibrado) pero **pierde en top-1** frente a la baseline trivial, un
hallazgo estructural del dataset que motiva el uso de estados ocultos
(O3) y features per-individual (O4).

## Contexto y motivación

- **Pregunta concreta:** ¿puede un Markov(1) global con estacionalidad
  mensual batir a la baseline trivial "mañana = hoy" en los datos de
  *Larus fuscus*?
- **Rol en el TFG:** baseline interpretable que los modelos de O3 y O4
  deberán mejorar. Las predicciones LOBO almacenadas en
  `data/processed/o2/predictions_lobo.parquet` las consume O5 para
  mapas y análisis comparativo del error.
- **Conexión con O1:** input directo del entregable `daily.parquet`
  (una posición por día, huecos explícitos).
- **Conexión con O3/O4:** el hallazgo de la alta residencialidad y las
  rutas individuales motiva la introducción de estados ocultos (HMM)
  y de features per-individual (ML supervisado).

## Decisiones tomadas

1. **D1 — Tamaño de celda: `cell_deg = 0.5°`**
   - Alternativas consideradas: 0,25°, 0,5°, 1°, 2°.
   - Criterio: balance entre resolución y densidad de transiciones por
     celda. A 0,5° se obtienen 1 217 celdas activas con un número de
     transiciones por mes suficiente para que el suavizado Laplace no
     domine completamente.
   - Evidencia: `reports/figures/o2_fig01_grid-size-tradeoff.png`,
     `reports/tables/o2_tab01_grid-size-tradeoff.csv`.

2. **F1 — Rol como baseline interpretable** (fijada antes del EDA):
   modelo global sin features per-individual; ML de O4 debe batirlo.

3. **F2 — Alcance global** (fijado antes del EDA):
   una única familia de matrices (12) entrenada sobre las 82 aves
   agregadas. Per-individual queda como ablación de evaluación.

4. **F3 — 12 matrices mensuales** (fijado antes del EDA):
   granularidad mensual captura el ciclo migratorio sin fragmentar el
   dataset por debajo del umbral estadístico mínimo.

5. **F5 — Saltar transición ante hueco** (fijado antes del EDA):
   semántica estricta de Markov(1) a paso 1 día; no se cuentan pares
   separados por más de 1 día.

6. **F6 — Suavizado Laplace α = 1** (fijado antes del EDA):
   valor estándar add-one; garantiza distribución válida para celdas
   sin observación y log-loss finito.

7. **F7 — LOBO (leave-one-bird-out)** (fijado antes del EDA):
   el sesgo temporal del dataset (82 aves en 2009, 1 en 2014-2015)
   descarta splits por año; LOBO evalúa generalización real.

8. **Criterio de aceptación revisado — log-loss** (decisión del autor
   tras el diagnóstico del EDA): el criterio original "Markov bate
   persistencia en top-1" se sustituyó por "Markov bate persistencia
   en log-loss" al comprobar que el top-1 es estructuralmente dominado
   por la alta residencialidad del dataset (73% self-loops) y no mide
   calibración probabilística. Ver sección "Hallazgo principal" abajo.

## Implementación

- **Módulos del paquete:** `src/tfg_aves/markov/` con submódulos
  `discretize`, `transition`, `smooth`, `predict`, `evaluate`, `build`.
- **Algoritmos clave:**
  - Discretización: `floor(coord / cell_deg)` → índice entero.
  - Conteo de transiciones: agregación con `numpy.add.at` en tensor
    `(12, n_cells, n_cells)`.
  - Suavizado: Laplace add-α por fila en cada mes.
  - Predicción: fila `P[month_int][cell_from_idx]`; fallback a
    distribución marginal del mes si `cell_from` no está en el espacio
    del fold de entrenamiento.
  - Distancia: Haversine entre centroide del top-1 y posición real.
- **Orquestador:** `build_o2(cell_deg=0.5, alpha=1.0)` en
  `src/tfg_aves/markov/build.py` — escribe los 5 ficheros de
  `data/processed/o2/`.
- **Parámetros finales:** `cell_deg=0.5`, `alpha=1.0`.

## Resultados y validación

### Métricas globales (LOBO, 82 folds)

| Métrica | Markov | Persistencia | Ganador |
|---|---|---|---|
| log-loss | **5,4509** | 5,5857 | **Markov** |
| top-1 accuracy | 0,4258 | **0,7305** | Persistencia |
| distancia mediana (km) | — | — | (ver C5) |

Markov bate persistencia en log-loss: es un modelo probabilístico
mejor calibrado. Pierde en top-1 por razones estructurales del dataset
(ver hallazgo principal abajo).

### Hallazgo principal — Alta residencialidad y rutas individuales

El dataset de *Larus fuscus* presenta **73% de self-loops** en las
transiciones día a día: los individuos permanecen frecuentemente en la
misma celda de 0,5°. Esto eleva la top-1 de la baseline de persistencia
hasta 0,7305 por construcción. Bajo LOBO, **el 43% de las predicciones
del modelo Markov caen en celdas no observadas en las 81 aves del fold
de entrenamiento** porque cada individuo sigue rutas migratorias
específicas; en esos casos el modelo cae al fallback de la distribución
marginal y predice celdas alejadas de la posición real, lo que castiga
severamente top-1 pero no log-loss (que mide la calibración global de
la distribución completa, no el argmax).

**Conclusión metodológica:** top-1 accuracy no es la métrica adecuada
para evaluar un Markov global LOBO en datos con alta residencialidad y
rutas individuales. Log-loss y distancia mediana km son métricas más
informativas para este modelo en este dataset. Este resultado se
documenta en la memoria como hallazgo honesto y transparente, no como
fallo del modelo.

### Caracterización de artefactos

- **C1 — Matriz ejemplo** (`o2_fig02_transition-matrix-example`):
  heatmap del mes de octubre (mayor número de transiciones). Diagonal
  visible (self-loops); off-diagonal dispersa (rutas migratorias).
- **C2 — 12 matrices mensuales** (`o2_fig03_monthly-matrices-overview`):
  grid 3×4 con los 12 meses. Lectura cualitativa del ciclo estacional:
  mayor dispersión fuera de la diagonal en meses de migración
  (primavera/otoño).
- **C3 — Accuracy vs. baseline** (`o2_fig04_accuracy-vs-baseline`):
  comparación Markov vs. persistencia en top-1, distancia mediana y
  log-loss, desglosada por mes.
- **C4 — Rendimiento por individuo** (`o2_fig05_per-bird-performance`):
  distribución por ave de top-1 y distancia mediana km; alta
  heterogeneidad individual confirma el valor de O4 per-individual.
- **C5 — Tabla resumen** (`o2_tab06_summary`): una fila por mes con
  métricas Markov y delta vs. persistencia.

### Validación

- 56 tests pasando (29 de O1 + 27 de O2). `uv run pytest -q` verde.
- `uv run ruff check src tests` verde.
- `build_o2(cell_deg=0.5)` reproducible desde `daily.parquet` en una
  sola llamada.

## Salida materializada

Ficheros en `data/processed/o2/` (gitignored, regenerables):

| Fichero | Dimensiones |
|---|---|
| `cells.parquet` | 1 217 filas (celdas activas) |
| `transitions_counts.npz` | `(12, 1217, 1217)` int32 |
| `transition_matrices.npz` | `(12, 1217, 1217)` float64 |
| `predictions_lobo.parquet` | ~44 000 filas (markov + persistence) |
| `metrics.parquet` | scope × model × agrupación |

## Conclusiones y limitaciones

### Lo que funciona bien

- Las 12 matrices mensuales capturan la estacionalidad del ciclo
  migratorio de forma interpretable.
- El modelo es reproducible, eficiente y sirve como baseline
  cuantificado para O3 y O4.
- En log-loss, Markov demuestra ser un modelo probabilístico mejor
  calibrado que la persistencia, lo cual es el objetivo teórico de
  un modelo Markov bien suavizado.

### Limitaciones detectadas

- **Alta sparsity:** 1 217 celdas × 0,19% pares observados por mes.
  La mayor parte de las predicciones descansa sobre el suavizado
  Laplace o la marginal del mes.
- **Supuesto de estacionalidad estable entre años:** las matrices
  mezclan 2009-2015, asumiendo que el comportamiento migratorio es
  reproducible año a año. No hay datos suficientes para contrastar.
- **Sesgo temporal del dataset:** 82 aves activas en 2009, ~10 en
  2011, 1 en 2014-2015. Las matrices están dominadas por 2009.
- **Alcance global:** el modelo ignora la heterogeneidad individual
  (rutas propias de cada ave). Este es el principal impulsor del
  castigo en top-1 bajo LOBO.

### Aspectos abiertos / futuras mejoras

- O3 (HMM): introducir estados ocultos "residente" vs. "en
  migración" puede resolver la heterogeneidad del comportamiento.
- O4 (ML): features per-individual (posición anterior, historial
  reciente, fecha) deberían mejorar top-1 significativamente sobre
  el modelo global.
- O5: los `predictions_lobo.parquet` de O2 se comparán
  cuantitativamente con los equivalentes de O3 y O4.

## Notas para la redacción final

- Citar `o2_fig01_grid-size-tradeoff` y `o2_tab01` en la justificación
  de la discretización.
- Dedicar un párrafo al hallazgo de la residencialidad y las rutas
  individuales — es el resultado más relevante de O2 para el hilo
  argumental del TFG.
- Conectar explícitamente con las motivaciones de O3 (HMM) y O4 (ML):
  el fallo de top-1 del Markov global no es un bug, es la evidencia
  de que se necesita modelar el comportamiento individual.
- Revisar si las métricas de distancia mediana km están en C5 con
  valores concretos para citar en el texto.
- Bibliografía pendiente: referencia estándar para cadenas de Markov
  finitas y suavizado Laplace (capítulo 2 de Bishop, o similar).
