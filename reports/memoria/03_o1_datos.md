# Capítulo 3 — O1: Preparación de datos GPS

> **Estado:** notas
> **Última actualización:** 2026-05-21

## Resumen ejecutivo

Se carga y limpia el dataset GPS de *Larus fuscus* del repositorio
Movebank (89 867 fixes, 126 individuos, 2009–2015), se filtra por
velocidad imposible y se construye una tabla diaria alineada a las
08:00 UTC ±60 min. El resultado son 21 823 filas diarias válidas
sobre 82 individuos, lista como entrada directa para la cadena de
Markov de O2.

## Contexto y motivación

- **Pregunta que responde este bloque:** ¿Cómo transformar la serie GPS
  densa e irregular de cada individuo en una secuencia de posiciones
  diarias homogéneas, sin huecos implícitos, apta para un modelo de
  Markov de primer orden?
- **Conexión con O2:** la tabla diaria es la única entrada de O2; las
  decisiones sobre representación de huecos y granularidad temporal
  condicionan directamente el diseño de la cadena.
- **Conexión con O3–O4:** los fixes limpios (`fixes_clean.parquet`)
  podrán alimentar features en O3 (HMM) y O4 (ML).

## Decisiones tomadas

### D1 — Umbral de velocidad: 120 km/h

- **Alternativas consideradas:** 80, 100, 120, 150 km/h; iteración
  convergente sin cap vs. con cap de iteraciones.
- **Criterio:** la velocidad de crucero de *Larus fuscus* es ~50 km/h;
  el umbral elegido deja margen holgado sobre el máximo observado con
  viento de cola y elimina únicamente artefactos GPS.
- **Evidencia:** `o1_fig03_speed-distribution`
  (`reports/figures/o1_fig03_speed-distribution.png`,
  `reports/tables/o1_tab03_speed-distribution.csv`).
- **Nota de ejecución:** el algoritmo iterativo alcanzó el cap de 20
  iteraciones sin converger (aviso `UserWarning`). Con el umbral de
  120 km/h solo se descartan 20 fixes (0,022 % del total); la
  no-convergencia es irrelevante en la práctica.

### D2 — Hora de referencia: 08:00 UTC

- **Alternativas consideradas:** elegir uno de los cuatro picos
  discretos del muestreo Movebank (05, 08, 14 o 20 UTC).
- **Criterio:** el muestreo de Movebank no es continuo; concentra los
  fixes en cuatro ventanas programadas con volúmenes muy similares
  (~15 500 fixes por pico). Se elige el pico de las **08:00 UTC**
  por tres razones convergentes:
  1. Coincide con un pico real del muestreo (15 591 fixes brutos),
     a diferencia de las 07:00 UTC del análisis preliminar, que era
     un compromiso entre los picos de 05:00 y 08:00.
  2. *Larus fuscus* es una especie diurna; las 08:00 UTC coinciden
     con el inicio del ciclo diario de actividad en gran parte de su
     rango migratorio (Europa septentrional durante la cría, costas
     atlánticas e ibéricas durante la migración). La posición a esa
     hora capta el **lugar de roost nocturno** o el punto
     inmediatamente posterior al despegue matinal — un estado
     espacialmente estacionario, bien definido.
  3. Para una cadena de Markov día-a-día, las transiciones son más
     informativas cuando se observan ambos extremos en el mismo
     punto del ciclo biológico diario. Las 08:00 UTC anclan las
     observaciones al inicio del día biológico del ave, evitando
     posiciones transitorias asociadas a fases activas de vuelo.
- **Evidencia:** `o1_fig04_hourly-coverage`
  (`reports/figures/o1_fig04_hourly-coverage.png`,
  `reports/tables/o1_tab04_hourly-coverage.csv`). Histograma horario
  con los cuatro picos resaltados y la ventana elegida sombreada.

### D3 — Tolerancia: ±60 min

- **Alternativas consideradas:** ±30, ±60, ±90, ±120, ±180 min.
- **Criterio:** la separación entre el pico elegido (08:00) y el
  pico vecino más cercano (05:00) es de 3 horas. Una tolerancia de
  ±60 min recoge el pico de las 08:00 y sus vecinos minoritarios
  (07:00 y 09:00) sin solapar con el pico vecino. Tolerancias
  mayores capturarían fixes de picos distintos, mezclando muestras
  en distintos puntos del ciclo diario. Frente a ±30 min, ±60 min
  recupera la pequeña fracción de fixes que el GPS emitió en las
  horas inmediatamente adyacentes al pico.
- **Evidencia:** `o1_fig05_tolerance-tradeoff`
  (`reports/figures/o1_fig05_tolerance-tradeoff.png`,
  `reports/tables/o1_tab05_tolerance-tradeoff.csv`).

### D4 — Mínimo de días válidos por individuo: 30

- **Alternativas consideradas:** 10, 20, 30, 50 días.
- **Criterio:** la figura `o1_fig06_valid-days-per-bird` muestra la
  distribución de días válidos por individuo; con el umbral de 30 se
  retienen 82 de los 126 individuos (65,1 %), manteniendo suficiente
  diversidad y descartando individuos con tracking demasiado escaso
  para aprender transiciones.
- **Evidencia:** `o1_fig06_valid-days-per-bird`
  (`reports/figures/o1_fig06_valid-days-per-bird.png`,
  `reports/tables/o1_tab06_valid-days-per-bird.csv`).

## Implementación

- **Módulos del paquete:**
  - `src/tfg_aves/data/load.py` — carga y limpieza de fixes
    (`load_fixes`, `drop_invalid_coords`, `drop_speed_outliers`).
  - `src/tfg_aves/data/daily.py` — construcción de la tabla diaria
    (`build_daily`, `coverage_by_hour`).
  - `src/tfg_aves/data/build.py` — orquestador `build_o1` que
    encadena carga → limpieza → resample → filtrado.
- **Algoritmos clave:**
  - Distancia Haversine vectorizada (`_haversine_km`) para calcular
    velocidades entre fixes consecutivos.
  - Selección del fix más cercano a la hora de referencia dentro de
    la ventana de tolerancia (no interpolación).
  - Huecos representados explícitamente como filas con
    `lat = NaN, lon = NaN` (no eliminados).
- **Parámetros finales:**
  `max_speed_kmh=120.0`, `reference_hour_utc=8`,
  `tolerance_min=60`, `min_valid_days=30`.

## Resultados y validación

### Métricas de `build_o1`

| Métrica | Valor |
|---|---|
| Fixes iniciales | 89 867 |
| Descartados (Movebank flag) | 0 |
| Descartados (visible=False) | 0 |
| Descartados (coords inválidas) | 0 |
| Descartados (duplicados) | 0 |
| Descartados (velocidad > 120 km/h) | 20 |
| **Fixes limpios** | **89 847** |
| Aves iniciales | 126 |
| **Aves conservadas (≥ 30 días)** | **82** |
| **Filas diarias totales** (incluye NaN-gap) | **24 444** |
| **Filas diarias válidas** (con fix real) | **21 823** |

### Caracterización del dataset original (C1)

Fuente: `o1_tab01_dataset-overview`
(`reports/tables/o1_tab01_dataset-overview.csv`).

- 89 867 fixes, 126 individuos, rango 2009-05-25 – 2015-08-27 UTC.
- Mediana de fixes por ave: 262; p10: 26; p90: 1 533.
  La distribución es muy asimétrica: hay aves densamente seguidas y
  aves con tracking esporádico.

### Distribución del Δt nativo (C2)

Fuente: `o1_fig02_fix-interval-distribution`
(`reports/figures/o1_fig02_fix-interval-distribution.png`).

El muestreo nativo de Movebank es irregular. La mayoría de intervalos
son cortos (moda ~1 h), pero hay huecos de varios días que hacen
inviable un modelo de tiempo continuo. Este análisis justifica la
elección de la abstracción de "posición diaria".

### Visión geográfica (C4)

Fuente: `o1_fig08_spatial-overview`
(`reports/figures/o1_fig08_spatial-overview.png`).

Los fixes cubren rutas de migración desde el norte de Europa hasta el
África subsahariana y costas atlánticas. La distribución espacial
confirma que el dataset es representativo de la migración completa de
la especie.

### Datos descartados (C3)

Fuente: `o1_tab07_discard-breakdown`
(`reports/tables/o1_tab07_discard-breakdown.csv`).

El único filtro activo es velocidad > 120 km/h: se descartan 20 fixes
(0,022 % del total). No hay fixes con flags de Movebank, coordenadas
inválidas ni duplicados en este dataset.

A nivel de individuo, se descartan 44 aves (126 − 82) por tener menos
de 30 días válidos tras el resample. Representan el 34,9 % de los
individuos pero aportan escasa información de transición.

### Fragmentación de las series diarias (C5)

Fuente: `o1_fig09_streak-length-distribution`
(`reports/figures/o1_fig09_streak-length-distribution.png`,
`reports/tables/o1_tab09_streak-length-distribution.csv`).

| Percentil | Longitud de racha (días) |
|---|---|
| p10 | 1 |
| p25 | 3 |
| p50 | 12 |
| p75 | 61 |
| p90 | 142 |
| p95 | 261 |
| máx | 1 330 |

La mediana es 12 días y el p90 es 142 días. La distribución es muy
asimétrica: la mayoría de rachas son cortas (huecos frecuentes), pero
existe un subconjunto de aves con seguimiento casi continuo de meses.
Para O2, esto significa que la cadena de Markov dispondrá de muchas
transiciones de un solo paso pero pocas rachas largas en la mayor
parte de la muestra. El diseño de la cadena deberá considerar si
entrenar sobre todas las rachas o sólo sobre las de longitud mínima.

### Distribución temporal del seguimiento (C6)

Fuente: `o1_fig10_monthly-seasonal-coverage`
(`reports/figures/o1_fig10_monthly-seasonal-coverage.png`,
`reports/tables/o1_tab10_monthly-seasonal-coverage.csv`).

Distribución agregada de las filas diarias válidas por mes y por estación
del hemisferio norte. Permite ver el sesgo del seguimiento Movebank en el
ciclo anual: qué meses concentran más observaciones y qué fases
migratorias o de permanencia estacional están mejor representadas. Es un
contexto necesario para interpretar las transiciones de la cadena de
Markov, dado que las transiciones se aprenden con la representatividad
estacional que muestre este artefacto.

### Línea de tiempo de un individuo (C7)

Fuente: `o1_fig11_timeline-91916a`
(`reports/figures/o1_fig11_timeline-91916a.png`,
`reports/tables/o1_tab11_timeline-91916a.csv`).

Línea de tiempo del individuo 91916A como ejemplo de tracking
multi-anual (2009–2015). Permite visualizar la alternancia entre rachas
de continuidad y huecos. Es contexto cualitativo de la heterogeneidad
del dataset y justifica la decisión de representar los huecos de forma
explícita en lugar de elidirlos.

### Salidas materializadas

- `data/processed/daily.parquet` — 24 444 filas × 82 aves.
- `data/processed/fixes_clean.parquet` — 89 847 fixes.
- Ambos ficheros son reproducibles ejecutando:

```python
from tfg_aves.data import build_o1
build_o1(max_speed_kmh=120.0, reference_hour_utc=8,
         tolerance_min=60, min_valid_days=30)
```

### Validación

- 29 tests pasan (`uv run pytest -q`).
- Ruff limpio (`uv run ruff check src tests`).
- Los valores del dict de métricas son coherentes con las tablas del
  INDEX generadas en el EDA.

## Conclusiones y limitaciones

**Lo que funciona bien:**
- La tabla diaria está lista como entrada para O2 (Markov). Las cuatro
  decisiones (D1–D4) están validadas con figuras y son reproducibles.
- Los filtros de limpieza son mínimamente invasivos: se elimina
  <0,03 % de los fixes por calidad de señal.

**Limitaciones detectadas:**
1. **Sesgo temporal de la posición diaria al inicio de actividad.**
   Las 08:00 UTC se alinean con el inicio del ciclo diario de
   actividad de *Larus fuscus*. La "posición diaria" representa el
   lugar de roost nocturno o el punto inmediatamente posterior al
   despegue, no el centroide de actividad ni una posición durante
   el vuelo. Esta elección es coherente con la biología de la
   especie y proporciona un anclaje estable para la cadena de
   Markov, pero hay que tenerlo presente al interpretar transiciones
   como "lugares donde el ave duerme/despega de un día a otro".
2. **Descarte total de aves con < 30 días.** Se pierden 44 individuos
   (34,9 %); su tracking parcial no se aprovecha. Si O3/O4 demandan
   mayor diversidad de individuos, podría revisarse el umbral.
3. **No convergencia del outlier de velocidad.** El algoritmo iterativo
   alcanza el cap de 20 iteraciones. Con sólo 20 fixes eliminados
   (0,022 %) el impacto es despreciable, pero podría indicar que
   algún par de fixes tiene velocidades marginalmente superiores a
   120 km/h que no se eliminan en iteraciones posteriores. Susceptible
   de revisión si O2/O3 muestran transiciones anómalas.
4. **Fragmentación elevada.** La mediana de racha es 12 días (p50);
   el 50 % de las rachas tienen ≤ 3 días (p25). O2 deberá decidir si
   usa todas las rachas o aplica un umbral de longitud mínima.

**Aspectos abiertos / futuras mejoras:**
- Decisión pendiente para O2: longitud mínima de racha para entrenamiento
  vs. inferencia (flag antes de O3).
- Valorar en O3/O4 si incluir los 43 individuos descartados con una
  representación alternativa (e.g., k-NN temporal en lugar de resample).

## Notas para la redacción final

- Expandir la explicación del algoritmo Haversine y el criterio de
  selección del fix más cercano (con fórmula LaTeX).
- Citar bibliografía de Movebank (Wikelski et al. 2015) y las
  convenciones de formato Movebank.
- Conectar el anclaje de las 08:00 UTC con la biología de *Larus fuscus*
  (literatura sobre ritmos circadianos de gaviotas y roost nocturno).
- Tabla resumen de D1–D4 para el capítulo (una fila por decisión).
- Añadir nota sobre la limitación del cap de iteraciones del outlier
  con cita al código fuente.
