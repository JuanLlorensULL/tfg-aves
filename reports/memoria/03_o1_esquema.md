# Esquema de redacción — Capítulo 3: Preparación de los datos GPS (O1)

> **Estado:** esquema (sin prosa)
> **Última actualización:** 2026-05-25
> **Relación con las notas:** complementa a `03_o1_datos.md` (notas según
> plantilla). Este fichero fija la **estructura de redacción** del capítulo.

## Contexto

Cerrado O5, arranca la redacción final de la memoria. Este documento es el
**esquema** (secciones/subsecciones + resumen de puntos + figuras candidatas)
del capítulo 3 (desarrollo de O1), todavía **sin prosa**. Decisiones de enfoque
acordadas con el autor:

- **Ingeniería de software integrada** en la metodología (detalles algorítmicos
  inline; sin sección de arquitectura separada; solo una nota breve de
  reproducibilidad/tests en §3.5).
- **2 diagramas conceptuales nuevos** (ilustrativos, no "evidencia de decisión")
  además de los 9 artefactos ya en `reports/`.
- **2 tablas nuevas** inspiradas en la iteración previa (v2): selección/
  renombrado de variables (§3.2.2) y embudo del filtrado (§3.5.2).
- **Excluido a propósito** (revisado con el autor): traer la segmentación en
  trayectorias de la v2 como alternativa en §3.4.4.

Estructura elegida: **orientada al flujo de la pipeline** (crudo → limpieza →
tabla diaria → filtrado → caracterización del resultado), que es la del propio
código (`load`/`clean`/`daily`/`build`) y la de las notas existentes.

> Nota LaTeX: el destino es el capítulo de desarrollo de O1. La carpeta
> `latex/secciones/` tiene hoy un único `desarrollo.tex` (placeholder de la
> plantilla); habrá que crear/separar el fichero de O1 al redactar.

---

## Esquema

### 3.1 Introducción y objetivo del capítulo
- **Puntos:** el problema central (serie GPS densa, irregular y multi-individuo
  → secuencia diaria regular con huecos explícitos, apta para Markov/HMM/ML);
  principio rector (limpieza mínimamente invasiva, sin inventar datos, huecos
  nunca implícitos); visión de la pipeline en 4 etapas; conexión con O2 (única
  entrada), O3/O4 (`fixes_clean.parquet` para features).
- **Figuras:** **[NUEVA] Diagrama de la pipeline** `load → clean → daily →
  filter` (TikZ/conceptual).

### 3.2 El conjunto de datos de partida
- **3.2.1 Origen, especie y experimento** — Movebank, *navigation experiments
  in lesser black-backed gulls* (Wikelski et al. 2015); *Larus fuscus* única
  especie; 126 individuos; rango 2009-05-25 – 2015-08-27. (Cita bibliográfica.)
- **3.2.2 Formato Movebank y taxonomía de variables** — columnas del formato
  Movebank estándar presentadas **agrupadas por categoría**:
  *identificación y contexto* (`event-id`, `individual-local-identifier`,
  `sensor-type=gps`), *espacio-temporales* (`timestamp`, `location-long/lat`),
  *calidad y control* (`visible` —con su duplicado `visible.1`—,
  `manually-marked-outlier`) y *ambientales* (`veg_low/high` de ECMWF; `NCEP
  veg` 100 % NaN). Selección y renombrado a snake_case; descarte de covariables
  ambientales y de metadatos administrativos. `bird_id` como strings tipo
  `91732A`.
- **3.2.3 Caracterización inicial** — volumen (89 867 fixes); fixes/ave muy
  asimétrico (mediana 262, p10 26, p90 1 533); muestreo nativo irregular.
  Mención breve de la cobertura espacial amplia (Europa septentrional ↔ África
  occidental), con la figura detallada en §3.6. *Adelanto:* el muestreo no solo
  es irregular sino **discreto en 4 ventanas** → se desarrolla en §3.4.
- **Figuras/tablas:** **[NUEVA] tabla de selección/renombrado** (columna
  Movebank original → nombre v3 → uso; + columnas descartadas y motivo);
  `o1_tab01` (overview del dataset); `o1_fig02` (distribución del Δt nativo).

### 3.3 Limpieza de los fixes GPS
- **3.3.1 Filtros de calidad Movebank y coordenadas** — flags `visible` y
  `manually_marked_outlier`, coordenadas fuera de rango, duplicados
  `(bird_id, timestamp)`. **Hallazgo:** 0 descartes (dataset pre-limpio); el
  conteo con precedencia evita doble contabilización.
- **3.3.2 Filtro de velocidad imposible** — fórmula Haversine (LaTeX, inline);
  velocidad respecto al fix anterior del mismo ave; **eliminación iterativa
  golosa** (solo el "peor" fix por iteración, porque un outlier envenena su
  salto de entrada y de salida); umbral 120 km/h (crucero de *L. fuscus*
  ~50 km/h); cap de 20 iteraciones y nota de no convergencia (20 fixes =
  0,022 %, impacto despreciable).
- **Figuras/tablas:** `o1_fig03` (distribución de velocidad → umbral 120 km/h),
  `o1_tab07` (desglose acumulado de descartes por causa).

### 3.4 Construcción de la tabla diaria
- **3.4.1 La abstracción de "posición diaria"** — por qué una posición por día
  (la irregularidad hace inviable un modelo de tiempo continuo); día calendario
  UTC; una fila por `(bird_id, date_utc)`.
- **3.4.2 Selección de la hora de referencia (08:00 UTC)** — los **4 picos
  discretos** del muestreo (05/08/14/20 UTC, ~15 500 fixes c/u); función
  `coverage_by_hour`; triple criterio convergente (pico de mayor volumen real +
  biología de *L. fuscus* diurna: roost/despegue matinal + transiciones Markov
  más informativas ancladas al mismo punto del ciclo diario). Definición de
  **`delta_minutes`** (distancia en minutos del fix elegido a la hora de
  referencia).
- **3.4.3 Ventana de tolerancia (±60 min)** — trade-off cobertura vs. precisión
  temporal; ±60 recoge el pico de las 08:00 y vecinos minoritarios (07/09) sin
  solapar con el pico vecino (05:00, a 3 h).
- **3.4.4 Selección por cercanía y huecos explícitos** — *nearest-fix* (no
  interpolación); reconstrucción del calendario completo por ave; huecos como
  filas `is_valid=False` con `lat/lon=NaN`; justificado por la heterogeneidad
  del tracking individual.
- **Figuras:** `o1_fig04` (cobertura horaria / 4 picos → 08:00), `o1_fig05`
  (trade-off de tolerancia → ±60 min), `o1_fig11` (timeline de 91916A →
  justifica los huecos explícitos); **[NUEVA] esquema del resample diario**
  (fixes de un día sobre el eje horario, hora de referencia, ventana ±60 min,
  fix elegido y un día-hueco).

### 3.5 Filtrado de individuos y dataset final
- **3.5.1 Umbral de días válidos (30)** — distribución de días válidos por ave;
  126 → 82 (65,1 % retenidos); trade-off diversidad vs. aprendibilidad de
  transiciones.
- **3.5.2 Dataset final, salidas y reproducibilidad** — tabla de métricas de
  `build_o1` (89 867 → 89 847 fixes; 82 aves; 24 444 filas diarias; 21 823
  válidas ~89 %); salidas `daily.parquet` (entregable) + `fixes_clean.parquet`
  (auditoría para O5); *one-liner* de reproducción. **Nota breve de ingeniería
  (integrada aquí):** diseño de funciones puras (solo `build_o1` escribe a
  disco) → testabilidad y reproducibilidad; 29 tests + ruff limpio.
- **Figuras/tablas:** `o1_fig06` (días válidos por ave → umbral 30); **[NUEVA]
  tabla embudo del filtrado** (reducción secuencial: 89 867 fixes → 89 847
  limpios → 24 444 filas diarias → 21 823 válidas; aves 126 → 82) que sustituye
  a la tabla plana de métricas de las notas y complementa a `o1_tab07`.

### 3.6 Caracterización del dataset resultante (entrada para O2)
- **3.6.1 Dominio espacial** — distribución geográfica de los fixes
  supervivientes; verifica el dominio (Europa septentrional ↔ África occidental)
  y la consistencia con las rutas migratorias conocidas de *L. fuscus*.
- **3.6.2 Fragmentación de las series** — longitud de racha consecutiva sin
  huecos (p25=3, p50=12, p90=142, máx 1 330); implicación para O2: muchas
  transiciones de un solo paso, pocas rachas largas → decisión pendiente
  (longitud mínima de racha para entrenamiento).
- **3.6.3 Sesgo estacional del seguimiento** — cobertura mensual/estacional de
  las filas válidas; condiciona la interpretación de las transiciones (qué
  fases migratorias/estacionales están mejor representadas).
- **Figuras/tablas:** `o1_fig08` (visión espacial), `o1_fig09` (longitud de
  racha), `o1_fig10` (cobertura mensual/estacional).

### 3.7 Conclusiones y limitaciones
- **Funciona bien:** limpieza mínimamente invasiva (<0,03 % de fixes); 4
  decisiones (D1–D4) justificadas con figura y reproducibles; tabla diaria lista
  como entrada de O2.
- **Limitaciones:** (1) sesgo temporal de la posición diaria (ancla
  roost/despegue, no centroide de actividad); (2) descarte de 44 aves (34,9 %);
  (3) no convergencia del filtro de velocidad; (4) fragmentación elevada.
- **Abierto para O2–O4:** longitud mínima de racha; revisar el umbral de aves
  si O3/O4 demandan más diversidad (eco del artefacto C5 de O3).

---

## Mapa figura/tabla → sección (resumen rápido)

| Artefacto | Tipo | Sección | Justifica |
|---|---|---|---|
| `o1_tab01_dataset-overview` | tabla | 3.2.3 | Caracterización del crudo |
| `o1_fig02_fix-interval-distribution` | figura | 3.2.3 | Muestreo nativo irregular |
| `o1_fig08_spatial-overview` | figura | 3.6.1 | Dominio espacial |
| `o1_fig03_speed-distribution` | figura | 3.3.2 | Umbral 120 km/h (D1) |
| `o1_tab07_discard-breakdown` | tabla | 3.3 | Descartes por causa |
| `o1_fig04_hourly-coverage` | figura | 3.4.2 | Hora ref. 08:00 (D2) |
| `o1_fig05_tolerance-tradeoff` | figura | 3.4.3 | Tolerancia ±60 min (D3) |
| `o1_fig11_timeline-91916a` | figura | 3.4.4 | Huecos explícitos |
| `o1_fig06_valid-days-per-bird` | figura | 3.5.1 | Mínimo 30 días (D4) |
| `o1_fig09_streak-length-distribution` | figura | 3.6.1 | Fragmentación → O2 |
| `o1_fig10_monthly-seasonal-coverage` | figura | 3.6.2 | Sesgo estacional |
| **[NUEVA] tabla selección/renombrado** | tabla | 3.2.2 | Variables Movebank → v3 |
| **[NUEVA] tabla embudo del filtrado** | tabla | 3.5.2 | Reducción por etapas |
| **[NUEVA] diagrama pipeline** | diagrama | 3.1 | Ilustrativo |
| **[NUEVA] esquema resample diario** | diagrama | 3.4 | Ilustrativo |

Cobertura: **9/9 figuras + tab01 + tab07** colocadas; las tablas pareadas
(`tab02–06`, `tab09–11`) acompañan a su figura. Las 4 decisiones justificadas
con figura (D1–D4) caen en 3.3–3.5.

---

## Notas para la redacción final (recordatorios, de las notas existentes)
- Fórmula Haversine en LaTeX + criterio de selección por cercanía.
- Citas: Movebank / Wikelski et al. 2015; literatura de ritmos circadianos y
  roost nocturno de gaviotas (anclaje 08:00).
- Tabla resumen D1–D4 (una fila por decisión) como cierre de la parte
  metodológica.
- Nota de la limitación del cap de iteraciones con referencia al código.

## Verificación de cobertura editorial
(1) las 9 figuras + 2 tablas independientes están todas asignadas a una sección;
(2) cada una de las 4 decisiones D1–D4 tiene su figura en la sección correcta;
(3) el orden de secciones reproduce el flujo real de `build_o1`
(`load → clean → build_daily → filter_birds_by_validity`).

## Artefactos nuevos pendientes de crear
- [x] Diagrama de la pipeline (§3.1) — `latex/figuras/o1_pipeline.{tex,pdf}`.
- [x] Esquema del resample diario (§3.4) — `latex/figuras/o1_resample_diario.{tex,pdf}`.
- [ ] Tabla de selección/renombrado de variables (§3.2.2).
- [ ] Tabla embudo del filtrado (§3.5.2).

> Diagramas en TikZ `standalone` (vectoriales). Para incluirlos en el capítulo:
> `\includegraphics{figuras/o1_pipeline.pdf}` (o `\input` del `tikzpicture` si el
> documento carga `tikz`). Recompilar: `pdflatex o1_pipeline.tex`. Requieren
> `lmodern` (ya en el preámbulo del standalone).
