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
  asimétrico (mediana 262, p10 26, p90 1 533) → anticipa el filtro por individuo.
  (El análisis del muestreo horario y los 4 picos se trata en §3.4; la cobertura
  espacial, en §3.6. `o1_fig02` retirada por el autor 2026-05-25 por redundante
  con `o1_fig04`.)
- **Figuras/tablas:** **[NUEVA] tabla de selección/renombrado** (columna
  Movebank original → nombre v3 → uso; + columnas descartadas y motivo);
  `o1_tab01` (overview del dataset).

### 3.3 Limpieza de los fixes GPS (sin subsecciones)
- **Comprobaciones de calidad y coordenadas, de pasada** — flags `visible` /
  `manually_marked_outlier`, rango de coordenadas y duplicados
  `(bird_id, timestamp)`. **0 descartes** → el conjunto venía ya algo depurado
  desde el origen. Mención breve; **sin tabla de desglose** (decisión del autor
  2026-05-25: `o1_tab07` ya no se incluye en la memoria).
- **Filtro de velocidad (cuerpo de la sección)** — fórmula Haversine (ecuación
  numerada); velocidad respecto al fix anterior del mismo ave; eliminación
  iterativa golosa (solo el "peor" fix por iteración, porque un outlier contamina
  sus dos segmentos); umbral 120 km/h fundamentado con percentiles reales
  (mediana 0,66; p99,5 = 50,6 ≈ crucero; p99,9 = 134); no convergencia (tope 20
  iter, 20 fixes = 0,022 %, marginal), declarada sin tono apologético.
- **Figuras:** `o1_fig03` (distribución de velocidad → umbral 120 km/h).

### 3.4 Construcción de la tabla diaria
> Revisado 2026-05-25: **eliminados** el diagrama del resample y la subsección de
> tolerancia; la tolerancia pasa a mención de pasada. Hora de referencia
> reformulada (volumen NO discrimina entre picos; decide la biología).
- **Opening (1 frase)** — una posición por día (UTC): fix más cercano a la hora
  de referencia dentro de tolerancia; días sin fix próximo = huecos. (Sin
  diagrama de resample.)
- **3.4.1 Hora de referencia (08:00 UTC)** — la hora ha de **coincidir con un
  pico de muestreo** (fuera de ellos casi no hay fixes → cobertura). Los 4 picos
  (05/08/14/20 UTC) tienen **volumen casi idéntico (15 466–15 656 fixes)**, así
  que el volumen NO decide; la elección concreta de las 08:00 es **biológica**
  (inicio de actividad de *L. fuscus* → roost/post-despegue, posición estable) +
  anclaje para Markov. Figura `o1_fig04`.
- **3.4.2 Selección por cercanía y huecos explícitos** — *nearest-fix* por
  `delta_minutes` (no interpolación, principio de no inventar datos); **tolerancia
  ±60 mencionada de pasada** (sin subsección ni figura): holgada para no
  fragmentar de más (94,6 % con ±60 vs 66 % con ±30 estricta; no más de ±60 para
  no tocar el pico vecino de 05:00, a 3 h). Reconstrucción del
  calendario + huecos explícitos (`is_valid=False`, NaN); figura `o1_fig11`.
- **Figuras:** `o1_fig04` (cobertura horaria → 08:00), `o1_fig11` (timeline
  91916A → huecos). **Retiradas:** el diagrama del resample y `o1_fig05`
  (tolerancia, ya no se incluye).

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
  huecos. **Datos reales (`tab09`): p25=3, p50=8, p90=96, máx 1326** (las
  notas/CLAUDE.md decían 12/142/1330 — desactualizadas; usar el artefacto).
  Implicación para O2: muchas transiciones de un solo paso, pocas rachas largas
  → decisión pendiente (longitud mínima de racha para entrenamiento).
- **3.6.3 Sesgo estacional del seguimiento** — cobertura mensual/estacional de
  las filas válidas; condiciona la interpretación de las transiciones (qué
  fases migratorias/estacionales están mejor representadas).
- **Figuras/tablas:** `o1_fig08` (visión espacial), `o1_fig09` (longitud de
  racha), `o1_fig10` (cobertura mensual/estacional).

### 3.7 Conclusiones y limitaciones — ELIMINADA (2026-05-25)
> El autor retiró la sección de conclusiones del capítulo: repetía lo ya dicho
> en §3.1–§3.6 sin aportar. Las conclusiones globales irán en el capítulo 9
> (`conclusiones.tex`). Las limitaciones relevantes quedan ya integradas en su
> sección (anclaje 08:00 en §3.4; descarte de 44 aves en §3.5; fragmentación en
> §3.6). El capítulo O1 termina en §3.6.

---

## Mapa figura/tabla → sección (resumen rápido)

| Artefacto | Tipo | Sección | Justifica |
|---|---|---|---|
| `o1_tab01_dataset-overview` | tabla | 3.2.3 | Caracterización del crudo |
| `o1_fig02_fix-interval-distribution` | figura | — | Retirada: redundante con o1_fig04 |
| `o1_fig08_spatial-overview` | figura | 3.6.1 | Dominio espacial |
| `o1_fig03_speed-distribution` | figura | 3.3.2 | Umbral 120 km/h (D1) |
| `o1_tab07_discard-breakdown` | tabla | — | Retirada: mención de pasada, sin tabla |
| `o1_fig04_hourly-coverage` | figura | 3.4.2 | Hora ref. 08:00 (D2) |
| `o1_fig05_tolerance-tradeoff` | figura | — | Retirada: tolerancia mencionada de pasada |
| `o1_fig11_timeline-91916a` | figura | 3.4.4 | Huecos explícitos |
| `o1_fig06_valid-days-per-bird` | figura | 3.5.1 | Mínimo 30 días (D4) |
| `o1_fig09_streak-length-distribution` | figura | 3.6.1 | Fragmentación → O2 |
| `o1_fig10_monthly-seasonal-coverage` | figura | 3.6.2 | Sesgo estacional |
| **[NUEVA] tabla selección/renombrado** | tabla | 3.2.2 | Variables Movebank → v3 |
| **[NUEVA] tabla embudo del filtrado** | tabla | 3.5.2 | Reducción por etapas |
| **[NUEVA] diagrama pipeline** | diagrama | 3.1 | Ilustrativo |
| **[NUEVA] esquema resample diario** | diagrama | — | Retirada del capítulo (artefacto sigue en latex/figuras) |

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
- [~] Esquema del resample diario — creado pero **retirado del capítulo**
  (2026-05-25, decisión del autor); el artefacto sigue en `latex/figuras/`.
- [ ] Tabla de selección/renombrado de variables (§3.2.2).
- [ ] Tabla embudo del filtrado (§3.5.2).

> Diagramas en TikZ `standalone` (vectoriales). Para incluirlos en el capítulo:
> `\includegraphics{figuras/o1_pipeline.pdf}` (o `\input` del `tikzpicture` si el
> documento carga `tikz`). Recompilar: `pdflatex o1_pipeline.tex`. Requieren
> `lmodern` (ya en el preámbulo del standalone).
