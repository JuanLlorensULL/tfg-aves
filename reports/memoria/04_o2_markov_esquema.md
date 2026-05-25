# Esquema de redacción — Capítulo 4: Predicción con cadenas de Markov visibles (O2)

> **Estado:** esquema (sin prosa)
> **Última actualización:** 2026-05-25
> **Relación con las notas:** complementa a `04_o2_markov.md` (notas según
> plantilla). Este fichero fija la **estructura de redacción** del capítulo, con
> los datos ya verificados contra `data/processed/o2/` y los artefactos.

## Contexto y enganche con la narrativa

- **Hereda de O1 (cap. 3):** consume `daily.parquet` (una posición por día UTC,
  huecos explícitos). Cierra los hilos H1 (longitud mínima de racha) y H2 (tamaño
  de celda 0,5°); arrastra H5 (transiciones roost-a-roost) y H6 (sesgo estacional
  del seguimiento). Ver `hilos_abiertos.md`.
- **Aporta a O3/O4/O5:** primer modelo interpretable y **baseline cuantificado**
  que O3 (HMM) y O4 (ML) deben superar. `predictions_lobo.parquet` lo consume O5.
- **Enfoque (igual que O1):** orientado al flujo del propio código
  (`discretize → transition → smooth → predict → evaluate`); ingeniería integrada
  en una nota breve, sin sección de arquitectura aparte. Tono investigador, no
  apologético: el "Markov pierde el top-1" es un **hallazgo estructural**, no un
  fracaso.

> **Frontera con Preliminares (cap. 2, aún sin redactar):** la teoría general de
> cadenas de Markov finitas vive en cap. 2. En O2 se da solo la definición
> **operativa aplicada** (propiedad de Markov + matriz de transición sobre celdas)
> y se referenciará a cap. 2 cuando exista. Sembrar hilo en `hilos_abiertos.md`.

## Datos verificados (recalculados desde los .parquet — NO de las notas)

| Dato | Valor verificado | Fuente |
|---|---|---|
| Celdas activas a 0,5° | 1 217 | `o2_tab01`, `cells.parquet` |
| Transiciones día-a-día (1 paso) | 21 214 | `metrics.parquet` global |
| Self-loops a 0,5° | 73,0 % | `o2_tab01` |
| % pares observados a 0,5° | 0,19 % | `o2_tab01` |
| log-loss Markov / persistencia | 5,451 / 5,586 (**Markov gana**) | `metrics.parquet` global |
| top-1 Markov / persistencia | 0,426 / 0,730 (persistencia gana) | global |
| top-3 Markov / persistencia | 0,524 / 0,730 | global |
| dist. mediana km Markov / pers. | 35,4 / 20,5 | global |
| dist. p90 km Markov / pers. | 3 173 / 55,9 | global |
| **Origen no observado en el fold ese mes** | **34 %** (NO 43 %) | recalculado LOBO |
| top-1 en origen no observado | 0,0001 (≈ azar) | recalculado |
| top-1 en origen observado | 0,64 (≈ persistencia) | recalculado |
| dist. mediana km origen no observado | 1 970 | recalculado |
| top-1 por ave (Markov) | min 0,06 · mediana 0,43 · máx 1,00 | `metrics.parquet` bird_month |

> **CORRECCIÓN:** las notas y CLAUDE.md decían "43 % de predicciones en celdas no
> observadas". El recálculo riguroso (bucle LOBO + señal `prob_top1==1/1217`) da
> **34 %** (7 203 de 21 214). Usar 34 %.

**Markov gana log-loss en abr–oct, pierde en nov–mar** (de `o2_tab06`): el triunfo
global viene de los meses de movimiento (abril: persistencia log-loss 8,1, Markov
6,0), porque en invierno sedentario "mañana = hoy" acierta casi siempre
(persistencia log-loss ~2,5). Ata con el sesgo estacional H6 y la coherencia
biológica de *L. fuscus*.

## Esquema

### 4.1 Introducción y objetivo
- **Puntos:** pregunta concreta (¿puede un Markov(1) global con estacionalidad
  mensual batir a la baseline trivial "mañana = hoy"?); rol de baseline
  interpretable que O3/O4 deben superar; enganche con O1 (entrada `daily.parquet`,
  posición roost-a-roost a 08:00, huecos explícitos); visión de la canalización
  (discretizar → contar → suavizar → predecir → evaluar LOBO).
- **Formalismo (compacto, aplicado):** propiedad de Markov de orden 1 + matriz de
  transición sobre celdas (ecuación). Remite a cap. 2 para la teoría general.

### 4.2 Discretización del espacio (cierra H2)
- **Puntos:** de `(lat, lon)` continua a índice de celda entero
  (`floor(coord/cell_deg)`, id tipo `122_49`). **Decisión D1: `cell_deg = 0,5°`**.
  Trade-off resolución ↔ densidad estadística sobre 4 candidatos (0,25/0,5/1/2°):
  celdas mayores inflan artificialmente los self-loops (62,9 → 86,1 %) y, con
  ellos, el top-1 de la persistencia; celdas menores fragmentan los conteos. A
  0,5°: 1 217 celdas activas, 0,19 % de pares observados. Citar solo las cifras
  que mueven la decisión.
- **Figuras/tablas:** `o2_fig01_grid-size-tradeoff` + `o2_tab01`.

### 4.3 Estimación de las matrices de transición (cierra H1, arrastra H6)
- **Puntos:** conteo de transiciones de 1 paso → **12 matrices mensuales**
  (tensor `12 × 1217 × 1217`, 21 214 transiciones). **F3 mensual:** capta el
  ciclo migratorio sin fragmentar por debajo del mínimo estadístico (vs estacional
  o global). **Enganche con §3.6.3 de O1 (pedido del autor):** la estacionalidad
  del movimiento (migración primavera/otoño vs residencia verano/invierno)
  motiva las 12 matrices mensuales frente a una única global, que promediaría
  ambos regímenes; el sesgo de muestreo descrito en §3.6.3 (otoño ≫ primavera)
  añade el matiz de que algunos meses se estiman con más datos que otros (hilo
  H6). **F5 gap-aware:** solo cuentan pares de días válidos consecutivos
  (paso 1 día); ningún par cruza un hueco de O1 (semántica estricta de Markov(1)).
  **H1 resuelta:** se entrena con **todas** las transiciones de 1 paso, sin exigir
  racha mínima (Markov(1) solo necesita pares consecutivos; exigir rachas largas
  tiraría la mayoría de los datos, mediana de racha 8 días de O1).
- **Lectura cualitativa:** `o2_fig02`/`o2_fig03` **regeneradas como scatter**
  (índice origen vs destino; el heatmap original era ilegible, 99,8 % vacío,
  decisión del autor 2026-05-25). El scatter SÍ muestra la banda diagonal y, además,
  la variación estacional: junio-julio sedentarios (puntos en un tramo corto de la
  diagonal) vs abril-mayo y sep-oct migración (dispersión a lo largo y fuera de la
  diagonal). Refuerza la justificación de las matrices mensuales.
- **Figuras:** `o2_fig02_transition-matrix-example` (scatter de un mes, septiembre),
  `o2_fig03_monthly-matrices-overview` (scatter de los 12 meses). Mismos
  nombres/slugs; regeneradas con `save_artifact(..., overwrite=True)`.

### 4.4 Suavizado de Laplace
- **Puntos:** problema de sparsity (frecuencias crudas dan prob = 1 a celdas vistas
  una vez y 0 al resto → sobreconfianza estadística + log-loss infinito). **F6
  Laplace α = 1** (add-one): no como truco para el cálculo, sino para dar una
  distribución sensata (transición no vista = improbable, no imposible); no altera
  el top-1 (conserva el orden de la fila). Ecuación de la fila suavizada. Fila de
  origen sin observar → uniforme (gancho a §4.6). **NO citar la iteración v2**
  (decisión del autor 2026-05-25: fuera de la prosa).
- **Figuras:** ninguna propia (se apoya en 4.2/4.3).

### 4.5 Protocolo de evaluación
- **Puntos:** **F7 LOBO (leave-one-bird-out)**, 82 folds: el sesgo temporal del
  dataset (82 aves en 2009, 1 en 2014-15, de O1 §3.6.3) descarta splits por año;
  LOBO mide generalización a un ave estructuralmente nueva. Predicción: fila
  `P[mes, celda_origen]` → top-1 (argmax), top-3 y distribución completa;
  fallback a marginal/uniforme si el origen no se vio en el fold. Distancia =
  Haversine (ref. ecuación de O1, NO redefinir) entre centroide del top-1 y la
  posición real. Baseline de persistencia ("mañana = hoy"). **Métricas:** top-1,
  top-3, distancia mediana km, **log-loss** (definición; criterio heredado por
  O3/O4). El cambio de criterio (top-1 → log-loss) se justifica en 4.6 con la
  evidencia, no aquí.
- **Figuras:** ninguna propia.

### 4.6 Resultados y discusión
- **4.6.1 Métricas globales (tabla propia inline):** Markov vs persistencia en
  top-1, top-3, dist. mediana/p90, log-loss. Markov **gana log-loss** (mejor
  calibrado), pierde top-1 y distancia.
- **4.6.2 Hallazgo estructural (por qué el top-1 está dominado):** 73 % self-loops
  → top-1 de persistencia 0,730 por construcción; **34 %** de las transiciones de
  test parten de una celda no vista por las otras 81 aves ese mes (rutas
  individuales) → fila uniforme → top-1 ≈ 0 y dist. ~1 970 km; en el origen
  observado Markov sube a top-1 0,64 (≈ persistencia). Conclusión metodológica:
  el top-1 no es la métrica adecuada para un Markov global LOBO con rutas
  individuales; log-loss mide la calibración de la distribución completa. Aquí se
  justifica el cambio de criterio.
- **4.6.3 Lectura estacional (fig04 + tab06):** Markov gana log-loss en abr–oct
  (movimiento real) y pierde en nov–mar (sedentarismo, persistencia casi cierta);
  el triunfo global viene de los meses de migración donde la persistencia es
  catastrófica (abril). Ata con H6 (sesgo estacional) y la fenología de
  *L. fuscus*.
- **4.6.4 Heterogeneidad por ave (fig05):** top-1 por ave de 0,06 a 1,00
  (residentes ≈ 1, migratorias bajas) → motiva estados ocultos (O3) y features
  per-individual (O4). **Cierre/hilo adelante** (sin sección de conclusiones
  propia, como en O1): el Markov global está bien calibrado pero su argmax falla
  porque ignora la ruta individual y el régimen de comportamiento → O3 y O4.
- **Figuras/tablas:** `o2_fig04_accuracy-vs-baseline` + `o2_tab06_summary`,
  `o2_fig05_per-bird-performance`.

> **Sin sección 4.7 de conclusiones** (igual que O1: las conclusiones globales van
> al cap. 9). El capítulo termina en 4.6 con el hilo hacia O3/O4.

## Mapa figura/tabla → sección

| Artefacto | Tipo | Sección | Justifica |
|---|---|---|---|
| `o2_fig01_grid-size-tradeoff` + `o2_tab01` | fig+tab | 4.2 | D1: celda 0,5° |
| `o2_fig02_transition-matrix-example` | fig | 4.3 | Matriz ejemplo (self-loops) |
| `o2_fig03_monthly-matrices-overview` | fig | 4.3 | Estacionalidad cualitativa |
| (tabla global inline) | tab | 4.6.1 | Resultado headline |
| `o2_fig04_accuracy-vs-baseline` + `o2_tab06` | fig+tab | 4.6.3 | Comparación por mes |
| `o2_fig05_per-bird-performance` | fig | 4.6.4 | Heterogeneidad → O3/O4 |

Cobertura: 5/5 figuras + tab01 + tab06 colocadas (tab04/tab05 respaldan sus
figuras, no van sueltas).

## Notas para la redacción
- Referenciar la ecuación de Haversine de O1 (`eq:haversine`), no redefinirla.
- Definir log-loss aquí (criterio heredado por O3/O4); top-1/top-3 de pasada.
- Mención de pasada al bosquejo v2 (sin suavizar → sparsity) como motivación del
  Laplace, sin tabla ni figura propia.
- Cero rayas (—); paréntesis. Verificar con grep antes de cerrar.
- Cita bibliográfica pendiente: cadenas de Markov finitas / suavizado add-one
  (Bishop o similar) — coordinar con cap. 2.
