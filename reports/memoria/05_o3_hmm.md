# Capítulo 5 — O3: Detección de comportamiento con HMM

> **Estado:** notas
> **Última actualización:** 2026-05-22 (post-rework)

## Resumen ejecutivo

Se entrenan dos modelos de Markov oculto (HMM) gaussianos como ablación
comparativa sobre el dataset diario de *Larus fuscus* (82 aves, 20 672
observaciones válidas). El **Modelo A** usa únicamente features cinemáticas
(`step_length_km` y `cos_turning_angle`); el **Modelo B** añade contexto
ambiental y temporal (`veg_low`, `veg_high`, `daylight_hours`). Ambos
detectan dos estados ocultos: **estacionario** y **migración**.

Tras una primera ejecución que evidenció un fallo metodológico (ver §6),
el preprocesado se corrigió: las features se pasan al HMM en escala
original (sin `StandardScaler`) y el desplazamiento sin transformación
logarítmica. Con esta corrección, ambos modelos descubren estados
biológicamente coherentes — Modelo A separa "ave en el sitio" (~4 km/día)
de "ave volando" (~129 km/día) con un patrón estacional defensible;
Modelo B replica esa estructura añadiendo un refinamiento marginal del
contexto. El acuerdo entre modelos es del 93 %, lo que confirma que las
features contextuales aportan poco una vez que el step en km puede
dominar la inicialización por k-means. El entregable principal es
`features.parquet` con 17 columnas — incluyendo `state_a`, `state_b` y
las probabilidades posteriores — que O4 y O5 consumirán directamente.

## Contexto y motivación

- **Pregunta concreta:** ¿pueden estados ocultos gaussianos descubrir
  regímenes comportamentales biológicamente coherentes (estacionario vs
  migración) en series de posiciones diarias de *Larus fuscus*, sin
  supervisión?
- **Motivación desde O2:** el Markov(1) global perdió en top-1 frente a
  la persistencia trivial por el 73 % de self-loops del dataset y las
  rutas individuales. Introducir estados ocultos permite capturar el
  régimen comportamental de cada ave: en estado estacionario la
  predicción óptima se acerca a la persistencia; en estado migración el
  modelo podrá explotar la direccionalidad del movimiento. Además los
  estados `state_a`/`state_b` son features discretas de alto nivel que
  O4 (ML supervisado) puede consumir para mejorar la predicción
  per-individual.
- **Conexión con O1:** input directo de `data/processed/daily.parquet`
  (posición diaria por ave, huecos explícitos).
- **Conexión con O4:** `features.parquet` proporciona `state_a`,
  `state_b` y las cuatro probabilidades posteriores como features de
  entrada para los modelos de ML.
- **Conexión con O5:** los estados detectados permiten estratificar el
  análisis del error por régimen comportamental.

## Decisiones tomadas

1. **D1 — Número de estados: `n_components = 2`**
   - Alternativas consideradas: 2, 3, 4 (sweep D1).
   - Criterio: alineación con el proposal (estacionario vs migración) e
     interpretabilidad biológica. El sweep AIC/BIC prefiere
     monótonamente n>2, pero los estados adicionales no admiten
     etiquetado biológico claro con sólo 2 features cinemáticas y se
     documentan como follow-up.
   - Evidencia: `reports/figures/o3_fig01_nstates-aic-bic-sweep.png`,
     `reports/tables/o3_tab01_nstates-aic-bic-sweep.csv`.

2. **F1 — Detección binaria (estacionario vs migración)**
   - La tutora sugirió tres estados ("residente, migración, forrajeo").
     El autor cerró a 2 estados: el forrajeo se incluye en
     "estacionario" como sub-régimen de baja velocidad. Si D1 hubiera
     sugerido N=3 con mejora interpretable, se habría abierto este punto.

3. **F2 — Dos modelos como ablación (Modelo A vs Modelo B)**
   - Modelo A: features cinemáticas puras (`step_length_km`,
     `cos_turning_angle`).
   - Modelo B: A + contexto (`veg_low`, `veg_high`, `daylight_hours`).
   - La ablación responde a la propuesta de la tutora de usar
     vegetación y horas de luz, y al riesgo de circularidad geográfica:
     si los estados de B se reducen a "verano vs invierno" usando
     contexto como discriminador principal, la señal cinemática habría
     quedado dominada por la estacionalidad. El resultado tras el rework
     (§6) muestra que con preprocesado correcto el step en kilómetros
     domina la inicialización y B coincide con A en el 93 % de las
     observaciones, lo que confirma que el contexto añade poca
     información discriminativa adicional.

4. **F3 — Alcance global (un HMM por modelo, 82 aves comparten parámetros)**
   - Alternativas: per-individuo (82 HMMs), muestra representativa.
   - Criterio: LOBO demostró en O2 que las rutas individuales son
     idiosincráticas; sin embargo, el HMM opera sobre features
     cinemáticas comparables entre aves (no sobre posiciones absolutas),
     por lo que un modelo global es más justificable que en O2. Produce
     un entregable único y sirve de baseline para O4 per-individual.

5. **F4 — `GaussianHMM` de `hmmlearn 0.3.3`, `covariance_type='diag'`**
   - Práctica estándar en análisis de movimiento animal (Patterson et
     al. 2017). Robustez numérica superior a `'full'`; varianza por
     feature directamente interpretable. Ver §9.3.

6. **F5 — Inicialización con k-means + 10 restarts EM**
   - K-means proporciona una semilla sensible que reduce la
     probabilidad de óptimos locales degenerados. 10 restarts con
     seeds distintos y retención del mejor LL en train. Ver §9.4 y §9.5.

7. **F6 — Sin estandarización (revisado tras hallazgo empírico)**
   - Las features se pasan al HMM en su escala original
     (`step_length_km` en km, `cos_turning_angle` adimensional,
     `daylight_hours` en horas, `veg_*` en [0, 1]). Ver §9.2 para la
     justificación completa: la decisión inicial de aplicar
     `StandardScaler` resultó ser el origen de un fallo metodológico
     descrito en §6.

8. **F7 — Split 80/20 por ave, estratificado por días válidos**
   - 65 aves en entrenamiento, 17 en holdout. Estratificación garantiza
     representación proporcional de aves con tracking rico y pobre.
     Cambio respecto al LOBO de O2 justificado en §9.8.

## Implementación

- **Módulos del paquete:** `src/tfg_aves/hmm/` con submódulos
  `features`, `fit`, `evaluate`, `build`, `_paths`.
- **Algoritmos y fórmulas clave:**
  - `step_length_km`: `haversine(pos_t, pos_{t+1})` en kilómetros. Sin
    transformación logarítmica (ver §6 y §9.2 para la justificación).
  - `cos_turning_angle`: `cos(bearing(t, t+1) − bearing(t-1, t))`,
    normalizado a `[-1, 1]`. Valor +1 = vuelo rectilíneo (rumbo
    sostenido), −1 = inversión completa, 0 = giro de 90°. Ver §9.2 bis.
  - `daylight_hours`: fórmula astronómica clásica (declinación solar
    con corrección de Cooper). No requiere datos externos.
  - `veg_low`, `veg_high`: columnas ECMWF del dataset Movebank,
    cargadas directamente sin transformación (ya normalizadas 0-1).
  - `fit_hmm_with_restarts()`: k-means init → EM → selección por LL
    train. Devuelve `(GaussianHMM, best_ll, all_lls)` — sin scaler.
  - `relabel_states()`: re-etiquetado determinista por menor
    `μ[step_length_km]` (el estado de menor desplazamiento medio recibe
    la etiqueta `estacionario`).
  - `viterbi_per_bird()`: secuencia de estados más probables por ave
    sobre tramos consecutivos.
- **Orquestador:** `build_o3(holdout_frac=0.20, n_restarts=10,
  random_state=0)` en `src/tfg_aves/hmm/build.py` — escribe los 3
  ficheros de `data/processed/o3/` y devuelve un `BuildO3Result`.
- **Parámetros finales:** `n_components=2`, `covariance_type='diag'`,
  `n_restarts=10`, `holdout_frac=0.20`, `random_state=0`.

## Resultados y validación

### Medias aprendidas por estado (unidades originales)

**Modelo A** (`step_length_km`, `cos_turning_angle`):

| Estado | μ[step_length_km] | μ[cos_turning_angle] |
|---|---|---|
| 0 (estacionario) | 4,24 km | −0,265 |
| 1 (migración) | **128,9 km** | +0,077 |

**Modelo B** (`step_length_km`, `cos_turning_angle`, `veg_low`,
`veg_high`, `daylight_hours`):

| Estado | μ[step_km] | μ[cos_turn] | μ[veg_low] | μ[veg_high] | μ[daylight_h] |
|---|---|---|---|---|---|
| 0 (estacionario) | 5,84 | −0,273 | 0,226 | 0,334 | 13,07 |
| 1 (migración) | **164,4** | +0,242 | 0,173 | 0,184 | 12,30 |

Ratio de medias en step entre estados: **~30×** para A, **~28×** para B.
La separación es ahora cinemática genuina: el HMM descubre "ave parada"
(unidades de km/día) vs "ave volando" (más de 100 km/día).

### Métricas de ajuste (holdout 17 aves)

| Modelo | LL por observación (holdout) |
|---|---|
| Modelo A | −5,498 |
| Modelo B | −8,501 |

**Nota importante:** las LL no son comparables directamente entre
modelos porque operan sobre espacios de observación distintos
(2 features vs 5). A tiene LL más alta en su escala propia; B la tiene
más baja porque mide la probabilidad conjunta de 5 features, no 2. Ver
§9.10.

### Estadísticas del entregable

- **n_birds_train:** 65 | **n_birds_holdout:** 17
- **n_observations válidas:** 20 672 (de 24 444 totales)
- **% migración global:** 21,2 % (Modelo A) | 15,3 % (Modelo B)
- **% acuerdo A-B:** **93,0 %** — el contexto añade poca información
  discriminativa una vez que el step en km domina la inicialización.

### Patrón estacional (% migración por mes, Modelo B)

| Mes | % migr | Mes | % migr |
|---|---|---|---|
| Ene | 8 % | Jul | **2,7 %** |
| Feb | 8 % | Ago | 14 % |
| Mar | 14 % | Sep | 29 % |
| Abr | **29 %** | Oct | 29 % |
| May | 18 % | Nov | 16 % |
| Jun | **3,3 %** | Dic | 9 % |

El valle en jun-jul (cría en colonias del norte de Europa) y los picos
de abr y sep-oct (paso migratorio) coinciden con la fenología conocida
de *Larus fuscus*. El patrón también muestra la baja relativa de
dic-feb (invernada en zonas africanas) — comportamiento de
desplazamiento local de corto rango, predominantemente clasificado como
"estacionario".

### Monotonía estado vs desplazamiento real (Modelo A)

| Bin de step | n | % migración |
|---|---|---|
| <10 km | 14 268 | 0,18 % |
| 10-50 km | 4 320 | 48,2 % |
| 50-100 km | 757 | 100,0 % |
| >100 km | 1 327 | 100,0 % |

La clasificación es **monótona** en step y **biológicamente razonable**:
desplazamientos por debajo de 10 km/día son casi inequívocamente
estacionarios; por encima de 50 km/día son inequívocamente migración.
El bin 10-50 km contiene la zona ambigua donde el cos_turning aporta
discriminación adicional.

## 6. Hallazgo metodológico: la advertencia de circularidad confirmada

**Esta sección documenta un hallazgo que el TFG defiende como
contribución metodológica.** La primera ejecución de O3 — con el
preprocesado tal y como lo describía la spec original (F6 con
`StandardScaler`, F9 con `log_displacement_km`) — produjo un modelo
aparentemente "biológicamente coherente" cuyo análisis post-ejecución
reveló que era una **falsa positiva**: el HMM no estaba detectando
comportamiento sino geografía. La ablación A vs B (F2) estaba
explícitamente diseñada para detectar este riesgo, y lo detectó.

### Síntoma observable en la primera ejecución

- Las medias aprendidas para los dos estados de Modelo B caían a
  `μ[disp_km]` = 5,4 km ("migración") y 3,8 km ("estacionario"), una
  separación de apenas 1,4×.
- La separación efectiva no la marcaba el desplazamiento sino
  `daylight_hours` (17,3 h vs 12,0 h) y `veg_high` (0,94 vs 0,09) —
  features que codifican estación + bioma.
- En enero, el modelo asignaba ~100 % de las observaciones a "migración".
  Esto era biológicamente implausible: en enero las gaviotas están en
  invernada en África, no migrando.

### Diagnóstico de causa raíz

Tras inspeccionar las medias del modelo y comparar con la versión
previa del proyecto (`v2/notebooks/HMM5.ipynb`, que había documentado
explícitamente la lección), se identificaron dos errores de
preprocesado:

1. **`log_displacement_km` comprime la separación natural**. El step
   real tiene una bimodalidad clara en km: modo "parado" en torno a 2-5
   km/día y modo "volando" en torno a 100-1 500 km/día. Aplicar
   `log1p` colapsa este factor 30-100× a un factor 3-4× en log-espacio.
2. **`StandardScaler` iguala las varianzas y mata el dominio del step**.
   Tras escalar, las 5 features tienen σ=1, así que la inicialización
   por k-means (que usa distancia euclídea sin ponderar) ya no clusteriza
   por step sino por la combinación que da el cluster más limpio en
   espacio estandarizado. Esa combinación resultó ser
   `daylight + veg_high` — exactamente la circularidad geográfica.

### Resolución

Se revirtieron ambas decisiones (F6 y F9), replicando la convención
validada empíricamente por el autor en `v2/HMM5.ipynb`:

- `step_length_km` en kilómetros sin transformación.
- Sin `StandardScaler`. Las features se pasan al HMM en escala original.
- Se cambió además `|turning_angle|` → `cos(turning_angle)` por
  suavidad de la emisión gaussiana (§9.2 bis).

El re-entrenamiento con estas correcciones produjo los resultados
descritos en §5 — bimodalidad cinemática genuina, patrón estacional
coherente con la fenología conocida.

### Por qué esto es un resultado, no un error que ocultar

La ablación A vs B existía **precisamente** para detectar el riesgo
de circularidad geográfica. Cuando la primera ejecución produjo un
Modelo B con apariencia de "coherencia estacional", la inspección de
las medias aprendidas reveló que la separación no la dominaba la
cinemática sino el contexto — el escenario exacto que F2 quería
descartar. La metodología comparativa funcionó: detectó el problema
antes de propagarlo a O4. La corrección está documentada, versionada
en `docs/superpowers/specs/2026-05-22-o3-hmm-design.md §3 bis` y §9.2,
y reproducible.

### Implicación para O4 y O5

La recomendación canónica para downstream es **Modelo A** (o
indistintamente Modelo B, dado el 93 % de acuerdo). La aportación del
contexto es marginal una vez que el step en km puede dominar; la
narrativa del TFG es que el ave se comporta de manera
detectable-en-cinemática y el contexto añade refinamiento, no
información esencial.

## Caracterización de artefactos

- **D1** (`o3_fig01_nstates-aic-bic-sweep`): AIC/BIC para n∈{2,3,4}.
  Respalda la elección de n=2 como decisión interpretativa (n>2 mejora
  la verosimilitud monótonamente pero no admite etiquetado biológico
  claro).
- **C1** (`o3_fig02_features-by-state-a`): histogramas de las dos
  features cinemáticas por estado A. Separación nítida: estado
  estacionario concentra masa en `step_length_km` bajo y
  `cos_turning_angle` negativo/aleatorio; estado migración, patrón
  contrario (step alto, cos cercano a +1).
- **C2** (`o3_fig03_features-by-state-b`): histogramas de las cinco
  features del Modelo B por estado. Las dos cinemáticas replican C1; las
  tres de contexto (`daylight`, `veg_*`) muestran diferencias mucho más
  modestas entre estados, lo que confirma cuantitativamente que el step
  domina la separación.
- **C3** (`o3_fig04_state-vs-biology`): principal artefacto de
  validación. % migración por mes y por bin de latitud, para A y B.
  Tabla adjunta con valores numéricos
  (`o3_tab04_state-vs-biology.csv`).
- **C4** (`o3_fig05_ab-agreement`): matriz de confusión A vs B +
  histograma de desacuerdos por `step_length_km`. Cuantifica la
  aportación del contexto: el 93 % de acuerdo se distribuye con sesgo
  hacia "B clasifica más estacionario" en la zona ambigua 10-50 km/día.
- **C5** (`o3_fig06_per-bird-state-proportions`): dispersión
  proporciones por ave en A vs B. Revela heterogeneidad individual:
  cada ave tiene una fracción distinta de días en migración (algunas
  casi exclusivamente estacionarias, otras con migración fuerte). La
  diagonal y=x está poblada uniformemente, lo que confirma que A y B
  coinciden en la mayoría de las aves; los puntos alejados son aves
  donde el contexto cambia la inferencia.

### Validación

- 78 tests pasando (`uv run pytest -q`). Ruff limpio (`uv run ruff
  check src tests`). `build_o3()` reproducible desde `daily.parquet`
  en una sola llamada con `random_state=0`.

## Salida materializada

Ficheros en `data/processed/o3/` (gitignored, regenerables):

| Fichero | Contenido |
|---|---|
| `features.parquet` | 24 444 filas × 17 columnas: features, estados, posteriores, flags |
| `models_a_b.pkl` | dict con HMM A, HMM B, label_map A, label_map B, train_bird_ids, holdout_bird_ids, random_state |
| `metrics.parquet` | LL por observación (holdout) + estadísticas de acuerdo A-B |

Las 17 columnas de `features.parquet`:
`bird_id`, `date_utc`, `lat`, `lon`, `step_length_km`,
`cos_turning_angle`, `daylight_hours`, `veg_low`, `veg_high`,
`state_a`, `state_b`, `posterior_a_estacionario`,
`posterior_a_migracion`, `posterior_b_estacionario`,
`posterior_b_migracion`, `is_observation_valid`, `in_holdout`.

## 9. Justificación de decisiones metodológicas

*Esta sección replica la sección 8 del spec de O3 en formato citable
para el capítulo 5 de la memoria LaTeX. Cada entrada: decisión, razón,
alternativa descartada.*

### 9.1 No combinar `veg_low` y `veg_high`

- **Decisión:** usar ambas columnas como features separadas en Modelo B
  (5 features, no 4).
- **Razón:** `veg_low` (pastos, cultivos, matorral) y `veg_high`
  (bosques) representan biomas físicamente distintos. Una estepa
  `(0.8, 0.0)` y un bosque `(0.0, 0.8)` tienen la misma suma pero
  biomas opuestos. Combinarlas impondría una asunción no respaldada por
  los datos sobre cómo se relacionan los dos estratos en el
  comportamiento del ave.
- **Alternativa descartada:** suma o ratio `veg_low + veg_high`.
  Colapsa tres biomas distintos en valores no distinguibles.

### 9.2 Sin estandarización: las features se pasan al HMM en escala original (revisado)

- **Decisión:** **no** aplicar `StandardScaler`. Cada feature entra al
  HMM en sus unidades naturales (`step_length_km` en km,
  `cos_turning_angle` adimensional en [-1, 1], `daylight_hours` en
  horas, `veg_*` en [0, 1]).
- **Razón:** el razonamiento inicial favorable a escalar resultó ser
  **incorrecto** en la práctica (ver §6). El experimento empírico de
  la primera ejecución mostró que escalar destruye la bimodalidad
  natural del `step_length_km` (varianza ~10⁴ km², dos modos a ~5 km y
  ~170 km separados por dos órdenes de magnitud) y desplaza la
  inicialización por k-means hacia la combinación de features
  contextuales que codifican estación + bioma. El HMM resultante no
  detecta comportamiento sino geografía/estacionalidad — exactamente la
  circularidad que F2 estaba diseñada para diagnosticar.
- **Por qué `covariance_type='diag'` lo permite:** cada estado tiene
  su propia σ por feature; un estado puede ser una Gaussiana estrecha
  centrada en 4 km (σ pequeña) y el otro una Gaussiana ancha centrada
  en 130 km (σ grande). El likelihood discrimina por sí solo aunque la
  marginal global no sea gaussiana.
- **Precedente:** `v2/notebooks/HMM5.ipynb` documentó explícitamente
  esta lección: *"Sin escalar: la varianza de `step_length` debe
  seguir dominando para no diluir la bimodalidad natural"*.
- **Alternativa descartada (la opción anterior):** aplicar
  `StandardScaler` ajustado en train. Refutada empíricamente: con
  escalado, el step pierde el contraste de varianzas que el HMM
  necesitaba para separar los dos regímenes.

### 9.2 bis Linealización del cambio de rumbo: `cos(turning_angle)` en lugar de `|turning_angle|`

- **Decisión:** codificar el cambio de rumbo entre días consecutivos
  como `cos_turning_angle = cos(bearing_out - bearing_in)` ∈ [-1, 1] en
  lugar de `|turning_angle|` ∈ [0, π].
- **Razón:** dos motivos. (1) `cos` es **suave en toda la recta real**
  y se aproxima mejor a una emisión gaussiana, mientras que el valor
  absoluto tiene una "esquina" en 0 que distorsiona la forma de la
  distribución condicional al estado. (2) `cos` colapsa giros
  simétricos (izquierda vs derecha) igual que el absoluto, pero
  conserva la semántica natural del producto escalar: `cos = +1` →
  mismo rumbo (vuelo rectilíneo, característico de migración),
  `cos = -1` → inversión, `cos = 0` → giro de 90°.
- **Alternativa descartada:** `|turning_angle|` ∈ [0, π]. La marginal
  condicional al estado tiende a un half-normal truncado en 0, peor
  ajustada por una gaussiana con `covariance_type='diag'`.
- **Precedente:** convención de `v2/HMM5.ipynb`.

### 9.3 Covarianza diagonal (`covariance_type='diag'`)

- **Decisión:** `GaussianHMM` con `covariance_type='diag'`. Cada
  estado tiene varianza propia por feature, sin modelar correlaciones
  entre features.
- **Razón:** práctica estándar en HMMs de movimiento animal (Patterson
  et al. 2017; `moveHMM`). Features diseñadas para ser semánticamente
  ortogonales; correlaciones intra-estado son débiles. Mayor robustez
  estadística que `'full'` y directa interpretabilidad (varianza por
  feature es legible).
- **Alternativa descartada:** `'full'`. Mayor flexibilidad pero peor
  estabilidad numérica y más difícil de explicar en la memoria.

### 9.4 Inicialización con k-means

- **Decisión:** ejecutar k-means con k=2 sobre las features en escala
  original antes del EM. Centroides como `means_init`, varianza
  intra-cluster como varianza inicial.
- **Razón:** EM converge a óptimos locales. Sin inicialización sensata
  puede caer en soluciones degeneradas (un estado absorbe todo).
  K-means es un pre-clustering rápido que da a EM un punto de partida
  que ya separa los datos en dos grupos razonables.
- **Alternativa descartada:** inicialización aleatoria. Requeriría
  20-50 restarts para confianza equivalente.

### 9.5 Diez restarts de EM

- **Decisión:** repetir (k-means init + EM) 10 veces con seeds
  distintos. Retener el modelo con mayor LL en train.
- **Razón:** incluso con k-means init, EM puede converger a óptimos
  ligeramente distintos. 10 restarts es defensivo (~2 min total para
  ambos modelos sobre 21 000 observaciones) y suficiente para descartar
  trayectorias patológicas.
- **Alternativa considerada:** 5 restarts. Estadísticamente suficiente
  pero con menor margen; la diferencia de tiempo es mínima.

### 9.6 Re-etiquetado de estados por menor `μ[step_length_km]`

- **Decisión:** tras entrenar, el estado con menor media en
  `step_length_km` recibe la etiqueta `estacionario` (0); el otro,
  `migración` (1). Aplicado en A y B.
- **Razón:** `hmmlearn` asigna etiquetas internas `0`/`1`
  arbitrariamente según el seed. Sin re-etiquetado, `state_a` y
  `state_b` serían inconsistentes entre runs. La regla es
  biológicamente correcta (menor desplazamiento = más estático) e
  independiente del estado del modelo.
- **Alternativa descartada:** re-etiquetado manual post-inspección.
  Frágil, irreproducible, sesgo del autor.

### 9.7 `n_components = 2` a priori (respaldado por D1)

- **Decisión:** dos estados en ambos modelos.
- **Razón:** el proposal especifica detección binaria. N=2 alinea el
  modelo con la pregunta de investigación y maximiza interpretabilidad.
  D1 muestra que AIC/BIC mejoran monótonamente con n>2 pero los estados
  adicionales no admiten etiquetado biológico claro con las features
  disponibles.
- **Alternativa descartada:** dejar que AIC/BIC elijan N. Más
  predictivo pero los estados extra son sub-clusters de "se mueve poco"
  que no aportan a la narrativa del TFG.

### 9.8 Holdout 80/20 estratificado en lugar de LOBO

- **Decisión:** 65 aves en train / 17 en holdout. Split por ave,
  estratificado por número de días válidos. No se usa LOBO (82 folds).
- **Razón:** (1) LOBO en O2 castiga por rutas individuales
  idiosincráticas; en O3 el HMM opera sobre features cinemáticas
  comparables entre aves, lo que reduce pero no elimina ese problema.
  (2) LOBO en O3 implicaría 82×2×10 = 1 640 fits, excesivo para el
  aporte marginal. (3) La estratificación garantiza representación
  proporcional.
- **Alternativa descartada:** LOBO completo. Más riguroso
  estadísticamente pero impracticable y no justificado dado el
  aprendizaje de O2.

### 9.9 Log-likelihood por observación, no total

- **Decisión:** reportar `LL_total / n_obs_holdout`.
- **Razón:** la LL total depende del tamaño del dataset. La LL por
  observación es comparable entre datasets, configuraciones y
  reproducciones.
- **Alternativa descartada:** LL total. Engañosa cuando se compara
  entre datasets de diferente tamaño.

### 9.10 No comparar LL entre Modelo A y Modelo B

- **Decisión:** las LL se reportan por separado en su propia escala,
  no como métrica de comparación.
- **Razón:** Modelo A mide `log P(2 features | modelo A)` y Modelo B
  mide `log P(5 features | modelo B)`. Los espacios de observación son
  distintos; la LL no es directamente comparable. La comparación se
  hace con criterios independientes: coherencia biológica (C3) y
  acuerdo (C4).
- **Alternativa descartada:** AIC/BIC para comparar. Requeriría asumir
  que las features de B son extensión de las de A, lo que es discutible.

### 9.11 Coherencia biológica como criterio principal sin ground truth

- **Decisión:** la validación principal de los estados es su
  coherencia con el conocimiento fenológico de *Larus fuscus*, no
  contra etiquetas verdaderas.
- **Razón:** no existe ground truth diario sobre el comportamiento de
  las 82 aves. Sí existe conocimiento poblacional sobre la fenología de
  la especie (Wikelski et al. 2015) que actúa como criterio externo
  independiente.
- **Limitación reconocida:** criterio poblacional, no individual — no
  detecta fenologías atípicas de aves concretas.
- **Alternativa descartada:** etiquetas humanas en sub-muestra.
  Costoso para el alcance del TFG y dependiente de la pericia del
  anotador.

### 9.12 Acuerdo A-B como porcentaje crudo

- **Decisión:** `% acuerdo = n_coincidencias / n_total`. No se usa
  Cohen's kappa.
- **Razón:** Cohen's kappa corrige por acuerdo esperado por azar,
  apropiado cuando los clasificadores son independientes y se evalúan
  contra ground truth. A y B no son independientes (comparten dos
  features) y no hay ground truth. El porcentaje crudo es directamente
  interpretable.
- **Alternativa descartada:** Cohen's kappa. Más sofisticado, menos
  interpretable, no necesario para el análisis cualitativo de los
  desacuerdos.

## Conclusiones y limitaciones

### Lo que funciona bien

- Tras corregir el preprocesado (§6), tanto Modelo A como Modelo B
  descubren estados cinemáticamente coherentes con la fenología de
  *Larus fuscus* sin supervisión: el valle de jun-jul (cría) y los
  picos de paso (abr, sep-oct) son nítidos en ambos modelos.
- La separación de medias en step (4 km vs 129 km en Modelo A) es de
  dos órdenes de magnitud, una bimodalidad genuina del comportamiento
  diario.
- La monotonía de % migración por bin de step (<10 km → 0,18 %, >50 km
  → 100 %) confirma que la clasificación captura la magnitud del
  movimiento como discriminador principal.
- La ablación A vs B detectó empíricamente el riesgo de circularidad
  geográfica en la primera ejecución y orientó la corrección. Es un
  resultado defensible del propio diseño experimental.
- El entregable `features.parquet` es modular y listo para consumo
  directo por O4 y O5.

### Limitaciones detectadas

- **Aporte marginal del contexto:** el 93 % de acuerdo entre Modelo A
  y Modelo B indica que `daylight_hours`, `veg_low` y `veg_high`
  añaden poco una vez que el step en km puede dominar la
  inicialización. Esto es coherente con el resultado de §6 y con la
  observación de v2.
- **Validación poblacional, no individual:** la coherencia biológica se
  juzga a nivel agregado; aves con fenologías atípicas quedan sin
  validación individual.
- **Asunción de covarianza diagonal:** no captura correlaciones entre
  features dentro de un estado.
- **Sensibilidad de EM a la inicialización:** mitigada por 10 restarts
  pero no eliminada; soluciones subóptimas son posibles.
- **Modelo global:** ignora que distintas aves pueden tener umbrales
  de desplazamiento diferentes para el mismo régimen comportamental.
  O4 per-individual abordará esta limitación.

### Aspectos abiertos / futuras mejoras

- O4 (ML): usar `state_a` (o `state_b`, ~equivalentes dado el 93 % de
  acuerdo) y las cuatro posteriores como features de entrada en los
  modelos de RF, XGBoost y LightGBM. El estado HMM añade información
  sobre el régimen actual del ave que el Markov(1) global de O2 no
  capturaba.
- O5 (evaluación): estratificar el error de predicción por estado HMM
  para ver si los modelos de O4 cometen errores distintos en estado
  estacionario vs migración.
- Extensión futura (fuera del alcance del TFG): HMM por-individuo o
  HMM jerárquico (parámetros compartidos pero iniciados per-ave) para
  capturar la heterogeneidad inter-individual visible en C5.

## Notas para la redacción final

- La sección §6 ("Hallazgo metodológico") es el resultado defensible
  más fuerte del capítulo 5 ante tribunal. Dedicarle un párrafo
  completo: la ablación A vs B funcionó como diagnóstico, identificó
  la circularidad antes de cerrar el modelo, y la corrección está
  validada empíricamente por dos iteraciones independientes del autor
  (v2 y v3 post-rework).
- Conectar explícitamente con O2: "el HMM resuelve la ambigüedad de
  comportamiento que el Markov(1) global no podía capturar, porque el
  argmax de una distribución mezcla sin estados ocultos colapsa hacia
  el self-loop dominante".
- Citar Wikelski et al. (2015) en la sección de validación biológica y
  Patterson et al. (2017) en la elección de `covariance_type='diag'`.
- Aclarar en el texto que LL Modelo A ≠ LL Modelo B porque los
  espacios de observación son distintos (§9.10).
- La recomendación de qué modelo usar como "el modelo O3" (A o B) puede
  ser una sola: dado el 93 % de acuerdo y la marginalidad de las
  features contextuales, Modelo A es suficiente y más parsimonioso.
  Si se prefiere mantener la propuesta original de la tutora, Modelo B
  produce resultados ~equivalentes y es defendible. Conversación
  pendiente.
- Bibliografía pendiente:
  - Wikelski M et al. (2015): dataset Movebank.
  - Patterson et al. (2017): HMM para movimiento animal.
  - Rabiner (1989): tutorial clásico HMM.
