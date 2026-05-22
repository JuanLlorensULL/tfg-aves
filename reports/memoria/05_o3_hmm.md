# Capítulo 5 — O3: Detección de comportamiento con HMM

> **Estado:** notas
> **Última actualización:** 2026-05-22

## Resumen ejecutivo

Se entrenan dos modelos de Markov oculto (HMM) gaussianos como ablación
comparativa sobre el dataset diario de *Larus fuscus* (82 aves, 20 672
observaciones válidas). El Modelo A usa únicamente features cinemáticas
(desplazamiento y cambio de rumbo); el Modelo B añade contexto ambiental
y temporal (vegetación baja, vegetación alta, horas de luz). Ambos
detectan dos estados ocultos: **estacionario** y **migración**. El
Modelo B revela una separación estacional muy clara (migración concentrada
en primavera y otoño) coherente con la fenología conocida de *Larus
fuscus*; el Modelo A produce una separación cinemática correcta pero con
menor discriminación temporal, lo que sugiere que el contexto refina
genuinamente la señal cinemática sin quedar atrapado en circularidad
geográfica. El resultado principal es un entregable `features.parquet` con
17 columnas — incluyendo `state_a`, `state_b` y las probabilidades
posteriores — que O4 y O5 consumirán directamente.

## Contexto y motivación

- **Pregunta concreta:** ¿pueden estados ocultos gaussianos descubrir
  regímenes comportamentales biológicamente coherentes (estacionario vs
  migración) en series de posiciones diarias de *Larus fuscus*, sin
  supervisión?
- **Motivación desde O2:** el Markov(1) global perdió en top-1 frente a
  la persistencia trivial por el 73 % de self-loops del dataset y las
  rutas individuales. Introducir estados ocultos permite capturar el
  régimen de comportamiento de cada ave: en estado estacionario, la
  predicción óptima se acerca a la persistencia; en estado migración,
  el modelo puede explotar la direccionalidad del movimiento. Además, los
  estados `state_a`/`state_b` son features discretas de alto nivel que O4
  (ML supervisado) puede consumir para mejorar la predicción per-individual.
- **Conexión con O1:** input directo de `data/processed/daily.parquet`
  (posición diaria por ave, huecos explícitos).
- **Conexión con O4:** `features.parquet` proporciona `state_a`, `state_b`
  y las cuatro probabilidades posteriores como features de entrada para los
  modelos de ML.
- **Conexión con O5:** los estados detectados permiten estratificar el
  análisis del error por régimen comportamental.

## Decisiones tomadas

1. **D1 — Número de estados: `n_components = 2`**
   - Alternativas consideradas: 2, 3, 4 (sweep D1).
   - Criterio: alineación con el proposal (estacionario vs migración) y
     interpretabilidad biológica. El artefacto D1 confirma que añadir un
     tercer o cuarto estado no produce una mejora sustancial de AIC/BIC,
     por lo que la restricción a 2 es razonable y no artificial.
   - Evidencia: `reports/figures/o3_fig01_nstates-aic-bic-sweep.png`,
     `reports/tables/o3_tab01_nstates-aic-bic-sweep.csv`.

2. **F1 — Detección binaria (estacionario vs migración)**
   - La tutora sugirió tres estados ("residente, migración, forrajeo").
     El autor cerró a 2 estados: el forrajeo se incluye en "estacionario"
     como sub-régimen de baja velocidad. Si D1 hubiera sugerido N=3 con
     mejora estadística clara, se habría abierto este punto.

3. **F2 — Dos modelos como ablación (Modelo A vs Modelo B)**
   - Modelo A: features cinemáticas puras (`log_displacement_km`,
     `abs_turning_angle_rad`).
   - Modelo B: A + contexto (`daylight_hours`, `veg_low`, `veg_high`).
   - La ablación responde a la propuesta de la tutora de usar vegetación y
     horas de luz y al riesgo de circularidad geográfica: si los estados B
     se reducen a "verano vs invierno" en C3, la señal cinemática habría
     sido dominada por el contexto. El resultado muestra el patrón opuesto:
     Modelo B tiene mayor coherencia biológica estacional.

4. **F3 — Alcance global (un HMM por modelo, 82 aves comparten parámetros)**
   - Alternativas: per-individuo (82 HMMs), muestra representativa.
   - Criterio: LOBO demostró en O2 que las rutas individuales son
     idiosincráticas; sin embargo, el HMM opera sobre features cinemáticas
     comparables entre aves (no sobre posiciones absolutas), por lo que un
     modelo global es más justificable que en O2. Produce un entregable
     único y sirve de baseline para O4 per-individual.

5. **F4 — `GaussianHMM` de `hmmlearn 0.3.3`, `covariance_type='diag'`**
   - Práctica estándar en análisis de movimiento animal (Patterson et al.
     2017). Robustez numérica superior a `'full'`; varianza por feature
     directamente interpretable. Ver decisión 8.3 más abajo.

6. **F5 — Inicialización con k-means + 10 restarts EM**
   - K-means proporciona una semilla sensible que reduce la probabilidad de
     óptimos locales degenerados. 10 restarts con seeds distintos y retención
     del mejor LL en train. Ver decisiones 8.4 y 8.5 más abajo.

7. **F6 — `StandardScaler` ajustado en train, aplicado a holdout**
   - Previene data leakage y equilibra la escala de features antes del HMM
     y del k-means de inicialización. Ver decisión 8.2 más abajo.

8. **F7 — Split 80/20 por ave, estratificado por días válidos**
   - 65 aves en entrenamiento, 17 en holdout. Estratificación garantiza
     representación proporcional de aves con tracking rico y pobre.
     Cambio respecto a LOBO de O2: justificado en decisión 8.8 más abajo.

## Implementación

- **Módulos del paquete:** `src/tfg_aves/hmm/` con submódulos
  `features`, `fit`, `evaluate`, `build`, `_paths`.
- **Algoritmos y fórmulas clave:**
  - `log_displacement_km`: `log(1 + haversine(pos_t, pos_{t-1}))`. Comprime
    el rango [0, 2 000 km] en [0, 7.6].
  - `abs_turning_angle_rad`: `|bearing_t - bearing_{t-1}|` reducido a
    `[0, π]`. Bearing calculado por fórmula de rumbo ortodrómico.
  - `daylight_hours`: fórmula astronómica de Duffie & Beckman (2013) a
    partir de latitud y día del año. No requiere datos externos.
  - `veg_low`, `veg_high`: columnas ECMWF del dataset Movebank, cargadas
    directamente sin transformación (están normalizadas 0-1 de origen).
  - `fit_hmm_with_restarts()`: k-means init → EM → selección por LL train.
  - `relabel_states()`: re-etiquetado determinista por menor
    `μ[log_displacement_km]`.
  - `viterbi_per_bird()`: secuencia de estados más probables por ave.
- **Orquestador:** `build_o3(holdout_frac=0.20, n_restarts=10,
  random_state=0)` en `src/tfg_aves/hmm/build.py` — escribe los 3 ficheros
  de `data/processed/o3/` y devuelve un `BuildO3Result`.
- **Parámetros finales:** `n_components=2`, `covariance_type='diag'`,
  `n_restarts=10`, `holdout_frac=0.20`, `random_state=0`.

## Resultados y validación

### Métricas de ajuste (holdout 17 aves)

| Modelo | LL por observación |
|---|---|
| Modelo A (cinemático) | -2,6192 |
| Modelo B (cinemático + contexto) | -5,7461 |

**Nota importante:** las LL no son comparables directamente entre modelos
porque operan sobre espacios de observación distintos (2 features vs 5).
A tiene LL más alta en su escala propia; B la tiene más baja porque mide
la probabilidad conjunta de 5 features, no 2. Ver decisión 8.10.

### Estadísticas del entregable

- **n_birds_train:** 65 | **n_birds_holdout:** 17
- **n_observations válidas:** 20 672 (de 24 444 totales)
- **% acuerdo A-B:** 60,98 % — B añade migración sobre A=estacionario en el
  73,1 % de desacuerdos; B añade estacionario sobre A=migración en el 22,5 %.

### Hallazgo principal — Coherencia biológica de Modelo B

El artefacto C3 (`o3_fig04_state-vs-biology`) revela el resultado más
relevante de O3:

- **Modelo A** asigna entre el 60 % y el 73 % de las observaciones a
  estado "migración" en todos los meses sin patrón estacional claro.
  Discrimina bien por magnitud de desplazamiento (C1) pero no reconstruye
  la fenología esperada.
- **Modelo B** muestra un patrón estacional nítido: estado "migración"
  ocupa el 84-100 % de las observaciones en otoño (septiembre: 84,4 %,
  octubre: 96,7 %) e invierno (noviembre-febrero: 98-100 %), cae en verano
  (junio: 29,4 %, julio: 23,9 %) y los meses de paso primaveral (mayo:
  56 %, abril: 88,3 %).

La fenología de *Larus fuscus* (gaviota sombría) es bien conocida:
individuos de la subsp. *intermedius* y *fuscus* migran hacia África
subsahariana entre agosto y noviembre y regresan entre marzo y mayo
(Wikelski et al. 2015). El patrón de Modelo B coincide: verano son los
meses en la zona de cría (baja migración), otoño e invierno son migración
activa o invernada africana, primavera es retorno. La semántica de
"estacionario" en B captura el período estival en las zonas de cría; la
de "migración" captura el resto del año cuando el ave está en tránsito o
en la zona de invernada.

**Conclusión crítica:** el contexto ambiental (fotoperiodo + vegetación)
no ha producido circularidad sino que ha **refinado** la señal cinemática,
produciendo estados con mayor coherencia biológica. Modelo B es el que
se recomienda como feature para O4.

### Caracterización de artefactos

- **D1** (`o3_fig01_nstates-aic-bic-sweep`): AIC/BIC para n∈{2,3,4}.
  Respalda la elección de n=2.
- **C1** (`o3_fig02_features-by-state-a`): histogramas de las dos features
  cinemáticas por estado A. Separación clara: estado estacionario tiene
  log_displacement bajo y turning_angle alto; estado migración, patrón
  contrario.
- **C2** (`o3_fig03_features-by-state-b`): histogramas de las cinco features
  del Modelo B por estado. Permite detectar si el contexto domina o refina.
- **C3** (`o3_fig04_state-vs-biology`): principal artefacto de validación.
  Coherencia biológica: % migración por mes y por bin de latitud, para A y B.
  Tabla adjunta con valores numéricos (`o3_tab04_state-vs-biology.csv`).
- **C4** (`o3_fig05_ab-agreement`): matriz de confusión A vs B + histograma
  de desacuerdos por log_displacement. Cuantifica la aportación del
  contexto sobre la cinemática pura.
- **C5** (`o3_fig06_per-bird-state-proportions`): dispersión proporciones
  por ave en A vs B. Revela heterogeneidad individual: algunas aves son
  casi exclusivamente estacionarias, otras casi exclusivamente migratorias
  según ambos modelos; los puntos alejados de la diagonal y=x son aves donde
  el contexto cambia significativamente la inferencia.

### Validación

- 78 tests pasando (`uv run pytest -q`). Ruff limpio (`uv run ruff check
  src tests`). `build_o3()` reproducible desde `daily.parquet` en una sola
  llamada con `random_state=0`.

## Salida materializada

Ficheros en `data/processed/o3/` (gitignored, regenerables):

| Fichero | Contenido |
|---|---|
| `features.parquet` | 24 444 filas × 17 columnas: features, estados, posteriores, flags |
| `models_a_b.pkl` | dict con HMM A, HMM B, scaler A, scaler B, label_map A, label_map B |
| `metrics.parquet` | LL por observación (holdout) + estadísticas de acuerdo A-B |

Las 17 columnas de `features.parquet`:
`bird_id`, `date_utc`, `lat`, `lon`, `log_displacement_km`,
`abs_turning_angle_rad`, `daylight_hours`, `veg_low`, `veg_high`,
`state_a`, `state_b`, `posterior_a_estacionario`, `posterior_a_migracion`,
`posterior_b_estacionario`, `posterior_b_migracion`, `is_observation_valid`,
`in_holdout`.

## Justificación de decisiones metodológicas

*Esta sección replica la sección 8 del spec de O3 en formato citable para
el capítulo 5 de la memoria LaTeX. Cada entrada: decisión, razón,
alternativa descartada.*

### 8.1 No combinar `veg_low` y `veg_high`

- **Decisión:** usar ambas columnas como features separadas en Modelo B
  (5 features, no 4).
- **Razón:** `veg_low` (pastos, cultivos, matorral) y `veg_high` (bosques)
  representan biomas físicamente distintos. Una estepa `(0.8, 0.0)` y un
  bosque `(0.0, 0.8)` tienen la misma suma pero biomas opuestos. Combinarlas
  impondría una asunción no respaldada por los datos sobre cómo se relacionan
  los dos estratos en el comportamiento del ave.
- **Alternativa descartada:** suma o ratio `veg_low + veg_high`. Colapsa
  tres biomas distintos (suelo desnudo, prado, bosque) en valores no
  distinguibles.

### 8.2 Estandarización con `StandardScaler` ajustada en train

- **Decisión:** aplicar `StandardScaler` (μ=0, σ=1) a las features antes
  del HMM. El scaler se ajusta exclusivamente sobre train y se aplica con
  las mismas estadísticas al holdout.
- **Razón:** la emisión gaussiana del HMM es sensible a la escala.
  `daylight_hours` (rango 8-18) dominaría sobre `log_displacement_km`
  (rango 0-6) sin estandarización, sesgando k-means init y ralentizando EM.
  Ajustar solo en train evita *data leakage*: estadísticas del holdout no
  contaminan el preprocesamiento.
- **Alternativa descartada:** no escalar. Sesgo en inicialización difícil
  de corregir.

### 8.3 Covarianza diagonal (`covariance_type='diag'`)

- **Decisión:** `GaussianHMM` con `covariance_type='diag'`. Cada estado
  tiene varianza propia por feature, sin modelar correlaciones entre features.
- **Razón:** práctica estándar en HMMs de movimiento animal (Patterson et al.
  2017; `moveHMM`). Features diseñadas para ser semánticamente ortogonales;
  correlaciones intra-estado son débiles. Mayor robustez estadística que
  `'full'` y directa interpretabilidad (varianza por feature es legible).
- **Alternativa descartada:** `'full'`. Mayor flexibilidad pero peor
  estabilidad numérica y más difícil de explicar en la memoria.

### 8.4 Inicialización con k-means

- **Decisión:** ejecutar k-means con k=2 sobre features estandarizadas antes
  del EM. Centroides como `means_init`, varianza intra-cluster como varianza
  inicial.
- **Razón:** EM converge a óptimos locales. Sin inicialización sensata puede
  caer en soluciones degeneradas (un estado absorbe todo). K-means es un
  pre-clustering rápido que da a EM un punto de partida que ya separa los
  datos en dos grupos razonables.
- **Alternativa descartada:** inicialización aleatoria. Requeriría 20-50
  restarts para confianza equivalente.

### 8.5 Diez restarts de EM

- **Decisión:** repetir (k-means init + EM) 10 veces con seeds distintos.
  Retener el modelo con mayor LL en train.
- **Razón:** incluso con k-means init, EM puede converger a óptimos
  ligeramente distintos. 10 restarts es defensivo (~2 min total para ambos
  modelos sobre 21 000 observaciones) y suficiente para descartar
  trayectorias patológicas.
- **Alternativa considerada:** 5 restarts. Estadísticamente suficiente pero
  con menor margen; la diferencia de tiempo es mínima.

### 8.6 Re-etiquetado de estados por menor `log_displacement`

- **Decisión:** tras entrenar, estado con menor media en `log_displacement_km`
  recibe la etiqueta `estacionario`; el otro, `migración`. Aplicado en A y B.
- **Razón:** `hmmlearn` asigna etiquetas `0`/`1` arbitrariamente según el
  seed. Sin re-etiquetado, `state_a` y `state_b` serían inconsistentes entre
  runs. La regla es biológicamente correcta (menor desplazamiento = más
  estático) e independiente del estado del modelo.
- **Alternativa descartada:** re-etiquetado manual post-inspección.
  Frágil, irreproducible, sesgo del autor.

### 8.7 `n_components = 2` a priori (respaldado por D1)

- **Decisión:** dos estados en ambos modelos.
- **Razón:** el proposal especifica detección binaria. N=2 alinea el modelo
  con la pregunta de investigación y maximiza interpretabilidad. D1
  confirma que N∈{3,4} no produce mejora sustancial de AIC/BIC.
- **Alternativa descartada:** dejar que AIC/BIC elijan N. Más predictivo
  pero los estados extra no admiten etiquetado biológico claro y
  descalibran la narrativa.

### 8.8 Holdout 80/20 estratificado en lugar de LOBO

- **Decisión:** 65 aves en train / 17 en holdout. Split por ave,
  estratificado por número de días válidos. No se usa LOBO (82 folds).
- **Razón:** (1) LOBO en O2 castiga por rutas individuales idiosincráticas;
  en O3 el HMM opera sobre features cinemáticas comparables entre aves, lo
  que reduce pero no elimina ese problema. (2) LOBO en O3 implicaría 82×2×10
  = 1 640 fits, excesivo para el aporte marginal. (3) La estratificación
  garantiza representación proporcional.
- **Alternativa descartada:** LOBO completo. Más riguroso estadísticamente
  pero impracticable y no justificado dado el aprendizaje de O2.

### 8.9 Log-likelihood por observación, no total

- **Decisión:** reportar `LL_total / n_obs_holdout`.
- **Razón:** la LL total depende del tamaño del dataset. La LL por
  observación es comparable entre datasets, configuraciones y reproducciones.
- **Alternativa descartada:** LL total. Engañosa cuando se compara entre
  datasets de diferente tamaño.

### 8.10 No comparar LL entre Modelo A y Modelo B

- **Decisión:** las LL se reportan por separado en su propia escala, no
  como métrica de comparación.
- **Razón:** Modelo A mide `log P(2 features | modelo A)` y Modelo B mide
  `log P(5 features | modelo B)`. Los espacios de observación son distintos;
  la LL no es directamente comparable. La comparación se hace con criterios
  independientes: coherencia biológica (C3) y acuerdo (C4).
- **Alternativa descartada:** AIC/BIC para comparar. Requeriría asumir que
  las features de B son extensión de las de A, lo que es discutible.

### 8.11 Coherencia biológica como criterio principal sin ground truth

- **Decisión:** la validación principal de los estados es su coherencia con
  el conocimiento fenológico de *Larus fuscus*, no contra etiquetas verdaderas.
- **Razón:** no existe ground truth diario sobre el comportamiento de las 82
  aves. Sí existe conocimiento poblacional sobre la fenología de la especie
  (Wikelski et al. 2015) que actúa como criterio externo independiente.
- **Limitación reconocida:** criterio poblacional, no individual — no detecta
  fenologías atípicas de aves concretas.
- **Alternativa descartada:** etiquetas humanas en sub-muestra. Costoso para
  el alcance del TFG y dependiente de la pericia del anotador.

### 8.12 Acuerdo A-B como porcentaje crudo

- **Decisión:** `% acuerdo = n_coincidencias / n_total`. No se usa Cohen's
  kappa.
- **Razón:** Cohen's kappa corrige por acuerdo esperado por azar, apropiado
  cuando los clasificadores son independientes y se evalúan contra ground
  truth. A y B no son independientes (comparten dos features) y no hay
  ground truth. El porcentaje crudo es directamente interpretable.
- **Alternativa descartada:** Cohen's kappa. Más sofisticado, menos
  interpretable, no necesario para el análisis cualitativo de los desacuerdos.

## Conclusiones y limitaciones

### Lo que funciona bien

- El Modelo B descubre estados biológicamente coherentes con la fenología
  de *Larus fuscus* sin supervisión: la separación estacional primavera-otoño
  (migración activa) vs verano (cría) es nítida y coincide con la bibliografía.
- El diseño como ablación (A vs B) permite cuantificar el aporte del contexto
  ambiental sobre la cinemática pura: en este caso, el contexto refina la señal
  sin producir circularidad.
- El entregable `features.parquet` es modular y listo para consumo directo
  por O4 y O5.

### Limitaciones detectadas

- **Validación poblacional, no individual:** la coherencia biológica se
  juzga a nivel agregado; aves con fenologías atípicas quedan sin validación
  individual.
- **Asunción de covarianza diagonal:** no captura correlaciones entre features
  dentro de un estado (e.g., el vínculo entre desplazamiento alto y horas
  de luz cortas en invierno).
- **Sensibilidad de EM a la inicialización:** mitigada por 10 restarts pero
  no eliminada; soluciones subóptimas son posibles.
- **Modelo global:** ignora que distintas aves pueden tener umbrales de
  desplazamiento diferentes para el mismo régimen comportamental. O4
  per-individual abordará esta limitación.

### Aspectos abiertos / futuras mejoras

- O4 (ML): usar `state_a`, `state_b` y las cuatro posteriores como features
  de entrada en los modelos de RF, XGBoost y LightGBM. El estado HMM añade
  información sobre el régimen actual del ave que el Markov(1) global de O2
  no capturaba.
- O5 (evaluación): estratificar el error de predicción por estado HMM para
  ver si los modelos de O4 cometen errores distintos en estado estacionario
  vs migración.
- Extensión futura (fuera del alcance del TFG): HMM por-individuo o HMM
  jerárquico (parámetros compartidos pero iniciados per-ave).

## Notas para la redacción final

- El hallazgo de Modelo B (coherencia biológica nítida) es el resultado
  central del capítulo 5. Dedicarle un párrafo completo con los números
  de la tabla C3: migración al 84-100 % en septiembre-febrero y al 24-56 %
  en junio-julio.
- Conectar explícitamente con O2: "el HMM resuelve la ambigüedad de
  comportamiento que el Markov(1) global no podía capturar, porque el
  argmax de una distribución mezcla sin estados ocultos colapsa hacia el
  self-loop dominante".
- Citar Wikelski et al. (2015) en la sección de validación biológica y
  Patterson et al. (2017) en la elección de `covariance_type='diag'`.
- Aclarar en el texto que LL Modelo A ≠ LL Modelo B porque los espacios
  de observación son distintos (decisión 8.10).
- Revisar si la tutora tiene opinión sobre cuál modelo (A o B) debe
  citarse como "el modelo O3" en el resumen ejecutivo del TFG.
- Bibliografía pendiente:
  - Wikelski M et al. (2015): dataset Movebank.
  - Patterson et al. (2017): HMM para movimiento animal.
  - Duffie & Beckman (2013): fórmula de fotoperiodo astronómico.
  - Rabiner (1989): tutorial clásico HMM.
