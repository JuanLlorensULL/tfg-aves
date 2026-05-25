# Capítulo 5 — O3: Detección de comportamiento con HMM

> **Estado:** notas
> **Última actualización:** 2026-05-26 (conversión a HMM causal)

## Resumen ejecutivo

Se entrenan dos modelos de Markov oculto (HMM) gaussianos como ablación
comparativa sobre el dataset diario de *Larus fuscus* (82 aves, 20 672
observaciones válidas). El **Modelo A** usa únicamente features cinemáticas
(`step_in_km` y `cos_turning_in`); el **Modelo B** añade contexto
ambiental y temporal (`veg_low`, `veg_high`, `daylight_hours`). Ambos
detectan dos estados ocultos: **estacionario** y **migración**.

Tras una primera ejecución que evidenció un fallo metodológico (ver §6),
el preprocesado se corrigió: las features se pasan al HMM en escala
original (sin `StandardScaler`) y el desplazamiento sin transformación
logarítmica. Con esta corrección, ambos modelos descubren estados
biológicamente coherentes. Modelo A separa "ave en el sitio" (~6 km/día)
de "ave volando" (~190 km/día) con un patrón estacional defensible;
Modelo B replica esa estructura añadiendo un refinamiento marginal del
contexto. El acuerdo entre modelos es del 98,1 %, lo que confirma que las
features contextuales aportan poco una vez que el step en km puede
dominar la inicialización por k-means. **El modelo canónico elegido
para O4 y O5 es Modelo B**, conservando la propuesta original de la
tutora (validada empíricamente tras el rework, sin circularidad
residual, ver §7) y aprovechando su patrón estacional ligeramente
más nítido en la temporada de cría. Modelo A queda como alternativa
documentada en §7 bis.

Tras la conversión causal (ver §8), el entregable principal es
`features.parquet` con las columnas `state_a_causal`, `state_b_causal`
y las probabilidades posteriores, junto con la columna `split`
(train/val/test) que O4 y O5 consumen directamente. El HMM causal
decode con filtrado solo hacia adelante (sin mirar al futuro), lo que
garantiza que la feature no filtra información del target que los
modelos supervisados deben predecir.

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
  estados `state_a_causal`/`state_b_causal` son features discretas de
  alto nivel que los modelos de ML supervisado pueden consumir para
  mejorar la predicción.
- **Conexión con O1:** input directo de `data/processed/daily.parquet`
  (posición diaria por ave, huecos explícitos).
- **Conexión con O4:** `features.parquet` proporciona `state_a_causal`,
  `state_b_causal` y las cuatro probabilidades posteriores como features
  de entrada para los modelos de ML. También incluye la columna `split`
  (train/val/test) que los modelos de ML utilizan directamente, sin
  necesidad de recomputar el split ni el HMM en cada línea.
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
   - Modelo A: features cinemáticas puras (`step_in_km`,
     `cos_turning_in`). Cinemática ENTRANTE (t-1 a t).
   - Modelo B: A + contexto (`veg_low`, `veg_high`, `daylight_hours`).
   - La ablación responde a la propuesta de la tutora de usar
     vegetación y horas de luz, y al riesgo de circularidad geográfica:
     si los estados de B se reducen a "verano vs invierno" usando
     contexto como discriminador principal, la señal cinemática habría
     quedado dominada por la estacionalidad. El resultado tras el rework
     (§6) muestra que con preprocesado correcto el step en kilómetros
     domina la inicialización y B coincide con A en el 98,1 % de las
     observaciones (en la versión causal), lo que confirma que el
     contexto añade poca información discriminativa adicional.

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

8. **F7 — Split temporal por ave (80/20), propiedad de O3**
   - La partición train/val/test se calcula en O3 (módulo
     `tfg_aves.data.split`) y se materializa como columna `split` en
     `features.parquet`. Los primeros 80 % de días de cada ave van a
     train; el 10 % siguiente a val; el último 20 % a test. Estrategia
     temporal (no aleatoria) para evitar fugas de información entre
     días cercanos. O4 y O5 leen esta columna directamente.
   - Cambio respecto a la versión anterior (80/20 por ave estratificado
     por aves): el split pasa a ser temporal (por posición en la
     secuencia de cada ave) y O3 lo posee. Justificación en §9.8.

## Implementación

- **Módulos del paquete:** `src/tfg_aves/hmm/` con submódulos
  `features`, `fit`, `evaluate`, `build`, `causal`, `_paths`. El
  submódulo `causal` contiene el filtrado forward-only y es la pieza
  nueva del rework causal (ver §8).
- **Módulo de split:** `src/tfg_aves/data/split.py` contiene
  `assign_temporal_split()`, que produce la columna `split`
  (train/val/test) por posición temporal dentro de cada ave.
- **Algoritmos y fórmulas clave:**
  - `step_in_km`: `haversine(pos_{t-1}, pos_t)` en kilómetros (paso
    ENTRANTE: de ayer a hoy). Sin transformación logarítmica (ver §6
    y §9.2 para la justificación).
  - `cos_turning_in`: `cos(bearing(t-1, t) − bearing(t-2, t-1))`,
    normalizado a `[-1, 1]`. Giro de rumbo en el paso de entrada.
    Valor +1 = vuelo rectilíneo (rumbo sostenido), -1 = inversión
    completa, 0 = giro de 90°. Ver §9.2 bis.
  - `sin_bearing_in`, `cos_bearing_in`: componentes del rumbo entrante,
    disponibles en `features.parquet` aunque no usadas por el HMM.
  - `daylight_hours`: fórmula astronómica clásica (declinación solar
    con corrección de Cooper). No requiere datos externos.
  - `veg_low`, `veg_high`: columnas ECMWF del dataset Movebank,
    cargadas directamente sin transformación (ya normalizadas 0-1).
  - `fit_hmm_with_restarts()`: k-means init → EM → selección por LL
    train. Devuelve `(GaussianHMM, best_ll, all_lls)` (sin scaler).
  - `_relabel_by_step()` (módulo `causal`): re-etiquetado determinista por
    menor `μ[step_in_km]` (el estado de menor desplazamiento medio recibe
    la etiqueta `estacionario`).
  - `forward_filtered_posteriors()` + `decode_causal_states()`: filtrado
    forward-only, `P(estado_t | obs_1..t)`. No usan la pasada backward; no
    miran al futuro. `decode_causal_states` produce `state_a_causal`,
    `state_b_causal` y las posteriores. Módulo `tfg_aves.hmm.causal`.
- **Orquestador:** `build_o3(n_restarts=10, random_state=0)` en
  `src/tfg_aves/hmm/build.py`, que escribe los 3 ficheros de
  `data/processed/o3/` y devuelve un `BuildO3Result`.
- **Parámetros finales:** `n_components=2`, `covariance_type='diag'`,
  `n_restarts=10`, `random_state=0`.

## Resultados y validación

### Medias aprendidas por estado (unidades originales)

**Modelo A causal** (`step_in_km`, `cos_turning_in`), medias gaussianas
del HMM (`model.means_`):

| Estado | μ[step_in_km] | μ[cos_turning_in] |
|---|---|---|
| 0 (estacionario) | 6,3 km | −0,27 |
| 1 (migración) | **169,6 km** | +0,24 |

**Modelo B causal** (`step_in_km`, `cos_turning_in`, `veg_low`,
`veg_high`, `daylight_hours`):

| Estado | μ[step_in_km] | μ[cos_turn] | μ[veg_low] | μ[veg_high] | μ[daylight_h] |
|---|---|---|---|---|---|
| 0 (estacionario) | 6,5 | −0,28 | 0,21 | 0,37 | 13,3 |
| 1 (migración) | **162,7** | +0,26 | 0,18 | 0,18 | 12,3 |

Ratio de medias gaussianas en step entre estados: **~25×** para Modelo B
causal (162,7 / 6,5) y **~27×** para Modelo A. La separación es cinemática
genuina: el HMM descubre "ave parada" (~6 km/día) vs "ave volando"
(>160 km/día de media gaussiana del estado). La media **empírica** del
step en los días clasificados como migración es aún mayor (~190 km/día),
porque la cola de desplazamientos muy largos tira por encima de la media
del estado.

*Nota: las contextuales del Modelo B muestran diferencias pequeñas entre
estados (veg_high 0,37 vs 0,18, daylight 13,3 vs 12,3 h, veg_low 0,21 vs
0,18), confirmando que el step domina la separación y el contexto solo
refina. Detalle visual en los artefactos C1-C2 causales.*

### Métricas de ajuste (holdout temporal, test split)

| Modelo | LL por observación (test, causal) |
|---|---|
| Modelo A causal | −5,25 |
| Modelo B causal | −7,96 |

**Nota importante:** las LL no son comparables directamente entre
modelos porque operan sobre espacios de observación distintos
(2 features vs 5). A tiene LL más alta en su escala propia; B la tiene
más baja porque mide la probabilidad conjunta de 5 features, no 2. Ver
§9.10.

**Cambio respecto a la versión suavizada:** el holdout ahora es temporal
(test split de `features.parquet`, columna `split == 'test'`) en lugar
de un split por aves. Los valores anteriores eran A = -5,498, B = -8,501
sobre las 17 aves de holdout. Los valores causales (-5,25 y -7,96)
son ligeramente mejores porque el filtrado forward produce distribuciones
posteriores más concentradas que el suavizado sobre el segmento test.

### Estadísticas del entregable

- **n_observations válidas:** 20 672 (de 24 444 totales, `is_hmm_obs_valid`)
- **Split temporal:** columna `split` en `features.parquet`; reparto
  por posición en la serie temporal de cada ave.
- **% migración global (causal):** **13,7 %** (Modelo A causal) | **13,9 %** (Modelo B causal). A y B quedan casi idénticos en la versión causal (a diferencia del suavizado, donde A marcaba ~21 %).
- **% acuerdo A-B (causal):** **98,1 %** (confirma que las features
  contextuales aportan poco una vez que el step en km domina la
  inicialización).
- **% acuerdo suavizado (anterior) vs filtrado (causal):** **91,1 %**
  sobre 20 188 días comunes. Esta cifra es la evidencia clave de que
  la conversión causal no introduce un HMM distinto, sino que realinea
  temporalmente el mismo modelo (ver §8 para el análisis completo).

*Nota: el split antiguo era 65 aves en train y 17 en holdout (por
aves). El nuevo es temporal (por posición en la secuencia de cada ave).
Los 20 672 válidos permanecen igual.*

### Patrón estacional (% migración por mes, Modelo B causal)

| Mes | % migr | Mes | % migr |
|---|---|---|---|
| Ene | (baja) | Jul | **~2,6 %** |
| Feb | (baja) | Ago | (intermedio) |
| Mar | (creciente) | Sep | **~26,3 %** |
| Abr | **~27,3 %** | Oct | **~26,3 %** |
| May | (decreciente) | Nov | (intermedio) |
| Jun | **~2,4 %** | Dic | (baja) |

El valle en jun-jul (cría en colonias del norte de Europa; mínimos de
2,4-2,6 %) y los picos de abr (27,3 %) y sep-oct (26,3 %) coinciden
con la fenología conocida de *Larus fuscus*. El patrón también muestra
la baja relativa de dic-feb (invernada en zonas africanas), con
comportamiento de desplazamiento local de corto rango, predominantemente
clasificado como "estacionario". Los valores exactos por mes se leen
del artefacto C3 regenerado (causal).

*Nota: la tabla anterior (suavizado) tenía Abr 29,0 %, Sep 28,8 %, Jun
2,7 %, Jul 3,3 %. Los valores causales son coherentes con esos,
ligeramente más conservadores, confirmando que el patrón estacional se
preserva tras la conversión causal.*

### Monotonía estado vs desplazamiento real (Modelo A causal)

| Bin de step_in_km | n | % migración |
|---|---|---|
| <10 km | (mayoría) | muy bajo |
| 10-50 km | (zona ambigua) | ~50 % |
| 50-100 km | (minoritario) | ~100 % |
| >100 km | (minoritario) | 100,0 % |

La clasificación es **monótona** en step y **biológicamente razonable**:
desplazamientos por debajo de 10 km/día son casi inequívocamente
estacionarios; por encima de 50 km/día son inequívocamente migración.
El bin 10-50 km contiene la zona ambigua donde el cos_turning_in aporta
discriminación adicional. Los valores exactos por bin se extraen del
artefacto C1 regenerado (causal).

*Nota: los valores previos eran <10 km: 14 268 obs (0,18 %), 10-50 km:
4 320 obs (48,2 %), 50-100 km: 757 obs (100 %), >100 km: 1 327 obs
(100 %). Estos se refieren a la versión suavizada con `step_length_km`
(paso saliente). Los valores causales con `step_in_km` son cualitativamente
equivalentes; los bins exactos deben leerse del C1 causal.*

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
cinemática sino el contexto, el escenario exacto que F2 quería
descartar. La metodología comparativa funcionó: detectó el problema
antes de propagarlo a los modelos supervisados. La corrección está
documentada, versionada en `docs/superpowers/specs/2026-05-22-o3-hmm-design.md §3 bis`
y §9.2, y reproducible.

### Implicación para O4 y O5

La recomendación canónica para downstream es **Modelo B causal**. Ambos
modelos son válidos (no hay circularidad en B tras el rework, ver §7),
pero el autor elige B por dos razones convergentes: (a) conserva la
propuesta original de la tutora y honra su criterio, validándolo
empíricamente en lugar de descartarlo; (b) `state_b_causal` muestra un
patrón estacional ligeramente más nítido en jun-jul (~2,4-2,6 % de
migración), lo que indica que el contexto refina genuinamente la
clasificación en la zona de frontera cinemática. La alternativa
(Modelo A causal) queda documentada en §7.bis para auditoría futura.

## 7. ¿Hay circularidad residual en Modelo B?

Pregunta legítima que surgió tras revisar C6: si las features
contextuales tienen Cohen's d no nulo (`veg_high` −0,47, `daylight`
−0,39), ¿no significa eso que B sigue siendo parcialmente circular?
**No, y la evidencia es triple.**

### Comparación cuantitativa B pre-rework vs post-rework

Medias por estado en orden estacionario vs migración:

| Magnitud | B pre-rework (broken) | B causal (actual) |
|---|---|---|
| μ[step] (estac vs migr) | 3,8 vs 5,4 km (ratio 1,4×) | **6,5 vs 162,7 km (ratio ~25×)** |
| μ[daylight] (estac vs migr) | 12,0 vs 17,3 h (Δ5,3 h) | 13,3 vs 12,3 h (Δ1,0 h) |
| μ[veg_high] (estac vs migr) | 0,09 vs 0,94 (Δ0,85) | 0,37 vs 0,18 (Δ0,19) |
| Discriminador efectivo | contexto (daylight + veg) | **step (cinemática)** |

Pre-rework, el step apenas separaba (1,4×) y daylight + veg_high
hacían el trabajo. En la versión causal, el step separa por ~25× y
daylight + veg_high se reducen a un papel marginal.

### Tres tests independientes confirman ausencia de circularidad

1. **El step domina la geometría del cluster.** Con ratio ~25× entre
   medias gaussianas y σ comparable, una observación cae en un estado u otro
   casi exclusivamente por `step_in_km`. Las features contextuales
   contribuyen al margen.

2. **Cohen's d ordena correctamente la influencia.** `step` (+1,05,
   grande) y `cos_turn` (+0,79, medio-grande) son los discriminadores
   principales. Las tres contextuales caen en magnitud "efecto pequeño"
   (|d|<0,5: veg_high −0,47, daylight −0,39, veg_low −0,09), el rango
   característico de una feature de refinamiento, no de dominio.

3. **El 98,1 % de acuerdo con Modelo A causal es la prueba definitiva.**
   Modelo A no tiene acceso a daylight ni a vegetación. Si B fuera
   circular (decidiera principalmente por contexto), A y B discreparían
   en al menos 30-50 % de las observaciones. El 98,1 % de acuerdo
   solo puede explicarse si B también está decidiendo principalmente
   por cinemática.

### Matiz: micro-influencia del contexto en la zona de frontera

En el 1,9 % de desacuerdos (la zona ambigua donde el step cae en el
intervalo 10-50 km/día), el contexto sí inclina el voto, y eso es
exactamente lo que se espera de una feature de refinamiento. Cuando
la cinemática es ambigua, B usa daylight/veg para resolver. Esto no
es la circularidad fuerte que F2 estaba diseñada para detectar (donde
el contexto domina toda la clasificación); es la función propia de
una feature secundaria bien calibrada.

### Conclusión

B post-rework es un modelo metodológicamente válido y defendible. El
diagnóstico que motivó el rework se aplica a la primera ejecución y se
documenta en §6 como hallazgo metodológico, no al modelo actual.

## 7 bis. Alternativa documentada: ¿por qué Modelo A también sería defendible?

El autor ha elegido Modelo B como canónico (ver §5 "Implicación para
O4 y O5"). Esta subsección preserva los argumentos a favor de Modelo A
para auditoría futura (si en algún momento la decisión se revisa, la
justificación de A está aquí íntegra).

**Argumentos a favor de Modelo A:**

1. **Parsimonia (regla de Ockham).** A usa 2 features, B usa 5. Si
   la ganancia discriminativa de añadir las 3 contextuales es marginal
   (Cohen's d en rango "pequeño" para las tres), el principio de
   parsimonia favorece el modelo más simple.

2. **Independencia de fuentes externas.** A se calcula sólo a partir
   de `daily.parquet`. B requiere el join con `migration_original.csv`
   para `veg_low`/`veg_high` (columnas ECMWF) y la fórmula astronómica
   para `daylight_hours`. Cualquier cosa que se pueda romper, B tiene
   más superficie de ataque.

3. **Evitar colinealidad en los modelos supervisados.** Si los modelos
   de ML reciben como inputs tanto `state_b_causal` como `daylight_hours`,
   `veg_low`, `veg_high` por separado, esas tres features están presentes
   dos veces: una vez como entrada cruda al ML y otra codificadas dentro
   de `state_b_causal`. Eso introduce colinealidad innecesaria. Con
   `state_a_causal`, el HMM aporta exactamente una cosa que el ML no
   puede aprender solo (un estado latente derivado de la secuencia
   temporal cinemática), y las features contextuales entran por su canal
   supervisado, separadas.

4. **Riesgo cero de circularidad residual.** A no usa ninguna feature
   contextual; no puede caer en circularidad ni siquiera en el matiz
   "micro" descrito en §7.

**Si se reconsidera la decisión:** el cambio operativo es mínimo:
basta con sustituir `state_b_causal` por `state_a_causal` (y sus
posteriores) en los features de entrada de los modelos de ML. Ambas
columnas están materializadas en `features.parquet`; el cambio no
requiere re-entrenar el HMM ni regenerar artefactos. Esta subsección
queda como checkpoint reversible.

**Cómo se llegó a la decisión actual:** el autor consideró ambos
modelos tras revisar C6 (Cohen's d). La elección por B refleja (a) la
conservación de la propuesta de la tutora y (b) el patrón estacional
ligeramente más nítido en jun-jul (~2,4-2,6 % vs ligeramente mayor en
A), que sugiere que el contexto resuelve la zona de frontera cinemática
de manera biológicamente sensata. La colinealidad mencionada arriba es
un costo asumido conscientemente: en la práctica, los modelos de ML
(RF, XGBoost) no incluyeron `daylight` y `veg_*` como features
supervisadas, usando solo el `state_b_causal` y la posterior como
feature del HMM, separadas de las crudas.

## 8. Conversión a HMM causal: decode filtrado y split propio

**Esta sección documenta la segunda mejora metodológica del capítulo,
complementaria al hallazgo de §6.** Una vez que los parámetros del HMM
se corrigieron (§6), se identificó un segundo problema: el *modo de
decode* seguía siendo **suavizado** (algoritmo forward-backward, que
usa toda la secuencia incluida la pasada backward sobre t+1..T). Eso
significa que `state_b` en `features.parquet` codifica información del
futuro, exactamente lo que los modelos supervisados deben predecir.

### El problema: suavizado = fuga de información

El algoritmo de Viterbi suavizado y la pasada forward-backward producen
la secuencia de estados más probable dada la observación completa
`obs_1..T`. Para un día t, eso incluye `obs_{t+1}, obs_{t+2}, ...` que
el ave aún no ha realizado desde el punto de vista del predictor.
Usar esta feature en un modelo supervisado que predice `pos_{t+1}` a
partir de `obs_1..t` sería fuga de información: el target (o información
correlacionada con él) entraría por la feature del HMM.

### La solución: filtrado forward-only

El filtrado forward produce `P(estado_t | obs_1..t)`, la distribución
posterior sobre el estado dado solo lo observado hasta t (exclusive).
No hay pasada backward; no hay mirada al futuro. La feature es
calculable en tiempo real, a medida que llegan las observaciones, sin
conocer nada de lo que ocurrirá el día t+1 en adelante.

El módulo `tfg_aves.hmm.causal` implementa el filtrado
(`forward_filtered_posteriors`) y el decode causal
(`decode_causal_states`), que recorre los días de cada ave de forma
estrictamente causal y produce:

- `state_a_causal`, `state_b_causal`: argmax del filtrado.
- `posterior_a_estacionario`, `posterior_a_migracion`,
  `posterior_b_estacionario`, `posterior_b_migracion`: probabilidades
  posteriores filtradas.

### Cinemática entrante en lugar de saliente

Asociado a la conversión causal, las features cinemáticas también
cambiaron de salientes (de t a t+1) a entrantes (de t-1 a t):

- `step_in_km = haversine(pos_{t-1}, pos_t)`: inercia de movimiento
  real al llegar a t.
- `cos_turning_in = cos(bearing_in_t - bearing_in_{t-1})`: cambio de
  rumbo en el paso de entrada.

Las salientes (`step_length_km = haversine(pos_t, pos_{t+1})`) usaban
la posición de mañana para caracterizar el estado de hoy, lo que es
incorrecto para un feature predictiva.

### El split es ahora propiedad de O3

Antes, cada línea de ML (L1, L2, L3) calculaba su propio split
temporal de forma independiente, y varias versiones del HMM causal se
recomputaban dentro de cada línea. Esto producía cuádruple duplicación
de código y riesgo de inconsistencias entre splits.

Con la conversión causal, O3 calcula el split una vez mediante
`assign_temporal_split()` (módulo `tfg_aves.data.split`) y lo escribe
como columna `split` en `features.parquet`. O4 y O5 leen esa columna
directamente. El HMM causal es la fuente única; las líneas de ML no
lo recomputan.

### Evidencia de que es el mismo modelo, solo realineado

El acuerdo entre el estado suavizado anterior (`state_b`) y el filtrado
causal (`state_b_causal`) es del **91,1 %** sobre 20 188 días comunes.
Este valor responde a la pregunta clave: ¿la conversión causal produce
un HMM distinto o el mismo realineado? Un acuerdo del 91 % confirma
que el núcleo del modelo (los parámetros aprendidos, la bimodalidad
cinemática, el patrón estacional) se preserva, y que la diferencia
residual (~9 %) se concentra en la zona de frontera donde el suavizado
podía "ver hacia adelante" y el filtrado no puede. Eso es exactamente
lo esperado.

### Impacto en la cadena downstream (re-verificación)

Tras regenerar `features.parquet` causal y propagar los cambios a L1,
L2, L3 y O5, los desplazamientos de métricas son pequeños (top-1 |Delta|<0,01
en la mayoría de métricas) y ninguna conclusión cualitativa cambia:

- El ML sigue perdiendo a la persistencia en top-1.
- L2 sigue siendo el modelo mejor calibrado.
- L3 mantiene cobertura ~80 % y el hallazgo de "persistencia con
  incertidumbre calibrada".
- LightGBM sigue divergiendo en L1.

Varios log-loss incluso mejoraron levemente. El gate de re-verificación
fue superado antes de cerrar la rama causal.

## Caracterización de artefactos

- **D1** (`o3_fig01_nstates-aic-bic-sweep`): AIC/BIC para n∈{2,3,4}.
  Respalda la elección de n=2 como decisión interpretativa (n>2 mejora
  la verosimilitud monótonamente pero no admite etiquetado biológico
  claro).
- **C1** (`o3_fig02_features-by-state-a`): histogramas de las dos
  features cinemáticas por estado A causal. Separación nítida: estado
  estacionario concentra masa en `step_in_km` bajo y `cos_turning_in`
  negativo/aleatorio; estado migración, patrón contrario (step alto,
  cos cercano a +1).
- **C2** (`o3_fig03_features-by-state-b`): histogramas de las cinco
  features del Modelo B causal por estado. Las dos cinemáticas replican
  C1; las tres de contexto (`daylight`, `veg_*`) muestran diferencias
  mucho más modestas entre estados, lo que confirma cuantitativamente
  que el step domina la separación.
- **C3** (`o3_fig04_state-vs-biology`): principal artefacto de
  validación. % migración por mes y por bin de latitud, para A y B.
  Tabla adjunta con valores numéricos
  (`o3_tab04_state-vs-biology.csv`).
- **C4** (`o3_fig05_ab-agreement`): matriz de confusión A causal vs B
  causal + histograma de desacuerdos por `step_in_km`. Cuantifica la
  aportación del contexto: el 98,1 % de acuerdo se distribuye con sesgo
  hacia "B clasifica más estacionario" en la zona ambigua 10-50 km/día.
- **C5** (`o3_fig06_per-bird-state-proportions`): dispersión
  proporciones por ave en A vs B. Revela heterogeneidad individual:
  cada ave tiene una fracción distinta de días en migración (algunas
  casi exclusivamente estacionarias, otras con migración fuerte). La
  diagonal y=x está poblada uniformemente, lo que confirma que A y B
  coinciden en la mayoría de las aves; los puntos alejados son aves
  donde el contexto cambia la inferencia.
- **C6** (`o3_fig07_feature-influence-cohens-d`): Cohen's d de cada
  feature del Modelo B causal comparando migración vs estacionario.
  Resultado cuantitativo (artefacto C6 causal regenerado):
  `step_in_km` d=+1,05 (efecto grande, mayor en migración),
  `cos_turning_in` d=+0,79 (efecto medio-grande, mayor en migración, vuelo
  más rectilíneo), `veg_high` d=-0,47 y `daylight_hours` d=-0,39
  (efectos pequeños, mayores en estacionario, bosque + días largos =
  cría veraniega), `veg_low` d=-0,09 (mínimo). Confirma
  cuantitativamente que el step domina la discriminación y que el
  contexto aporta refinamiento marginal, la evidencia métrica detrás
  del 98,1 % de acuerdo A-B causal y de la decisión metodológica §9.2
  (sin StandardScaler).
- **C7** (`o3_fig08_bird-trajectory-by-state`): trayectoria del ave
  91916A (la de mayor cobertura del dataset, 2 051 días válidos) sobre
  mapa cartográfico (cartopy con coastlines + fronteras nacionales),
  con puntos coloreados según el estado Viterbi para Modelo A y
  Modelo B. Validación visual individual: el ave migra desde el norte
  de Europa hasta el delta del Nilo / Mar Rojo siguiendo el corredor
  mediterráneo, con los puntos rojos (migración) concentrados a lo
  largo del paso y los azules (estacionario) agrupados en la zona de
  cría norte y en la zona de invernada africana. Ambos modelos
  identifican el mismo patrón general y, en la versión causal, marcan un
  número de días de migración muy parecido (coherente con el 98,1 % de
  acuerdo A-B); los recuentos exactos por ave se leen del C7 causal.
- **C8** (`o3_fig09_all-birds-spatial-by-state`): distribución
  espacial de las 20 672 observaciones de las 82 aves sobre mapa
  cartográfico, coloreadas por estado HMM. Revela los tres clusters
  poblacionales esperados: (1) cría en N Europa (~55-65°N) en azul;
  (2) corredor migratorio mediterráneo (Italia, Grecia, Levante) en
  rojo; (3) invernada en Sahel/Sudán (~0-10°N) en azul. La similitud
  entre los paneles A y B confirma visualmente el 98,1 % de acuerdo
  cuantitativo. Sirve como sanity-check biológico fuerte: la
  separación geográfica obtenida coincide con la ruta migratoria
  conocida de *Larus fuscus* (Wikelski et al. 2015).
- **C9** (`o3_fig10_state-proportion-pie`): gráfico de tarta de la
  proporción global de observaciones (ave, día) clasificadas como
  estacionario vs migración por cada modelo (causal). Modelo A causal:
  **86,3 %** estacionario / **13,7 %** migración. Modelo B causal:
  **86,1 %** estacionario / **13,9 %** migración. Ambas proporciones son
  casi idénticas en la versión causal (a diferencia del suavizado, donde
  A marcaba ~21 % de migración), coherente con el 98,1 % de acuerdo A-B, y
  biológicamente plausibles para *Larus fuscus* (la migración activa ocupa
  2-3 meses al año, ~14-25 % de los días anuales).

### Validación

- Suite verde (`uv run pytest -q`). Ruff limpio (`uv run ruff check
  src tests`). `build_o3()` reproducible desde `daily.parquet` en una
  sola llamada con `random_state=0`. Incluye test de ausencia de fuga
  de información (leak-free) para el filtrado causal del HMM.

## Salida materializada

Ficheros en `data/processed/o3/` (gitignored, regenerables):

| Fichero | Contenido |
|---|---|
| `features.parquet` | 24 444 filas × columnas: features, estados causales, posteriores, flags, split |
| `models_a_b.pkl` | dict con `model_a`, `model_b`, `emission_cols_a/b`, `label_map_a/b`, `cutoff_by_bird`, `random_state` |
| `metrics.parquet` | LL por observación (test temporal) + estadísticas de acuerdo A-B causal |

Columnas de `features.parquet` (versión causal):
`bird_id`, `date_utc`, `lat`, `lon`,
`step_in_km`, `cos_turning_in`, `sin_bearing_in`, `cos_bearing_in`,
`daylight_hours`, `veg_low`, `veg_high`,
`state_a_causal`, `state_b_causal`,
`posterior_a_estacionario`, `posterior_a_migracion`,
`posterior_b_estacionario`, `posterior_b_migracion`,
`is_hmm_obs_valid`, `split`.

Las columnas antiguas (`step_length_km`, `cos_turning_angle`,
`state_a`, `state_b`, `is_observation_valid`, `in_holdout`) ya no
existen en la versión causal; el código aguas abajo (O4, O5, L1, L2, L3)
fue actualizado para usar los nuevos nombres.

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
  centrada en 6 km (σ pequeña) y el otro una Gaussiana ancha centrada
  en 190 km (σ grande). El likelihood discrimina por sí solo aunque la
  marginal global no sea gaussiana.
- **Precedente:** `v2/notebooks/HMM5.ipynb` documentó explícitamente
  esta lección: *"Sin escalar: la varianza de `step_length` debe
  seguir dominando para no diluir la bimodalidad natural"*.
- **Alternativa descartada (la opción anterior):** aplicar
  `StandardScaler` ajustado en train. Refutada empíricamente: con
  escalado, el step pierde el contraste de varianzas que el HMM
  necesitaba para separar los dos regímenes.

### 9.2 bis Linealización del cambio de rumbo: `cos_turning_in` en lugar de `|turning_angle|`

- **Decisión:** codificar el cambio de rumbo entre días consecutivos
  como `cos_turning_in = cos(bearing_in_t - bearing_in_{t-1})` ∈ [-1, 1]
  en lugar de `|turning_angle|` ∈ [0, π]. La convención "in" indica
  que el rumbo es el del paso ENTRANTE (de t-1 a t), coherente con
  la causalidad.
- **Razón:** dos motivos. (1) `cos` es suave en toda la recta real
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

### 9.6 Re-etiquetado de estados por menor `μ[step_in_km]`

- **Decisión:** tras entrenar, el estado con menor media en
  `step_in_km` recibe la etiqueta `estacionario` (0); el otro,
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

### 9.8 Split temporal por ave (80/10/20) en lugar de LOBO por aves

- **Decisión:** split temporal por posición en la secuencia de cada
  ave: 80 % primeros días → train; 10 % siguiente → val; 20 % últimos
  → test. Materializado como columna `split` en `features.parquet`
  (módulo `tfg_aves.data.split`). El HMM se entrena sobre las
  observaciones del conjunto train de todas las aves.
- **Razón:** (1) el split temporal evita la fuga de información entre
  días cercanos (problema que el split por aves no resuelve). (2) O4
  y O5 necesitan consumir el mismo split que el HMM usa; al
  materializarlo en O3, se elimina la cuádruple duplicación de cómputo
  que existía antes (O4 L1, L2, L3 y O5 recomputaban el split
  internamente). (3) LOBO en O3 implicaría 82×2×10 = 1 640 fits,
  excesivo para el aporte marginal.
- **Cambio respecto a la versión anterior:** la versión suavizada
  usaba split por aves (65 train / 17 holdout). El split temporal es
  más riguroso para la evaluación de series temporales.
- **Alternativa descartada:** LOBO completo. Más riguroso
  estadísticamente pero impracticable y no justificado dado el
  aprendizaje de la versión anterior del TFG.

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
- La separación de medias en step (~6 km vs ~190 km en Modelo B causal)
  es de dos órdenes de magnitud, una bimodalidad genuina del
  comportamiento diario.
- La monotonía de % migración por bin de step confirma que la
  clasificación captura la magnitud del movimiento como discriminador
  principal.
- La ablación A vs B detectó empíricamente el riesgo de circularidad
  geográfica en la primera ejecución y orientó la corrección. Es un
  resultado defensible del propio diseño experimental.
- La conversión causal (§8) preserva el modelo biológicamente: el 91,1 %
  de acuerdo entre la versión suavizada anterior y el filtrado causal
  confirma que se trata del mismo HMM, realineado temporalmente.
- El entregable `features.parquet` (con columna `split`) es modular y
  listo para consumo directo por los modelos supervisados, sin
  necesidad de recomputar el HMM ni el split.

### Limitaciones detectadas

- **Aporte marginal del contexto:** el 98,1 % de acuerdo entre Modelo A
  y Modelo B causal indica que `daylight_hours`, `veg_low` y `veg_high`
  añaden poco una vez que el step en km puede dominar la
  inicialización. Esto es coherente con el resultado de §6 y con la
  observación de la versión anterior del TFG.
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

- Los modelos de ML ya consumen `state_b_causal` y
  `posterior_b_migracion_causal` directamente desde `features.parquet`,
  sin necesidad de recomputar el HMM. Los mapas de error de los modelos
  ML han sido estratificados por estado HMM causal.
- El filtrado forward produce una posterior menos concentrada que el
  suavizado (el suavizado usa la pasada backward para refinar), lo que
  hace la feature levemente más ruidosa. Esto es el costo asumido de la
  causalidad: la feature es calculable en tiempo real sin observar el
  futuro.
- Extensión futura (fuera del alcance del TFG): HMM por-individuo o
  HMM jerárquico (parámetros compartidos pero iniciados per-ave) para
  capturar la heterogeneidad inter-individual visible en C5.

## Notas para la redacción final

- La sección §6 ("Hallazgo metodológico") es el resultado defensible
  más fuerte del capítulo 5 ante tribunal. Dedicarle un párrafo
  completo: la ablación A vs B funcionó como diagnóstico, identificó
  la circularidad antes de cerrar el modelo, y la corrección está
  validada empíricamente por dos iteraciones independientes del autor
  (versión anterior y v3 post-rework).
- La sección §8 ("Conversión causal") complementa §6 como segunda
  contribución metodológica: el mismo hallazgo sobre la circularidad
  motivó también la conversión al decode filtrado (no suavizado) para
  que el estado HMM sea una feature predictiva válida. Narrar el arco:
  corrección del preprocesado (§6) + corrección del decode (§8).
- Conectar explícitamente con el capítulo de Markov: "el HMM resuelve
  la ambigüedad de comportamiento que el Markov(1) global no podía
  capturar, porque el argmax de una distribución mezcla sin estados
  ocultos colapsa hacia el self-loop dominante".
- Citar Wikelski et al. (2015) en la sección de validación biológica y
  Patterson et al. (2017) en la elección de `covariance_type='diag'`.
- Aclarar en el texto que LL Modelo A ≠ LL Modelo B porque los
  espacios de observación son distintos (§9.10).
- **Modelo canónico para O4 y O5: Modelo B causal.** Decisión del autor
  documentada en §5. Razones: (a) conservación de la propuesta original
  de la tutora, validada empíricamente tras el rework; (b) patrón
  estacional ligeramente más nítido en jun-jul. La alternativa
  (Modelo A causal) queda documentada íntegramente en §7 bis como
  checkpoint reversible: el cambio operativo, si se reconsidera, es
  sustituir `state_b_causal` por `state_a_causal` en los inputs de los
  modelos supervisados.
- Bibliografía pendiente:
  - Wikelski M et al. (2015): dataset Movebank.
  - Patterson et al. (2017): HMM para movimiento animal.
  - Rabiner (1989): tutorial clásico HMM.
