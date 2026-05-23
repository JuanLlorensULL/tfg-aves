# Diseño del rework causal de O4 — cinemática entrante e inercia real

- **Fecha:** 2026-05-23
- **Objetivo del TFG:** O4 (65 h) — predicción de celda siguiente con
  modelos supervisados (Random Forest, XGBoost, LightGBM)
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado, pendiente de implementación
- **Sustituye a:** `2026-05-22-o4-ml-design.md` en lo relativo a las
  features. El resto del diseño de O4 (split, familias, baselines,
  criterio log-loss, arquitectura de módulos) se mantiene.

## 1. Resumen

La versión actual de O4 emplea features cinemáticas (`step_length_km`,
`cos_turning_angle`) y un estado HMM (`state_b`, `posterior_b_migracion`)
que se calculan **hacia adelante** (desplazamiento `t → t+1`). Como el
target de O4 es la celda del día `t+1`, esas features **codifican el
propio target** (su magnitud y, parcialmente, su dirección): no son
conocibles en el momento real de predicción. Es una fuga de información
(*data leakage*).

Este rework redefine **toda la cinemática como entrante/causal**
(desplazamiento `t-1 → t`, conocido el día `t`), lo que (a) elimina la
fuga y (b) introduce la **inercia real del movimiento** como señal
predictiva. El estado HMM se reconstruye de forma causal: emisión sobre
la cinemática entrante y **decodificado por filtrado forward-only**
(`P(estado_t | obs_1..t)`), que no mira días futuros.

La versión con fuga se trata como un error de implementación
pre-publicación: no se documenta como episodio metodológico ni se
genera comparativa antes/después. La versión causal pasa a ser **la**
O4 canónica; todos los números y hallazgos de O4 se regeneran sobre
ella.

**Fuera de alcance:** la línea de viento L1. Sus artefactos en
`O4_L1V1_DIR` quedan como están (conocidos-contaminados) y no se tocan
en este rework.

## 2. Contexto y constraints

- **La fuga, en concreto.** En `src/tfg_aves/hmm/features.py:109`,
  `step_length_km(t) = haversine(pos(t), pos(t+1))`; y
  `cos_turning_angle(t)` usa el rumbo saliente `bearing(t → t+1)`
  (`hmm/features.py:114`). El target de O4
  (`ml/features.py:59-66`) es la celda de `(lat[t+1], lon[t+1])`.
  Por tanto `step_length_km(t)` es literalmente la distancia
  ave→target. Además `viterbi_per_bird` decodifica con
  `predict`/`predict_proba` (forward-backward, `hmm/evaluate.py:49-50`),
  con lo que `state_b(t)`/`posterior_b_migracion(t)` heredan la fuga de
  la emisión **y** además incorporan observaciones de `t+1, t+2, …`
  por el suavizado. Doble contaminación para el target predictivo.
- **O3 no se modifica.** O3 es un modelo descriptivo: etiquetar el
  régimen del día `t` con el desplazamiento de ese día y suavizado
  global es legítimo porque O3 no predice el futuro. La fuga sólo
  existe al reusar su salida *como predictor* en O4. Por tanto el
  arreglo vive en el lado de O4. `tfg_aves.hmm` y los artefactos de O3
  (`data/processed/o3/*`, tag `v0.3.1-o3-rework`, memoria §5-7,
  artefactos C1-C9/D1) quedan intactos.
- **Restricción de la tutora (Adriana).** El estado HMM **debe** entrar
  como feature (Modelo B es su propuesta). Por eso no se elimina: se
  reconstruye causal. Además pidió usar **el rumbo y la variabilidad
  del ángulo del rumbo**; la versión actual sólo incluía la
  variabilidad (vía `cos_turning_angle`), no el rumbo. El rework añade
  el rumbo.
- **Entregables disponibles (sin cambios):**
  `data/processed/o3/features.parquet` (fuente de `lat`, `lon`,
  `veg_low`, `veg_high`, `daylight_hours`, `is_observation_valid` y la
  cinemática que se **recalcula**), `data/processed/o2/cells.parquet`
  (grid 0,5°), `data/processed/daily.parquet`.
- **Convenciones del proyecto (CLAUDE.md):** decisiones no triviales
  con figura/tabla + caption castellano vía `save_artifact()`; commits
  castellano sin trailer de IA; identificadores en inglés; tests
  pytest sobre datasets sintéticos; módulos puros + orquestador
  `build_o4`. Restricción de simplicidad (sin sobreingeniería).
- **Coherencia biológica como criterio durable:** todo resultado se
  contrasta con la fenología de *Larus fuscus* (jun-jul cría, abr-may
  y sep-oct migración, dic-feb invernada africana).

## 3. Decisiones de diseño fijadas

Acordadas en el brainstorming previo a este spec:

| # | Decisión | Valor |
|---|---|---|
| R1 | Sentido de la cinemática | **Entrante/causal**: el desplazamiento, rumbo y giro del día `t` describen el tramo `t-1 → t` (conocido el día `t`) |
| R2 | Inercia — magnitud | `step_in_km = haversine(pos(t-1), pos(t))` |
| R3 | Inercia — rumbo | `sin_bearing_in`, `cos_bearing_in` = sin/cos de `bearing(t-1 → t)` (**nueva**, petición de la tutora) |
| R4 | Inercia — variabilidad del rumbo | `cos_turning_in = cos(bearing(t-1→t) − bearing(t-2→t-1))` (giro instantáneo causal; análogo causal del antiguo `cos_turning_angle`) |
| R5 | Estado HMM | `state_b_causal`, `posterior_b_migracion_causal` de un **HMM causal** (§6); reemplazan a `state_b`/`posterior_b_migracion` |
| R6 | Features supervisadas de O4 | `lat`, `lon`, `sin_doy`, `cos_doy`, `step_in_km`, `sin_bearing_in`, `cos_bearing_in`, `cos_turning_in`, `state_b_causal`, `posterior_b_migracion_causal` (10 features) + `bird_id` en personalizado |
| R7 | Emisión del HMM causal | `step_in_km`, `cos_turning_in`, `veg_low`, `veg_high`, `daylight_hours` (5, paralela a Modelo B). **Sin rumbo absoluto** → invariante a dirección, evita reintroducir circularidad geográfica |
| R8 | Decodificado del HMM causal | **Filtrado forward-only**: `P(estado_t | obs_1..t)`. No usa suavizado backward |
| R9 | Ajuste del HMM causal | Sobre el tramo de **train temporal** de O4. Reutiliza `fit_hmm_with_restarts`, `relabel_states` de O3 |
| R10 | Máscara de validez | Fila candidata sólo si `(t-2, t-1, t, t+1)` son válidos y consecutivos en calendario (racha de 4 días) |
| R11 | Tratamiento de la versión con fuga | Error pre-publicación; no se narra ni se compara. La causal es la canónica |
| R12 | Invariantes heredados de O4 | Split temporal 80/10/20 por ave (F4-F5), familias RF/XGB/LGBM (F2), dos modos personalizado/poblacional (F3), baselines persistencia + Markov(1) (F11), criterio log-loss (F13). **Sin cambios** |

## 4. Features y semántica causal

Para predecir la posición del día `t+1`, toda feature debe ser
conocible al final del día `t` (observadas las posiciones hasta `t`
inclusive).

### 4.1 Features causales (conocidas en `t`)

| Feature | Definición | Requiere |
|---|---|---|
| `lat`, `lon` | posición en `t` | `t` válido |
| `sin_doy`, `cos_doy` | día del año cíclico de `t` | `t` válido |
| `step_in_km` | `haversine(pos(t-1), pos(t))` | `t-1, t` válidos+consecutivos |
| `sin_bearing_in`, `cos_bearing_in` | `bearing(t-1 → t)` | `t-1, t` válidos+consecutivos |
| `cos_turning_in` | `cos(bearing(t-1→t) − bearing(t-2→t-1))` | `t-2, t-1, t` válidos+consecutivos |
| `state_b_causal` | estado filtrado del HMM causal en `t` | secuencia hasta `t` |
| `posterior_b_migracion_causal` | `P(migración_t | obs_1..t)` | secuencia hasta `t` |
| `bird_id` | identificador (sólo personalizado) | — |

### 4.2 Por qué el rumbo va fuera del HMM

El **rumbo absoluto** (N/S/E/O) es una señal potente para los modelos
supervisados —un ave en migración tiende a mantener el rumbo, así que
indica *hacia dónde* estará el destino—, por eso entra como feature
cruda de O4. Pero **dentro de la emisión del HMM es peligroso**: el HMM
podría aprender "va al sur" vs. "va al norte" en lugar de
"migra vs. estacionario", reintroduciendo la circularidad geográfica
que motivó el rework de O3. La emisión del HMM (R7) usa por tanto sólo
magnitud + variabilidad del rumbo (giro), invariantes a la dirección.

### 4.3 Máscara de validez (racha de 4 días)

`cos_turning_in` exige `t-2`; la inercia exige `t-1`; el target exige
`t+1`. Una fila plenamente poblada necesita `(t-2, t-1, t, t+1)`
válidos y consecutivos. Es más estricta que la máscara actual (racha
de 3) pero, con mediana de racha de 12 días y p90 de 142 (O1), la
pérdida de muestra es pequeña; se cuantifica en el notebook con
`save_artifact`. Como en la versión actual, **ninguna fila atraviesa un
hueco de calendario** (se mantienen las barreras gap-aware del spec
original §8.11).

Para mantener coherencia entre familias (RF no tolera NaN), se exige
que **todas** las features cinemáticas estén presentes; las filas que
no completen la racha de 4 se descartan.

## 5. Arquitectura y flujo de datos

```
┌─ data/processed/o3/features.parquet  (lat, lon, veg_*, daylight, is_observation_valid)
│  └─ data/processed/o2/cells.parquet  (grid 0,5°, target)
│
├─ src/tfg_aves/ml/
│  ├─ features.py     ── recalcula cinemática causal (step_in, bearing_in, turning_in)
│  │                     desde posiciones; aplica máscara de 4 días; asigna celdas;
│  │                     split temporal por ave (sin cambios).
│  ├─ hmm_causal.py   ── NUEVO. Ajusta HMM causal sobre train (emisión R7) y
│  │                     decodifica por filtrado forward-only. Reutiliza primitivas
│  │                     de tfg_aves.hmm.fit. No toca tfg_aves.hmm.
│  ├─ train.py        ── sin cambios (RF/XGB/LGBM con config §8.6 de O4).
│  ├─ evaluate.py     ── sin cambios (métricas, baselines).
│  └─ build.py        ── orquesta: features causales → fit HMM causal en train →
│                        filtrar todo → merge state/posterior → entrenar 6 modelos.
│
└─ data/processed/o4/  ── REGENERADO: model_*.pkl, predictions_test.parquet, metrics.parquet
```

`tfg_aves.hmm` queda **congelado**: el decodificado filtrado vive en
`ml/hmm_causal.py` para no introducir ningún cambio de comportamiento
en el `build_o3()` actual.

## 6. HMM causal

### 6.1 Emisión

`GaussianHMM` (`covariance_type='diag'`, 2 estados), idéntica
configuración que O3, sobre las 5 features de R7:
`[step_in_km, cos_turning_in, veg_low, veg_high, daylight_hours]`.
Es la estructura del Modelo B de la tutora, sólo realineada al sentido
entrante. Como el conjunto de desplazamientos diarios es el mismo
(cada tramo entre dos días consecutivos es "saliente" para `t` y
"entrante" para `t+1`, sólo cambia el día al que se atribuye), se
**espera que los parámetros del HMM salgan casi idénticos a O3**
(medias ≈ 5,8 km estacionario / ≈ 164 km migración). Esto se verifica
en el notebook como sanity-check.

### 6.2 Ajuste

- Se ajusta **sólo sobre el tramo de train temporal de O4** (primeros
  ~72 % de días de cada ave, antes de la partición de validación). Al
  ser no supervisado, el HMM nunca ve el target; el ajuste sobre train
  es higiene estándar de transformación de features.
- Reutiliza `build_sequences`, `fit_hmm_with_restarts` (K-means + 10
  restarts) y `relabel_states` (etiqueta estacionario/migración por
  media de `step`) de `tfg_aves.hmm.fit`.

### 6.3 Decodificado por filtrado forward-only

Para cada ave, sobre su secuencia ordenada de días válidos, se calcula
el **posterior filtrado** `γ_t = P(estado_t | obs_1..t)`. Esto usa sólo
observaciones hasta `t`, a diferencia del posterior suavizado de O3
(`P(estado_t | obs_1..T)`).

Se implementa como recursión forward en log-espacio (estándar,
~15 líneas), usando `log(startprob_)`, `log(transmat_)` y la
log-verosimilitud de emisión del modelo ajustado
(`model._compute_log_likelihood(X)`), sin depender de API privada de
hmmlearn para el filtrado:

```
log_alpha[0] = log_startprob + log_emission[0]
log_alpha[t] = log_emission[t] + logsumexp_j(log_alpha[t-1] + log_transmat[:, j])
γ_t = softmax(log_alpha[t])          # posterior filtrado en t
state_t = argmax_k γ_t[k]            # MAP filtrado
```

`state_b_causal` = `state_t` relabelizado; `posterior_b_migracion_causal`
= `γ_t[idx_migración]`.

Decodificar requiere las observaciones del ave **hasta** `t`, incluidos
días del tramo de train cuando `t` cae en val/test. Esto es correcto y
realista (en uso real se dispone del historial del ave); los
**parámetros** del HMM, en cambio, sólo se ajustan en train (§6.2).

## 7. Pipeline `build_o4` (cambios)

1. Cargar `features.parquet`, `cells.parquet`.
2. **Recalcular cinemática causal** (`features.py`) desde `lat`/`lon`
   por `(bird_id, date_utc)`: `step_in_km`, `sin/cos_bearing_in`,
   `cos_turning_in`. Aplicar máscara de 4 días (R10).
3. Asignar `cell_id_t`, `cell_id_t_next`; añadir `sin/cos_doy`.
4. Construir matrices para ambos modos; **split temporal por ave**
   (sin cambios respecto a O4 base).
5. **Ajustar HMM causal sobre train** (§6.2) y **filtrar todas** las
   filas (§6.3); fusionar `state_b_causal`, `posterior_b_migracion_causal`.
6. Entrenar las 6 combinaciones (RF/XGB/LGBM × 2 modos) con la config
   §8.6 de O4 base.
7. Métricas globales train+test; baselines persistencia + Markov(1)
   sobre el mismo split (sin cambios).
8. Escribir artefactos en `O4_OUT_DIR` (regenera los actuales).

El flag `with_wind` y la rama L1 (`O4_L1V1_DIR`) permanecen en el
código pero **no se ejercitan** en este rework.

## 8. Decisiones metodológicas sin figura

Para citación directa en la memoria:

- **8.1 Por qué la cinemática entrante elimina la fuga.** El target es
  `pos(t+1)`. La cinemática saliente `t→t+1` es función de `pos(t+1)`;
  la entrante `t-1→t` es función de `pos(t-1), pos(t)`, ambas conocidas
  en `t`. La entrante es, además, la noción física de **inercia**: a
  dónde y con qué intensidad venía moviéndose el ave.
- **8.2 Por qué filtrado y no suavizado.** El suavizado
  `P(s_t | obs_1..T)` incorpora observaciones futuras, lo que es
  look-ahead inadmisible en predicción. El filtrado `P(s_t | obs_1..t)`
  es la inferencia en línea correcta: la creencia sobre el régimen
  actualizada con lo visto hasta hoy. Es la forma honesta de usar un
  HMM como feature predictiva.
- **8.3 Por qué el HMM no se elimina.** Aunque `step_in_km` crudo ya
  separa regímenes (Cohen's d de O3, C6), el estado HMM es la feature
  propuesta por la tutora y aporta una compresión probabilística del
  régimen. Se conserva, reconstruido sin fuga.
- **8.4 Por qué el rumbo fuera del HMM** — ver §4.2.
- **8.5 Por qué O3 no se reajusta.** El estado descriptivo de O3 es
  válido para su propósito; el HMM causal es una pieza de O4 con
  distinto split (temporal vs por-ave), distinto decodificado
  (filtrado vs suavizado) y distinto sentido cinemático. Mantener O3
  congelado evita invalidar su capítulo y su tag.

## 9. Tests

Pytest sobre datasets sintéticos pequeños.

### 9.1 `tests/test_ml_features.py` (ampliar)
- `step_in_km` coincide con haversine manual sobre un ave sintética.
- `sin_bearing_in`/`cos_bearing_in` correctos y con `sin²+cos² ≈ 1`.
- `cos_turning_in` usa **sólo** `t-2, t-1, t` (causal): cambiar
  `pos(t+1)` no altera `cos_turning_in(t)`.
- La máscara de 4 días descarta filas sin la racha completa y ninguna
  fila atraviesa un gap artificial.
- El target sigue siendo la celda de `(lat[t+1], lon[t+1])`.

### 9.2 `tests/test_ml_hmm_causal.py` (nuevo)
- **Propiedad leak-free (test central):** sobre una secuencia
  sintética, el posterior filtrado en `t` **no cambia** al modificar
  observaciones en `t+1..T`; el suavizado sí cambiaría (test de
  contraste).
- El filtrado forward coincide con una recursión de referencia
  independiente en un HMM de parámetros conocidos.
- `relabel_states` asigna migración al estado de mayor media de `step`.
- Sobre datos reales (test de integración ligero): los parámetros del
  HMM causal quedan próximos a los de O3 (sanity de §6.1).

### 9.3 `tests/test_ml_build.py` (ampliar)
- `build_o4` corre end-to-end en sintético, regenera los 3 artefactos
  y la matriz de features no contiene NaN.
- Las columnas de O4 son exactamente las 10 de R6 (+`bird_id` en
  personalizado); no aparecen `step_length_km`, `cos_turning_angle`,
  `state_b`, `posterior_b_migracion`.

## 10. Entregables y evidencia

- **Código:** `ml/features.py` (cinemática causal), `ml/hmm_causal.py`
  (nuevo), `ml/build.py` (orquestación). `train.py`/`evaluate.py` sin
  cambios funcionales.
- **Artefactos regenerados:** `data/processed/o4/{model_*.pkl,
  predictions_test.parquet, metrics.parquet}` desde el modelo causal.
- **Evidencia (`save_artifact`), sobre datos limpios:**
  - Impacto de la máscara de 4 días en el tamaño de muestra.
  - Tabla comparativa final (6 modelos + baselines) — **reemplaza** la
    tabla de O4 con fuga.
  - Análisis de error por estado HMM (estacionario vs migración) de
    los dos ganadores, sobre `state_b_causal`.
  - Importancia de features de los ganadores (¿aporta el rumbo? ¿y el
    estado HMM causal?).
- **No se genera** comparativa antes/después de la fuga (R11).
- **Notas de memoria:** actualizar `reports/memoria/06_o4_ml.md` con
  los resultados causales; el hallazgo metodológico que se capitaliza
  es positivo (inercia entrante + filtrado en línea), no el episodio
  de la fuga.
- **Actualizar `CLAUDE.md`** (sección O4) con los números causales,
  marcando que sustituyen a los anteriores.
- **Entrada en `reports/ai-log/`** describiendo la ingeniería de
  features causales de O4 (trabajo sustantivo del TFG).
- **Cierre:** tag nuevo (propuesta `v0.4.2-o4-rework-causal`). Sin
  reescritura de historial; `v0.4`/`v0.4.1` permanecen como historia.

## 11. Riesgos

| # | Riesgo | Mitigación |
|---|---|---|
| R-A | Tras quitar la fuga, el ML deja de batir a persistencia incluso en log-loss | Resultado honesto y esperado; se reporta como límite estructural (se enmarca como en O4 base: línea de trabajo futuro con modelos de secuencia/viento) |
| R-B | El HMM causal no converge igual que O3 o relabeliza distinto | Sanity-check de parámetros (§6.1) y de coherencia biológica del % migración por mes |
| R-C | La máscara de 4 días reduce demasiado la muestra | Se cuantifica antes de entrenar; si fuera severa, se reconsidera `cos_turning_in` (decisión reversible documentada) |
| R-D | Filtrado forward mal implementado (estabilidad numérica) | Log-espacio + `logsumexp`; test contra recursión de referencia y contra `score_samples` en el caso suavizado |
