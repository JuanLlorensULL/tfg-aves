# Diseño — Conversión de O3 a HMM causal que alimenta O4 sin fuga

- **Fecha:** 2026-05-25
- **Objetivo del TFG:** O3 (detección de comportamiento con HMM) —
  rework que unifica el HMM de O3 con el HMM causal de O4.
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado, pendiente de plan de implementación.
- **Supersede:** el estado canónico `v0.3.1-o3-rework` (suavizado). El
  HMM causal que hoy vive duplicado dentro de O4 (`ml/hmm_causal.py`)
  desaparece y su lógica pasa a ser propiedad de O3.

## 1. Resumen

Hoy hay **dos HMM** en el TFG que son el mismo modelo conceptual
(Modelo B, gaussiano de 2 estados estacionario/migración) decodificado
de dos maneras:

- **O3** lo decodifica con **suavizado** (Viterbi + `predict_proba`,
  forward-backward) sobre cinemática **saliente** (`t → t+1`) y lo ajusta
  con un split **por aves** (holdout de aves enteras). Sirve para
  *describir* comportamiento ya observado.
- **O4** lo **reajusta** internamente (`ml/hmm_causal.py`) con cinemática
  **entrante** (`t-1 → t`), decodificado **filtrado forward-only**
  (`P(estado_t | obs_1..t)`) y split **temporal**, porque necesita una
  feature que no mire el futuro.

Esta duplicación deja a O3 desconectado del flujo: O4 no consume la
columna de O3, la recalcula. Y no solo O4 (L1): **L2 y L3 reajustan el
mismo HMM causal por separado** (cuádruple cómputo idéntico). Este rework
**elimina la duplicación**: O3 pasa a ser causal por completo (cinemática
entrante + filtrado forward-only + split temporal), produce
`state_a_causal` y `state_b_causal`, y las líneas de O4 **leen esas
columnas** en vez de recalcularlas. El Modelo B causal coincide con el HMM
que hoy usa L3 (y, por tanto, O5).

La dependencia se invierte a **O4 → O3**: la maquinaria causal sube a
`tfg_aves.hmm`, el split a `tfg_aves.data`, y las tres líneas de O4 se
vuelven consumidoras.

## 2. Contexto y constraints

- **Estado de partida (canónico):** `v0.3.1-o3-rework`. O3 ajusta
  Modelo A (`step_length_km`, `cos_turning_angle`) y Modelo B (A +
  `veg_low`, `veg_high`, `daylight_hours`) con `fit_hmm_with_restarts`
  (k-means + 10 restarts, sin estandarizar), split por aves
  (`stratified_holdout_split`, 80/20), y decodifica con
  `viterbi_per_bird` (suavizado). Escribe `data/processed/o3/{features,
  models_a_b,metrics}`.
- **HMM causal de O4 (a absorber):** `ml/hmm_causal.py`
  (`fit_causal_hmm`, `forward_filtered_posteriors`,
  `decode_causal_states`, `build_hmm_sequences`) +
  `compute_causal_kinematics` y `HMM_EMISSION_COLS` en `ml/features.py`.
  La emisión del HMM causal es `[step_in_km, cos_turning_in, veg_low,
  veg_high, daylight_hours]` (paralela al Modelo B, sin rumbo absoluto).
  El split temporal es `split_temporal_per_bird` (`ml/features.py`).
- **Regla anti-fuga (durable):** toda feature predictiva debe ser
  calculable sin observar el futuro. El `state_b` suavizado de O3 viola
  esto (el paso backward mira `t+1..T`) y la cinemática saliente codifica
  el target. Por eso O4 no puede leer la columna de O3 tal cual hoy.
- **Restricción de coherencia biológica:** todo resultado del HMM debe
  pasar sanity-check contra la fenología de *Larus fuscus* (cría jun-jul,
  migración abr-may y sep-oct, invernada africana dic-feb). Esta regla
  detectó la circularidad en el primer cierre de O3 y es criterio de
  aceptación.

## 3. Decisiones de diseño

| # | Decisión | Valor | Justificación |
|---|---|---|---|
| C1 | Alcance del cambio en O3 | **Reemplazo total a causal** (no doble modo) | Una única fuente de verdad del HMM en todo el TFG; elimina la duplicación. Decisión del autor 2026-05-25. |
| C2 | Modelos | **Se conservan A y B, ambos causales.** El **Modelo B causal es exactamente el HMM de L3** (misma emisión, mismos restarts/seed, filtrado forward-only, poblacional) | La ablación A vs B es la contribución metodológica que detectó la circularidad; se mantiene. O4/O5 consumen solo el B causal (idéntico al que ya usan L1/L2/L3/O5). Decisión del autor 2026-05-25 ("como el modelo HMM de L3", manteniendo la ablación). |
| C3 | Cinemática de emisión | **Entrante** (`step_in_km`, `cos_turning_in`) | Calculable sin ver el futuro (inercia observable en t). |
| C4 | Decodificado | **Filtrado forward-only** `P(estado_t \| obs_1..t)` | Sin look-ahead; régimen admisible para una feature predictiva. |
| C5 | Split | **O3 posee un split temporal propio aguas arriba**, sobre los días válidos de cada ave (mismas fracciones que hoy: 72/8/20 train/val/test por ave). O3 escribe una columna `split ∈ {train,val,test}` en `features.parquet`. El HMM se ajusta sobre los días `train`. Aguas abajo (L1/L2/L3) **leen esa columna** y particionan su matriz por ella | Es el corte limpio que NO depende de la matriz del clasificador de O4. Decisión del autor: priorizar arquitectura limpia frente a neutralidad exacta (ver §5). El split por aves filtraría fuga porque el ajuste sobre la serie completa de las aves de train incluiría días del periodo de test. |
| C6 | Propiedad del código | **`tfg_aves.hmm` es el dueño** del HMM causal; O4/L2/L3 consumen | Invierte la dependencia a O4 → O3; conecta el flujo O3 → O4 real y elimina la **cuádruple** duplicación (hoy L1, L2 y L3 reajustan el mismo HMM por separado). |
| C7 | Ubicación del split | **Módulo neutral upstream** (`tfg_aves.data`); `split_temporal_per_bird` se mueve allí desde `ml/features.py` | O3 no debe depender de O4. El split opera sobre `(bird_id, date_utc)`, concepto de datos, upstream de ambos. |
| C8 | LL de holdout de O3 | **Pasa a ser temporal** (sobre los días `test` del split) | El split deja de ser por aves; el holdout mide generalización al futuro, coherente con el uso predictivo. Se conserva el acuerdo A–B. |
| C9 | Validación biológica | **Regenerar y revalidar** C1–C9, D1, patrón estacional con estados filtrados | Criterio de aceptación; si chirría con la fenología, es bug. |

### 3.1 Decisiones sin figura (justificación para la memoria)

- **Por qué O4 no puede simplemente leer el `state_b` suavizado de O3:**
  el suavizado en el día t usa `obs_{t+1..T}` (paso backward), que es
  exactamente el horizonte que O4 predice. Reusarlo reintroduce la fuga
  que el rework causal de O4 eliminó.
- **Por qué tampoco basta con cambiar solo el decodificado y dejar el
  split por aves:** el ajuste del HMM sobre la serie completa de las aves
  de train incluiría días del periodo de test de O4 (de esas mismas aves);
  al leer la columna en O4, esos parámetros estarían contaminados de
  futuro. El leak no está solo en el decode, también en los datos de
  ajuste. De ahí el split temporal (C5).
- **Por qué O3 posee el split (no lo hereda de O4):** hoy el cutoff de
  ajuste del HMM se deriva de la matriz del clasificador de O4 (filtrada
  por celdas y por el target `cell_id_t_next`), una población que O3 no
  posee. Reproducirla exigiría que O3 importara `build_feature_matrix` +
  celdas de O2/O4 (acoplamiento aguas abajo). En su lugar O3 define un split
  limpio sobre días válidos y lo exporta; las líneas de O4 lo adoptan. El
  coste es que el límite train/test del clasificador puede desplazarse un
  poco respecto al actual → §5.
- **Por qué el suavizado es correcto para describir pero no para predecir:**
  describir comportamiento ya observado se beneficia de toda la secuencia
  (el futuro mejora la etiqueta de t). Predecir prohíbe ese futuro. O3
  cambia de rol descriptivo a "detector causal que sirve doble propósito:
  caracterización en tiempo real + feature de O4". Es un encuadre
  defendible y más unificado, al coste de estados algo más ruidosos en las
  transiciones.

## 4. Arquitectura

### 4.1 Movimiento de código (O4 → O3)

- A `tfg_aves.hmm`:
  - cinemática entrante causal (`compute_causal_kinematics` y
    `HMM_EMISSION_COLS`, hoy en `ml/features.py`),
  - fit + filtrado + decode (`fit_causal_hmm`,
    `forward_filtered_posteriors`, `decode_causal_states`,
    `build_hmm_sequences`, hoy en `ml/hmm_causal.py`). El módulo
    `ml/hmm_causal.py` **se elimina**.
- A `tfg_aves.data` (neutral upstream): `split_temporal_per_bird`.
- `tfg_aves.ml` importa el split desde `tfg_aves.data` y lee las columnas
  HMM **y la columna `split`** desde el `features.parquet` de O3. Se
  actualiza `ml/__init__.py` (deja de exportar `hmm_causal`,
  `split_temporal_per_bird`, `compute_causal_kinematics`).

### 4.2 Nueva pipeline de `build_o3`

1. `daily.parquet` + veg del CSV crudo → features causales entrantes
   (ventana `t-2, t-1, t`, máscara `is_hmm_obs_valid`).
2. Emisión: A = `[step_in_km, cos_turning_in]`; B = A + `[veg_low,
   veg_high, daylight_hours]` (= emisión de L3).
3. Split temporal propio (`split_temporal_per_bird` de `tfg_aves.data`
   sobre días válidos por ave, 72/8/20) → columna `split` y
   `cutoff_by_bird` (fin del bloque `train`).
4. Ajuste A y B sobre los días `train` (k-means + 10 restarts, sin
   estandarizar; reusa `fit_hmm_with_restarts`).
5. Decodificado filtrado forward-only de **todos** los días válidos →
   `state_a_causal`, `state_b_causal`, `posterior_{a,b}_migracion`,
   `posterior_{a,b}_estacionario`.
6. Métricas: LL holdout temporal (A y B) + acuerdo A–B.

### 4.3 Cambios en `features.parquet` y `models_a_b.pkl`

- Columnas de estado: `state_a`/`state_b` (suavizado) →
  `state_a_causal`/`state_b_causal` (filtrado). Las cinemáticas
  salientes (`step_length_km`, `cos_turning_angle`) → entrantes
  (`step_in_km`, `cos_turning_in`). Se añaden `is_hmm_obs_valid` y `split`.
- `models_a_b.pkl` guarda los modelos causales + `cutoff_by_bird` usado.

### 4.4 Consumo en `build_o4`, `build_l2`, `build_l3`

Las tres líneas (hoy cada una reajusta su HMM) pasan a consumir O3:
- Se elimina el bloque de cinemática + re-decodificado
  (`compute_causal_kinematics` + `fit_causal_hmm` + `decode_causal_states`
  + `split_temporal_per_bird` sobre la matriz).
- Leen de `features.parquet` de O3 las cinemáticas entrantes, las columnas
  HMM (`state_b_causal`, `posterior_b_migracion_causal`) y la columna
  `split`; construyen la matriz de features y la **particionan por
  `split`** (merge `m:1`, manteniendo la validación "ninguna fila
  candidata sin estado").

## 5. Re-verificación aguas abajo (la neutralidad NO se garantiza)

Decisión del autor (2026-05-25): se prioriza la **arquitectura limpia**
frente a la neutralidad exacta. Como el split del HMM pasa a definirse
sobre días válidos (no sobre la matriz del clasificador), el límite
train/test del clasificador puede desplazarse en algunas aves cerca de la
frontera. **Los números de L1/L2/L3/O5 pueden moverse** (se espera que
poco) y hay que re-verificarlos, no darlos por idénticos.

**Gate de re-verificación (obligatorio antes de cerrar):**
1. Snapshot de los entregables actuales de L1/L2/L3 (`metrics.parquet`,
   `predictions_test.parquet`) y de O5.
2. Regenerar O3 causal → regenerar L1, L2, L3 leyendo O3 → regenerar O5.
3. **Comparar** métricas clave (top-1, log-loss, dist mediana, pinball,
   cobertura) antes/después. Cuantificar el desplazamiento.
4. Criterio de aceptación: el desplazamiento es pequeño y **no cambia
   ninguna conclusión cualitativa** del TFG (ML pierde a persistencia en
   top-1, gana/pierde donde corresponde, calibración ~80%, etc.). Si una
   conclusión cambia, **parar y avisar al autor** antes de re-taguear.
5. Revalidar la **coherencia biológica** de los estados filtrados.

**Riesgo principal de implementación:** alinear exactamente el conjunto de
días `train`/`val`/`test` entre la columna `split` de O3 y lo que cada
línea de O4 necesita, para que el merge no deje filas candidatas sin
estado ni sin etiqueta de split.

## 6. Impacto en artefactos, memoria y tests

- **Artefactos O3 (C1–C9, D1):** regenerar con estados filtrados y
  **revalidar coherencia biológica** (gate de aceptación). `save_artifact`
  para la decisión "decodificado causal en O3", con una figura/tabla de
  acuerdo entre estado suavizado y filtrado (evidencia de que es el mismo
  modelo realineado, no uno nuevo).
- **Memoria:** el capítulo de O3 en LaTeX aún no está redactado (solo notas
  `reports/memoria/05_o3_hmm.md`). Actualizar las notas para narrar el
  régimen causal y su porqué (alimenta a O4 sin fuga). Se conserva el relato
  del rework/circularidad. La memoria no narra la fuga de O4 (registro
  interno), pero sí puede presentar el HMM causal como detector en tiempo
  real con doble propósito.
- **Tests:** adaptar `tests/test_hmm*` al esquema causal (nombres de
  features y columnas de estado); mover/adaptar `tests/test_ml_hmm_causal.py`
  a `tests/test_hmm_causal.py` (el módulo cambia de paquete); ajustar
  `tests/test_ml_build*.py`, `test_ml_features.py` al nuevo consumo. Heredar
  el test leak-free del filtrado forward-only. Suite verde y `ruff` limpio.
- **Tag:** nuevo hito `v0.3.2-o3-causal` que supersede `v0.3.1-o3-rework`
  (conservado como historia). Re-taguear/regenerar entregables de
  L1/L2/L3/O5 tras la re-verificación.
- **ai-log:** entrada nueva (trabajo técnico sustantivo de O3).

## 7. Fuera de alcance (YAGNI)

- No se conserva el modo suavizado como alternativa (decisión C1).
- No se cambia el número de estados (sigue 2), ni la familia (`GaussianHMM`
  diag), ni los restarts (10), ni la política de no-estandarización.
- No se reabre la elección Modelo B vs A como canónico (B sigue siendo el
  que alimenta O4/O5; B causal = HMM de L3).
- No se reentrenan ni re-tunean los modelos ML de O4 (solo se regeneran con
  el nuevo input para la re-verificación).

## 8. Criterios de aceptación

1. `build_o3` produce `features.parquet` causal (estados filtrados +
   cinemática entrante + columnas `is_hmm_obs_valid` y `split`) y
   `models_a_b.pkl` causal.
2. `build_o4`, `build_l2`, `build_l3` ya no reajustan el HMM ni recalculan
   la cinemática causal; leen las columnas y el `split` de O3.
3. `ml/hmm_causal.py` eliminado; `split_temporal_per_bird` reside en
   `tfg_aves.data`; `ml/__init__.py` actualizado.
4. Gate de re-verificación pasado: desplazamiento de métricas cuantificado
   y sin cambios cualitativos en las conclusiones (o autor avisado).
5. Coherencia biológica de los estados filtrados revalidada.
6. Suite de tests verde + `ruff` limpio.
7. Artefactos C1–C9/D1 regenerados; decisión causal documentada con
   `save_artifact`; notas de memoria de O3 actualizadas; entrada de ai-log;
   tag `v0.3.2-o3-causal`.
