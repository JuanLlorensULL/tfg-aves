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
columna de O3, la recalcula. Este rework **elimina la duplicación**:
O3 pasa a ser causal por completo (cinemática entrante + filtrado
forward-only + split temporal), produce `state_a_causal` y
`state_b_causal`, y O4 **lee esas columnas** en vez de recalcularlas.

La dependencia se invierte a **O4 → O3**: la maquinaria causal sube a
`tfg_aves.hmm` y O4 se convierte en consumidor.

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
| C2 | Modelos | **Se conservan A y B, ambos causales** | La ablación A vs B es la contribución metodológica que detectó la circularidad; se mantiene como tal. |
| C3 | Cinemática de emisión | **Entrante** (`step_in_km`, `cos_turning_in`) | Calculable sin ver el futuro (inercia observable en t). |
| C4 | Decodificado | **Filtrado forward-only** `P(estado_t \| obs_1..t)` | Sin look-ahead; régimen admisible para una feature predictiva. |
| C5 | Split | **Temporal compartido con O4** (reusar `split_temporal_per_bird`). El HMM se ajusta sobre el **bloque train (72%)**, el mismo cutoff que usa O4 hoy (`cutoff_by_bird` derivado de `train_pob`, que excluye el 8% de val) | Si O3 ajustara por aves (serie completa), los días del periodo de test de O4 entrarían en el ajuste del HMM → fuga al leer la columna en O4. El corte temporal lo impide. El cutoff debe ser exactamente el de O4 (fin del 72%, no del 80%) para que el diff de neutralidad salga idéntico. |
| C6 | Propiedad del código | **`tfg_aves.hmm` es el dueño**; O4 consume | Invierte la dependencia a O4 → O3; conecta el flujo O3 → O4 real. |
| C7 | Ubicación del split | **Módulo neutral upstream** (`tfg_aves.data`) | O3 no debe depender de O4. El split opera sobre `(bird_id, date_utc)`, concepto de datos, upstream de ambos. |
| C8 | LL de holdout de O3 | **Pasa a ser temporal** (sobre el test del split temporal) | El split deja de ser por aves; el holdout mide generalización al futuro, coherente con el uso predictivo. Se conserva el acuerdo A–B. |
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
  ajuste. De ahí el split temporal compartido (C5).
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
  - cinemática entrante causal (`compute_causal_kinematics`, hoy en
    `ml/features.py`),
  - fit + filtrado + decode (`fit_causal_hmm`,
    `forward_filtered_posteriors`, `decode_causal_states`,
    `build_hmm_sequences`, hoy en `ml/hmm_causal.py`).
- A `tfg_aves.data` (neutral upstream): `split_temporal_per_bird` y un
  helper `cutoff_by_bird(...)`.
- `tfg_aves.ml` deja de tener `hmm_causal.py`; importa el split desde
  `tfg_aves.data` y las columnas HMM desde el `features.parquet` de O3.

### 4.2 Nueva pipeline de `build_o3`

1. `daily.parquet` + veg del CSV crudo → features causales entrantes
   (ventana `t-2, t-1, t`, máscara `is_hmm_obs_valid`).
2. Emisión: A = `[step_in_km, cos_turning_in]`; B = A + `[veg_low,
   veg_high, daylight_hours]`.
3. Split temporal compartido → `cutoff_by_bird` (fin del bloque train,
   72%, excluyendo val; mismo cutoff que O4).
4. Ajuste A y B sobre el train (k-means + 10 restarts, sin estandarizar;
   reusa `fit_hmm_with_restarts`).
5. Decodificado filtrado forward-only de **todos** los días válidos →
   `state_a_causal`, `state_b_causal`, `posterior_{a,b}_migracion`,
   `posterior_{a,b}_estacionario`.
6. Métricas: LL holdout temporal (A y B) + acuerdo A–B.

### 4.3 Cambios en `features.parquet` y `models_a_b.pkl`

- Columnas de estado: `state_a`/`state_b` (suavizado) →
  `state_a_causal`/`state_b_causal` (filtrado). Las cinemáticas
  salientes (`step_length_km`, `cos_turning_angle`) → entrantes
  (`step_in_km`, `cos_turning_in`). Se añade `is_hmm_obs_valid`.
- `models_a_b.pkl` guarda los modelos causales + `cutoff_by_bird` usado.

### 4.4 Consumo en `build_o4`

- Se elimina el bloque de re-decodificado (`fit_causal_hmm` +
  `decode_causal_states`).
- O4 hace `merge` `m:1` de `state_b_causal` /
  `posterior_b_migracion_causal` desde el `features.parquet` de O3 (misma
  validación de "ninguna fila candidata sin estado").

## 5. Afirmación de neutralidad y verificación

**Hipótesis:** si O3 replica el ajuste causal que O4 hace hoy (misma
emisión, mismo `cutoff_by_bird`, mismos restarts/seed, mismo conjunto de
días `is_hmm_obs_valid`), `state_b_causal` sale **idéntico** → O4 y O5 no
cambian de resultados.

**Verificación obligatoria (gate):**
1. Regenerar O3 causal.
2. Regenerar O4 leyendo O3.
3. **Diff** de `state_b_causal` y `posterior_b_migracion_causal` contra
   los valores que O4 producía con su re-decodificado interno (snapshot
   previo). Debe coincidir (salvo tolerancia numérica en el posterior).
4. Si no coincide, **reconciliar antes de seguir** (no se da por buena la
   neutralidad sin el diff verde).

**Riesgo principal:** el `cutoff_by_bird` de O4 se deriva hoy de
`train_pob`, que sale del `matrix` ya filtrado (gap-aware + asignación de
celdas). O3 no asigna celdas. Hay que calcular el cutoff desde la **misma
fuente temporal** (split sobre `daily`) y comprobar que el conjunto de días
`is_hmm_obs_valid` coincide entre O3 y el que O4 usaba. Es el punto donde
puede romperse la neutralidad y donde se concentra el cuidado de
implementación.

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
  features y columnas de estado); heredar el test leak-free del filtrado
  forward-only. La suite completa debe quedar verde y `ruff` limpio.
- **Tag:** nuevo hito `v0.3.2-o3-causal` que supersede `v0.3.1-o3-rework`
  (conservado como historia). Re-tag/regenerar entregables de O4 si el diff
  obliga.
- **ai-log:** entrada nueva (trabajo técnico sustantivo de O3).

## 7. Fuera de alcance (YAGNI)

- No se conserva el modo suavizado como alternativa (decisión C1).
- No se cambia el número de estados (sigue 2), ni la familia (`GaussianHMM`
  diag), ni los restarts (10), ni la política de no-estandarización.
- No se toca O5 salvo que el diff de neutralidad revele cambios.
- No se reabre la elección Modelo B vs A como canónico (B sigue siendo el
  que alimenta O4).

## 8. Criterios de aceptación

1. `build_o3` produce `features.parquet` causal (estados filtrados, split
   temporal) y `models_a_b.pkl` causal.
2. `build_o4` ya no reajusta el HMM; lee las columnas de O3.
3. Diff de neutralidad verde (o discrepancias reconciliadas y explicadas).
4. Coherencia biológica de los estados filtrados revalidada.
5. Suite de tests verde + `ruff` limpio.
6. Artefactos C1–C9/D1 regenerados; decisión causal documentada con
   `save_artifact`; notas de memoria de O3 actualizadas; entrada de ai-log.
