# Diseño de L2 (objetivo O4) — Modelo de dos etapas (régimen → posición)

- **Fecha:** 2026-05-23
- **Objetivo del TFG:** O4 — segunda línea de mejora (L2) del pipeline
  supervisado base. Sustituye el clasificador monolítico de O4 por una
  arquitectura en dos etapas (decisor de movimiento + decisor de
  destino condicional) para atacar la causa estructural **D1**
  ("dominio de self-loops, 73 %") diagnosticada en
  `reports/memoria/06_o4_ml.md` §"Análisis estructural del techo de
  rendimiento".
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado el 2026-05-23, pendiente de implementación.
  Implementación a arrancar cuando L1 (features de viento) cierre, para
  evitar conflictos en `src/tfg_aves/ml/`.

> **⚠ ACTUALIZACIÓN 2026-05-23 — rework causal de O4.** Las "8 features
> de O4 base" que este spec cita (`step_length_km`, `cos_turning_angle`,
> `state_b`, `posterior_b_migracion` + lat/lon/sin_doy/cos_doy) tenían
> *data leakage*: las cinemáticas eran salientes (`t → t+1`) y el estado
> HMM se decodificaba con suavizado, codificando el target. Se corrigió
> con el rework causal (ver `2026-05-23-o4-rework-causal-design.md`,
> §12). **Antes de implementar L2:** sustituir el feature set base por
> las **10 features causales** (R6 del rework): `lat, lon, sin_doy,
> cos_doy, step_in_km, sin_bearing_in, cos_bearing_in, cos_turning_in,
> state_b_causal, posterior_b_migracion_causal` (+ `bird_id` en
> personalizado). El desglose por estado HMM usa `state_b_causal`. El
> baseline L2-v0 debe re-derivarse del **O4 causal**, no del contaminado
> `v0.4-o4-completo`. Las menciones a las features antiguas en este spec
> deben leerse con esta sustitución. Registro interno: NO se traslada a
> la memoria.

## 1. Resumen

L2 descompone el problema "predecir `cell_{t+1}`" en dos preguntas
secuenciales:

1. **¿El ave se mueve hoy?** — clasificador binario `clf_move` sobre
   las 8 features de O4 base. Predice `p_move ∈ [0, 1]`.
2. **Si se mueve, ¿adónde?** — clasificador multiclase `clf_dest` sobre
   las 849 celdas activas, entrenado **sólo** sobre filas de train donde
   verdaderamente hubo movimiento. Predice `p_2B(cell | x)`.

Ambas etapas se combinan en una distribución final sobre las 849 celdas
mediante dos reglas evaluadas en paralelo:

- **Soft (canónica):**

  ```
  p_final(cell_t) = 1 − p_move
  p_final(cell)   = p_move · p_2B(cell | x) / (1 − p_2B(cell_t | x))     ∀ cell ≠ cell_t
  ```

  La división renormaliza la masa que `p_2B` asignaba a `cell_t`
  (clf_dest fue entrenado sobre filas con `y_move = 1`, pero sigue
  teniendo prior geográfico via `lat`/`lon` que puede colocar masa
  significativa sobre `cell_t`). Sin esa renormalización, la suma total
  sería `1 − p_move · p_2B(cell_t | x) < 1`. Para evitar división por 0
  cuando `p_2B(cell_t | x) → 1`, se aplica clipping defensivo a
  `1 − ε` con `ε = 1e-7`.

- **Hard (ablación):** si `p_move < τ` entonces `argmax = cell_t`; si no
  `argmax = argmax(p_2B)`. `τ` se barre en `{0,3, 0,5, 0,7}` sobre val y
  se selecciona el de mayor top-1 (NO log-loss; ver §6.1 y §9 sobre por
  qué log-loss no es métrica honesta para hard).

L2 **no toca features, ni target, ni split, ni hiperparámetros, ni
familias** respecto a O4 base. La única variable manipulada es la
**estructura del decisor**. Esta restricción mantiene la ablación
limpia y permite atribuir cualquier mejora exclusivamente a la
arquitectura de dos etapas.

Tras la implementación, el sistema produce dos conjuntos de artefactos
comparables sobre **el mismo test split temporal por ave** que O4 base:

- **L2-v0** (sin cambios): los resultados ya producidos por el cierre
  de O4 (tag `v0.4-o4-completo`, commit `4ed7401`). Sirven como
  baseline contra el que se compara la mejora.
- **L2-v1**: el pipeline en dos etapas. Genera nuevos modelos `.pkl`,
  `predictions_test_{soft,hard}.parquet` y `metrics.parquet` bajo
  `data/processed/o4/l2_v1/`.

La comparativa **L2-v0 vs L2-v1** sobre cuatro vistas de métricas
(global, por estado HMM, "moves only", gap train-test) cuantifica el
aporte aislado de la arquitectura. La narrativa fuerte para la memoria
es: *"el HMM no sólo aporta features supervisadas, **determina la
estructura del decisor**"*, cerrando el círculo Markov → HMM → ML.

L2 NO ataca D4 (sin viento), D5 (sin destino) ni Mo1 (target
categórico), reservadas a L1 y L3 respectivamente. Cualquier
combinación L2 × L1 o L2 × L3 queda como **trabajo futuro** dentro del
TFG sólo si los cierres individuales lo justifican.

## 2. Vocabulario

- **L2-v0** — estado base sin dos etapas. Equivale al cierre actual de
  O4 (tag `v0.4-o4-completo`). No requiere implementación, sólo
  preservación de los artefactos existentes y referenciación explícita
  como punto de comparación.
- **L2-v1** — pipeline O4 base reorganizado en dos etapas
  (`clf_move` + `clf_dest`). Es lo que se implementa en este spec.
- **`clf_move`** — clasificador binario poblacional de etapa 1.
  Target `y_move = 𝟙{cell_{t+1} ≠ cell_t}`. Único modelo por familia.
- **`clf_dest`** — clasificador multiclase de etapa 2B sobre las 849
  celdas activas. Dos modos: personalizado (con `bird_id`) y
  poblacional. Una pareja por familia.
- **Regla soft / Regla hard** — las dos formas evaluadas de combinar
  `p_move` y `p_2B` en una distribución final sobre las 849 celdas.
- **Subset "moves only"** — restricción del test al subconjunto de
  filas donde verdaderamente `y_move = 1`. Sirve para medir la calidad
  intrínseca de `clf_dest` aislando el efecto de la persistencia
  trivial.
- **Comparativa L2-v0 vs L2-v1** — tabla y figuras que cuantifican el
  aporte de la arquitectura de dos etapas. Entregable narrativo central
  de L2 para la memoria.

## 3. Contexto y constraints

**Datos disponibles** (todos verificados al 2026-05-23):

- **Pipeline O4 base completo** en `data/processed/o4/` con los 6
  modelos `.pkl` (RF/XGB/LGBM × personalizado/poblacional),
  `predictions_test.parquet` (32 296 filas) y `metrics.parquet` (14
  filas). LightGBM descartado por divergencia; L2 no lo reabre.
- **`predictions_test.parquet` de O4 base** contiene ya las columnas
  `cell_id_t` (celda del día t) y `cell_id_target` (celda del día t+1).
  El target binario `y_move` se deriva trivialmente de ambas sin
  recálculo geométrico.
- **Features de O3** en `data/processed/o3/features.parquet` (24 444 ×
  17): contiene las 8 features base de O4 (`lat`, `lon`, `sin_doy`,
  `cos_doy`, `step_length_km`, `cos_turning_angle`, `state_b`,
  `posterior_b_migracion`). NO se añaden features nuevas en L2.

**Frecuencias relevantes del dataset (calculadas sobre O4 base):**

- Persistencia trivial top-1 = **0,773** sobre test → ~77 % de filas
  test tienen `y_move = 0`. Frecuencia base `p(y_move = 1) ≈ 0,23-0,27`
  (variable según split; calcularla exactamente sobre train en
  implementación es trivial).
- Tamaño esperado del set de entrenamiento de `clf_dest`:
  `n_train_o4 × p(y_move=1) ≈ 22 600 × 0,27 ≈ 6 100 filas`. Compárese
  con las 22 600 que ve `clf_move`. Esta reducción motiva el
  monitorizar gap train-test específicamente sobre `clf_dest` (R2 en
  §7).

**Restricciones heredadas de O4 base (no se replantean en L2):**

- **Features**: las 8 de O4 base, exactamente. No se reabre F7 (no se
  re-incluyen `veg_low/high`, `daylight_hours`).
- **Target multiclase de etapa 2B**: las 849 celdas activas del grid
  0,5° de O2, codificadas con el mismo `LabelEncoder` que O4 base.
- **Split temporal 80/10/20 por ave**, gap-aware (cuatro barreras
  heredadas de §8.11 del spec O4).
- **Familias**: RF + XGBoost. LightGBM descartado (F7 de O4).
- **Hiperparámetros**: configuración conservadora §8.6 de O4. Sin
  tuning nuevo (consistente con la decisión post-cierre de O4 de NO
  invertir en tuning).
- **Modos**: personalizado / poblacional, con la misma definición que
  O4 base (`bird_id` como feature categórica en personalizado).

**Restricciones nuevas introducidas en L2:**

- Etapa 1 (`clf_move`) **siempre poblacional**. Sin variante
  personalizada (decisión F3 en §4).
- Etapa 2B (`clf_dest`) **sólo se entrena sobre filas train con
  `y_move = 1`**. En inferencia se aplica a TODAS las filas test,
  independientemente del valor verdadero de `y_move` (la regla soft
  pondera su contribución).
- Compensación obligatoria del desequilibrio en etapa 1 vía
  `class_weight='balanced'` (RF) y `scale_pos_weight = n_neg / n_pos`
  (XGBoost). Decisión F8.
- Calibración isotonic obligatoria en etapa 1 vía
  `CalibratedClassifierCV(method='isotonic', cv=3)`. Decisión F8.

## 4. Decisiones de diseño fijadas

### F1 — Dos etapas, no más

L2 implementa **exactamente dos etapas**: clasificador binario de
movimiento + clasificador multiclase condicional. Se descartaron
arquitecturas más profundas (e.g. tres etapas: estacionario / migración
corta / migración larga) por dos razones:

- La heterogeneidad cinemática del dataset ya está capturada por
  `state_b` y `posterior_b_migracion` como features, no requiere
  decisiones jerárquicas adicionales.
- Cualquier estratificación adicional reduciría aún más el dataset de
  la etapa más profunda, agravando R2 (sobreajuste por dataset
  pequeño).

### F2 — Feature set idéntico a O4 base, sin excepción

Las 8 features fijadas en F6 del spec de O4 base son las únicas que
ven `clf_move` y `clf_dest`:

```
lat, lon, sin_doy, cos_doy,
step_length_km, cos_turning_angle,
state_b, posterior_b_migracion
(+ bird_id en modo personalizado de clf_dest)
```

L2 ataca exclusivamente la causa **D1** y mantiene constante todo lo
demás. Las features de viento de L1, si L1 cierra antes de L2, NO se
incorporan a L2-v1; cualquier combinación L1+L2 se trataría como
experimento separado fuera del scope de este spec (ver §12).

### F3 — Etapa 1 siempre poblacional

`clf_move` se entrena una sola vez, sin `bird_id`. Razonamiento:

- La decisión "hoy migra o no" es un fenómeno **estructural-biológico**
  (régimen fenológico, condiciones del calendario, estado HMM) más que
  individual.
- Reduce a la mitad el número de modelos de etapa 1 y simplifica el
  análisis sin renunciar a expressividad relevante.
- La asimetría entre etapa 1 (poblacional) y etapa 2B (dos modos) es
  intencional y se documenta en la memoria como decisión de diseño.

Esta decisión se cierra sin ablación: el coste-beneficio de probar
también etapa 1 personalizada no compensa la complicación narrativa.

### F4 — Etapa 2B en dos modos (personalizado + poblacional)

`clf_dest` se entrena en **los dos modos de O4 base** para mantener
simetría con el cierre actual y comparar L2-v1 contra los mismos
ganadores (RF personalizado, XGB poblacional). Esto produce 4 modelos
de etapa 2B (RF/XGB × pers/pob), más los 2 modelos de etapa 1 (RF, XGB
poblacionales), total **6 modelos** entrenados en L2-v1.

### F5 — Familias RF + XGBoost; LightGBM no se reabre

L2 utiliza RF y XGBoost en ambas etapas. LightGBM sigue descartado por
la misma razón que en O4 base (incompatibilidad con la configuración
conservadora §8.6, divergencia en val). Reabrirlo en L2 mezclaría dos
contribuciones (cambio de arquitectura + cambio de familia) y haría
imposible atribuir las mejoras.

### F6 — Combinación soft canónica + hard como ablación

Las dos reglas de combinación se evalúan siempre, sobre los mismos
modelos entrenados:

- **Soft**: regla principal, justifica el enfoque probabilístico que ya
  ganó en log-loss en O2.
- **Hard con τ barrido**: ablación que cuantifica el coste de renunciar
  a la mezcla probabilística. `τ ∈ {0,3, 0,5, 0,7}` sobre val; se elige
  `τ*` por mínimo log-loss en val y se reporta el resultado de hard
  con `τ*` en test.

El esfuerzo extra es ~30 LOC y un artefacto (`L2-v1-D2`).

### F7 — Hiperparámetros heredados §8.6 de O4 base, sin tuning

Ambas etapas usan la configuración conservadora §8.6 fijada en el
spec de O4. Etapa 1 (binaria) y etapa 2B (multiclase) usan los mismos
hyperparámetros que la respectiva familia en O4 base. Esta decisión
es consistente con la política post-cierre de O4 de **no invertir
tiempo en tuning** (la causa real del techo es estructural, no de
configuración).

### F8 — Compensación de clases + calibración isotonic en etapa 1 (con val temporal)

Dos higienes obligatorias en `clf_move`:

- **Compensación del desequilibrio**: `class_weight='balanced'` (RF) y
  `scale_pos_weight = n_neg / n_pos` (XGBoost). Sin estas, el modelo
  degenera a "siempre predigo 0" y la etapa 2B nunca se activa en la
  combinación soft.
- **Calibración isotonic sobre el val temporal**: `clf_move` se entrena
  primero sobre `X_train` sin calibrar, y a continuación se envuelve en
  `CalibratedClassifierCV(base, method='isotonic', cv='prefit')` y se
  hace `fit(X_val, y_val_move)`. Usar `cv='prefit'` (en lugar de
  `cv=3` con KFold aleatorio) garantiza que la calibración respeta la
  estructura temporal por ave heredada de O4 base: el train sigue
  estrictamente antes que el val, y val antes que el test. Sin
  calibración, los valores numéricos de `p_move` no son probabilidades
  fiables y la regla soft mezcla mal los dos componentes.

Sin compensación + sin calibración temporalmente correcta, el
experimento estaría sesgado por construcción.

**Consecuencia operativa:** `X_val` se "gasta" en calibrar la etapa 1.
Como F7 prohíbe el tuning de hiperparámetros, esto NO entra en
conflicto con ningún otro uso de val: el único uso restante de val es
el barrido de `τ` para la regla hard (`sweep_tau`), que se hace sobre
las predicciones ya calibradas de val.

### F9 — Split temporal idéntico a O4 base

Sin cambios. Mismo 80/10/20 por ave con cuatro barreras gap-aware.
Garantiza comparabilidad total entre L2-v0 y L2-v1 sobre el mismo
test set.

### F10 — Target `y_move` derivado de columnas existentes

`y_move_t = 𝟙{cell_id_target_t ≠ cell_id_t}`. Ambas columnas ya
existen en `predictions_test.parquet` de O4 base y en sus análogos
internos de train/val (calculados durante `build_o4`). La derivación
es trivial y no introduce dependencias de cálculo nuevas.

## 5. Cobertura de causas raíz

Causas raíz diagnosticadas en `reports/memoria/06_o4_ml.md` §"Análisis
estructural del techo":

| Causa | Cobertura L2 | Comentario |
|---|---|---|
| D1 self-loops dominantes (73 %) | **Directa** | Núcleo del diseño. La etapa 1 absorbe la decisión "no se mueve" y la etapa 2B se especializa en "adónde". |
| D2 snapshot diario (1 punto/día) | — | Fuera de scope (trabajo futuro: modelo de secuencia). |
| D3 rutas individuales heterogéneas | Parcial | El modo personalizado de etapa 2B sigue capturando heterogeneidad como en O4 base. |
| D4 sin viento | — | Cubierto por L1. |
| D5 sin destino | — | Fuera de scope (trabajo futuro). |
| Mo1 target categórico | — | Cubierto por L3 (regresión cuantiles). |
| Mo2 horizonte 1 día | — | Fuera de scope. |
| Mo3 sin historia multi-día | — | Fuera de scope. |

L2 es **monofocal**: ataca exclusivamente D1 para que cualquier
mejora medida en la comparativa L2-v0 vs L2-v1 sea atribuible sin
ambigüedad a la arquitectura.

## 6. Pipeline e integración

### 6.1 Módulos nuevos

```
src/tfg_aves/ml/
    two_stage.py     # combinación soft/hard + helpers (~150 LOC)
    build_l2.py      # orquestador build_o4_l2() (~200 LOC)
```

**`two_stage.py`** (funciones puras):

```python
def derive_y_move(predictions_df: pd.DataFrame) -> pd.Series:
    """Devuelve una Series booleana y_move = (cell_id_target != cell_id_t)."""

def combine_soft(
    p_move: np.ndarray,           # (n_rows,)
    p_2b: np.ndarray,             # (n_rows, n_classes)
    cell_t_idx: np.ndarray,       # (n_rows,), índice de cell_t en LabelEncoder
    n_classes: int,
    eps: float = 1e-7,
) -> np.ndarray:
    """Aplica la regla soft canónica con renormalización. Devuelve
    (n_rows, n_classes) tal que cada fila suma 1.0 (±1e-6).

    Para cada fila:
        p_final[cell_t]      = 1 − p_move
        p_final[cell≠cell_t] = p_move · p_2b[cell] / (1 − p_2b[cell_t])

    El denominador se evalúa como max(1 − p_2b[cell_t], eps) para
    evitar división por 0 si clf_dest colapsa sobre cell_t."""

def combine_hard(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    tau: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Aplica la regla hard con umbral tau. Devuelve (n_rows, n_classes)
    con masa 1−eps en la celda predicha (cell_t si p_move<tau, si no
    argmax(p_2b)) y eps/(n_classes−1) en el resto.

    El clipping existe sólo por compatibilidad con APIs que esperan
    distribuciones estrictamente positivas (sklearn.metrics.log_loss);
    el log-loss numérico resultante NO es una métrica honesta para
    hard (cada fallo de argmax contribuye ~22.86 al log-loss por
    construcción del clipping). Sólo top-1, top-3 y dist_med_km son
    métricas comparables entre hard y soft."""

def sweep_tau(
    p_move_val: np.ndarray,
    p_2b_val: np.ndarray,
    cell_t_idx_val: np.ndarray,
    y_true_val: np.ndarray,
    n_classes: int,
    taus: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> tuple[float, pd.DataFrame]:
    """Devuelve (tau_star, tabla_de_top1_por_tau). tau* es el tau con
    mayor top-1 sobre val. Se usa top-1 (NO log-loss) porque hard
    devuelve one-hot y log-loss sería degenerado para evaluarlo."""
```

**`build_l2.py`** (orquestador):

```python
def build_o4_l2(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    o4_predictions_path: Path = PREDICTIONS_O4_PARQUET,
    out_dir: Path = O4_OUT_DIR / "l2_v1",
) -> BuildO4L2Result:
    """Orquesta el pipeline L2-v1.

    Por cada familia f ∈ {RF, XGB}:
      1. Entrena base_move = train_f(X_train, y_train_move).
      2. Envuelve clf_move = CalibratedClassifierCV(base_move,
         method='isotonic', cv='prefit').fit(X_val, y_val_move).
      3. Filtra X_train_moves, y_train_dest = subset(X_train, y_train,
         y_train_move == 1) y entrena clf_dest_pers y clf_dest_pob.
      4. Aplica combine_soft y combine_hard (con sweep_tau sobre val
         calibrado) sobre el test set.
      5. Persiste model_clf_move_f.pkl y dos model_clf_dest_f_*.pkl.

    Tras procesar ambas familias, genera predictions_test_soft.parquet,
    predictions_test_hard.parquet, tau_sweep.parquet y metrics.parquet."""
```

### 6.2 Cambios en módulos existentes

**Mínimos.** L2 reutiliza sin modificación:

- `src/tfg_aves/ml/features.py`: `build_feature_matrix`,
  `split_temporal_per_bird`.
- `src/tfg_aves/ml/train.py`: `train_random_forest`, `train_xgboost`
  (se reusan tal cual; la diferencia binario vs multiclase la maneja
  scikit-learn / XGBoost a partir del shape del target).
- `src/tfg_aves/ml/evaluate.py`: `top_k_accuracy`, `dist_median_km`,
  `evaluate_global`, `compute_persistence_baseline`,
  `compute_markov_baseline`.

**Único cambio menor en `evaluate.py`:** añadir un helper
`evaluate_moves_only(predictions_df, y_move_true) -> dict[str, float]`
que filtra el DataFrame de predicciones a las filas donde verdad
`y_move = 1` y devuelve las mismas métricas que `evaluate_global`.
~20 LOC.

### 6.3 Estructura de outputs

```
data/processed/
    o4/                                      # L2-v0 (existente, no se toca)
        model_*.pkl                           # 6 modelos
        predictions_test.parquet
        metrics.parquet
    o4/l2_v1/                                 # L2-v1 (nuevo)
        model_clf_move_rf.pkl                 # etapa 1, una por familia
        model_clf_move_xgb.pkl
        model_clf_dest_rf_personalizado.pkl   # etapa 2B, 4 modelos
        model_clf_dest_rf_poblacional.pkl
        model_clf_dest_xgb_personalizado.pkl
        model_clf_dest_xgb_poblacional.pkl
        predictions_test_soft.parquet
        predictions_test_hard.parquet
        metrics.parquet
        tau_sweep.parquet                     # registro del barrido de τ
```

Todos los `.pkl` y `.parquet` de `l2_v1/` van a `.gitignore` (igual que
el resto de `data/processed/`).

### 6.4 Dependencias añadidas a `pyproject.toml`

Ninguna. Todas las herramientas necesarias
(`sklearn.calibration.CalibratedClassifierCV`, `xgboost`, etc.) ya
están instaladas por O4 base.

## 7. Riesgos identificados

| # | Riesgo | Mitigación / acción |
|---|---|---|
| R1 | Cascada de errores: `clf_move` se equivoca diciendo `p_move ≈ 0` cuando verdaderamente hubo movimiento, etapa 2B contribuye poco y se predice mal el destino real. | Regla soft (vs hard). La mezcla probabilística amortigua errores moderados de etapa 1. Diagnóstico empírico en C5. |
| R2 | Sobreajuste de `clf_dest` por dataset reducido (~6 100 filas train vs ~22 600 en O4 base). | Artefacto C4 mide gap train-test específicamente sobre etapa 2B. Si gap > 0,3 → flag en la memoria como limitación honesta. |
| R3 | Doble dependencia de O3: `state_b` y `posterior_b_migracion` son features de ambas etapas, y la métrica clave se desglosa por `state_b`. Si O3 fuera defectuoso, contamina dos veces. | Aceptable: O3 ya está validado triplemente (rework v0.3.1, tres tests independientes documentados en `reports/memoria/05_o3_hmm.md`). |
| R4 | Calibración isotonic puede degradar discriminación si la familia ya estaba bien calibrada. | Bajo riesgo: RF/XGB nunca lo están en datasets de este tamaño. Si AUC val baja > 0,02 respecto a raw, registrar en D1 y discutir en memoria. |
| R5 | Doble computación de `predict_proba` sobre 849 clases en inferencia (etapa 2B se aplica a TODAS las filas test, no sólo a las que sí se mueven). Coste de tiempo. | Aceptable: O4 base tarda ~5 min en evaluar; L2-v1 ~10-12 min estimados. No es bloqueante. |
| R6 | F7 sigue cerrado: `veg_*/daylight` NO se reincorporan en L2. | Decisión consciente; documentar en spec (este §) y memoria. Reabrirlo mezclaría contribuciones. |
| R7 | L2 y L1 implementan modificaciones en `src/tfg_aves/ml/`; ejecutar ambas en paralelo sin coordinación puede generar conflictos de merge. | L2 espera al cierre de L1. Si necesidad de paralelismo, usar git worktree (decisión meta del autor). |

## 8. Validación y artefactos

### 8.1 Tests automáticos

En `tests/test_ml_two_stage.py` (nuevo):

- `test_derive_y_move_basic`: para un DataFrame con celdas `[A, A,
  B, B, A]`/`[A, B, B, A, A]`, `derive_y_move` devuelve
  `[F, T, F, T, F]`.
- `test_combine_soft_extremes`: si `p_move = 0` para una fila, la
  masa entera va a `cell_t`. Si `p_move = 1`, la masa va a `p_2b`
  excluyendo `cell_t` (renormalizada).
- `test_combine_soft_sums_to_one`: para entradas válidas, cada fila
  de la salida suma 1,0 (tolerancia 1e-6).
- `test_combine_soft_renormalizes_when_p2b_has_mass_on_cellt`:
  con `p_2b[cell_t] = 0,3` y `p_move = 0,5`, la suma sigue siendo
  1,0 (no `1 − 0,5·0,3 = 0,85`). El test verifica explícitamente que
  el denominador `(1 − p_2b[cell_t])` se aplica.
- `test_combine_soft_handles_p2b_cellt_near_one`: con `p_2b[cell_t]
  = 1 − 1e-10` y `p_move = 0,5`, no se lanza `ZeroDivisionError` ni
  aparecen NaN/Inf en la salida (clipping defensivo a `1 − ε`).
- `test_combine_hard_threshold`: con `p_move = [0,1, 0,9]` y `τ =
  0,5`, el primer argmax es `cell_t` y el segundo es `argmax(p_2b)`.
- `test_sweep_tau_returns_best`: con un val sintético donde `τ = 0,5`
  maximiza top-1, `sweep_tau` lo devuelve como `tau*`.

En `tests/test_ml_evaluate.py` (extensión):

- `test_evaluate_moves_only_filters_correctly`: con un DataFrame
  donde 3 de 10 filas tienen `y_move = 1`, `evaluate_moves_only`
  computa métricas sólo sobre esas 3.

En `tests/test_ml_build_l2.py` (nuevo, integración):

- `test_build_o4_l2_artifacts`: `build_o4_l2(...)` produce los 6
  `.pkl`, los 2 `predictions_test_*.parquet`, `metrics.parquet` y
  `tau_sweep.parquet` esperados, con las columnas documentadas.
- `test_build_o4_l2_idempotent`: dos ejecuciones consecutivas
  producen los mismos archivos (mismo `random_state`).

Criterio: todos los tests existentes (99 actuales) más los nuevos
deben pasar; ruff limpio.

### 8.2 Artefactos `save_artifact` planificados

| ID | Tipo | Decisión / hallazgo documentado |
|---|---|---|
| L2-v1-D1 | tabla | Calidad bruta de `clf_move` sobre test: AUC, accuracy, Brier, precision/recall por clase. Para ambas familias (RF, XGB). |
| L2-v1-D2 | figura + tabla | Barrido de `τ ∈ {0,3, 0,5, 0,7}` sobre val. Curva log-loss(τ) con línea horizontal en log-loss de la regla soft. Justifica `τ*` elegido. |
| **L2-v1-C1** | **tabla + barplot** | **Comparativa L2-v0 vs L2-v1 global** — métricas globales side-by-side para los 8 puntos comparados (ver esquema 8.3). Entregable narrativo central. |
| **L2-v1-C2** | **figura + tabla** | **Comparativa por estado HMM** (estacionario vs migración). Núcleo del éxito de L2: aquí debe aparecer el lift en top-1 migración. |
| L2-v1-C3 | tabla | Métricas sobre subset "moves only" (subset test con verdad `y_move = 1`). Aísla la calidad de `clf_dest`. |
| L2-v1-C4 | figura | Gap train-test por modelo de L2-v1, comparado con O4 base. Detecta sobreajuste por dataset reducido en etapa 2B (R2). |
| L2-v1-C5 | figura | Matriz de confusión `state_b verdad × cambia_celda predicho` para L2-v1 (soft) vs O4 base. Diagnóstico cualitativo de cuándo y por qué L2 mejora. |

### 8.3 Esquema exacto de L2-v1-C1 (la tabla central)

| modelo | modo | versión | regla | top-1 | top-3 | log-loss | dist_med_km |
|---|---|---|---|---|---|---|---|
| persistencia | — | baseline | — | 0,773 | 0,773 | 5,586 | 20,9 |
| markov(1) | — | baseline | — | 0,550 | — | — | 24,8 |
| RF | personalizado | L2-v0 | argmax monolítico | 0,644 | 0,754 | 5,22 | 23,1 |
| RF | personalizado | L2-v1 | soft | ? | ? | ? | ? |
| RF | personalizado | L2-v1 | hard (τ*) | ? | ? | —¹ | ? |
| XGB | poblacional | L2-v0 | argmax monolítico | 0,610 | 0,707 | 5,52 | 24,0 |
| XGB | poblacional | L2-v1 | soft | ? | ? | ? | ? |
| XGB | poblacional | L2-v1 | hard (τ*) | ? | ? | —¹ | ? |

¹ Log-loss de la regla hard no se reporta como métrica comparable.
Por construcción, hard devuelve una decisión one-hot clipada con
`ε = 1e-7`, lo que implica que cada fallo de argmax añade
`−ln(ε/(n−1)) ≈ 22,86` al log-loss. El número resultante mide el
artefacto del clipping, no la calidad del modelo. Hard se compara
contra soft únicamente en top-1, top-3 y dist_med_km, que sí son
métricas honestas para una regla determinista.

Esta tabla y su variante por estado HMM (L2-v1-C2) son la evidencia
primaria que entra en el capítulo 6 de la memoria, sección "L2 —
Mejora con dos etapas".

## 9. Métricas de éxito y criterio de aceptación

Tres criterios independientes:

1. **Primaria — log-loss (sólo soft).** L2-v1 soft mejora a L2-v0 si
   reduce log-loss ≥ 0,20 en al menos uno de los ganadores (RF
   personalizado o XGB poblacional). Hard se excluye de este criterio
   por la nota ¹ de §8.3.
2. **Secundaria — top-1 migración (soft y hard).** L2-v1 mejora si
   sube ≥ +5 pp absolutos en al menos uno de los ganadores. Es el
   régimen donde la descomposición en dos etapas debería aportar más.
   Se evalúa para soft y hard por separado.
3. **Diagnóstica — coherencia interna.** El gap train-test de
   `clf_dest` (C4) se mantiene en rango sano (gap ≤ gap O4 base + 0,1
   absoluto). Si se dispara, la mejora del log-loss es sospechosa y
   se discute en la memoria.

Interpretación honesta del resultado para la memoria:

- **3/3 cumplidos** → L2-v1 funciona, narrativa "la descomposición en
  dos etapas mejora calibración y captura mejor el régimen de
  migración".
- **2/3 cumplidos** (1+2 sin 3) → L2-v1 mejora pero sobreajustando,
  narrativa "el lift está acompañado de gap train-test elevado, queda
  cuantificado como limitación".
- **2/3 cumplidos** (1+3 sin 2) → L2-v1 mejora en calibración global
  pero no específicamente en migración, narrativa "la decomposición
  beneficia la calibración pero el techo de migración tiene otras
  causas".
- **1/3 cumplidos** → L2-v1 ambiguo, discutir caso a caso.
- **0/3 cumplidos** → L2-v1 falla; narrativa "la arquitectura sola no
  resuelve D1 — el techo es más profundo y exige modelos de secuencia
  (trabajo futuro)".

Cualquier resultado es válido para el TFG
(`feedback-memoria-tone`).

## 10. Decisiones metodológicas sin figura (van a la memoria como "decisiones documentadas en el spec")

- **Por qué etapa 1 siempre poblacional y etapa 2B en dos modos.** La
  asimetría es intencional: la decisión de movimiento es estructural
  (régimen fenológico, no individuo); el destino sí es individual
  (preferencias de hábitat, ruta migratoria propia). Esta separación
  da una narrativa biológicamente plausible al lector.
- **Por qué no se reabre F7 de O4 base.** L2 se restringe a atacar D1
  para mantener ablación limpia. Reactivar `veg_low/high` y
  `daylight_hours` contaminaría la conclusión de "cuánto aporta la
  arquitectura de dos etapas".
- **Por qué calibración a priori (no data-driven).** Garantiza que la
  fórmula soft tiene sentido probabilístico para cualquier familia
  sin tener que validar empíricamente cada caso. El coste (entrenar
  un isotonic regressor sobre val) es marginal.
- **Por qué `cv='prefit'` sobre `X_val` en lugar de `cv=3` con KFold
  aleatorio.** `CalibratedClassifierCV(cv=3)` baraja aleatoriamente
  las filas del input, sin respetar `bird_id` ni `date_utc`. Sobre un
  dataset de panel temporal como este, eso introduce un pequeño
  leakage futuro→pasado dentro del train y viola la regla heredada de
  O4 de "splits respetan estructura temporal por ave". `cv='prefit'`
  desacopla las dos etapas (entreno sobre `X_train`, calibro sobre
  `X_val`) y respeta estrictamente el orden train < val < test.
- **Por qué renormalizar `combine_soft` en lugar de dejar que sume
  `1 − p_move · p_2B(cell_t)`.** Sin renormalización, la salida de
  `combine_soft` no es una distribución de probabilidad y el log-loss
  resultante queda sesgado (subestima la confianza del modelo). La
  renormalización está demostrada matemáticamente en §1 y validada en
  el test `test_combine_soft_sums_to_one`.
- **Por qué `sweep_tau` usa top-1 en lugar de log-loss.** Hard
  devuelve one-hot, así que su log-loss queda dominado por el clipping
  `ε = 1e-7` (~22,86 por cada fallo de argmax). Top-1 sí es honesto
  para una regla determinista; minimizar log-loss sobre val
  seleccionaría `τ` por un artefacto numérico, no por calidad del
  modelo.
- **Por qué barrido de `τ` discreto y pequeño** (`{0,3, 0,5, 0,7}`).
  Un barrido continuo no aporta — la función log-loss(τ) es discreta
  por construcción (cambia de comportamiento cuando `p_move` cruza el
  umbral). Tres puntos cubren los regímenes "favorecer estacionario",
  "neutral", "favorecer movimiento".
- **Por qué no se entrena un modelo unificado etapa 1 + etapa 2B end-to-end.**
  Un modelo end-to-end (e.g. multitask con dos cabezas) perdería la
  interpretabilidad de la descomposición, no permitiría el ablation
  hard/soft, y exigiría tuning específico. Las dos etapas separadas
  permiten contar la historia metodológica como Markov → HMM → ML →
  ML estructurado por HMM.
- **Por qué LightGBM sigue descartado en L2:** mezclar reabrir LGBM
  con la arquitectura en dos etapas haría imposible atribuir las
  mejoras. Si en el futuro se quiere reabrir LGBM, sería un experimento
  separado.
- **Por qué no se considera regla "blend" intermedia entre soft y
  hard** (e.g. soft truncado por encima de cierto umbral). Habría que
  justificar el umbral con otra figura y la mejora esperada sobre
  soft puro es marginal. YAGNI.

## 11. Entregables al cerrar L2

Al cerrar L2-v1 (tag esperado `v0.4.2-o4l2-dos-etapas`), el repositorio
debe contener:

- `src/tfg_aves/ml/two_stage.py` y `src/tfg_aves/ml/build_l2.py`
  implementados y commiteados.
- Helper `evaluate_moves_only` añadido a `src/tfg_aves/ml/evaluate.py`.
- Tests nuevos en `tests/test_ml_two_stage.py`, `tests/test_ml_build_l2.py`
  y extensión de `tests/test_ml_evaluate.py`. Todos pasando.
- Notebook `notebooks/04b_eda_o4l2.py` (jupytext percent) que regenera
  las figuras y tablas vía `save_artifact`.
- 7 artefactos `L2-v1-{D1, D2, C1..C5}` en `reports/figures/`,
  `reports/tables/`, `reports/captions/` y `reports/INDEX.md`.
- Notas de memoria en `reports/memoria/06_o4_ml.md` (sección "L2 —
  Mejora con dos etapas") integradas en el flujo del capítulo.
- Entrada en `reports/ai-log/` (`00NN-o4l2-dos-etapas.md`) según
  política de scope.
- Commit(s) en castellano + tag `v0.4.2-o4l2-dos-etapas`.

## 12. Trabajo futuro (fuera del scope de L2)

- **L2 × L1**: re-entrenar L2-v1 sobre el feature set ampliado de
  L1-v1 (con viento). Sólo se aborda si **ambos** cierres
  individuales superan sus criterios primarios (§9). Decisión que se
  pospone al cierre de L1 y L2.
- **L2 con tres etapas** (estacionario / migración corta / migración
  larga): explorar si la estratificación adicional vale el coste de
  reducir aún más el dataset de la etapa más profunda. Sólo si L2-v1
  muestra que las dos etapas funcionan pero el techo en migración
  larga sigue alto.
- **`clf_move` con features de secuencia** (lags de `step_length_km`,
  rolling de cambio de celda): pertenece al trabajo futuro de
  "modelos de secuencia" identificado en O4 base, no se aborda en L2.
- **Calibración alternativa** (sigmoid, beta calibration): sólo si el
  reliability diagram de L2-v1-D1 muestra problemas residuales con
  isotonic.

## 13. Conexiones con el resto del TFG

- **O3 (HMM)**: L2 consume `state_b` y `posterior_b_migracion` como
  features y la métrica clave se desglosa por `state_b`. El éxito de
  L2 refuerza la utilidad del HMM como insumo estructural del
  pipeline ML, narrativa que se ancla en `[[project-o3-outcome]]`.
- **O4 base**: L2 reutiliza features, split, hiperparámetros y
  evaluación de O4 base. Los artefactos `data/processed/o4/` (L2-v0)
  son input no-modificable.
- **L1 (features de viento)**: L1 y L2 son **ortogonales por diseño**.
  La combinación L1+L2 queda como trabajo futuro condicional (§12).
- **L3 (regresión cuantiles)**: L3 se arrancará tras L2. Si L2 mejora
  significativamente, L3 hereda la arquitectura de dos etapas como
  baseline; si no, L3 vuelve al monolítico de O4 base. Decisión que
  se toma al arrancar L3.
- **O5 (mapas y evaluación del error)**: si L2-v1 (soft) supera a O4
  base, se convierte en el ganador candidato a exponer en la UI
  interactiva. La regla soft devuelve una distribución completa sobre
  las 849 celdas, ideal para los heatmaps planeados en O5.

---

Conecta con [[project-o4-improvement-lines]] (L2 es una de las tres
líneas paralelas), [[project-o4-outcome]] (estado de cierre de O4
base, sirve como referencia para comparar mejoras),
[[feedback-memoria-tone]] (tono investigador al narrar los
resultados de L2), [[feedback-decisions-must-be-justified]] (cada
decisión sin figura va a §10 y a la memoria) y
[[feedback-methodological-justifications]] (los riesgos identificados
en §7 van a la memoria como decisiones razonadas).
