# L4: HMM A + vegetación cruda en el ML (diseño)

- **Fecha:** 2026-05-26
- **Estado:** diseño aprobado, pendiente de implementación.
- **Naturaleza:** línea **exploratoria** (curiosidad del autor). **No** forma
  parte del trabajo fundamental del TFG: sin `save_artifact()`/`INDEX`, sin
  entrada en `reports/ai-log/`, sin notas de memoria, sin tag de hito y sin
  notebook. Vive solo como código + tests + artefactos en `data/processed/o4/l4/`.

## Pregunta de investigación

L4 es L3 con dos cambios:

1. El estado latente lo aporta el **HMM A** (emisión cinemática pura:
   `step_in_km`, `cos_turning_in`) en vez del **HMM B**.
2. La vegetación (`veg_low`, `veg_high`) entra como **feature cruda del
   regresor ML**, no dentro de la emisión del HMM.

`daylight_hours` se **omite**: es esencialmente una función determinista de la
latitud y el día del año, y la matriz de features ya incluye `lat`, `lon`,
`sin_doy` y `cos_doy`, así que un modelo de árboles puede reconstruirlo. Dentro
del HMM B sí aportaba (su emisión no veía lat ni día del año); en el ML es
redundante. La vegetación, en cambio, es señal ambiental real no derivable de
lat + día del año.

La pregunta: ¿es mejor que el HMM absorba la vegetación en un estado conductual
discreto (L3, modelo B), o darle al regresor la vegetación cruda y un estado
puramente cinemático (L4, modelo A)?

## Lo que NO se reentrena

`data/processed/o3/features.parquet` ya persiste `state_a_causal`,
`posterior_a_migracion`, `veg_low` y `veg_high`. L4 solo **lee** de O3. Cero
cambios en O3 y cero recálculo del HMM.

## Diseño técnico

Reutiliza el grueso de L3; los cambios en código compartido son
retrocompatibles (defaults que preservan el comportamiento actual de L3).

| Pieza | Cambio |
|---|---|
| `ml/features.py::attach_o3_state_and_split` | Param nuevo `suffix="b"`. Con `"a"` pega `state_a_causal` + `posterior_a_migracion` (renombrado a `posterior_a_migracion_causal`). Además arrastra `veg_low`/`veg_high` desde O3. |
| `ml/evaluate.py::build_regression_predictions` | Param nuevo `state_col="state_b_causal"`; lee `meta[state_col]` y emite una columna con ese nombre. |
| `ml/evaluate.py::compute_persistence_baseline` | Param nuevo `state_col="state_b_causal"`. |
| `ml/evaluate.py::evaluate_by_state` | Ya parametrizado (`state_col`). Sin cambios. |
| `ml/build_l4.py` (nuevo) | Copia fina de `build_l3.py`. Mismas 3 familias (xgb/lgbm/rf) + modo individual con xgb (espejo completo de L3). |
| `ml/_paths.py` | `O4_L4_DIR = ROOT/"data"/"processed"/"o4"/"l4"`. |

### Feature set de L4

```
FEATURES_KINEMATIC (8, idénticas a L3)
  + state_a_causal
  + posterior_a_migracion_causal
  + veg_low
  + veg_high
```

Es decir: cinemáticas + estado A + posterior A + 2 features de vegetación. **No**
incluye `state_b_causal`, `posterior_b_migracion_causal` ni `daylight_hours`.

### Flujo de datos

`features.parquet` (O3) → `build_feature_matrix` (cinemáticas + celdas +
target `cell_id_t_next`) → `attach_o3_state_and_split(suffix="a")` (estado A +
split + veg) → por familia y eje, `fit_quantile_axis` →
`predict_quantiles` → `build_regression_predictions(state_col="state_a_causal")`
→ `metrics.parquet` + `predictions_test.parquet` + `model_*.pkl` en `o4/l4/`.

### Naming honesto

Las columnas de estado conservan el sufijo real (`state_a_causal`), sin
disfrazarlas de `b`. Esto se sostiene parametrizando `state_col` en las dos
funciones compartidas en vez de renombrar A como B.

## Artefactos de salida (en `data/processed/o4/l4/`)

- `metrics.parquet`: mismo esquema que L3 (familia, modo, scope, top1, top3,
  dist_centroide_km, dist_nativa_km, pinball_lat/lon, coverage_lat/lon).
- `predictions_test.parquet`: predicciones de test por familia y modo.
- `model_{familia}_{modo}_{dlat,dlon}.pkl`.

## Tests (`tests/test_ml_build_l4.py`)

Espeja `test_ml_build_l3.py` con fixtures sintéticos. Aserciones clave:

- El feature set efectivo contiene las 4 features de contexto de L4
  (`state_a_causal`, `posterior_a_migracion_causal`, `veg_low`, `veg_high`).
- **No** contiene `state_b_causal` ni `daylight_hours`.
- Se entrenan las familias correctas y se escriben los artefactos esperados.
- Retrocompatibilidad: los tests existentes de L3 siguen verdes (los defaults
  `suffix="b"` / `state_col="state_b_causal"` no alteran su comportamiento).

## Fuera de alcance

- Notebook de exploración (decisión del autor: solo `src` + tests).
- `save_artifact()`, `reports/INDEX.md`, `reports/ai-log/`, notas de memoria,
  tag de hito.
- Cableado en el orquestador canónico de O4 (`build.py`): L4 queda como
  callable independiente + test, fuera del pipeline fundamental.
- Tuning de hiperparámetros (se heredan los conservadores de L3).

## Entrega

Al terminar: reporte de métricas L4 vs L3 en la conversación y propuesta de un
único commit etiquetado como experimental (lo confirma el autor).
