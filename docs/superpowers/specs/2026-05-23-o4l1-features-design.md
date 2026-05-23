# Diseño de L1 (objetivo O4) — Features de viento sobre el pipeline ML

- **Fecha:** 2026-05-23
- **Objetivo del TFG:** O4 — primera línea de mejora (L1) del pipeline
  supervisado base. Añade features de viento reanalysis ECMWF a 850 hPa al
  set de features actual para atacar la causa estructural **D4** ("ausencia
  de features de viento") diagnosticada en `reports/memoria/06_o4_ml.md`
  §"Análisis estructural del techo de rendimiento".
- **Autor:** Juan Llorens
- **Estado:** diseño aprobado el 2026-05-23, pendiente de implementación

## 1. Resumen

L1 amplía el pipeline supervisado de O4 con **tres features de viento**
extraídas de archivos NetCDF ECMWF reanalysis ECMWF que el autor ya descargó
en una iteración previa del proyecto (v2). Las features se calculan
mediante interpolación bilinear espacial del grid 0,5° al punto exacto
del fix diario de cada ave y se incorporan al pipeline existente como
columnas adicionales sin tocar arquitectura, target, split temporal ni
hiperparámetros.

Tras la implementación, el sistema produce dos conjuntos de artefactos
comparables sobre **el mismo test split temporal por ave** que O4 base:

- **L1-v0** (sin cambios): los resultados ya producidos por el cierre de
  O4 (tag `v0.4-o4-completo`, commit `4ed7401`). Sirven como baseline
  contra el que se compara la mejora.
- **L1-v1**: el pipeline con las tres features de viento añadidas.
  Genera nuevos modelos `.pkl`, `predictions_test.parquet` y
  `metrics.parquet` bajo un subdirectorio diferenciado.

La comparativa **L1-v0 vs L1-v1** sobre las cuatro métricas globales
(top-1, top-3, log-loss, distancia mediana km), más el desglose por
estado HMM (estacionario / migración), cuantifica el aporte aislado del
viento como predictor sin contaminar con otras decisiones.

L1 NO ataca D5 (ausencia de destino) ni Mo3 (ausencia de historia
multi-día), descartados en el brainstorm por mantener una ablación
limpia (una causa por línea). Estas dos causas se cubrirán en líneas
futuras (L2 ataca D1, L3 ataca Mo1) o quedarán como trabajo futuro
fuera del TFG.

## 2. Vocabulario

- **L1-v0** — estado base sin viento. Equivale al cierre actual de O4
  (tag `v0.4-o4-completo`). No requiere implementación, sólo
  preservación de los artefactos existentes y referenciación explícita
  como punto de comparación.
- **L1-v1** — pipeline O4 base + tres features de viento. Es lo que se
  implementa en este spec.
- **Comparativa L1-v0 vs L1-v1** — tabla y figuras que cuantifican el
  aporte del viento. Es el entregable narrativo central de L1 para la
  memoria.

## 3. Contexto y constraints

**Datos disponibles** (todos verificados al 2026-05-23):

- **Pipeline O4 base completo** en `data/processed/o4/` con los 6
  modelos `.pkl` (RF/XGB/LGBM × personalizado/poblacional),
  `predictions_test.parquet` (32 296 filas) y `metrics.parquet` (14
  filas). LightGBM descartado por divergencia; L1 no lo reabre.
- **Features de O3** en `data/processed/o3/features.parquet` (24 444 ×
  17): contiene las 8 features base de O4 más `daylight_hours`,
  `veg_low`, `veg_high` (descartadas en O4 por decisión F7 — colinealidad
  con `state_b` — y NO se reincorporan en L1).
- **Dataset Movebank** cubre `2009-05-25` a `2015-08-27`.
- **Wind data v2** en `/home/jllorens/Desktop/TFG/version2/data/raw/wind/`:
  7 archivos `wind_{YYYY}.nc` para años 2009-2015. Variables: `u`, `v`
  en m/s a presión 850 hPa. Resolución espacial 0,5° (139 lat × 93 lon).
  Resolución temporal: 1 snapshot diario a 12:00 UTC. Bbox: lat ∈
  [-3°, 66°] × lon ∈ [7°, 53°]. **Cobertura completa del dataset
  Movebank** (ningún día queda fuera). Atributos NetCDF confirman
  origen ECMWF (`GRIB_centre: ecmf`, `expver: 0001`, `GRIB_dataType:
  an`); el producto exacto (ERA-Interim vs ERA5 downsampled) queda
  pendiente de confirmar con el autor en la fase de implementación —
  no afecta al diseño de L1 pero sí a la cita bibliográfica de la
  memoria.

**Restricciones heredadas de O4 base (no se replantean en L1):**

- Target: celda 0,5° de O2 (~849 clases activas en train).
- Split temporal 80/10/20 por ave.
- Familias: RF + XGBoost (LightGBM sigue descartado).
- Modos: personalizado (con `bird_id`) + poblacional (sin él).
- Configuración fija de hiperparámetros por familia (sin rejilla).
- Pipeline gap-aware: ninguna fila atraviesa hueco calendario.
- Criterio principal: log-loss.

**Principio de simplicidad** (`feedback-keep-solutions-simple`): L1 añade
únicamente lo que ataca la causa D4. Cualquier tentación de mezclar
otras mejoras (lags, destino, reactivación de vegetación) se rechaza
para mantener la ablación limpia.

## 4. Decisiones de diseño fijadas

Decididas en el brainstorming del 2026-05-23, sin EDA previa.

### F1 — Tres features de viento, nada más

Se añaden exactamente tres columnas al set actual:

| Feature | Definición | Justificación |
|---|---|---|
| `wind_u_850` | Componente U (eastward) del viento a 850 hPa en m/s | Señal atómica raw del `.nc`. Permite a RF/XGB modelar interacciones espaciales. |
| `wind_v_850` | Componente V (northward) del viento a 850 hPa en m/s | Junto con U reconstruye el vector completo. |
| `wind_speed_850` | `sqrt(u² + v²)` en m/s | Magnitud interpretable directamente en figuras y feature importance. |

Descartadas explícitamente:

- `wind_direction` (ángulo en radianes): colineal con (U, V) y con
  discontinuidad ±π. RF/XGBoost reconstruyen el ángulo de U y V sin
  pérdida.
- `tailwind_component` (proyección sobre el bearing del ave): requiere
  definir un bearing de referencia, cada elección es discutible y
  añade engineering decisions. RF/XGBoost modelan la interacción (U,
  V) × (lat, lon, doy) implícitamente.
- Lags de viento (`wind_speed_lag1`, etc.): introducirían debate de
  lags multi-día que se cerró fuera de L1.

### F2 — Fuente: viento de v2 a 850 hPa, sin re-descarga

Se usan los `.nc` de v2 tal cual. Implicaciones documentadas:

- Nivel 850 hPa (≈1,5 km de altitud), **no** viento de superficie a
  10 m. Defendible biológicamente: dentro del rango de altitud de
  vuelo migratorio de *Larus fuscus* (0-1,5 km). Documentado como
  riesgo R2.
- Resolución temporal: 1 snapshot diario a 12:00 UTC. Mismatch de
  4 h con los fixes Movebank a 08:00 UTC. Documentado como R1.

La opción de descargar viento de superficie a 10 m vía Copernicus CDS
API se discutió y se descartó por simplicidad. Queda como trabajo
futuro si L1-v1 no muestra mejora suficiente.

### F3 — Localización: copia a `data/raw/wind/`

Los `.nc` se copian desde v2 a `data/raw/wind/` (no symlink).
Autocontención del proyecto. `.gitignore` cubre `data/raw/wind/`
igual que `migration_original.csv`.

### F4 — Match espacio-temporal: bilinear interpolation

Para cada fila `(bird_id, date_utc, lat, lon)` válida:

1. Selección del archivo `wind_{date_utc.year}.nc`.
2. Selección temporal: snapshot con `valid_time = date_utc 12:00 UTC`.
3. Match espacial: `xarray.Dataset.interp(latitude=lat,
   longitude=lon, method='linear')` interpola bilinealmente entre los
   4 puntos del grid 0,5° más próximos.
4. Las tres features se calculan a partir del U y V interpolados.

Edge cases:

- Fix con `lat=NaN` o `lon=NaN` (huecos de O1): features → NaN
  propagado, filtrado por gap-aware downstream.
- Fix fuera del bbox del viento: features → NaN. Verificable con
  L1-v1-D1.
- Año fuera de [2009, 2015]: imposible (verificado, dataset cubre
  2009-05-25 a 2015-08-27).

### F5 — Decisión F7 de O4 base se mantiene

`veg_low`, `veg_high`, `daylight_hours` siguen **descartadas** en L1.
F7 de O4 base argumentaba colinealidad con `state_b`. L1 no reabre esa
discusión: se añade sólo viento, lo demás idéntico.

### F6 — Misma arquitectura, mismo split, mismos hiperparámetros

L1 NO modifica:

- Las dos familias entrenadas (RF + XGBoost).
- Los modos (personalizado / poblacional).
- El split temporal 80/10/20 por ave.
- La configuración de hiperparámetros por familia (§8.6 del spec de O4).
- El criterio de selección de ganador (log-loss).

L1 SÍ amplía el set de features de 8 a 11 (poblacional) o de 9 a 12
(personalizado).

### F7 — LightGBM no se reabre

LightGBM sigue descartado en L1. Cualquier propuesta de retuneo
(p.ej. `learning_rate=0,01`, `min_child_samples=5`) queda fuera del
scope de L1 para mantener la ablación limpia. Si se quisiera rescatar
LightGBM, sería sub-tarea independiente con spec propio, no parte de
L1.

## 5. Cobertura de causas raíz

Según la nomenclatura de `06_o4_ml.md` §"Análisis estructural":

| Causa | ¿L1-v1 la ataca? | Cómo |
|---|---|---|
| M1 Persistencia óptima en estacionario | no | techo de definición |
| M2 Log-loss vs top-1 | n/a | métrica, no causa |
| D1 73 % de self-loops | no | reservado a L2 (modelo de dos etapas) |
| D2 Snapshot diario | no | requiere reprocesar O1 a paso intradía; fuera de TFG |
| D3 Rutas individuales no vistas | no | reservado a trabajo futuro |
| **D4 Sin viento** | **sí — directa** | tres features de viento reanalysis ECMWF 850 hPa |
| D5 Sin destino | no | descartado del scope de L1 por simplicidad |
| Mo1 Target categórico | no | reservado a L3 (regresión cuantiles) |
| Mo2 Horizonte 1 día | no | requiere modelo de secuencia, fuera de TFG |
| Mo3 Sin historia multi-día | no | descartado del scope de L1 (lags) por simplicidad |

L1-v1 ataca **una sola causa raíz (D4)** con feature engineering puro.
Esta decisión maximiza la interpretabilidad de la ablación: si L1-v1
mejora a L1-v0, la mejora es atribuible al viento. Si no mejora,
sabemos que el viento solo no basta y queda cuantificado para la
memoria.

## 6. Pipeline e integración

### 6.1 Módulos nuevos

```
src/tfg_aves/meteo/
    __init__.py
    wind.py        # load_wind_dataset, interpolate_wind_to_fixes
    build_wind.py  # orquestador → wind_per_fix.parquet
```

**`wind.py`** (funciones puras):

```python
def load_wind_dataset(
    years: list[int],
    base_dir: Path,
) -> xr.Dataset:
    """Carga los .nc de los años pedidos y devuelve un Dataset unido
    en la dimensión temporal."""

def interpolate_wind_to_fixes(
    wind_ds: xr.Dataset,
    fixes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Para cada (bird_id, date_utc, lat, lon) válida, interpola
    bilinealmente el viento del día (12:00 UTC) y devuelve un
    DataFrame con wind_u_850, wind_v_850, wind_speed_850."""
```

**`build_wind.py`** (orquestador):

```python
def build_wind(
    daily_path: Path,
    wind_raw_dir: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Lee daily.parquet, carga los .nc, interpola y guarda
    wind_per_fix.parquet."""
```

### 6.2 Cambios en módulos existentes

**`src/tfg_aves/ml/features.py`:**

- Extender la constante `_FEATURES_BASE` con las tres nuevas columnas
  (cuando `with_wind=True`).
- Nueva función `merge_wind_features(matrix, wind_df) -> DataFrame`
  que hace `LEFT JOIN` por `(bird_id, date_utc)`.

**`src/tfg_aves/ml/build.py`:**

- `build_o4()` recibe un nuevo parámetro `with_wind: bool = False`.
- Si `with_wind=True`, carga `data/processed/wind/wind_per_fix.parquet`
  y hace merge antes de construir la matriz de features.
- Los outputs se escriben a `data/processed/o4/l1_v1/` (en lugar del
  directorio raíz `data/processed/o4/`, que sigue conteniendo los
  artefactos de L1-v0 sin tocar).

### 6.3 Estructura de outputs

```
data/processed/
    o4/                              # L1-v0 (existente, no se toca)
        model_*.pkl                  # 6 modelos
        predictions_test.parquet
        metrics.parquet
    o4/l1_v1/                        # L1-v1 (nuevo)
        model_rf_personalizado.pkl   # 4 modelos (RF/XGB × 2 modos)
        model_rf_poblacional.pkl     # LightGBM NO se entrena (F7)
        model_xgb_personalizado.pkl
        model_xgb_poblacional.pkl
        predictions_test.parquet
        metrics.parquet
    wind/                            # nuevo, intermedio
        wind_per_fix.parquet
```

`data/raw/wind/*.nc` y `data/processed/wind/*.parquet` van a
`.gitignore` (igual que el dataset Movebank crudo).

### 6.4 Dependencias añadidas a `pyproject.toml`

- `xarray` (regular).
- `netCDF4` (regular, backend para leer `.nc`).

## 7. Riesgos identificados

| # | Riesgo | Mitigación / acción |
|---|---|---|
| R1 | Mismatch temporal 08:00 UTC fix vs 12:00 UTC wind | Asumir estabilidad sinóptica a 850 hPa en 4 h. Documentar en la memoria como decisión motivada. |
| R2 | Viento a 850 hPa (≈1,5 km), no de superficie | Defendible biológicamente. Documentar en la memoria. |
| R3 | Edge case: fix fuera del bbox del viento | NaN propagado → filtrado por gap-aware. Verificar cobertura en L1-v1-D1; debe ser 0 filas perdidas. |
| R4 | Resolución diaria, no horaria | Pierde brisa diurna, captura escala sinóptica (adecuado para migración). Documentar. |
| R5 | Dependencias nuevas (`xarray`, `netCDF4`) | Añadir a `pyproject.toml`, ejecutar `uv sync`, verificar tests. |
| R6 | Pérdida silenciosa de filas en el merge | Test específico que cuenta filas pre/post merge. Build falla si no cuadra. |
| R7 | F7 NO se reabre en L1 (descartar veg_*/daylight) | Decisión consciente; documentar en spec (este §) y memoria. |

## 8. Validación y artefactos

### 8.1 Tests automáticos

En `tests/test_meteo_wind.py` (nuevo):

- `test_load_wind_dataset_combines_years`: carga 2010 y 2011, verifica
  shape y rango temporal del dataset combinado.
- `test_interpolate_wind_known_point`: para un punto sintético en el
  centro de cuatro nodos del grid, la interpolación bilinear devuelve
  la media de los cuatro nodos.
- `test_interpolate_wind_out_of_bbox`: fix fuera del bbox → NaN en
  las tres features.
- `test_interpolate_wind_nan_input`: fix con `lat=NaN` o `lon=NaN`
  → NaN propagado.
- `test_build_wind_idempotent`: dos ejecuciones consecutivas producen
  el mismo `wind_per_fix.parquet`.

En `tests/test_ml_features.py` (extensión):

- `test_merge_wind_features_preserves_rows`: tras el merge, el número
  de filas no cambia (LEFT JOIN sobre claves únicas).
- `test_build_o4_with_wind_artifacts`: `build_o4(with_wind=True)`
  produce los outputs esperados en `data/processed/o4/l1_v1/`.

Criterio: todos los tests existentes (99 actuales) más los nuevos
deben pasar; ruff limpio.

### 8.2 Artefactos `save_artifact` planificados

| ID | Tipo | Decisión / hallazgo documentado |
|---|---|---|
| L1-v1-D1 | figura + tabla | Cobertura espacial: bbox del viento (lat ∈ [-3°, 66°] × lon ∈ [7°, 53°]) superpuesto a los fixes. Verifica que 0 fixes quedan fuera. |
| L1-v1-D2 | figura | Distribución de `wind_speed_850` por mes. Estacionalidad esperada. |
| L1-v1-C1 | tabla | Correlación Pearson entre `wind_u/v/speed_850` y `state_b`, `posterior_b_migracion`, `step_length_km`. Detecta colinealidad. |
| L1-v1-C2 | figura | Feature importance comparada L1-v0 vs L1-v1. Las 3 features de viento deben aparecer en top-10. |
| **L1-v1-C3** | **tabla** | **Comparativa L1-v0 vs L1-v1 global** — métricas globales side-by-side (top-1, top-3, log-loss, dist mediana). Entregable central. |
| L1-v1-C4 | figura + tabla | Comparativa por estado HMM (estacionario vs migración). El lift esperado del viento está en migración. |
| L1-v1-C5 | figura | Análisis post-hoc de aprendizaje del viento: % de aciertos cuando viento está alineado con la dirección de migración fenológica vs cuando no. |

### 8.3 Esquema exacto de L1-v1-C3 (la tabla central)

| modelo | modo | versión | top-1 | top-3 | log-loss | dist_med_km |
|---|---|---|---|---|---|---|
| RF | personalizado | L1-v0 | 0,644 | 0,754 | 5,22 | 23,1 |
| RF | personalizado | L1-v1 | ? | ? | ? | ? |
| RF | poblacional | L1-v0 | 0,557 | 0,708 | 5,65 | 25,2 |
| RF | poblacional | L1-v1 | ? | ? | ? | ? |
| XGB | personalizado | L1-v0 | 0,622 | 0,710 | 5,48 | 23,8 |
| XGB | personalizado | L1-v1 | ? | ? | ? | ? |
| XGB | poblacional | L1-v0 | 0,610 | 0,707 | 5,52 | 24,0 |
| XGB | poblacional | L1-v1 | ? | ? | ? | ? |

Esta tabla y su variante por estado HMM (L1-v1-C4) son la evidencia
primaria que entra en el capítulo 6 de la memoria, sección "L1 — Mejora
con viento".

## 9. Métricas de éxito y criterio de aceptación

Tres criterios independientes:

1. **Primaria — log-loss.** L1-v1 mejora a L1-v0 si reduce log-loss
   ≥ 0,10 en al menos uno de los ganadores (RF personalizado o XGB
   poblacional).
2. **Secundaria — top-1 migración.** L1-v1 mejora si sube ≥ +3 pp
   absolutos en al menos uno de los ganadores. Es el régimen donde el
   viento debería aportar más.
3. **Diagnóstica — feature importance.** Las 3 features de viento
   aparecen en el top-10 (de 11-12 totales) en al menos uno de los
   ganadores. Verifica que el modelo realmente las usa.

Interpretación honesta del resultado para la memoria:

- **3/3 cumplidos** → L1-v1 funciona, narrativa "el viento aporta
  consistentemente al pipeline".
- **2/3 cumplidos** → L1-v1 mejora parcial, narrativa "el viento aporta
  pero modestamente, queda cuantificado".
- **1/3 cumplidos** → L1-v1 ambiguo, discutir caso a caso.
- **0/3 cumplidos** → L1-v1 falla pero cuantifica el techo del viento
  solo, narrativa "el viento solo no basta — confirmado experimentalmente,
  abre línea para L2/L3 con confianza".

Cualquier resultado es válido para el TFG (`feedback-memoria-tone`).

## 10. Decisiones metodológicas sin figura (van a la memoria como
"decisiones documentadas en el spec")

- **Por qué no surface wind 10 m:** v2 ya tenía 850 hPa descargado;
  re-descargar de CDS habría introducido tiempos de cola y trabajo de
  registro/configuración sin ganancia clara (850 hPa está dentro del
  rango de altitud de vuelo de *Larus fuscus*).
- **Por qué no `wind_direction` ni `tailwind_component`:** ambas son
  derivadas redundantes con (U, V) que RF/XGBoost pueden modelar
  implícitamente. Añadirlas no aporta información y sí engineering
  decisions discutibles.
- **Por qué no se reabre F7 de O4 base:** L1 se restringe a atacar D4
  para mantener ablación limpia. Reactivar `veg_low/high` y
  `daylight_hours` contaminaría la conclusión de "cuánto aporta el
  viento".
- **Por qué LightGBM sigue descartado en L1:** retuneo de LightGBM
  ataca una causa distinta (incompatibilidad de hiperparámetros) y se
  mezclaría con la ablación del viento. Si se desea rescatarlo, será
  spec/plan/tag independiente, no parte de L1.
- **Por qué no LOBO en L1:** el split temporal por ave heredado de O4
  base es coherente; LOBO castiga por construcción al modelo
  personalizado. Cambiar el split en L1 mezclaría dos cambios.
- **Por qué `data/processed/o4/l1_v1/` y no sobrescribir:** preservar
  los artefactos de L1-v0 es necesario para que la comparativa
  L1-v0 vs L1-v1 sea reproducible bit a bit y para que el tag
  `v0.4-o4-completo` siga siendo válido.

## 11. Entregables al cerrar L1

- Tag `v0.4.1-o4l1-viento` sobre el commit final de L1.
- Entrada nueva en `reports/ai-log/` (siguiente número disponible)
  documentando L1.
- Sección nueva en `reports/memoria/06_o4_ml.md`:
  "L1 — Mejora con viento reanalysis ECMWF 850 hPa" con la tabla L1-v1-C3
  embebida y el análisis honesto del resultado.
- Actualización de `MEMORY.md` y `project_o4_improvement_lines.md`
  con el resultado de L1.

## 12. Trabajo futuro (fuera del scope de L1)

- **D5 (sin destino)** — features de distancia a centroides históricos
  de cría e invernada. Descartado de L1 por simplicidad. Puede
  retomarse como L1-v2 si L1-v1 muestra mejora insuficiente, o
  quedarse como trabajo futuro.
- **Mo3 (lags)** — historia multi-día explícita. Cubierto parcialmente
  por L2 (la etapa 2 ve indirectamente el régimen previo vía
  `state_b`) y plenamente por modelos de secuencia (LSTM/Transformer)
  identificados como trabajo futuro en `06_o4_ml.md`.
- **D4 alternativo** — viento de superficie 10 m descargado vía CDS,
  o múltiples niveles (850 hPa + 925 hPa + surface) para que el
  modelo elija. Sólo si L1-v1 a 850 hPa no muestra mejora.
- **Tailwind alineado con bearing fenológico** como feature derivada.
  Sólo si L1-v1 con (U, V, speed) raw no aprende la interacción
  espacial implícitamente.

## 13. Conexiones con el resto del TFG

- **Memoria §"Análisis estructural del techo de rendimiento"
  (06_o4_ml.md):** L1 es la primera materialización de la "asignación
  de causas a posibles mejoras" de la matriz en §"Asignación de causas
  a posibles mejoras". Esta spec se cita como referencia.
- **Memoria del log de IA:** L1 produce una entrada en
  `reports/ai-log/`.
- **L2 y L3:** independientes; no bloquean ni son bloqueadas por L1.
  Cada una con su propio spec/plan/tag.
- **O5 (visualización):** consumirá los modelos de L1-v1 si la
  comparativa muestra mejora ≥ los criterios de aceptación.
  En caso contrario seguirá consumiendo los modelos de L1-v0
  (=O4 base). Decisión documentada en O5 cuando se llegue.
