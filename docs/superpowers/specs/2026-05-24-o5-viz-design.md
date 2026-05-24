# Diseño de O5 — Mapas interactivos y análisis del error

## 1. Resumen

O5 es el objetivo de **visualización y análisis del error** (44 h). No
entrena modelos nuevos: consume las predicciones ya calculadas de L3 y las
convierte en (1) **mapas interactivos folium** de la predicción por ave y
(2) **mapas y tablas de análisis del error** sobre el conjunto de test.

La base cartográfica es **L3 cuantil, familia LightGBM, modo poblacional**
(`data/processed/o4/l3_v2/`, `familia=='lgbm'`, `modo=='poblacional'`).
Cada predicción es un punto continuo `p50` `(lat,lon)` más una banda
rectangular `[p10,p90]ₗₐₜ × [p10,p90]ₗₒₙ`. Esta decisión **supersede** el
puntero previo de `CLAUDE.md` a `l3_v1`.

Dos pilares:

- **Pilar 1 — demo de predicción** por ave: vista principal a **un día**
  (punto p50 + banda + capas de comparación) con *time-slider* sobre los
  días de test, más una **demo multi-paso encadenada** etiquetada como
  exploratoria.
- **Pilar 2 — análisis del error**: coropletas de error y de calibración
  por celda 0,5°, overlay de vectores de fallo en migración, y tablas de
  métricas por régimen HMM y por mes (eco de C5/C6 de O3).

## 2. Contexto y constraints

- **Stack:** solo **folium** para mapas (ya en deps). Sin backend: los
  HTML son estáticos servibles desde fichero. La interactividad es
  pan/zoom, capas conmutables (`LayerControl`), popups y *time-slider*
  (`TimestampedGeoJson`). **No** hay desplegable que recalcule el modelo en
  vivo: todas las predicciones del test **ya están** precalculadas en
  `l3_v2/predictions_test.parquet`.
- **Evidencia para la memoria:** `save_artifact` guarda PNG (matplotlib) y
  CSV, no HTML. Por tanto:
  - Los mapas folium se guardan como `.html` versionado en
    `reports/figures/` (son evidencia citada).
  - Para citar un mapa en el `.tex` se usa un **screenshot PNG manual**
    (el autor elige zoom/encuadre). Sin dependencia headless nueva.
  - **Todo número que justifique una decisión va a una tabla
    `save_artifact`** (CSV + caption + INDEX). La regla "decisión con
    datos" se cumple por las tablas; los mapas ilustran.
- **Coherencia biológica:** todo resultado pasa sanity-check contra la
  fenología de *Larus fuscus* (cría jun-jul, migración abr-may y sep-oct,
  invernada africana dic-feb) antes de aceptarse.
- **Simplicidad:** proporcionado al TFG; sin sobreingeniería. La demo
  multi-paso es la pieza más compleja y la primera candidata a recortar.

### 2.1 Interfaz de los artefactos de L3 (entrada de O5)

- Modelos: `model_lgbm_poblacional_d{lat,lon}.pkl` — dict joblib con
  `{"model": _PerQuantileAxis, "feature_cols", "familia", "mode", "axis"}`.
  `_PerQuantileAxis.predict_raw(X) → (n,3)` con `[p10,p50,p90]`.
- Features de entrada: `FEATURES_KINEMATIC + FEATURES_HMM` (las 10 causales
  de O4, sin `bird_id` en poblacional).
- Helpers reutilizables de `tfg_aves.ml.quantile`: `predict_quantiles`,
  `point_to_cell`, `build_regression_predictions`, `interval_coverage`.
- `predictions_test.parquet` (filtrado `familia=='lgbm'`,
  `modo=='poblacional'`) trae por fila: `bird_id`, `date_utc`, `true_cell`,
  `pred_cell_top1`, `pred_lat`, `pred_lon`, `dist_native_km`,
  `dlat_p{10,50,90}`, `dlon_p{10,50,90}`, `in_interval_lat`,
  `in_interval_lon`, `state_b_causal`.

## 3. Decisiones de diseño fijadas

| # | Decisión | Valor |
|---|---|---|
| V1 | Base cartográfica | L3 cuantil **LightGBM poblacional** (`l3_v2`). Supersede `l3_v1` |
| V2 | Stack | **Solo folium HTML**; tablas vía `save_artifact` |
| V3 | Modo de predicción | **A (un día)** principal + **B (multi-paso)** demo etiquetada |
| V4 | Banda de incertidumbre | **Rectángulo** `[p10,p90]ₗₐₜ × [p10,p90]ₗₒₙ` (cuantiles por eje) |
| V5 | Capas de comparación (vista A) | real t+1 (siempre) · persistencia · Markov(1) **apagada por defecto** |
| V6 | Aves de la demo | **4 con más histórico**: 91916A, 91752A, 91823A, 91763A |
| V7 | Mapas de error | A error/celda · B calibración/celda · C vectores de fallo **solo migración** |
| V8 | Desgloses tabulares | métricas por **régimen HMM** y por **mes** (`save_artifact`) |
| V9 | Evidencia memoria | HTML versionado + **screenshot PNG manual** + tablas |

### 3.1 Justificación de V6 (aves curadas)

Criterio del autor: las **4 aves con más histórico**. Por días válidos en
`daily.parquet` (idéntico ranking que por días de test):

| Ave | Días válidos | Días test | % migración (B) |
|---|---|---|---|
| 91916A | 2111 | 405 | 12,5 % |
| 91752A | 1439 | 286 | 3,2 % (casi residente) |
| 91823A | 1403 | 269 | 12,8 % |
| 91763A | 1283 | 249 | 10,1 % |

El criterio da de regalo variedad de régimen (91752A casi residente vs el
resto ~10-13 %). 91916A aporta continuidad con O3 (C7) y el modelo
individual de L3. La selección se documenta con la tabla
`o5_tab01_aves-curadas`.

## 4. Arquitectura y flujo de datos

Paquete nuevo `tfg_aves.viz`, separando **cómputo puro** (testeable) de
**render folium** (difícil de testear unitariamente), siguiendo el patrón
de O1–O4:

```
src/tfg_aves/viz/
  error.py    # puro: error/celda, calibración/celda, métricas por régimen y mes
  chain.py    # puro: encadenado multi-paso (B) → trayectoria sintética + cono
  maps.py     # render folium: vista A, coropletas, vectores, demo B
  build.py    # orquestador build_o5(): genera HTML + dispara save_artifact
  __init__.py
notebooks/05_eda_o5.py        # jupytext percent, conduce build_o5
tests/test_viz.py             # sobre funciones puras de error.py y chain.py
```

Flujo:

1. `build_o5()` carga `predictions_test.parquet` (filtro lgbm/poblacional),
   `cells.parquet` (grid 0,5°), `daily.parquet` (trayectoria real),
   `features.parquet` de O3 (régimen por día) y los modelos lgbm.
2. **Pilar 1:** para cada ave de V6, `maps.py` construye el HTML de la vista
   A (slider sobre sus días de test); para 1-2 aves, `chain.py` + `maps.py`
   construyen la demo B.
3. **Pilar 2:** `error.py` agrega error y calibración por celda y métricas
   por régimen/mes; `maps.py` renderiza las coropletas A/B y los vectores C;
   las tablas se guardan con `save_artifact`.
4. Salidas a `reports/figures/*.html` (mapas) y `reports/tables/*.csv`
   (tablas). Captions castellanos en `reports/captions/`.

### 4.1 Contrato de cada unidad

- `error.py` recibe el DataFrame de predicciones (+ régimen) y devuelve
  DataFrames agregados; no toca folium ni disco.
- `chain.py` recibe los modelos lgbm, el HMM causal y una fila de partida;
  devuelve un DataFrame con la trayectoria p50 y los anchos de banda por
  paso; no toca folium ni disco.
- `maps.py` recibe DataFrames ya agregados y devuelve objetos
  `folium.Map`; no calcula métricas.
- `build.py` es el único que escribe a disco.

## 5. Pilar 1 — Demo de predicción

### 5.1 Vista A (un día) — principal

Por ave, un `folium.Map` centrado en su área. Capas:

- **Trayectoria real** de los días de test (línea + marcadores tenues).
- **Slider temporal** (`TimestampedGeoJson`) que, para el día seleccionado:
  - marcador del **inicio** (posición real en `t`),
  - marcador **p50** y línea inicio→p50,
  - **rectángulo** de banda `[p10,p90]` (V4),
  - marcador **real t+1**,
  - círculo **persistencia** (= inicio),
  - celda **Markov(1)** (polígono 0,5°, capa apagada por defecto).
- `LayerControl` para conmutar real / persistencia / Markov.
- Popups con métricas del día (distancia p50→real, dentro/fuera de banda).

La **persistencia** es trivial (= posición de inicio, marcador coincidente).
**Markov(1)** se obtiene del baseline mensual reentrenado en O4 (argmax de la
fila de transición de la celda de inicio); si no está serializado por fila,
`build_o5` lo recalcula a partir de las matrices de transición. No se
entrenan baselines nuevas.

### 5.2 Vista B (multi-paso encadenado) — demo etiquetada

Exploratoria, claramente rotulada **"demo exploratoria · banda ilustrativa
no calibrada"**. Para 1-2 aves (incl. 91916A), `chain.py`:

1. Parte de un día real de test.
2. En cada paso `i` (hasta `k≈7`): construye las 10 features causales para
   la posición actual (recomputando `step_in`, `*_bearing_in`,
   `cos_turning_in` desde el movimiento sintético, avanzando `sin/cos_doy`,
   y re-decodificando `state_b_causal` / `posterior_b_migracion_causal` con
   el **HMM causal filtrado forward-only**); predice `(dlat_p50,dlon_p50)`
   con el modelo lgbm; realimenta la nueva posición.
3. Acumula los anchos `[p10,p90]` por paso para dibujar un **cono
   ilustrativo** (no es una banda calibrada; los cuantiles no componen).

`maps.py` dibuja el track p50 encadenado + cono + track real superpuesto.

Riesgo asumido: la complejidad vive aquí. Si el coste se dispara, B se
recorta sin afectar al resto de O5 (V3 ya lo contempla).

## 6. Pilar 2 — Análisis del error

Sobre el test poblacional completo de lgbm (n=4037 transiciones).

### 6.1 Mapa A — error por celda

Coropleta del grid 0,5°: cada celda activa coloreada por **distancia
mediana** `p50→real` (haversine) de las predicciones cuyo inicio cae en
ella. Escala verde→rojo. Responde *dónde* acierta/falla geográficamente.

### 6.2 Mapa B — calibración por celda

Coropleta de la **cobertura empírica marginal por celda**: media de
`in_interval_lat` e `in_interval_lon` (fracción de días con el real dentro
del intervalo [p10,p90] de cada eje; ideal ≈ 0,80). Resalta dónde la
incertidumbre de L3 es honesta — el punto fuerte de la línea (cobertura
global lgbm 79,8/80,0 % ≈ 80 % nominal).

**Matiz metodológico que se documenta:** la banda dibujada (V4) es el
**rectángulo** = producto de los dos intervalos marginales al 80 %. Su
cobertura *conjunta* (real dentro del rectángulo, `in_interval_lat ∧
in_interval_lon`) es **menor** que 80 % por construcción y NO debe leerse
como el nominal. El mapa reporta la cobertura **marginal** (la calibrada a
≈80 %); la conjunta se menciona aparte en la tabla por régimen para no
inducir a error.

### 6.3 Mapa C — vectores de fallo (solo migración)

Sobre los días con `state_b_causal == migración`, flechas
`p50 → real` de los mayores errores. Responde *cuándo* el modelo falla en
catástrofe (coherente con el hallazgo de O4: top-1 ~0,13 en migración).

### 6.4 Tablas (`save_artifact`)

- **Por régimen HMM:** top-1, distancia mediana, cobertura para
  estacionario vs migración (eco de C5/C6 de O3 y de la tabla maestra de
  O4). Sanity-check de coherencia biológica.
- **Por mes:** mismas métricas mensuales, para cruzar el error con la
  fenología (picos de migración abr-may y sep-oct).

## 7. Decisiones metodológicas sin figura

- **Por qué LightGBM poblacional y no RF/XGB:** las tres familias convergen
  (top-1 0,764–0,769); LightGBM es marginalmente la mejor en top-1/top-3 y
  **la mejor calibrada** (cobertura 79,8/80,0 % ≈ 80 % nominal), lo idóneo
  para una banda cartografiable. Contrapartida documentada: 254 cruces de
  cuantil (vs 0 de RF), ya corregidos en el pipeline; se vigila el cono. No
  contradice el descarte de LightGBM en O4-base (aquello era clasificación;
  esto es regresión y aquí no diverge).
- **Por qué rectángulo y no elipse para la banda:** los cuantiles son por
  eje; una elipse implicaría una distribución conjunta no modelada. El
  rectángulo es la representación fiel.
- **Por qué un día como vista principal y multi-paso solo como demo:** la
  banda de un día es la calibrada y evaluada en O4; el encadenado acumula
  error y descompone la calibración. La honestidad metodológica manda.
- **Por qué sin backend / sin selector vivo:** "solo folium HTML" (V2) y
  YAGNI; las predicciones del test ya están precalculadas. Un punto de
  partida arbitrario exigiría sintetizar features causales dependientes de
  la trayectoria reciente — fuera de alcance.
- **Por qué screenshot manual y no export headless:** evita una dependencia
  pesada (selenium/driver) para un puñado de figuras; el autor controla el
  encuadre.

## 8. Tests

`tests/test_viz.py`, sobre las funciones **puras** (el render folium no se
testea unitariamente; a lo sumo se asserta que el `folium.Map` contiene las
capas esperadas):

- `error.py`: error por celda y cobertura por celda sobre un fixture
  sintético con valores conocidos; agregados por régimen y por mes con
  recuentos esperados; manejo de celdas sin observaciones.
- `chain.py`: la trayectoria encadenada tiene longitud `k`; cada paso
  recomputa features causales sin mirar el futuro (test leak-free heredado
  de O4); el cono ilustrativo crece monótonamente; partir de un día real
  reproduce el p50 de `predictions_test` en el paso 1.
- Coherencia: las métricas por régimen reproducen los agregados globales ya
  conocidos de la tabla maestra de O4 (control cruzado).

## 9. Entregables y evidencia

Mapas folium (`reports/figures/`, `.html` versionado):

- `o5_fig01_prediccion-91916A` … (vista A por ave de V6).
- `o5_fig0X_demo-multipaso-91916A` (vista B etiquetada).
- `o5_figXX_error-por-celda`, `o5_figXX_calibracion-por-celda`,
  `o5_figXX_vectores-fallo-migracion` (coropletas y vectores).

Tablas (`save_artifact`, `reports/tables/`):

- `o5_tab01_aves-curadas` (justifica V6).
- `o5_tabXX_error-por-regimen`, `o5_tabXX_error-por-mes`.

Cada artefacto con caption castellano y fila en `reports/INDEX.md`. Para la
memoria: screenshots PNG manuales de los HTML clave.

## 10. Riesgos

- **Demo multi-paso (B):** mayor complejidad (recomputar features causales +
  re-decodificar HMM por paso). Mitigación: alcance mínimo (1-2 aves, k≈7),
  cono ilustrativo, recortable sin tocar el resto.
- **`TimestampedGeoJson` con huecos de calendario:** las rachas con gaps no
  deben dibujar segmentos que atraviesen huecos (heredar la lógica
  gap-aware de O1/O4). Mitigación: segmentar por rachas válidas.
- **Volumen de los HTML:** folium embebe referencias a tiles, no los tiles;
  tamaño moderado. Aceptable versionarlos como evidencia.
- **Cruces de cuantil de LightGBM en la vista A:** ya corregidos en
  `predictions_test`; el rectángulo usa los cuantiles ordenados. Se
  añade un sanity-check de que `p10 ≤ p90` por eje antes de dibujar.
