# Capítulo 7 — O5: Mapas interactivos y análisis del error

> **Estado:** notas
> **Última actualización:** 2026-05-24

## Resumen ejecutivo

O5 convierte las predicciones de L3 (regresión de cuantiles del
desplazamiento, familia **LightGBM poblacional**) en mapas folium
interactivos y en un análisis del error georreferenciado. No entrena
modelos: consume los artefactos de O4. Entrega dos pilares —una demo de
predicción por ave (punto p50 + banda de incertidumbre [p10,p90], con
slider temporal) y un análisis del error (coropletas de error y de
calibración por celda, vectores de fallo en migración, tablas por régimen
biológico y por mes)— que confirman cartográficamente el hallazgo central
del TFG: el modelo predice bien el destino diario cuando el ave es
estacionaria y falla cuando migra.

## Contexto y motivación

- **Pregunta que responde el capítulo:** ¿*dónde* y *cuándo* acierta o
  falla el mejor modelo del TFG, y cómo se comunica su predicción y su
  incertidumbre sobre un mapa? Cierra el TFG dando interpretabilidad
  espacial a los números de O4.
- **Conexión con objetivos previos:** consume L3 (O4) como base
  cartografiable (punto continuo + banda); reusa el grid 0,5° de O2, el
  régimen HMM causal de O3 (estacionario/migración) para segmentar el
  error, y las trayectorias de O1 (`daily.parquet`).

## Decisiones tomadas

Cada decisión cita su evidencia en `reports/INDEX.md`.

1. **Base cartográfica = L3 cuantil, familia LightGBM, modo poblacional.**
   - **Alternativas:** RF (QRF) y XGBoost poblacionales; `l3_v1` (L3
     original de familia única).
   - **Criterio:** las tres familias convergen (top-1 0,764–0,769), pero
     LightGBM es marginalmente la mejor en top-1/top-3 y **la mejor
     calibrada** (cobertura marginal 79,8/80,0 % ≈ 80 % nominal), lo idóneo
     para una banda cartografiable. Contrapartida: es la que más cruces de
     cuantil produce (254, ya corregidos por ordenación). No contradice el
     descarte de LightGBM en O4 base (aquello era clasificación, donde sí
     divergía; esto es regresión y no diverge).
   - **Evidencia:** `o4_tab31` (comparativa de familias L3).
2. **Stack = solo folium HTML; tablas vía `save_artifact`.**
   - **Criterio:** "mapas interactivos" del proposal + sin backend (YAGNI);
     las predicciones del test ya están precalculadas. Las figuras de la
     memoria son screenshots manuales de los HTML; los números que
     justifican decisiones van a tablas CSV.
3. **Banda de incertidumbre = rectángulo [p10,p90]ₗₐₜ × [p10,p90]ₗₒₙ.**
   - **Alternativas:** elipse.
   - **Criterio:** los cuantiles son por eje; una elipse implicaría una
     distribución conjunta no modelada. El rectángulo es la representación
     fiel. **Matiz que se documenta:** la cobertura *conjunta* del
     rectángulo es menor que el 80 % nominal (que es marginal por eje).
   - **Evidencia:** `o5_tab02` (cobertura marginal vs conjunta por régimen).
4. **Predicción a un día como vista principal; multi-paso solo como demo
   etiquetada.**
   - **Criterio:** la banda de un día es la calibrada y evaluada en O4; el
     encadenado acumula error y descompone la calibración. La honestidad
     metodológica manda.
5. **Aves de la demo = las 4 con más histórico** (91916A, 91752A, 91823A,
   91763A).
   - **Criterio:** decisión del autor; el máximo histórico da de regalo
     variedad de régimen (91752A casi residente, 3,2 % migración, vs el
     resto ~10-13 %) y un slider rico en días.
   - **Evidencia:** `o5_tab01` (aves curadas).

## Implementación

- **Módulos del paquete `tfg_aves.viz`** (cómputo puro separado del render,
  patrón O1-O4):
  - `error.py` — agregados puros: error/celda, calibración marginal/celda,
    métricas por régimen y por mes. La posición de origen (día t) se
    recupera invirtiendo el p50 (`lat_t = pred_lat − dlat_p50`).
  - `chain.py` — encadenado multi-paso (demo B): recomputa la cinemática
    entrante por paso y realimenta el p50.
  - `maps.py` — 7 constructores `folium.Map` (coropletas, vista A con
    `TimestampedGeoJson`, vectores de fallo, demo).
  - `build.py` — orquestador `build_o5` + `build_o5_tables`.
- **Refinamiento (demo multi-paso):** las features del HMM
  (`state_b_causal`, `posterior_b_migracion_causal`) se **congelan** en el
  encadenado, porque la emisión del HMM causal exige covariables
  ambientales (veg/daylight) atadas a posiciones reales, no disponibles
  para posiciones sintéticas futuras. El estado se fija al del día de
  arranque y el posterior a un valor heurístico (0,8 si migración, 0,1 si
  no). Es coherente con el carácter ilustrativo, no calibrado, de la demo.
- **Simplificación (vista A):** la capa Markov(1) (que el diseño preveía
  "apagada por defecto") se omite; la persistencia, que es la baseline
  relevante (gana en top-1), sí se dibuja.

## Resultados y validación

- **Métricas por régimen (`o5_tab02`, test poblacional lgbm):**
  - Estacionario (n=3566): top-1 **0,840**, dist mediana nativa **2,1 km**,
    cobertura marginal 0,806, conjunta 0,702.
  - Migración (n=471): top-1 **0,229**, dist mediana **61,7 km**, cobertura
    marginal 0,747, conjunta 0,609.
- **Métricas por mes (`o5_tab03`):** top-1 mínimo en los picos migratorios
  (abril 0,62; sep 0,67; oct 0,72) y máximo en invierno (ene 0,91; feb
  0,93); la distancia mediana empeora en abril (6,8 km) y otoño.
- **Figuras citadas:** `o5_fig01..04` (vista A por ave), `o5_fig20`
  (error/celda), `o5_fig21` (calibración/celda), `o5_fig22` (vectores de
  fallo en migración), `o5_fig30` (demo multi-paso).
- **Tablas citadas:** `o5_tab01` (aves curadas), `o5_tab02` (por régimen),
  `o5_tab03` (por mes).
- **Validación:** 17 tests unitarios de `viz` (183 en total), ruff limpio;
  revisión spec + calidad por módulo y revisión final holística;
  **sanity-check de coherencia biológica** superado (la caída en migración
  y el patrón mensual cuadran con la fenología de *Larus fuscus*).

## Conclusiones y limitaciones

- **Lo que funciona bien:** la cartografía hace tangible el hallazgo del
  TFG —el modelo es una "persistencia con incertidumbre calibrada" que
  acierta el destino estacionario (~2 km de error) y falla en migración
  (~62 km)—; la banda [p10,p90] está bien calibrada en lo marginal
  (~80 %); el error y la calibración por celda localizan geográficamente
  dónde el modelo es fiable.
- **Limitaciones:** la cobertura conjunta del rectángulo es menor que la
  marginal (consecuencia de modelar los ejes por separado); la demo
  multi-paso es exploratoria (cono ilustrativo no calibrado, HMM
  congelado); no hay selector vivo (sin backend), la navegación es por
  fichero + slider.
- **Aspectos abiertos:** un encadenado multi-paso riguroso requeriría un
  modelo de las covariables ambientales en posiciones futuras o un modelo
  de secuencia (trabajo futuro de O4); una banda conjunta calibrada
  requeriría modelar la dependencia entre ejes.

## Notas para la redacción final

- Expandir el matiz **marginal vs conjunta** de la calibración: es un punto
  fino que el tribunal puede preguntar; `o5_tab02` lo soporta con números.
- Justificar el **posterior congelado heurístico 0,8/0,1** de la demo
  multi-paso como decisión consciente de una demo no calibrada (no inventar
  rigor donde no lo hay).
- Conectar el mapa de calibración (`o5_fig21`) con la elección de LightGBM
  por calibración (cap. 6, `o4_tab31`): O5 confirma cartográficamente la
  ventaja de calibración que motivó la elección de familia.
- Conectar el desglose por régimen/mes con C3 de O3 (% migración por mes):
  el error sigue la fenología, cerrando el círculo O3→O4→O5.
- Imágenes: insertar screenshots de los HTML (la memoria no embebe HTML).
