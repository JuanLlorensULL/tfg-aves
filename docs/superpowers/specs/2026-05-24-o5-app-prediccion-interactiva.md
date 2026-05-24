# O5 — App interactiva de predicción (`o5_prediccion_app.html`)

> **Propósito de este documento:** registrar TODAS las pautas, decisiones y
> preferencias que el autor dio al construir la app interactiva a medida de
> O5, de forma que el trabajo se pueda **repetir o reconstruir** en otra
> sesión. Es un complemento del spec de O5
> (`2026-05-24-o5-viz-design.md`), específico de la app Leaflet.

## Qué es

App web **estática, sin backend**, generada por
`tfg_aves.viz.build.build_prediction_app` inyectando datos JSON en la
plantilla `src/tfg_aves/viz/templates/prediccion_app.html`. Salida:
`reports/figures/o5_prediccion_app.html`. Construida con la skill
`frontend-design`. **Es ADITIVA**: NO sustituye los mapas folium por ave
(`o5_fig01..04`) ni el índice (`o5_prediccion_index.html`); se conservan.

Base de datos: L3 **LightGBM poblacional** (`o4/l3_v2/`,
`familia=='lgbm'`, `modo=='poblacional'`) — punto continuo p50 + banda
[p10,p90] por eje. Mismo origen que el resto de O5.

## Estética (decisión de diseño, frontend-design)

- **Instrumento de campo naturalista**: panel lateral oscuro (slate) con
  texto pergamino, título serif (**Fraunces**), UI en **IBM Plex Sans**,
  lecturas/datos en **IBM Plex Mono**. Mapa claro (CartoDB positron) a la
  derecha. Sobrio, legible, no genérico.
- Fuentes y Leaflet por CDN (necesita internet, como el resto de mapas).

## Layout

Barra lateral izquierda (≈320 px) = menú; mapa Leaflet ocupa el resto.
Orden del panel:

1. **Título** + subtítulo (Larus fuscus · L3 · 1 día).
2. **Selector de ave** (`<select>`).
3. **Ventana de días**: un **único slider de DOS TOPES** (inicio + fin en
   la misma línea, con relleno entre ambos; no se cruzan) + **dos
   datepickers** (inicio → fin) para fijar día exacto sin arrastrar.
4. **Reproducción**: lectura "día actual" + **slider** + **datepicker del
   día actual** (acotado a la ventana) + botones ⏮ / ▶Reproducir-⏸Pausar /
   ⏭ + control de **velocidad (fps)**.
5. **Caja de fecha** (solo la fecha) y **caja de ERROR aparte, en ROJO**
   (km, distancia p50→real, de `dist_native_km`).
6. **Toggles** (checkboxes): "Mostrar punto de Markov", "Mostrar ruta real
   del ave", "Estado HMM (O3)".
7. **Leyenda** por secciones con cabeceras (ver abajo).

## Controles y comportamiento (pautas del autor)

- **Selector de ave:** las **82 aves del test** seleccionables, **ordenadas
  por histórico descendente** (días válidos en `daily`; las de más días
  arriba). Primera opción especial: **"Todas (rutas reales)"**.
- **Ventana de días = ventana de reproducción**: el play anima SOLO de
  inicio→fin, un día cada vez. Slider de dos topes + datepickers de inicio
  y fin, todos sincronizados. El **día actual** tiene su propio slider y su
  propio datepicker (acotado a [inicio,fin]); si la fecha cae en hueco,
  salta al día con dato más cercano.
- **Reproducción pausable** (▶/⏸), paso a paso (⏭), reinicio (⏮),
  velocidad fps. Durante el play, slider y datepicker del día actual se
  actualizan.
- **Punto de Markov (conmutable):** rombo (violeta) = centroide de la celda
  0,5° que predice el **baseline Markov(1) de O4** (`compute_markov_baseline`,
  mensual, reentrenado sobre el train temporal poblacional). Coherente con
  los números de la memoria.
- **Ruta real del ave (conmutable, SOLO en vista por ave):** la trayectoria
  real del periodo de test; **coloreada por año** si el periodo cubre >1
  año. El botón/checkbox de ocultar ruta va **en la vista por ave, NO en
  "Todas"**.
- **Estado HMM de O3 (conmutable, en AMBAS vistas):** punto por día
  coloreado por `state_b_causal` — **azul marino = estacionario, magenta =
  migración**. En "Todas" cubre los días de las 82 aves; por ave, la
  ventana visible.
- **Vista "Todas":** dibuja las rutas reales de las 82 aves a la vez,
  **un color por ave** (paleta categórica), líneas finas con **resaltado al
  pasar el ratón** (engorda + tooltip con el id). Sin predicción/animación;
  los controles por día se desactivan. **No** hay botón de ocultar rutas en
  esta vista.
- **Persistencia: ELIMINADA** (a petición del autor; antes era un círculo
  verde en el origen).

## Colores de datos (paleta SIN conflictos — regla firme)

Definidos en `:root` (CSS) y duplicados en el objeto `C` del script (deben
coincidir):

| Elemento | Color |
|---|---|
| p50 (predicción) y banda | azul cielo `#4dabf7` (banda translúcida) |
| posición real | ámbar `#f59f00` |
| Markov (rombo) | violeta `#9775fa` |
| **error (p50→real)** | **rojo `#fa5252`** — *el error SIEMPRE en rojo* |
| estacionario (estado O3) | azul marino `#1b3a6b` |
| migración (estado O3) | **magenta `#d6336c`** |
| rutas "Todas" | paleta categórica 16 colores (cíclica) |
| ruta por año | ámbar + violeta + verde-azulado + naranja + rosa + cian |

**Conflictos resueltos (regla del autor):** el **error se queda rojo**;
por eso **migración pasó de rojo a magenta** (no puede haber dos rojos), y
el **p50 (azul cielo) se separó del estacionario (azul marino)** (no dos
azules confundibles). La paleta por año arranca en el ámbar de "real" y
**evita el rojo puro** para no chocar con el error.

## Leyenda ("el índice") — por secciones

Reorganizada en bloques con cabeceras (`.leglabel`):
- **Predicción del día**: p50, banda [p10,p90], error p50→real, real, Markov.
- **Ruta real del ave**: colores por año (o "ventana de test").
- **Rutas reales (test)** (solo vista "Todas").
- **Estado HMM (O3)**: estacionario / migración (solo al activar el toggle).

## Datos embebidos (`prediction_app_data`)

`{"birds":[...], "all":[...], "curated":[...]}`:
- `birds`: TODAS las aves del test, ordenadas por histórico desc; cada una
  `{"id", "days":[...]}`. Cada día: `date`, `year`, `o` (origen=pred−p50),
  `p` (p50), `band` [[sur,oeste],[norte,este]], `r` (real t+1 o null),
  `m` (punto Markov o null), `e` (error km), `s` (estado O3 0/1).
- `all`: rutas reales de las 82 aves para la vista "Todas" (`track` +
  `st` estado por punto).
- `curated`: ids de las 4 aves de más histórico (referencia).
- Markov: `markov_points_for_test(features_o3, cells)` calcula el punto para
  TODO el test (reusa el baseline de O4).

## Decisiones descartadas

- **Pestaña L3 / L1** (alternar entre L3 cuantil y el mejor de L1 = XGB
  poblacional, top-1 0,581 / log-loss 5,827, mostrando la celda 0,5°):
  **propuesta y RECHAZADA** por el autor ("mejor no"). L1 está dominada
  para O5; si se retoma, sería solo una pestaña de comparación didáctica.
- **App tipo dashboard con backend** (streamlit/dash): descartada antes;
  todo es estático sin servidor.

## Cómo reconstruir

1. `tfg_aves.viz.build.build_prediction_app(preds, daily, out_dir,
   features_o3=..., cells=...)` con `preds = load_lgbm_predictions()`.
2. La plantilla `templates/prediccion_app.html` lleva todo el HTML/CSS/JS;
   `__PRED_DATA__` se sustituye por el JSON.
3. Tests en `tests/test_viz_build.py` (export de datos puro). Validación
   extra: `node --check` del `<script>` inline + JSON embebido parseable.
4. Tras regenerar, los HTML de folium NO se tocan (cambian solo IDs de
   folium); restaurarlos con `git checkout` si se regeneran por error.

## Método de trabajo con el autor (preferencia observada)

El autor **revisa la app renderizada** y pide ajustes concretos e
iterativos (mover un control, separar el error, cambiar colores, añadir un
toggle). Patrón: implementar el cambio mínimo, validar (`node --check` +
tests), regenerar, commitear con mensaje en castellano, y pedirle que
recargue (Ctrl+F5) y confirme. Mantener SIEMPRE el trabajo anterior
(aditivo), no borrar.
