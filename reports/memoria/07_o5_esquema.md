# Capítulo 7 — Mapas interactivos y análisis del error (O5)

> **Estado:** **ENTREGADO** (2026-05-31). Capítulo breve orientado a la
> herramienta: §7.1 (funcionalidades y uso) + §7.2 (desarrollo). Sin §7.3 de
> síntesis y sin análisis del error (decisión del autor; vive en el cap. 6).
> Figura única: `o5_prediccion_app.png` (captura de la app, ya enlazada).
> Hilos sincronizados en `hilos_abiertos.md`.
> **Última actualización:** 2026-05-31
> Fuente de contenido: `07_o5_viz.md` (notas, 2026-05-24, **cifras desfasadas**),
> `reports/INDEX.md`, tablas `o5_tab01..03` (**fuente de verdad**), figuras
> `o5_fig01..30`.
>
> **CIFRAS VERIFICADAS contra `o5_tab02.csv` (no las notas):**
> - Estacionario (n=3509): top-1 0,835; dist. mediana 2,1 km; cobertura
>   marginal 0,821; conjunta 0,725.
> - Migración (n=451): top-1 0,197; dist. mediana 61,0 km; cobertura marginal
>   0,742; conjunta 0,596.
> Por mes (`o5_tab03.csv`): top-1 mínimo en abril (0,595) y septiembre (0,659);
> máximo en enero (0,883) y febrero (0,888); dist. mediana peor en abril
> (6,6 km). Aves curadas (`o5_tab01.csv`): 91916A (11,7 % migr.), 91752A
> (2,9 %), 91823A (10,4 %), 91763A (8,4 %).

## Estructura final (orientada a la HERRAMIENTA, decidida con el autor)

> **Pivote (2026-05-31):** el autor pide centrar el capítulo en las
> **funcionalidades de la herramienta, cómo usarlas y cómo se desarrollaron
> (brevemente)**. NO se analiza el error (ya está en el cap. 6): nada de tablas
> por régimen/mes (`o5_tab02/03` fuera) ni sección de error. El error en km se
> lee *sobre el mapa* (lectura destacada en rojo). Título: «Visualización del
> movimiento en mapas interactivos».

- **§7.1 Funcionalidades y uso.** Dos productos: (a) mapas folium generados
  por ave + índice (vista mínima reproducible); (b) **aplicación a medida**
  (Leaflet) como pieza central. Lista de funcionalidades y su uso: selección de
  ave (82 + «Todas»), predicción del día + **error en km (caja roja)**,
  reproducción temporal (ventana, play/pausa/paso/velocidad, estela), capas
  conmutables (ruta real por año, Markov, estado del modelo oculto). Código de
  color firme (error rojo, real verde, p50 granate). Demo multi-paso de pasada.
  Figura: `o5_prediccion_app` (placeholder).
- **§7.2 Desarrollo (breve).** `tfg_aves.viz` (cómputo puro vs render); folium
  para los mapas por ave; app = plantilla HTML + JSON embebido, estática sin
  backend (predicciones precalculadas). Tres decisiones técnicas plegadas aquí:
  predictor recomendado por calibración, banda = rectángulo por eje, horizonte
  a un día (multi-paso solo demo).
- **§7.3 Síntesis** (pendiente de aprobación de §7.1–§7.2): cierre breve, qué
  hace evidente la herramienta, límites, enganche con conclusiones; app y
  «¿misma ruta para todas?» de pasada.

## Fuentes de fidelidad de las funcionalidades

- App: `docs/superpowers/specs/2026-05-24-o5-app-prediccion-interactiva.md`
  (controles, capas, colores; persistencia ELIMINADA, Markov es la capa
  conmutable).
- Mapas folium + orquestación: `src/tfg_aves/viz/{maps,build}.py`.

## Figuras (placeholders, PNG pendientes del autor)

- `o5_prediccion_app` (app a medida) → §7.1 (figura principal).
- Opcional `o5_fig01` (mapa folium por ave) si el autor quiere una segunda.

## Tablas

- Ninguna en el capítulo (el análisis cuantitativo vive en el cap. 6).

---

## (Histórico) esquema propuesto inicial

## Principio de diseño del capítulo

Capítulo **breve y al grano** (encargo del autor). No reintroduce el modelo ni
las métricas (ya definidas en el capítulo anterior); entra directo a la tarea:
convertir las predicciones en mapas e identificar *dónde* y *cuándo* el modelo
acierta o falla. Cierra el arco narrativo comportamiento → predicción → mapa.

## Enganche con lo anterior y lo siguiente

- **Hereda:** la regresión de cuantiles del desplazamiento (mejor modelo
  cartografiable: punto p50 + banda [p10,p90]); el grid de medio grado; el
  régimen estacionario/migración del modelo de comportamiento; las trayectorias
  diarias. Reutiliza top-1, log-loss y la baseline de persistencia ya fijados.
- **Produce / cierra:** confirmación cartográfica del hallazgo transversal
  ("persistencia con incertidumbre calibrada"); el error sigue la fenología,
  cerrando el círculo comportamiento → predicción → error. Siembra para
  conclusiones: límites (cobertura conjunta, sin memoria multi-día) y trabajo
  futuro (secuencia, banda conjunta).

## Estructura propuesta (4 secciones)

### 7.1 De la predicción al mapa
- Qué hace el capítulo y qué NO: no entrena, consume el modelo de cuantiles.
- Dos entregables: (a) vista de predicción por ave (p50 + banda, slider
  temporal); (b) análisis georreferenciado del error.
- Nota breve de ingeniería: paquete `tfg_aves.viz`, cómputo puro separado del
  render; salida `folium` HTML (sin backend, predicciones precalculadas).
- Sin figura propia (o un único diagrama de flujo si hiciera falta; preferible
  no).

### 7.2 Visualización de la predicción y su incertidumbre
- **Decisión 1:** base cartográfica = modelo de cuantiles, familia LightGBM,
  modo poblacional. Criterio: las tres familias convergen pero esta es la mejor
  calibrada (cobertura marginal ~80 % ≈ nominal), lo idóneo para una banda.
  Evidencia: tabla de familias del capítulo anterior (`o4_tab31`), citada, no
  repetida.
- **Decisión 2:** banda = rectángulo [p10,p90] por eje (no elipse): los
  cuantiles son por eje; el rectángulo es la representación fiel. Matiz
  (se siembra, se cuantifica en 7.3): cobertura conjunta < marginal.
- **Decisión 3:** vista principal a un día (calibrada); multi-paso solo como
  demo etiquetada (acumula error, HMM congelado). Honestidad metodológica.
- **Figura:** `o5_fig01` (vista de predicción de una ave, p50 + banda + real).
  Screenshot del HTML. Posible mención de pasada a `o5_fig02..04` (otras aves,
  variedad de régimen) sin figura propia. Tabla `o5_tab01` (aves curadas) solo
  si aporta; probablemente mención de pasada.

### 7.3 Análisis georreferenciado del error  *(núcleo del capítulo)*
- **Por régimen (`o5_tab02`):** estacionario top-1 0,840 / 2,1 km mediana;
  migración 0,229 / 61,7 km. Cobertura marginal vs conjunta (0,806/0,702 y
  0,747/0,609): el matiz fino que el tribunal puede preguntar.
- **Por mes (`o5_tab03`):** top-1 mínimo en picos migratorios (abr/sep/oct),
  máximo en invierno (ene/feb); la distancia empeora en abril y otoño. Cruce
  con la fenología (cierra el arco con el capítulo de comportamiento).
- **Mapas de error:** `o5_fig20` (error por celda), `o5_fig21` (calibración por
  celda; conecta con la elección de familia por calibración), `o5_fig22`
  (vectores de fallo en migración: hacia dónde se equivoca).
- Decisión: ¿cuántas de las 3 figuras entran? Para "breve" propongo **fig20 +
  fig22** como principales y fig21 mencionada/secundaria, o las tres si el autor
  prefiere. **(decisión a confirmar)**

### 7.4 Síntesis: una persistencia con incertidumbre calibrada
- El modelo es una "persistencia con incertidumbre calibrada": acierta el
  destino estacionario (~2 km), falla en migración (~62 km), y el error sigue la
  fenología. Cierra comportamiento → predicción → error.
- Límite honesto + nota breve "¿la misma ruta para todas las aves?": no hay ruta
  fija común, sí una función poblacional compartida (sin `bird_id`); probable
  pregunta de defensa, se despacha en un párrafo.
- Limitaciones → trabajo futuro: cobertura conjunta < marginal (ejes modelados
  por separado); demo multi-paso ilustrativa (HMM congelado); sin selector vivo.
- Mención **de pasada** a la app interactiva a medida (`o5_prediccion_app.html`)
  como extensión, sin figura ni subsección propia (salvo que el autor quiera).

## Figuras/tablas candidatas (a fijar con el autor)

| Artefacto | Sección | ¿Entra? |
|---|---|---|
| `o5_fig01` predicción por ave | 7.2 | sí (1 screenshot) |
| `o5_fig02..04` otras aves | 7.2 | mención de pasada |
| `o5_tab01` aves curadas | 7.2 | mención de pasada |
| `o5_tab02` error por régimen | 7.3 | sí (tabla núcleo) |
| `o5_tab03` error por mes | 7.3 | sí (o resumida en prosa) |
| `o5_fig20` error por celda | 7.3 | sí |
| `o5_fig21` calibración por celda | 7.3 | sí o secundaria |
| `o5_fig22` vectores de fallo | 7.3 | sí |
| `o5_fig30` demo multi-paso | 7.2/7.4 | opcional (demo etiquetada) |
| app a medida | 7.4 | mención de pasada |

## Decisiones abiertas para el autor

1. **Alcance de "breve":** ¿4 secciones como arriba, o fundir 7.2 en 7.1 para
   un capítulo aún más corto (3 secciones)?
2. **Cuántas figuras** del análisis del error (7.3): ¿las tres (fig20/21/22) o
   solo dos?
3. **Demo multi-paso (`o5_fig30`):** ¿se incluye como figura etiquetada o solo
   se menciona en prosa?
4. **App a medida:** ¿mención de pasada (recomendado) o subsección breve?
5. **Screenshots:** las figuras son HTML; ¿los pego yo de capturas que tú
   generes, o uso placeholders `\includegraphics` a la espera de los PNG?
   (Pendiente del autor según CLAUDE.md.)
