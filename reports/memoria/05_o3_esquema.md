# Capítulo 5 — Detección de comportamiento con HMM: subesquema de redacción

> **Estado:** PROPUESTA, pendiente de aprobación del autor.
> **Fuente de verdad** del esquema del capítulo (sincronizar con cada poda).
> Fichero LaTeX destino: `latex/secciones/desarrollo_o3.tex` (label `ch:o3`).
> Notas de respaldo: `05_o3_hmm.md`. Teoría del HMM: cap. 2 §2.3 (`sec:prelim-hmm`).

## Decisiones meta tomadas con el autor (2026-05-26)

1. **Filtrado causal = mención breve, SIN sección propia, poco peso**
   (decisión del autor, 2026-05-26). Dentro de §5.2 se menciona que el estado
   se decodifica por **filtrado forward** para evitar *data leakage* cuando los
   modelos del cap. 6 lo consuman como feature (enganche con el cap. de ML). NO
   se narra el arco "suavizado→filtrado" ni el 91,1 % de acuerdo; solo la
   elección de diseño y su porqué (causalidad / no mirar al futuro).
2. **Circularidad geográfica = integrada en la justificación**, sin sección
   propia. Se cuenta (comprimida) dentro de la justificación de "sin
   estandarización / step en km" (§5.2).
3. **Matiz de coherencia:** las features cinemáticas se presentan como
   **entrantes por diseño causal** desde el principio (§5.1), SIN narrar que
   hubo una versión con cinemática saliente. Coherente con la postura del cap.
   de ML (que NO narra su fuga). Tanto el filtrado del decode como la cinemática
   entrante se presentan como decisiones de diseño causales, no como correcciones.

## Enganche con la narrativa (qué hereda y qué aporta)

- **Hereda de Markov (cap. 4):** los dos vacíos que la cadena visible no podía
  capturar (cierre §4.6): el **régimen de comportamiento** de cada día
  (residente vs migración, hilo **H11**) y la ruta individual (**H12**, para
  el cap. de ML). Este capítulo ataca el primero con estados ocultos.
  Hereda también el criterio de **log-loss** (H9) y la baseline de
  **persistencia** (H10), ya definidos: se referencian, no se redefinen.
- **Hereda de O1 (cap. 3):** entrada directa `daily.parquet` (H3); posición
  anclada a 08:00 / roost (H5); heterogeneidad inter-individual (**H7**).
- **Referencia a Preliminares (cap. 2 §2.3, hilo P1):** definición del HMM
  ($\pi, A, B$), emisión gaussiana diagonal (eq. `eq:prelim-emision`), los tres
  problemas (forward/Viterbi/Baum-Welch) y la distinción **filtrado vs
  suavizado**. NO se reexplica aquí; se aplica y se remite con `\ref`.
- **Convención de nombres (hilo I1):** estados = **estacionario / migración**
  (nunca "residente"). Verificado en `hmm/causal.py`.
- **Aporta a O4 (cap. 6):** `features.parquet` con `state_a_causal`,
  `state_b_causal`, posteriores y la columna `split`. O3 es la fuente única.
- **Aporta a O5 (cap. 7):** estratificación del error por régimen.

## Estructura de secciones propuesta

### Preámbulo sin título (introducción + objetivo integrados, SIN sección propia)
Decisión del autor (2026-05-26): **no hay sección "Introducción y objetivo"**; el
planteamiento se integra en un preámbulo de ~4 párrafos tras `\chapter`, que carga:
- Engancha con el cierre de Markov (dos vacíos; este capítulo ataca el régimen de
  comportamiento). Pregunta: ¿estados ocultos gaussianos descubren, sin
  supervisión, estacionario vs migración? (engancha H11).
- Por qué el HMM y no la cadena visible: la celda mezcla regímenes y el 73 % de
  self-loops colapsa el destino hacia el origen; el estado latente los separa.
- Dos estados, no tres (forrajeo subsumido; justificación remitida a §5.2-modelo).
- Alcance **global** justificado (cinemática comparable entre aves, no posiciones;
  engancha H7). Remite a §2.3 (teoría). Nombres estacionario/migración (I1).
- Entrada `daily.parquet` (cap. 3); salida de doble uso (feature para cap. 6 +
  estratificador del error para cap. 7).

### §5.1 Variables del movimiento (`sec:o3-features`)
- Vector de observación cinemático **entrante** (diseño causal):
  - `step_in_km`: haversine(pos_{t-1}, pos_t), en km, sin transformación.
  - `cos_turning_in`: cos del cambio de rumbo entrante; +1 rectilíneo
    (migración), −1 inversión, 0 = 90°. Linealización frente a
    `|ángulo|` por suavidad de la emisión gaussiana (just. 9.2bis).
- **Dos conjuntos de features** (mención breve, sin protagonismo): el
  cinemático (estado A) y el cinemático + contexto (estado B, con `veg_low`,
  `veg_high`, `daylight_hours`). Por qué el contexto (propuesta de la tutora);
  `daylight_hours` por fórmula astronómica (sin datos externos); `veg_*` de
  ECMWF; `veg_low`/`veg_high` separadas, no combinadas (9.1). El modelo
  canónico será B; la comparación entre ambos se zanja al final de §5.5.
- *Sin figura* (es el diseño de features; las distribuciones por estado van a
  resultados).

### §5.2 Especificación y ajuste del modelo (`sec:o3-modelo`)
- `GaussianHMM`, `covariance_type='diag'` (remite a eq. `eq:prelim-emision`;
  práctica estándar en movimiento animal, Patterson et al. 2017 / moveHMM;
  just. 9.3).
- `n_components = 2`: barrido D1 (**fig01**). AIC/BIC decrecen monótonamente con
  n (171 171 → 150 569 → 147 450 en AIC para n=2,3,4), pero n>2 no admite
  etiquetado biológico claro con las features disponibles; n=2 alinea con la
  pregunta binaria y maximiza interpretabilidad (9.7).
- Ajuste: k-means init + 10 restarts EM, retención del mejor LL en train
  (9.4, 9.5). Re-etiquetado determinista por menor `μ[step_in_km]`
  (estacionario = menor desplazamiento; 9.6), para reproducibilidad.
- **Sin `StandardScaler`, step en km** (9.2). Aquí se **integra comprimida** la
  circularidad: escalar + `log` comprimía la bimodalidad natural del step
  (modos ~5 km y ~170 km, factor 30-100× → 3-4× en log), desplazando el
  k-means hacia la combinación contexto (luz + veg) que codifica estación y
  bioma. El síntoma fue biológicamente implausible (~100 % migración en enero,
  cuando las gaviotas invernan en África); revertir a km sin escalar
  (replicando la lección de `v2/HMM5.ipynb`) restauró la bimodalidad cinemática
  genuina. (El contraste cinemática-sola vs +contexto ayudó a confirmar que la
  separación debía venir del step; se cuenta sin dar protagonismo a la
  ablación.)
- **Decode por filtrado forward (mención breve, poco peso).** El estado de cada
  día se infiere por **filtrado** (`P(estado_t | obs_{1:t})`, solo pasado y
  presente), no por suavizado: como lo consumirán los modelos del cap. 6 como
  feature, un decode que mirase al futuro introduciría *data leakage*. Una o dos
  frases, remitiendo a la distinción filtrado/suavizado de §2.3. SIN narrar el
  arco "suavizado→filtrado" ni el 91,1 %.
- Entregable (nota breve): `features.parquet` con estados, posteriores y columna
  **`split`** (temporal por ave, 80/10/20; just. 9.8). O3 es la fuente única;
  el cap. 6 y el cap. 7 la consumen sin recomputar.
- Nota de ingeniería breve: paquete `tfg_aves.hmm`, `build_o3()` reproducible
  (`random_state=0`).
- **Figura: fig01** (barrido n estados).

### §5.3 Los estados descubiertos: resultados y validación biológica (`sec:o3-resultados`)
- Medias gaussianas por estado (Modelo B causal): `μ[step]` **6,5 vs 162,7 km**
  (~25×); media empírica del step en días de migración **~190,8 km** (tab11).
  Contexto apenas separa.
- **Coherencia biológica como criterio principal** sin ground truth (9.11): no
  hay etiqueta diaria; la fenología de *Larus fuscus* (Wikelski et al. 2015)
  es el criterio externo.
- Patrón estacional (% migración por mes) = validación central: valle
  **jun-jul** (cría, mínimo en junio), picos **abr** (máximo) y **sep-oct**
  (paso), baja dic-feb (invernada). **Figura: fig04** (state-vs-biology) +
  tabla `o3_tab04`.
- Validación espacial poblacional: los tres clusters esperados (cría N Europa,
  corredor mediterráneo, invernada Sahel) coinciden con la ruta conocida.
  **Figura: fig09** (mapa de las 82 aves por estado).
- Separación cinemática genuina por estado. **Figura: fig02** (features por
  estado A) — bimodalidad nítida del step. *(Alternativa: fig03 para B.)*
- Proporción global 86,1 % / 13,9 % (B causal, tab10): plausible (migración
  activa 2-3 meses/año). *Mención en texto, sin figura propia (fig10/pie se
  omite por filtro de relevancia).*
- **Cierre breve A vs B (sin sección propia, decisión del autor):** el estado
  cinemático (A) y el de contexto (B) coinciden en el **98,1 %** de los días, y
  el contexto pesa poco (Cohen's d pequeño: veg_high −0,47, luz −0,39), de modo
  que B no es circular. Se adopta **B** como canónico (conserva y valida la
  propuesta de la tutora y afina algo más el patrón estacional de cría, visible
  en fig04); A queda como **alternativa reversible** (sustituir `state_b_causal`
  por `state_a_causal`). Un párrafo, respaldado por fig04 + las dos cifras
  inline. *Sin fig05/fig06/fig07 propias.*
- **Cierre del capítulo (integrado al final de §5.3, SIN sección de
  conclusiones; decisión del autor):** párrafo(s) de síntesis y handoff, al
  estilo del cierre de Markov (§4.6). Recoge en prosa: qué funciona (estados no
  supervisados coherentes con la fenología; bimodalidad de dos órdenes de
  magnitud); limitaciones (aporte marginal del contexto; validación poblacional
  no individual; covarianza diagonal; modelo global que ignora umbrales por
  ave); y el handoff (cierra **H11**; arrastra **H7** al cap. 6 per-individual;
  `features.parquet` con estados, posteriores y `split` listo para el cap. 6 y
  para estratificar el error del cap. 7). No es una sección aparte: son los
  últimos párrafos de §5.3.

## Inventario de figuras (filtro de relevancia)

| Fig | Artefacto | Sección | Estado |
|---|---|---|---|
| fig01 | nstates-aic-bic-sweep | §5.2 | **núcleo** (decisión n=2) |
| fig02 | features-by-state-a | §5.3 | **núcleo** (separación cinemática) |
| fig04 | state-vs-biology | §5.3 | **núcleo** (validación estacional + justifica B) |
| fig09 | all-birds-spatial-by-state | §5.3 | **núcleo** (validación espacial) |
| fig03 | features-by-state-b | — | mención (alternativa a fig02) |
| fig05 | ab-agreement | — | demotada (98,1 % a texto) |
| fig06 | per-bird-state-proportions | — | demotada |
| fig07 | feature-influence-cohens-d | — | demotada (Cohen's d a texto) |
| fig08 | bird-trajectory-by-state | — | mención (1 ave; fig09 lo cubre) |
| fig10 | state-proportion-pie | — | omitida (dato a texto) |

Total núcleo: **4 figuras** (fig01, fig02, fig04, fig09). A vs B sin figura propia.

## Bibliografía a añadir/usar (verificar metadatos)
- `wikelski2015` (dataset Movebank) — ya `% verificar` en biblio.
- `patterson2017` (HMM movimiento animal, `covariance_type='diag'`).
- `langrock2012`, `rabiner1989`, `norris1997` ya en cap. 2 (reusar `\cite`).

## Hilos que este capítulo debe cerrar/sembrar (`hilos_abiertos.md`)
- **Cierra H11** (régimen de comportamiento, sembrado en Markov §4.6.4).
- **Recoge H7** (heterogeneidad) parcialmente y lo **arrastra** al cap. 6.
- **Hereda H9/H10** (log-loss, persistencia) sin redefinir.
- **Honra I1** (estados estacionario/migración) y **P1** (teoría HMM §2.3).
- Riesgo I1 a resolver aparte: Markov §4 usa "casi residente"/"residente o en
  migración"; cambiar a "estacionario" cuando el autor lo autorice (no bloquea
  este capítulo).
