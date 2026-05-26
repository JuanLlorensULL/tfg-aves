# Hilos abiertos de la memoria (coherencia entre capítulos)

> Sembrado desde **O1 (cap. 3)** el 2026-05-25. Cada capítulo posterior debe
> **cerrar o arrastrar** explícitamente estos hilos (referenciando a O1), para que
> la memoria se lea como un único proyecto. Se amplía al redactar O2–O5.
> Ver la skill `redactar-memoria-tfg`.

## Hilos sembrados en O1

| # | Hilo | Tipo | En O1 | Se retoma en |
|---|---|---|---|---|
| H1 | Longitud mínima de racha para entrenar la cadena (todas las rachas vs. mínimo) | Decisión abierta | §3.6.2 (explícito) | **O2** |
| H2 | Discretizar el dominio (norte de Europa ↔ este de África) en celdas | Handoff | §3.6.1 | **O2** (tamaño de celda 0,5°) |
| H3 | `daily.parquet` como entrada del modelo | Entregable | §3.1, §3.5.2 | **O2** (lo consume) |
| H4 | `fixes_clean.parquet` reservado a resolución original | Entregable diferido | §3.1, §3.5.2 | **O5** (mapas intradía) |
| H5 | Posición anclada a las 08:00 (roost / post-despegue, no centroide) | Caveat interpretativo | §3.4.1 | **O2** (transiciones roost-a-roost), **O5** (lectura del error) |
| H6 | Sesgo estacional del seguimiento (otoño ≫ primavera) | Caveat interpretativo | §3.6.3 | **O2** (transiciones por estación), **O5** (error por mes) |
| H7 | Heterogeneidad inter-individual del tracking | Decisión futura | §3.2.3 + fig. timeline | **O3** (HMM global), **O4** (poblacional vs personalizado) |
| H8 | Umbral de 30 días válidos, revisable si se necesita más diversidad de aves | Decisión abierta latente | No está en el cap. (estaba en el esquema y en la §3.7 retirada) | **O3 / O4** si demandan más aves |

## Hilos cerrados y sembrados en O2 (cap. 4)

> Actualizado al cerrar el capítulo de Markov (2026-05-25).

**Cerrados en O2:**

| # | Hilo | Cómo se cierra |
|---|---|---|
| H1 | Longitud mínima de racha | §4.3: se entrena con **todas** las transiciones de 1 paso, sin racha mínima (Markov(1) solo necesita pares consecutivos; exigir racha tiraría la mayoría de los datos). |
| H2 | Tamaño de celda | §4.2: `cell_deg = 0,5°` (decisión D1, fig. del trade-off). |
| H3 | `daily.parquet` como entrada | §4.1: el modelo lo consume directamente. |

**Arrastrados (siguen abiertos para O5):**

| # | Hilo | Estado tras O2 |
|---|---|---|
| H5 | Posición roost-a-roost (08:00) | §4.1 lo recoge (las transiciones son roost-a-roost). Se retoma en **O5** (lectura del error). |
| H6 | Sesgo estacional del seguimiento | Recogido en §4.3 (matrices mensuales mejor/peor estimadas) y §4.6.3 (log-loss por estación). Se retoma en **O5** (error por mes). |

**Sembrados en O2 (para capítulos posteriores):**

| # | Hilo | Tipo | En O2 | Se retoma en |
|---|---|---|---|---|
| H9 | Criterio de **log-loss** como métrica principal | Convención transversal | §4.5, §4.6.2 | **O3, O4** (lo heredan) |
| H10 | Baseline de **persistencia** ("mañana = hoy") | Baseline transversal | §4.5, §4.6 | **O3, O4, O5** (referencia a batir) |
| H11 | Régimen de comportamiento (residente vs migración) no modelado | Limitación → motivación | §4.6.4 (cierre) | **O3** (estados ocultos) |
| H12 | Ruta individual de cada ave (34 % de orígenes no vistos) no modelada | Limitación → motivación | §4.6.2, §4.6.4 | **O4** (features por individuo) |
| H13 | Predicciones LOBO almacenadas como referencia del error | Entregable | §4.1, §4.6 cierre | **O5** (análisis comparativo del error) |
| H14 | Teoría general de cadenas de Markov finitas | Frontera con Preliminares | §4.1 (remite a cap. 2) | ✅ **CERRADO en cap. 2** (§2.2) |

## Hilos cerrados y sembrados en Preliminares (cap. 2)

> Actualizado al cerrar el capítulo de Preliminares (estado del arte + marco
> teórico) el 2026-05-26. Estructura entrelazada por método (2.1 movimiento animal /
> 2.2 Markov / 2.3 HMM / 2.4 aprendizaje supervisado), breve y al servicio de las
> referencias. Auditado con dos agentes (organización + verificador): sin hallazgos
> críticos.

**Cerrados en Preliminares:**

| # | Hilo | Cómo se cierra |
|---|---|---|
| H14 | Teoría general de cadenas de Markov finitas | §2.2: propiedad de Markov de primer orden, matriz de transición estocástica por filas, Chapman-Kolmogorov y estimación por máxima verosimilitud. El cap. de Markov la referencia y la aplica/refina (Laplace). |

**Sembrados en Preliminares (referencias que los capítulos de desarrollo deben usar):**

| # | Hilo | Tipo | En cap. 2 | Se retoma en |
|---|---|---|---|---|
| P1 | Teoría del HMM (parámetros $\pi,A,B$, emisión gaussiana diagonal, problemas forward/Viterbi/Baum-Welch, filtrado vs suavizado) | Referencia teórica | §2.3 (+ fig. del modelo gráfico) | **Cap. 5 (HMM)** la referencia en vez de reexplicarla; debe usar el filtrado causal allí anunciado |
| P2 | Teoría de aprendizaje supervisado (RF, gradient boosting XGBoost/LightGBM, regresión cuantílica + pérdida pinball) | Referencia teórica | §2.4 | **Cap. 6 (ML)** la referencia; la regresión cuantílica enlaza con la banda de incertidumbre que **Cap. 7** lleva al mapa |

**Pendiente menor (sugerencias de la auditoría de organización, opcionales):** reforzar
en §2.4 el enganche explícito con la tensión interpretabilidad↔capacidad de la intro;
y, si se desea, nombrar en §2.1 la especie/caso. No bloqueantes; el autor decidió
mantener el capítulo lean (se eliminó a propósito el párrafo-panorama de las tres
familias por repetir §1.2).

## Decisiones y hilos sembrados en la Introducción (cap. 1)

> Redactada tras O1–O5 (2026-05-26). La introducción fija la narrativa global del
> trabajo y algunas convenciones que los capítulos de desarrollo deben respetar.

| # | Decisión / hilo | Dónde (intro) | A respetar en |
|---|---|---|---|
| I1 | Estados del HMM = **"estacionario" / "migración"** (NO "residente"; "residente" era la propuesta de 3 estados descartada). Confirmado en `hmm/causal.py` y `05_o3_hmm.md`. | §1.2, §1.3 | **Cap. 5 (HMM)** debe usar estos nombres |
| I2 | Espina dorsal = enfoque híbrido: Markov visible = **baseline**; **el HMM alimenta al ML** (HMM+ML = predictor híbrido) que busca superar la baseline; más una herramienta de cartografía interactiva. | §1.2 + fig. 1.1 | Caps. 4–7 (deben encajar con esta arquitectura) |
| I3 | **Énfasis comparativo** (interés central, instrucción de la tutora): comparar las soluciones desarrolladas con métrica común y baselines explícitas. | §1.3 (obj. general) | Caps. 4–7 |
| I4 | Objetivo 5 = herramienta de cartografía interactiva (el análisis espacial/estacional del error se retiró del enunciado del objetivo, por decisión del autor). | §1.3 | Cap. 7 |

**Riesgo de coherencia a resolver (I1):** `desarrollo_o2.tex` usa "residente o en
migración" (línea 398, handoff al HMM) y "casi residente" (línea 380). Para una
sola narrativa conviene cambiarlos a "estacionario" cuando se revise el capítulo de
Markov (allí no nombra el estado del HMM sino el régimen conductual, pero el término
debe ser único en toda la memoria). **Pendiente de visto bueno del autor.**

## Notas

- **H4 — riesgo de coherencia a resolver:** §3.1 afirma que de `fixes_clean.parquet`
  "se extraerán variables cinemáticas para la detección de comportamiento y los
  modelos supervisados". Pero en el proyecto las cinemáticas de O3/O4
  (`step_length`, ángulo de giro) se calculan sobre las **posiciones diarias**, no
  sobre los fixes densos; estos se usan en **O5** (mapas detallados). Al redactar
  O3/O4/O5 hay que **confirmar y, si procede, corregir** ese claim de O1 (verificar
  contra el código antes de citarlo).
- **H8 — latente:** no se promete en el capítulo (se retiró con la §3.7). Si O3/O4
  acaban revisando el umbral de aves, conviene **reintroducir una frase en O1** que
  lo deje anticipado, para no romper la narrativa (un cambio aguas abajo que O1 no
  anunciaba).
- **H1 / H2** son los hilos que O2 recoge de forma más directa: la primera línea de
  O2 debe abrir conectando con `daily.parquet` (H3) y resolviendo H1 y H2.

## Cómo se usa este fichero

Al cerrar el capítulo de un objetivo, marcar aquí qué hilos cierra (y cómo) y qué
hilos nuevos siembra para los siguientes. Mantenerlo sincronizado es lo que
garantiza que "todo encaje en la narrativa de un único proyecto".
