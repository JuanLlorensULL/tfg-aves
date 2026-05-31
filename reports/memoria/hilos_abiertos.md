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

## Hilos cerrados y sembrados en el HMM (cap. 5)

> Actualizado al cerrar el capítulo del HMM (2026-05-31). Estructura preámbulo + §5.1
> (variables del movimiento) + §5.2 (especificación y ajuste) + §5.3 (estados
> descubiertos, con cierre integrado, sin sección de conclusiones aparte). Auditado con
> dos agentes (organización + verificador): APTO CON CAMBIOS MENORES, aplicados al
> cierre (poda de redundancia preámbulo p1↔p2, remisión a §2.3 para filtrado/suavizado,
> retirada del meta-comentario "iteración anterior", split 72/8/20 explicitado, cifras
> de Cohen una sola vez, handoff a H7 explícito, caption fig01 reconciliado con el
> cierre causal). Ninguna afirmación factual falseada; la causalidad del decode
> (forward-only) verificada contra `causal.py`.

**Cerrados en el HMM:**

| # | Hilo | Cómo se cierra |
|---|---|---|
| H11 | Régimen de comportamiento (residente vs migración) | §5.3 cierre: el HMM con dos estados ocultos recupera sin supervisión los dos regímenes (estacionario / migración) que la cadena visible promediaba; validación por coherencia fenológica con *Larus fuscus* (fig04, fig09). |
| P1 | Teoría del HMM | §5.1–§5.3 la **referencian** (`sec:prelim-hmm`, `eq:prelim-emision`) sin reexplicarla. El filtrado forward-only del decode se anuncia como aplicación causal de esa teoría. |
| I1 | Estados "estacionario / migración" | Honrado en todo el capítulo, incluido el código (`causal.py:191-193`). |

**Arrastrados (siguen abiertos para los capítulos siguientes):**

| # | Hilo | Estado tras el HMM |
|---|---|---|
| H7 | Heterogeneidad inter-individual del tracking | §5.3 cierre lo arrastra explícitamente al **cap. 6 (ML)** ("donde se retomará la heterogeneidad inter-individual que el capítulo \ref{ch:o1} dejó sembrada"). El HMM es global por diseño (cinemática comparable entre aves); el ML decidirá poblacional vs personalizado. |
| H9 | log-loss como criterio | El HMM no lo evalúa (validación no supervisada); el cap. 6 lo hereda intacto de Markov. |
| H10 | Baseline de persistencia | Igual: ni se usa ni se redefine en el HMM; sigue como referencia a batir en el cap. 6. |

**Sembrados en el HMM (para capítulos posteriores):**

| # | Hilo | Tipo | En el HMM | Se retoma en |
|---|---|---|---|---|
| H15 | `features.parquet` con `state_a_causal`, `state_b_causal`, posteriores `posterior_b_*` y columna `split` (72 % train / 8 % val / 20 % test por ave, sin solapamiento temporal) | Entregable | §5.2 final | **Cap. 6** (lo consume como variable de entrada de alto nivel y reutiliza el split sin recalcular) y **cap. 7** (estratifica el error por régimen) |
| H16 | Modelo B canónico (cinemática + contexto), A como alternativa reversible (98,1 % de acuerdo) | Decisión de diseño | §5.3 cierre A/B | **Cap. 6 / cap. 7** consumen `state_b_causal` por defecto; cambio a `state_a_causal` documentado como ablación reversible |

## Hilos cerrados y sembrados en la predicción supervisada (cap. 6)

> Actualizado al cerrar el capítulo de predicción supervisada (2026-05-31).
> Estructura: preámbulo + §6.1 (datos, variables y diseño) + §6.2 (predicción
> categórica) + §6.3 (predicción continua con cuantiles) + §6.4 (síntesis). Sin
> auditoría (queda a decisión del autor). Modelo de dos etapas excluido por
> decisión del autor (no relevante para la narrativa central). Naming firme: en
> prosa, títulos y captions no aparece nunca «L1/L2/L3» ni «línea 1/2/3»; las dos
> formulaciones se nombran descriptivamente.

**Cerrados en el capítulo 6:**

| # | Hilo | Cómo se cierra |
|---|---|---|
| H7 | Heterogeneidad inter-individual | §6.2.5 y §6.3.4: una prueba en cada formulación con un clasificador y un regresor entrenados únicamente con el ave de mayor histórico (91916A) muestra que el modelo poblacional ya la captura tan bien como un modelo dedicado. Modelo canónico poblacional, sin `bird_id`. |
| H12 | Rutas individuales no vistas en train | Cerrado conjuntamente con H7 por la misma evidencia. La heterogeneidad existe pero la distribución poblacional la cubre sin necesidad de personalizar. |
| P2 | Teoría ML (RF, boosting, regresión cuantílica) | §6.1.3 referencia §\ref{sec:prelim-ml} sin reexplicar; §6.3.1 referencia §\ref{sec:prelim-cuantil} y la ecuación \ref{eq:prelim-pinball}. |

**Arrastrados (siguen abiertos para el capítulo 7):**

| # | Hilo | Estado tras el capítulo 6 |
|---|---|---|
| H9 | log-loss como criterio principal | §6.1.4 lo consume como métrica probabilística complementaria de las de precisión; la elección entre familias pondera todas las métricas en conjunto. Sigue como referencia transversal. |
| H10 | Persistencia como baseline a batir | §6.2 confirma la dificultad estructural de batirla; §6.3 la iguala en agregado (distancia mediana 20,9\,km empatada, *top-1* a una o dos décimas) y revierte la pérdida amplia que la formulación categórica acumulaba. Sigue invicta en *top-1* global y en el régimen de migración. Se retoma en el cap. 7 sobre el mapa. |
| H15 | `features.parquet` + columna `split` (72/8/20 por ave) | §6.1.1 lo consume sin recalcular. Sigue siendo el entregable transversal del cap. 5. |
| H16 | Modelo B canónico del HMM | §6.1.2 lo consume (`state_b_causal`, `posterior_b_migracion_causal`). |

**Sembrados en el capítulo 6 (para el capítulo 7):**

| # | Hilo | Tipo | En el capítulo 6 | Se retoma en |
|---|---|---|---|---|
| H17 | Banda de incertidumbre `[p10, p90]` calibrada (cobertura empírica ≈ 80\,\% nominal) | Entregable | §6.3.3, §6.3.4, §6.4 | **Cap. 7** (capa de incertidumbre sobre el mapa) |
| H18 | Predicciones del modelo recomendado etiquetadas con el régimen del HMM | Entregable | §6.4 (entrega final) | **Cap. 7** (estratificación del error por régimen y por mes) |
| H19 | Modelo recomendado = regresión de cuantiles con LightGBM en modo poblacional, con XGBoost como alternativa equivalente | Decisión de diseño | §6.3.4 (modelo recomendado), §6.4 | **Cap. 7** lo consume como predictor base |
| H20 | Techo estructural común del enfoque: horizonte de un día con variables locales no contiene el predictor causal de un evento migratorio (viento, historia multi-día, destino) | Limitación → trabajo futuro | §6.4 (sin algoritmo ni reformulación que lo salve) | **Cap. 9 (conclusiones / trabajo futuro)** lo recoge y propone las tres direcciones |

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

## Hilos cerrados y sembrados en la visualización en mapas (cap. 7)

> Actualizado al cerrar el capítulo de visualización (2026-05-31). Capítulo
> **breve, orientado a la herramienta**: §7.1 (funcionalidades y uso de la
> aplicación interactiva) + §7.2 (desarrollo). Por decisión del autor NO repite
> el análisis cuantitativo del error (vive en el cap. 6): sin sección de error
> ni tablas; el error en kilómetros se lee sobre el mapa. Sin síntesis (§7.3)
> aparte. Figura única: captura de la app (`o5_prediccion_app.png`).

**Cerrados en el capítulo 7:**

| # | Hilo | Cómo se cierra |
|---|---|---|
| I4 | Objetivo = herramienta de cartografía interactiva | La aplicación web interactiva (82 aves, predicción a un día con banda, reproducción temporal, capas conmutables) materializa el objetivo. |
| H17 | Banda de incertidumbre `[p10,p90]` | Se lleva al mapa como el rectángulo de incertidumbre por eje. |
| H19 | Predictor recomendado (regresión de cuantiles, LightGBM poblacional) | Es la base de datos de la aplicación. |
| H18 | Predicciones etiquetadas con el régimen del HMM | Se usan como capa conmutable (estado estacionario/migración), no para una estratificación cuantitativa (que queda en el cap. 6). |

**No retomados en el capítulo 7** (por decisión del autor; el análisis del error vive en el cap. 6):

| # | Hilo | Estado |
|---|---|---|
| H5 | Lectura del error según la posición roost-a-roost | Pertenece a la interpretación del error del cap. 6. |
| H6 | Error por mes (cruce con fenología) | El desglose estacional del error queda en el cap. 6. |
| H13 | Predicciones LOBO de Markov como referencia del error | El baseline de Markov aparece como capa conmutable en la app, sin comparación cuantitativa. |
| H4 | Mapas intradía desde `fixes_clean.parquet` | No realizado; la herramienta trabaja con las posiciones diarias. |

**Arrastrados a conclusiones (cap. 9):**

| # | Hilo | Estado |
|---|---|---|
| H10 | Persistencia como baseline | En la app final se retiró la capa de persistencia (se muestra el baseline de Markov); la persistencia sigue siendo la referencia conceptual del cap. 6. |
| H20 | Techo estructural (horizonte de un día con variables locales) | Sigue pendiente para el cap. 9 (trabajo futuro: secuencia, viento, multi-paso). |

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
