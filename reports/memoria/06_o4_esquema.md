# Capítulo 6 — Predicción supervisada del movimiento diario

> **Estado:** **ENTREGADO** (2026-05-31). Aprobado sección a sección; commits
> 9e86c81 (preámbulo + §6.1), 00b2435 (§6.2), 7cc7865 (§6.3), ff6a2ef (§6.4).
> Hilos cerrados/sembrados sincronizados en `reports/memoria/hilos_abiertos.md`.
> El capítulo no pasó por la auditoría dos agentes (decisión del autor).
> **Fuente única de los datos:** `reports/memoria/06_o4_ml.md` (notas vivas) +
> artefactos en `reports/figures/` y `reports/tables/` (slugs `o4_*`). Toda
> cifra se recalculó contra `data/processed/o4/` antes de escribir.

## Por qué este esquema y no otro

El capítulo del HMM tenía una sola pregunta y se resolvió en tres secciones.
Este capítulo plantea **dos formulaciones** del mismo problema de predicción
del movimiento diario: una **categórica sobre la celda** (la natural por
simetría con la cadena de Markov) y otra **continua sobre el desplazamiento
con incertidumbre**. La narrativa va de la primera a la segunda porque la
segunda corrige una limitación geométrica de la primera y, sobre todo,
porque sólo la segunda entrega una predicción cartografiable con banda de
incertidumbre, que es lo que necesitan los mapas del capítulo siguiente. El
modelo de dos etapas que se exploró en el trabajo se ha **excluido** de la
memoria por decisión del autor: no es relevante para la narrativa central.

La preparación común (datos, variables, métricas, baselines) se aísla en una
sección previa que no se vuelve a tocar, y un cierre final compara las dos
formulaciones contra los mismos baselines y prepara los mapas del capítulo
siguiente.

**Convención firme de naming.** Las etiquetas internas "L1/L2/L3" o
"línea 1 / línea 2 / línea 3" **no aparecen nunca** en la prosa, en los
títulos de sección ni en los pies. Las dos formulaciones se nombran siempre
descriptivamente: "**clasificación categórica de la celda**" / "**predicción
categórica**" la primera, y "**regresión continua del desplazamiento con
cuantiles**" / "**predicción continua con incertidumbre**" la segunda. Los
*labels* internos del `.tex` pueden ser técnicos (p. ej. `sec:o4-categorica`,
`sec:o4-cuantiles`) sin propagar la jerga.

**Coherencia que se exige a este capítulo (hilos abiertos):**

| Hilo | Origen | Qué hace este capítulo |
|---|---|---|
| H7 — heterogeneidad inter-individual | Cap. \ref{ch:o1} (sembrado), HMM (arrastrado) | **Cierra**: una mención breve por formulación, sobre el ave de más histórico, basta para mostrar que el modelo poblacional ya la cubre → poblacional canónico. |
| H9 — log-loss como criterio principal | Cap. \ref{ch:o2} | **Consume** como métrica probabilística complementaria de las métricas de precisión. La elección entre familias pondera todas las métricas en conjunto (top-1, top-3, distancia mediana, *log-loss* y, en la formulación continua, cobertura y *pinball*), no decide una sola por sí sola. |
| H10 — persistencia como baseline a batir | Cap. \ref{ch:o2} | **Consume**: la referencia común; sigue invicta en agregado, pero el supervisado la mejora donde importa cartográficamente (distancia de Markov(1) reducida en uno o dos órdenes de magnitud). |
| H12 — rutas individuales no vistas | Cap. \ref{ch:o2} | **Cierra junto con H7**: el supervisado captura la heterogeneidad poblacionalmente sin requerir un modelo por ave. |
| H15 — `features.parquet` + columna `split` (72/8/20) | HMM | **Consume**: entrada directa; no se recalcula el HMM ni el *split*. |
| H16 — modelo B canónico del HMM | HMM | **Consume**: `state_b_causal` y `posterior_b_migracion_causal` por defecto. |
| P2 — teoría ML | Cap. \ref{ch:preliminares} | **Referencia**: bosques aleatorios, *boosting* (XGBoost/LightGBM) y regresión cuantílica (pérdida *pinball* y QRF) se citan, no se reexplican. |

**Hilos nuevos que este capítulo siembra (para el capítulo 7):**

- **H17** — banda de incertidumbre `[p10, p90]` de la formulación continua (cobertura ≈ 80 %): insumo natural de la capa de incertidumbre de los mapas.
- **H18** — predicciones del modelo recomendado guardadas con la etiqueta del régimen HMM, para que el capítulo 7 estratifique el error por régimen y por mes sin recalcular nada.
- **H19** — el modelo recomendado para los mapas es el de la formulación continua (concretamente la familia con mejor calibrado, anticipada en §6.3.5).

## Estructura del capítulo

### Preámbulo (sin sección)

- Engancha con el cierre del HMM: la cadena visible promediaba dos regímenes; el HMM los separó pero no predice posiciones. Este capítulo aborda la predicción supervisada del movimiento diario.
- Pregunta concreta: dadas las posiciones diarias y el régimen del día, ¿puede un modelo supervisado predecir adónde irá el ave mañana mejor que la persistencia trivial y la cadena de Markov?
- Estrategia: **dos formulaciones** sobre el mismo problema y los mismos baselines, la natural (categórica sobre la celda, por simetría con Markov) y una reformulación (continua sobre el desplazamiento con cuantiles), que corrige una limitación geométrica de la primera y entrega además una banda de incertidumbre interpretable. Adelanto del recorrido en un párrafo.
- Producto: predicciones por ave y día, con incertidumbre, que alimentan los mapas del capítulo \ref{ch:o5}.

### §6.1 Datos, variables y diseño experimental

**Propósito:** dejar fijado todo lo que es común a las dos formulaciones. No se vuelve a explicar.

- **§6.1.1 Entrada y partición.** `features.parquet` del capítulo \ref{ch:o3} (con `state_b_causal`, `posterior_b_migracion_causal` y la columna `split`) más las posiciones diarias del capítulo \ref{ch:o1}. *Split* temporal por ave 72/8/20 reutilizado del HMM, sin recalcular. Por qué no LOBO (heredado del cierre de Markov: 34 % de orígenes no vistos).
- **§6.1.2 Conjunto de variables predictivas.** Diez variables que respetan la causalidad temporal (no observan `t+1`): posición actual (`lat`, `lon`), ciclo anual (`sin_doy`, `cos_doy`), cinemática **entrante** del tramo `t-1 → t` (`step_in_km`, `sin_bearing_in`, `cos_bearing_in`, `cos_turning_in`) y dos derivadas del HMM (`state_b_causal`, `posterior_b_migracion_causal`). Justificar la causalidad como criterio firme: una variable predictiva no puede observar el día que se predice. Justificar la cinemática entrante (la inercia real disponible al final del día `t`).
- **§6.1.3 Familias y configuración.** Tres familias supervisadas (capítulo \ref{ch:preliminares}, §\ref{sec:prelim-rf}–§\ref{sec:prelim-cuantil}): bosque aleatorio, XGBoost y LightGBM. **Configuración fija y conservadora por familia**, sin rejilla (justificación metodológica: una rejilla optimizada sobre la validación interna introduce sesgo de selección sobre esa misma validación; el control del sobreajuste se hace con hiperparámetros conservadores + parada temprana + diagnóstico *train-test*). Cita a `o4_tab01_hyperparameter-config`.
- **§6.1.4 Métricas y baselines.** Top-1, top-3 y distancia mediana en kilómetros (heredadas de Markov). **Log-loss** heredado del cierre de Markov como métrica probabilística que complementa a las de precisión; la elección entre familias dentro de cada formulación pondera todas las métricas en conjunto (no la decide el *log-loss* en solitario). Cuatro cortes que se aplican a las dos formulaciones: *global*, *estacionario* (`state_b_causal=0`), *migración* (`state_b_causal=1`) y *días de movimiento real* (`y_move=1`). Baselines: **persistencia** y **Markov(1) reentrenado sobre el mismo *split***, para comparar en igualdad.

**Figuras/tablas en §6.1:** ninguna propia. Las cifras se citan en texto y se refieren al `o4_tab01_hyperparameter-config` como evidencia de la configuración.

### §6.2 Predicción categórica de la celda

**Propósito:** caracterizar el techo del problema con la formulación más directa (la celda 0,5° como clase, mismo *target* que Markov).

- **§6.2.1 Planteamiento.** Clasificación multiclase con la celda del día siguiente como objetivo (~849 clases activas en *train*). Misma resolución que Markov por simetría: las métricas son directamente comparables. Modo poblacional (sin `bird_id`), justificado al pie con una mención de pasada a las pruebas de alcance entrenadas con un solo ave (que se cierran en el cierre del capítulo).
- **§6.2.2 Resultados globales y descarte de LightGBM.** Comparativa de las ocho combinaciones (seis modelos ML + dos baselines) con `o4_fig04_models-comparison`. Hallazgo metodológico transparente: **LightGBM diverge desde la primera iteración** sobre este *target* multiclase de alta cardinalidad, lo que se interpreta como incompatibilidad entre la configuración conservadora y un *target* con muchas clases activas y distribución desbalanceada. Se descarta como candidato de esta formulación, con la evidencia abierta en `o4_fig03_learning-curves`. Anticipo de una línea (para evitar contradicción aparente con §6.3.5): el descarte es específico de la formulación categórica; en la regresión de cuantiles la familia se recupera.
- **§6.2.3 Lectura del techo.** Frente a Markov, **los modelos supervisados ganan en todo**; frente a la persistencia, **pierden en todas las métricas globales** y el log-loss del mejor ML (~5,5 sobre 849 clases) es informativo en absoluto (mejor que el uniforme 6,7) pero no traduce a ventaja en top-1. La persistencia explota el 87 % de días estacionarios.
- **§6.2.4 Dónde falla el modelo (estratificación por régimen del HMM).** Desglose por estado con `o4_fig07_error-by-state-poblacional`: el top-1 cae de ~0,64 en estacionario a ~0,14 en migración. **La persistencia gana también dentro del régimen de migración del HMM**, porque muchos días así etiquetados el ave aún no cambia de celda. El valor genuino del supervisado está en los **días de movimiento real** (`y_move=1`), donde la persistencia vale 0 por construcción.
- **§6.2.5 Importancia relativa de las variables.** `o4_fig09_feature-importance-winners`: la posición domina (lat + lon ≈ 0,78), el `posterior_b_migracion_causal` del HMM es la tercera variable más influyente (≈ 0,06 en poblacional) y el estado duro `state_b_causal` apenas se usa. Es el primer cierre cuantitativo del puente HMM → predicción supervisada: el HMM **aporta señal real, modesta y a través del posterior blando, no de la etiqueta dura**.
- **§6.2.6 Diagnóstico de las causas del techo y cierre del alcance.** Breve, sin replicar la taxonomía tripartita de las notas. **Dos causas estructurales** que motivan la reformulación de §6.3 y se retoman en el cierre: (i) **el *target* categórico desperdicia la geometría del espacio** (un error a celda vecina pesa igual que un error al otro extremo del dominio); (ii) **el horizonte a un día con variables locales** no contiene los predictores causales del fenómeno migratorio (viento, historia multi-día, destino), límite común a las dos formulaciones que se argumenta en §6.4. **Una sola frase final** sobre el alcance del entrenamiento: el modelo entrenado únicamente con el ave de más histórico (91916A) no supera al poblacional evaluado sobre sus mismas filas (log-loss 2,72 vs 4,33; empate en top-1), primera evidencia para el cierre de H7/H12 que la formulación continua confirma. Sin figura ni tabla propia.

**Figuras/tablas que se muestran en §6.2:** `o4_fig04_models-comparison`, `o4_fig03_learning-curves`, `o4_fig07_error-by-state-poblacional`, `o4_fig09_feature-importance-winners`. (Se descartan `o4_fig02_train-test-gap`, `o4_tab05_winner-recommendation`, `o4_fig06_error-by-state-individual` y `o4_fig08_individual-vs-poblacional` por no cambiar ninguna decisión que el texto no exprese ya. La cifra de la prueba con un solo ave se cita de pasada en §6.2.6.)

### §6.3 Predicción continua del desplazamiento con cuantiles

**Propósito:** corregir la limitación geométrica de la formulación categórica y entregar una predicción con incertidumbre interpretable. Cierra además, con la evidencia más limpia, la pregunta global vs per-individuo.

- **§6.3.1 Motivación y arquitectura.** El *target* deja de ser la celda y pasa a ser el desplazamiento diario `(Δlat, Δlon)` en grados (frente a (paso, rumbo) para evitar la discontinuidad angular). Tres cuantiles `{p10, p50, p90}` por eje, ajustados con la pérdida *pinball* (capítulo \ref{ch:preliminares}, §\ref{sec:prelim-cuantil}, ecuación \ref{eq:prelim-pinball}); `p50` es la predicción puntual y `[p10, p90]` la banda de incertidumbre nominal al 80 %. La monotonía `p10 ≤ p50 ≤ p90` se fuerza por ordenación a posteriori. Para comparar con la formulación categórica, el `p50` se discretiza al *grid* 0,5° (top-1 *mapeado*) y la distancia se reporta nativa (punto predicho → verdad) y vía centroide.
- **§6.3.2 Resultado de la reformulación.** `o4_fig26_distance-distribution`. Sólo por reformular el *target*, el top-1 mapeado sube de **0,58 a 0,76** (frente al XGBoost categórico hermano) y la distancia mediana baja a 20,9 km, **igualando a la persistencia**. La explicación es geométrica: con el ≈ 88 % de días estacionarios el regresor aprende a predecir desplazamiento ≈ 0, su `p50` recae en la celda de origen y reproduce el acierto de los *self-loops* que el clasificador no fijaba con su argmax. **La discretización del espacio costaba ~18 puntos de top-1** que la regresión recupera sin esfuerzo.
- **§6.3.3 Banda de incertidumbre calibrada.** `o4_fig25_calibration-coverage`. La cobertura empírica de `[p10, p90]` es 0,85 (latitud) y 0,84 (longitud), muy próxima al 80 % nominal. **Es la primera vez que el trabajo entrega una predicción con incertidumbre interpretable**; siembra el hilo **H17** para los mapas del capítulo \ref{ch:o5}.
- **§6.3.4 Convergencia entre familias y revisión del descarte de LightGBM.** `o4_tab31_comparativa-familias-l3` + `o4_fig29_quantile-crossing`. Las tres familias sobre la **misma** tarea de regresión: top-1 0,762–0,771 (todas algo por debajo de la persistencia 0,778), distancia ≈ 20,9 km, *pinball* indistinguible. **La elección de familia es secundaria en esta tarea.** Diferencias cualitativas: LightGBM es el más calibrado (0,84 / 0,79), el bosque aleatorio (vía bosque de cuantiles, \cite{meinshausen2006}) sobrecubre (0,86 / 0,86) y no cruza cuantiles por construcción (cero cruces), frente a 54 de XGBoost y 202 de LightGBM corregibles a posteriori. **LightGBM no diverge aquí**, lo que retrospectivamente confirma que su patología en §6.2 era específica del régimen multiclase de muy alta cardinalidad, no intrínseca a la familia.
- **§6.3.5 Lectura honesta y modelo recomendado para los mapas.** Esta formulación **iguala a la persistencia en agregado, no la supera**, y como la categórica falla en los días de migración (top-1 0,21, distancia 60 km). Su valor genuino frente a la persistencia es la **banda de incertidumbre calibrada** que entrega por construcción. Para el capítulo \ref{ch:o5}, el modelo recomendado es esta formulación con la familia LightGBM (mejor calibrado), con XGBoost como alternativa equivalente: su salida geométrica (`p50` + banda) es la única que se puede llevar al mapa directamente como punto + región. **Una sola frase** sobre el alcance del entrenamiento que confirma el cierre de H7/H12: el regresor entrenado únicamente con 91916A empata al poblacional evaluado sobre sus mismas filas (top-1 0,685 vs 0,687; distancia mediana 14,6 vs 14,4 km), reforzando lo visto en §6.2.6 sin figura propia.

**Figuras/tablas que se muestran en §6.3:** `o4_fig26_distance-distribution`, `o4_fig25_calibration-coverage`, `o4_tab31_comparativa-familias-l3`, `o4_fig29_quantile-crossing`. (Se descartan `o4_fig24_target-dist-bird-history`, `o4_tab27_comparativa-l3` y `o4_fig28_vectores-desplazamiento-91916a` por no añadir información que el texto no exprese ya o por solaparse con la tabla maestra del cierre.)

### §6.4 Síntesis y entrega al capítulo siguiente

**Propósito:** consolidar el techo común y dejar definido lo que recibe el capítulo \ref{ch:o5}.

- **§6.4.1 Comparativa final.** Tabla maestra `o4_tab30_comparativa-maestra-regimen` filtrada a las filas relevantes (persistencia, Markov(1), predicción categórica y predicción continua), evaluadas en los cuatro cortes (*global*, *estacionario*, *migración*, *días de movimiento*) con las mismas métricas. **Lectura única:** las dos formulaciones se igualan a la persistencia en agregado (la continua roza el empate, la categórica queda dos décimas por debajo), ambas chocan en migración y en días de movimiento, y ambas mejoran a Markov(1) en distancia por uno o dos órdenes de magnitud.
- **§6.4.2 Techo estructural común.** Las dos formulaciones convergen al mismo límite: **el horizonte a un día con variables locales y el muestreo discreto a 08:00 UTC no contienen el predictor causal del fenómeno migratorio.** No es un problema de algoritmo (lo confirma la convergencia entre familias de §6.3.5), ni del *target* (la regresión lo recupera): es del **planteamiento del problema a esta escala**.
- **§6.4.3 Decisiones cerradas en el capítulo.** Modelo canónico **poblacional** (cierra **H7** y **H12** con las dos menciones de §6.2.6 y §6.3.5: en ambas formulaciones, el modelo entrenado sólo con el ave de más histórico no supera al poblacional). Criterio **log-loss** consumido (**H9**). Persistencia sigue invicta en agregado (**H10**), pero la mejora estructural sobre Markov en distancia (orden de magnitud) y la banda de incertidumbre calibrada justifican la cadena Markov → HMM → predicción supervisada como infraestructura para los mapas.
- **§6.4.4 Lo que recibe el capítulo siguiente.** Las predicciones de la formulación continua (modelo recomendado, familia LightGBM por mejor calibrado, con XGBoost como alternativa equivalente), con su `p50` y su banda `[p10, p90]` (**H17**), y la etiqueta de régimen del HMM por fila (**H18**), permiten al capítulo \ref{ch:o5} cartografiar el error, estratificarlo por régimen y por mes, y representar la incertidumbre como una región sobre el mapa (**H19**). Reconocer brevemente las direcciones de trabajo futuro que el techo común abrió: secuencias multi-día, viento ECMWF, distancia a centroides de invernada, todas fuera del alcance temporal del trabajo.

**Figuras/tablas en §6.4:** `o4_tab30_comparativa-maestra-regimen` (filtrada para excluir la fila del modelo de dos etapas) más la actualización del registro de hilos abiertos.

## Inventario final de figuras y tablas mostradas

| Sección | ID artefacto | Tipo | Función |
|---|---|---|---|
| §6.2.2 | `o4_fig04_models-comparison` | fig | Comparativa global seis modelos ML + dos baselines. |
| §6.2.2 | `o4_fig03_learning-curves` | fig | Divergencia de LightGBM en multiclase (descarte transparente). |
| §6.2.4 | `o4_fig07_error-by-state-poblacional` | fig | Caída en migración (formulación categórica). |
| §6.2.5 | `o4_fig09_feature-importance-winners` | fig | Posición domina, posterior HMM aporta modesto. |
| §6.3.2 | `o4_fig26_distance-distribution` | fig | La regresión recupera 18 pp de top-1. |
| §6.3.3 | `o4_fig25_calibration-coverage` | fig | Cobertura empírica ≈ 80 % de la banda. |
| §6.3.4 | `o4_tab31_comparativa-familias-l3` | tabla | Convergencia entre familias. |
| §6.3.4 | `o4_fig29_quantile-crossing` | fig | Coherencia de cuantiles (bosque de cuantiles, cero cruces). |
| §6.4.1 | `o4_tab30_comparativa-maestra-regimen` | tabla | Comparativa final por régimen (filtrada). |

Total: **7 figuras + 2 tablas** mostradas. Las cifras puntuales del modelo entrenado con un solo ave se citan en texto sin figura propia, como mención de refuerzo al modelo poblacional. El resto se cita de pasada o se omite por no cambiar ninguna decisión.

## Riesgos a vigilar al redactar

- **Naming firme: cero "L1/L2/L3" y cero "línea 1 / línea 2 / línea 3" en prosa, títulos y captions.** Las dos formulaciones se nombran descriptivamente siempre. Verificación: `grep -nE "\bL[1-3]\b|\blínea [1-3]\b" desarrollo_o4.tex` debe dar 0.
- **No mencionar el modelo de dos etapas** en ninguna parte del capítulo (decisión del autor: no relevante para la narrativa).
- **No replicar la taxonomía tripartita (métrica / datos / modelado) de las notas:** §6.2.6 compacta las causas a dos.
- **No reabrir la "fuga de información" del primer O4:** la causalidad se presenta como criterio de diseño desde el inicio de §6.1.2.
- **Cero rayas (—)** y nada de "O1/O2/O3/O4/O5" en prosa; `\ref{ch:o1}` etc.
- **Tabla maestra `o4_tab30`:** sus `n` por régimen (3566 / 471 / 908) son ligeramente distintos de los de las tablas por línea (3509 / 451 / 880) por contar todas las filas test del *split* sin el filtro de la versión categórica. Recalcular antes de citar y nota al pie si es necesario; **no escribir números desde memoria**. Filtrar la fila del modelo de dos etapas al renderizar.
- **Recordatorio I1 (introducción):** los regímenes se llaman "estacionario" y "migración" en este capítulo también; nunca "residente".

## Cita de bibliografía nueva (a añadir a `biblio.bib` si no está)

Ya en `biblio.bib`: `breiman2001`, `friedman2001`, `chen2016`, `ke2017`, `koenker1978`, `meinshausen2006`, `wikelski2015`. Comprobar al redactar; si falta alguna entrada verificar metadatos antes de añadirla.
