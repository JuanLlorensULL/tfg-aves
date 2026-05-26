# Esquema del capítulo 1 — Introducción

> Fuente de verdad del esquema del capítulo de introducción. Mantener
> sincronizado con `secciones/intro.tex` al podar o reordenar.
> Capítulo **marco** (contexto + objetivos), NO de desarrollo: no calca la
> estructura de los capítulos de datos/Markov. No entra en teoría (Markov, HMM,
> ML): eso vive en el capítulo de Preliminares (`ch:preliminares`).
>
> **RESTRICCIÓN DE LONGITUD (autor):** introducción breve, clara y al grano.
> Objetivo ~2-3 páginas. 1.1: 2-3 párrafos; 1.2: 3-4 párrafos + diagrama; 1.3:
> general + 5 específicos en lista compacta. Sin relleno, sin cifras de dataset,
> sin teoría, sin recorrido de capítulos.

## Función del capítulo en la narrativa global

- Es la puerta de entrada: situar al tribunal en el problema, justificar por qué
  importa y enunciar los objetivos. El mapa del documento lo da el índice y la
  escalera de modelos de 1.2 (no hay sección de "estructura" propia).
- **Siembra** el hilo conductor de toda la memoria: el *enfoque híbrido* como una
  escalera de modelos en la que cada uno ataca una limitación concreta del
  anterior (Markov visible → HMM → ML supervisado), más la cartografía del error.
  Ese hilo es el que cosen los capítulos 4–7.
- No redefine nada que luego se detalla: el dataset, la baseline de persistencia,
  el criterio de log-loss, la teoría de cada modelo se **anuncian** aquí y se
  desarrollan en su capítulo (se referencian con `\ref`, no se explican).

## Decisiones de esquema (RESUELTAS)

- **D1 — Contribuciones:** intro **compacta de 4 secciones**, SIN sección de
  contribuciones propia. Confirmado por precedente de la v2 (armazón
  1.1–1.4 estándar ETSIINF que la tutora ya vio) y por el filtro de relevancia.
  El hallazgo transversal se anuncia en 1.2; las contribuciones se detallan en
  Conclusiones.
- **D2 — Figura conceptual (F1):** **SÍ**, diagrama TikZ de la escalera de
  modelos en 1.2 (aprobado por el autor). Es el mapa del documento y la mejora
  visual más clara sobre la v2.
- **D3 — Mapa del corredor:** no repetir `o1_fig08` aquí; describir el corredor
  en prosa y remitir al cap. de datos.

### Material transferido de la versión 2 (`tfg_latex_etsiinf_JuanLlorens.pdf`)

- Armazón de 4 secciones (Motivación / Definición del problema / Objetivos /
  Estructura): adoptado.
- Objetivo general y los 5 específicos: reutilizados y enriquecidos.
- Recurso de enumerar las dificultades del problema en 1.2: adoptado,
  concretado a nuestro caso (muestreo irregular, errores, movimiento que mezcla
  régimen sedentario y de migración, ruta individual).
- **Correcciones sobre la v2 (NO copiar):** la v2 dice "Argos/GPS" → aquí
  **solo GPS** (Movebank); la v2 es genérica → aquí se nombra *Larus fuscus* y
  el corredor; la v2 no tiene escalera de modelos ni marco honesto → se añaden.

## Estructura final (4 secciones, alineada con la v2)

### 1.1 Motivación  (`sec:intro-motivacion`)
- Ecología del movimiento: el seguimiento GPS (Movebank y similares) ha generado
  grandes volúmenes de trayectorias animales; el reto pasa de *recoger* datos a
  *modelar y anticipar* el movimiento.
- Por qué predecir el desplazamiento de aves importa (sin inflar, 2-3 usos
  concretos): conservación y diseño de áreas protegidas a lo largo de corredores
  migratorios, seguridad aérea (colisiones), y comprensión de la navegación y la
  respuesta al cambio climático.
- La especie y el caso de estudio: *Larus fuscus* (gaviota sombría), migrante de
  larga distancia entre las zonas de cría del norte de Europa y la invernada en el
  este de África; fenología marcada (cría jun-jul, paso abr-may y sep-oct,
  invernada dic-feb). Caso idóneo: una sola especie (homogeneidad) con un patrón
  estacional fuerte y datos públicos.
- Cifras de encuadre **mínimas** (sin tabla; se detallan en cap. de datos):
  origen Movebank, una sola especie. (Volumen exacto se da en el cap. de datos.)

### 1.2 Planteamiento del problema y enfoque híbrido  (`sec:intro-enfoque`)
- La pregunta concreta del trabajo: dado dónde está hoy un individuo, ¿dónde
  estará mañana? (predicción a un día sobre la posición diaria).
- Por qué es difícil y por qué un único modelo no basta: tensión entre
  interpretabilidad (probabilístico) y capacidad predictiva (ML); el movimiento
  mezcla días sedentarios y de migración; cada ave sigue su ruta.
- **El enfoque híbrido como escalera de modelos** (espina dorsal de la memoria):
  1. Cadenas de Markov visibles: baseline probabilística interpretable; fija el
     criterio de comparación y revela dos límites (no separa régimen de
     comportamiento ni ruta individual).
  2. Modelos ocultos de Markov (HMM): introducen un estado latente de
     comportamiento (residente vs migración) que Markov no veía.
  3. Aprendizaje supervisado (RF, XGBoost, LightGBM): incorpora el comportamiento
     del HMM (de ahí "híbrido": probabilístico + ML) y variables por individuo.
  4. Cartografía interactiva y análisis espacial del error: cierra el ciclo y lee
     el error contra la fenología.
- Anuncio honesto del hallazgo transversal (tono investigador): la persistencia
  ("mañana = hoy") es un rival sorprendentemente fuerte; el valor del trabajo está
  en la calibración probabilística, en aislar el régimen migratorio y en
  caracterizar un techo estructural de la predicción a un día. (Sin cifras aquí;
  se entregan en los capítulos.)
- **Figura candidata F1 (conceptual):** diagrama de la escalera de modelos (las
  cuatro fases y la limitación que cada una hereda y resuelve). TikZ standalone.
  *Opcional, a confirmar con el autor.*

### 1.3 Objetivos del trabajo  (`sec:intro-objetivos`)
- **Objetivo general:** diseñar y evaluar un enfoque híbrido (modelos
  probabilísticos + aprendizaje supervisado) para predecir el desplazamiento
  diario de *Larus fuscus* a partir de datos GPS, con una métrica probabilística
  común y baselines explícitas.
- **Objetivos específicos** (5, descritos por contenido; NADA de jerga "O1/O2"):
  1. Construir, a partir de los registros GPS en bruto, una secuencia diaria
     homogénea por individuo, con los huecos representados de forma explícita.
  2. Establecer una baseline predictiva interpretable y cuantificada con cadenas
     de Markov visibles, y fijar el criterio de evaluación (log-loss) y la
     referencia de persistencia.
  3. Detectar el régimen de comportamiento (estacionario vs migración) mediante
     modelos ocultos de Markov, validando su coherencia biológica.
  4. Predecir la posición del día siguiente con modelos supervisados que integran
     el comportamiento detectado y variables por individuo, sin fuga de
     información.
  5. Cartografiar las predicciones de forma interactiva y analizar la estructura
     espacial y estacional del error.
- Cada objetivo específico remite (con `\ref`) a su capítulo (3–7).

> **Sin sección "Estructura de la memoria"** (decisión del autor): el índice ya
> cumple esa función de mapa, y la lógica narrativa entre capítulos la da la
> escalera de modelos de 1.2 (texto + diagrama F1). No se gancha el anexo de IA
> desde la intro; el anexo se sostiene solo.

## Coherencia / hilos

- Engancha hacia delante con TODOS los capítulos; ninguno hacia atrás (es el
  primero). Debe sembrar el hilo "enfoque híbrido = escalera de modelos" que
  `hilos_abiertos.md` ya ve materializado en H11 (régimen no modelado → HMM) y
  H12 (ruta individual → ML).
- Terminología fijada que la intro debe respetar: *fix*, posición diaria, celda,
  self-loop, persistencia, log-loss, régimen estacionario/migración, *Larus
  fuscus*/gaviota sombría. Nombres de modelos: cadenas de Markov visibles,
  modelos ocultos de Markov (HMM), aprendizaje supervisado (RF, XGBoost,
  LightGBM).

## Verificación de datos pendiente (antes de prosa de cada sección)

- Cifras de encuadre si se citan (origen, especie única): ya verificadas en el
  cap. de datos. Volumen exacto NO se repite aquí.
- Fenología de *Larus fuscus* (cría/paso/invernada): coherente con lo usado en
  los capítulos de Markov y HMM.
