# Esquema del capítulo 2 — Preliminares (Estado del Arte y Marco Teórico)

> Fuente de verdad del esquema del capítulo de Preliminares. Mantener
> sincronizado con `secciones/preliminares.tex` al podar o reordenar.
> Capítulo **de fundamentos**: estado del arte + marco teórico. NO desarrolla el
> trabajo propio (eso vive en los capítulos 3–7); aquí se da el contexto en la
> literatura y la teoría que esos capítulos **usan en forma aplicada y
> referencian** con `\ref{ch:preliminares}`.
>
> **RESTRICCIÓN DE PROPÓSITO Y LONGITUD (autor, 2026-05-26):** capítulo
> **funcional y breve**. Su razón de ser es que los capítulos de desarrollo
> puedan **referenciarlo** y no reexplicar la teoría; NO es un tratado. Regla
> operativa: **solo entra teoría que algún capítulo de desarrollo vaya a
> referenciar**. Cada sección, lo justo (pocos párrafos + la ecuación o el
> esquema imprescindible). Si un concepto no se va a citar desde O1–O5, se omite
> o se menciona de pasada. Sin relleno.

## Función del capítulo en la narrativa global

- Es el segundo capítulo, entre la Introducción (`ch:intro`) y el de datos
  (`ch:o1`). Hay que **insertarlo** en el maestro `tfg_etsiinf_JuanLlorens.tex`
  entre `\input{secciones/intro}` y `\input{secciones/desarrollo_o1}`, y crear
  `secciones/preliminares.tex` con `\label{ch:preliminares}`.
- **Cierra el hilo H14:** el capítulo de Markov ya remite aquí para "la teoría
  general de las cadenas de Markov finitas" (`desarrollo_o2.tex:42`). La intro
  delega aquí toda la teoría (Markov, HMM, ML); los capítulos de desarrollo la
  usan aplicada y la referencian, no la reexplican.
- **Carga fuerte de bibliografía** del TFG (instrucción de la skill): aquí van
  las citas del estado del arte y del marco teórico. Se forma `biblio.bib` a la
  par que se redacta cada sección, con metadatos verificados.
- No redefine lo ya fijado: el **dataset** (cap. 3), la **baseline de
  persistencia** y el **criterio de log-loss** (cap. 4) se definen donde se usan;
  aquí, como mucho, se nombran de pasada al situar los métodos.

## Decisiones de esquema (RESUELTAS con el autor, 2026-05-26)

- **D1 — Profundidad: rigor matemático selectivo.** Definiciones formales y las
  ecuaciones/algoritmos clave (propiedad de Markov + matriz de transición +
  estimación por máxima verosimilitud; supuestos del HMM + los tres problemas
  canónicos a nivel de esquema; pérdida cuantílica). **Sin demostraciones
  largas ni derivaciones** (no derivar Baum-Welch). El grado es Matemáticas e
  Informática, así que cabe rigor; pero solo la teoría que el trabajo usa.
- **D2 — Estructura: entrelazada por método** (estilo v2, que la tutora ya vio,
  pero con rigor y citas). Cada sección de método mezcla su **contexto en la
  literatura** (estado del arte) y su **teoría** (marco teórico). NO hay una
  sección de "estado del arte" separada de la teoría.
- **D3 — Métricas: log-loss se queda en el capítulo de Markov** (hilo H9). NO se
  define aquí entropía cruzada / log-loss. A lo sumo, mención de pasada a las
  reglas de puntuación propias al situar la evaluación, sin ecuación ni tabla.
- **D4 — Sin sección "Tecnologías utilizadas"** (la v2 la tenía, 2.5). El stack
  (Python, pandas, scikit-learn, hmmlearn, xgboost, lightgbm, folium…) va como
  **nota breve de ingeniería integrada en la metodología** del cap. de datos, no
  como sección de Preliminares (regla firme de la skill).

## Material de la versión 2 (`tfg_latex_etsiinf_JuanLlorens.pdf`, cap. 2)

- **Esqueleto adoptado** (la tutora lo vio): 2.1 movimiento animal / 2.2 Markov /
  2.3 HMM / 2.4 ML (RF + boosting). Se mantiene el orden, que es el de la escalera
  de modelos de la introducción.
- **Correcciones sobre la v2 (NO copiar):**
  - v2 sin ecuaciones ni citas → v3 con rigor selectivo (D1) y bibliografía real.
  - v2 dice "Argos/GPS" → aquí **solo GPS**.
  - v2 genérica ("un ave") → nombrar *Larus fuscus* y su caso al situar el campo.
  - v2 tenía 2.5 "Tecnologías utilizadas" → **se elimina** (D4).
  - v2 no menciona regresión cuantílica → **añadir** (la línea L3 la usa para la
    incertidumbre calibrada que alimenta la cartografía del cap. 7).
  - Terminología única en toda la memoria: estados del HMM = **estacionario /
    migración** (decisión I1), nunca "residente".

## Estado: capítulo REDACTADO y cerrado (2026-05-26)

Las 4 secciones están escritas en `secciones/preliminares.tex`, aprobadas sección a
sección y committeadas. Capítulo insertado en el maestro entre intro y datos.
Auditado con dos agentes (organización + verificador escéptico): ecuaciones y
afirmaciones teóricas correctas, coherentes con `src/tfg_aves/`, bibliografía
verificada (corregidas las páginas de `ke2017` a 3149--3157). H14 cerrado en
`hilos_abiertos.md`; sembrados P1 (teoría HMM → cap. 5) y P2 (teoría ML → caps. 6-7).

## Estructura propuesta (4 secciones, entrelazadas)

> Cada sección de método = contexto en la literatura (con citas) + teoría con
> rigor selectivo. Rótulos de sección con el nombre real del método (sin jerga
> O1/O2…).

### 2.1 El análisis y la predicción del movimiento animal  (`sec:prelim-movimiento`)
*Encuadre del campo (estado del arte general). No es un método; sitúa el problema.*
- La ecología del movimiento como marco (paradigma de Nathan et al. 2008): el
  movimiento como proceso a explicar; la revolución del seguimiento GPS y los
  repositorios tipo Movebank (engancha con la motivación de la intro, `kays2015`).
- Del **describir** trayectorias al **modelar y predecir** posiciones futuras:
  por qué es difícil (muestreo irregular, mezcla de regímenes, ruta individual).
- Panorama de familias de enfoques en la literatura, que motiva las tres del
  trabajo: (a) modelos espaciales discretos / cadenas de Markov sobre celdas;
  (b) modelos de estados latentes para segmentar comportamiento (HMM en ecología
  del movimiento); (c) aprendizaje automático para predicción tabular de
  trayectorias. Cierra anunciando que las tres se desarrollan en las secciones
  siguientes y se comparan (énfasis comparativo, decisión I3).
- **Citas candidatas:** `nathan2008`, `kays2015` (ya en .bib), posiblemente
  `demsar2015` (análisis/visualización de movimiento) — verificar antes.
- Sin figura propia (encuadre en prosa).

### 2.2 Cadenas de Markov  (`sec:prelim-markov`)
*Contexto + teoría general de cadenas finitas (cierra H14).*
- Contexto: las cadenas de Markov como herramienta probabilística interpretable
  para el movimiento entre zonas discretas del espacio.
- Teoría (rigor selectivo):
  - Proceso estocástico sobre espacio de estados finito; **propiedad de Markov**
    de primer orden (el futuro depende solo del presente). Definición formal.
  - **Matriz de transición** \(P=(p_{ij})\), estocástica por filas
    (\(\sum_j p_{ij}=1\)). Homogeneidad. (Esta es la teoría que el cap. de Markov
    usa en forma aplicada: `eq:o2-markov` allí; aquí, la general.)
  - Transiciones a \(n\) pasos / Chapman-Kolmogorov (breve, una línea). Mención
    de la distribución estacionaria (sin desarrollarla; no se usa a fondo).
  - **Estimación por máxima verosimilitud** de \(p_{ij}\) por frecuencias de
    transición observadas (lo que hace el cap. de Markov al contar pares
    consecutivos).
- **Citas candidatas:** un texto de cadenas de Markov (p. ej. `norris1997` o
  `ross_probability`) — verificar edición/año.
- Figura candidata F2 (opcional, baja prioridad): diagrama de transición de una
  cadena pequeña (TikZ). Probable que no aporte; decidir al redactar.

### 2.3 Modelos ocultos de Markov  (`sec:prelim-hmm`)
*Contexto en ecología del movimiento + teoría del HMM.*
- Contexto: el HMM como extensión natural de la cadena de Markov para inferir un
  **estado de comportamiento no observado** (estacionario / migración) a partir
  de variables del movimiento; su uso consolidado en ecología del movimiento
  para segmentar trayectorias (moveHMM y trabajos asociados).
- Teoría (rigor selectivo):
  - Definición: cadena de Markov **oculta** \(\{S_t\}\) + proceso de
    **observación/emisión** \(\{O_t\}\). Parámetros \((\pi, A, B)\); supuestos de
    independencia condicional.
  - **Emisiones gaussianas** (covarianza diagonal): la forma que usa el cap. de
    HMM. Una ecuación de la densidad de emisión.
  - **Los tres problemas canónicos** (Rabiner 1989) a nivel de esquema, sin
    derivar: evaluación (**algoritmo forward**), decodificación (**Viterbi**,
    MAP global; y **filtrado forward-only**, posterior causal en línea),
    aprendizaje (**Baum-Welch / EM**, solo qué hace).
  - **Filtrado vs suavizado** (una frase): distinción que el cap. de HMM explota
    para su decodificación causal forward-only (engancha con I-causalidad).
- **Citas candidatas:** `rabiner1989` (tutorial canónico), `langrock2012` y/o
  `mcclintock2018` (HMM en ecología del movimiento, moveHMM), `zucchini2016`
  (libro HMM) — verificar.
- **Figura candidata F1 (recomendada):** modelo gráfico del HMM (cadena oculta
  \(S_{t-1}\to S_t\to S_{t+1}\) con emisiones \(O_t\)). TikZ standalone. Es la
  figura más defendible y clásica del capítulo.

### 2.4 Aprendizaje automático supervisado  (`sec:prelim-ml`)
*Contexto + teoría de los modelos que usa el cap. de ML.*
- Contexto: el aprendizaje supervisado como vía complementaria que capta
  relaciones no lineales y combina muchas variables, frente a la transparencia de
  los modelos probabilísticos (tensión interpretabilidad ↔ capacidad, ya anunciada
  en la intro). Métodos basados en árboles por su robustez en problemas tabulares.
- Teoría (rigor selectivo):
  - Marco supervisado: predictores → objetivo; clasificación vs regresión;
    sobreajuste y generalización (breve, lo justo para enmarcar).
  - **Árboles de decisión** como bloque base (idea de partición recursiva).
  - 2.4.x **Random Forest** (bagging + subespacios aleatorios de variables;
    Breiman 2001). Reduce varianza; importancia de variables.
  - 2.4.x **Gradient boosting** (XGBoost, LightGBM): modelo aditivo de árboles que
    corrigen el residuo; regularización; crecimiento por niveles vs por hojas e
    histogramas (mención, no detalle de implementación).
  - 2.4.x **Regresión cuantílica** (pérdida pinball / cuantílica; quantile
    regression forests, Meinshausen 2006): predice cuantiles y por tanto una
    banda de incertidumbre, no solo la media. Sustenta la salida calibrada de la
    línea L3 que alimenta la cartografía del cap. 7.
- **Citas candidatas:** `breiman2001` (RF), `friedman2001` (GBM), `chen2016`
  (XGBoost), `ke2017` (LightGBM), `koenker1978` (regresión cuantílica),
  `meinshausen2006` (QRF), `hastie2009` (ESL, opcional) — verificar todas.
- Sin figura propia salvo que un esquema de ensemble aporte (probable que no;
  decidir al redactar). Filtro de relevancia.

## Lo que este capítulo NO incluye (para no duplicar)

- Definición de **log-loss** y de la **baseline de persistencia** (cap. 4, H9/H10).
- Descripción del **dataset** y de la especie en cifras (cap. 3).
- Detalles de **implementación / stack** (nota de ingeniería en cap. 3, D4).
- Decisiones propias del trabajo (nº de estados del HMM, tamaño de celda,
  hiperparámetros): se justifican en cada capítulo de desarrollo, no aquí.

## Coherencia / hilos

- **Cierra H14** (teoría de Markov finita). Engancha hacia atrás con la intro
  (retoma la tensión interpretabilidad↔capacidad y la escalera de modelos) y hacia
  delante con los caps. 4 (Markov), 5 (HMM), 6 (ML), que referenciarán secciones
  de aquí en vez de reexplicar la teoría.
- Terminología única: *Larus fuscus*/gaviota sombría, celda, posición diaria,
  cadena de Markov visible, modelo oculto de Markov (HMM), estado estacionario /
  migración (I1), aprendizaje supervisado (Random Forest, XGBoost, LightGBM),
  regresión cuantílica. Notación de la matriz de transición \(P=(p_{ij})\)
  consistente con `eq:o2-markov` del cap. de Markov.
- Tras redactar, **actualizar `hilos_abiertos.md`**: marcar H14 cerrado y anotar
  que los caps. 4–6 deben referenciar las secciones de teoría de aquí.

## Bibliografía a formar (incremental, al redactar cada sección)

Verificar autores/título/año/DOI por búsqueda web antes de dar por buena cada
entrada (o marcar `% verificar`). Estilo IEEE (biblatex/biber en el build final).
Candidatas por sección listadas arriba.

## Verificación antes de cerrar cada sección (recordatorio de la skill)

- `grep -o "—" preliminares.tex | wc -l` == 0.
- `grep -nE "\bO[1-5]\b" preliminares.tex` == 0.
- Compila en el preview (driver sin biblatex) sin `!`; `??` a caps. aún sin
  redactar son esperadas durante la escritura.
- Cada afirmación citable lleva su `\cite` con entrada ya en `biblio.bib`.
- Esquema sincronizado con el `.tex`.
- El autor aprueba cada sección antes de pasar a la siguiente; commit por sección.
