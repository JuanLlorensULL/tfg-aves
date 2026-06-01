# Esquema del capítulo de Conclusiones y trabajo futuro (cap. 9)

> Fuente de verdad del esquema. Último capítulo de contenido (antes del anexo de IA).
> Redactado tras cerrar caps. 1–8 y el resumen. Engancha hacia atrás con todos los
> capítulos de desarrollo; no hay capítulo posterior que enganchar.

## Principio rector (evitar redundancia)
- El **impacto (cap. 8)** ya trata: las tres aportaciones metodológicas (diseño causal,
  coherencia biológica, techo estructural), los ODS y la repercusión por contextos.
  Las conclusiones **no repiten** eso; lo dan por dicho y, si acaso, lo referencian.
- El **resumen** ya sintetiza el todo en una página. Las conclusiones son más
  específicas: cierran objetivo por objetivo y desarrollan el trabajo futuro con detalle.
- Lo genuino de este capítulo: (1) balance del cumplimiento de cada objetivo, (2) la
  conclusión de investigación enunciada de forma definitiva, (3) trabajo futuro en
  profundidad (lo que el impacto solo nombró de pasada).

## Hilos que aterrizan aquí
- **H20** (techo estructural: horizonte de un día con variables locales). Las tres
  direcciones de trabajo futuro (modelos de secuencia, viento, predicción multi-paso)
  se desarrollan aquí, no solo se nombran.
- **H10** (persistencia como baseline): referencia conceptual, mencionada de pasada
  (el autor pidió no sobre-centrar el discurso en ella).
- Hilos no realizados que pueden reaparecer como trabajo futuro: **H4** (mapas/análisis
  intradía desde `fixes_clean.parquet`), **H8** (ampliar el umbral de aves), **H7/H12**
  (personalización por individuo, ya cerrados como "la población basta", pero
  reabribles con más datos).

## Decisiones del autor (2026-06-01)
1. **Dos secciones**: 9.1 fusiona cumplimiento de objetivos + conclusión de
   investigación; 9.2 trabajo futuro.
2. **Cualitativo** (sin cifras), coherente con el resumen.
3. **Trabajo futuro**: las tres direcciones centrales en profundidad + las reabribles
   en una frase.

## Estructura aprobada (2 secciones, sin figuras ni tablas nuevas)

### 9.1 Conclusiones
Funde el balance de objetivos con la conclusión de investigación. En prosa:
- Arranque: se planteó predecir el desplazamiento diario de la gaviota sombría con un
  enfoque híbrido comparado con métrica común y baselines explícitas; el objetivo general
  se ha alcanzado.
- Recorrido por los cinco objetivos específicos como hilo del balance, cada uno con su
  resultado clave y remisión al capítulo (\ref), sin reexplicar el método: secuencia
  diaria homogénea; baseline de Markov + log-loss + persistencia; régimen de
  comportamiento con el HMM validado por fenología; predicción supervisada causal,
  poblacional, con incertidumbre calibrada; herramienta de cartografía interactiva.
- Conclusión de investigación (tono investigador): el enfoque híbrido funciona como
  sistema integrado; el límite de la predicción a un día con información local es
  estructural, no de configuración; el valor está en la incertidumbre calibrada y en
  aislar el régimen de migración. Persistencia solo de pasada.
- Las lecciones metodológicas (coherencia biológica como validación, diseño causal de
  variables) se mencionan como aprendizajes, referidas al cap. de impacto sin
  reexplicarlas.

### 9.2 Trabajo futuro
Tres direcciones centrales en profundidad (H20) + reabribles en una frase:
- **Modelos de secuencia** (memoria temporal multi-día; recurrentes o de atención) para
  capturar la antesala de un evento migratorio que un paso local no ve.
- **Variables de viento y ambientales** con diseño causal (la ablación de viento de la
  línea base quedó pendiente de rework causal): el viento es un predictor físico del
  desplazamiento migratorio.
- **Predicción multi-paso / a horizontes mayores**, propagando la incertidumbre.
- (El párrafo de reabribles intradía/más aves/personalización se redactó pero el autor
  decidió retirarlo: el capítulo cierra con las tres direcciones centrales.)
