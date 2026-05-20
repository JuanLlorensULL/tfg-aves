# Memoria del TFG

Carpeta de la memoria del TFG. Versionada en git porque es la salida
principal del proyecto.

## Flujo de trabajo (modo híbrido)

1. **Durante cada objetivo (O1..O5):** se mantiene el archivo
   `0N_<slug>.md` correspondiente con **notas estructuradas** según
   `_plantilla.md`. No es prosa final; son notas suficientemente
   detalladas para reconstruir el razonamiento meses más tarde.
2. **Al cerrar el trabajo técnico (post-O5):** las notas se redactan
   en prosa final dentro del mismo archivo.
3. **Entrega:** conversión a LaTeX con `pandoc` + plantilla UPM y
   generación del PDF.

## Estructura de capítulos

| Archivo | Capítulo |
|---|---|
| `00_introduccion.md` | Introducción, motivación, objetivos, planificación |
| `01_o1_datos.md` | O1 — Preparación del dataset GPS |
| `02_o2_markov.md` | O2 — Predicción con cadenas de Markov visibles |
| `03_o3_hmm.md` | O3 — Detección de comportamiento con HMM |
| `04_o4_ml.md` | O4 — Predicción con Machine Learning |
| `05_o5_viz.md` | O5 — Visualización en mapas y análisis del error |
| `06_conclusiones.md` | Conclusiones, limitaciones, líneas futuras |
| `07_anexo_ia.md` | Anexo de uso de IA (alimentado por `reports/ai-log/`) |

Los archivos se crean cuando se inicia el objetivo correspondiente, no
todos a la vez.

## Cómo citar evidencia

Las figuras y tablas del proyecto están registradas en
`reports/INDEX.md`. Para citarlas en la memoria, referenciar la ruta:

```markdown
Como muestra la Figura X (`reports/figures/o1_fig03_gap-distribution.png`),
la distribución de huecos temporales...
```

El caption castellano en `reports/captions/o1_fig03_gap-distribution.md`
es directamente reutilizable como pie de figura.

## Anexo de uso de IA

Se redacta a partir de `reports/ai-log/` al cierre del trabajo. La
referencia bibliográfica y la plantilla del anexo están descritas en
`reports/ai-log/README.md`.
