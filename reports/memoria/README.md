# Memoria del TFG

Carpeta de la memoria del TFG. Versionada en git porque es la salida
principal del proyecto.

## Flujo de trabajo (modo híbrido)

1. **Durante cada objetivo (O1..O5):** se mantiene el archivo
   `0N_<slug>.md` correspondiente con **notas estructuradas** según
   `_plantilla.md`. No es prosa final; son notas suficientemente
   detalladas para reconstruir el razonamiento meses más tarde.
2. **Al cerrar el trabajo técnico (post-O5):** las notas se redactan
   en prosa final, esta vez **directamente en los ficheros `.tex`** de
   `latex/secciones/`.
3. **Entrega:** compilación de la plantilla oficial ETSIINF →
   `pdflatex + biber + pdflatex + pdflatex` → PDF final.

## Estructura de capítulos (alineada con la plantilla oficial ETSIINF)

| # | Notas Markdown | Capítulo en LaTeX |
|---|---|---|
| 0 | `00_resumen.md` | `latex/secciones/resumen.tex` (Resumen ES + Abstract EN, máx. 2 pp. c/u) |
| 1 | `01_introduccion.md` | `latex/secciones/intro.tex` (contexto, motivación, objetivos) |
| 2 | `02_preliminares.md` | `latex/secciones/preliminares.tex` (estado del arte: Markov, HMM, ML para series espacio-temporales) |
| 3 | `03_o1_datos.md` | `latex/secciones/desarrollo_o1.tex` (O1 — preparación dataset) |
| 4 | `04_o2_markov.md` | `latex/secciones/desarrollo_o2.tex` (O2 — cadenas visibles) |
| 5 | `05_o3_hmm.md` | `latex/secciones/desarrollo_o3.tex` (O3 — HMM) |
| 6 | `06_o4_ml.md` | `latex/secciones/desarrollo_o4.tex` (O4 — ML supervisado) |
| 7 | `07_o5_viz.md` | `latex/secciones/desarrollo_o5.tex` (O5 — mapas y evaluación) |
| 8 | `08_impacto.md` | `latex/secciones/impacto.tex` (personal / social / económico / medioambiental / cultural / ODS) |
| 9 | `09_conclusiones.md` | `latex/secciones/conclusiones.tex` (resultados, conclusiones, trabajo futuro) |
| A | `10_anexo_ia.md` | `latex/secciones/anexo_ia.tex` (declaración de uso de IA, alimentada por `reports/ai-log/`) |

Los archivos Markdown se crean cuando se inicia el objetivo
correspondiente, no todos a la vez. Los archivos LaTeX se rellenan en
la fase de redacción final (post-O5).

## Plantilla LaTeX

`latex/` contiene una copia íntegra de la **plantilla oficial ETSIINF
versión 2023.06.01** (no clonada del repo upstream — copiada según la
instrucción del README de la plantilla). Fichero principal renombrado a
`tfg_etsiinf_JuanLlorens.tex` por convención.

- **`datos_tfg.tex`:** metadatos del TFG (parcialmente cumplimentados;
  varios campos `[PENDIENTE]` a resolver).
- **`secciones/`:** contenido capítulo a capítulo. De momento contiene
  el material de ejemplo de la plantilla; se irá sustituyendo en la
  fase de redacción final.
- **`portada/`:** portada con escudo UPM/ETSIINF.
- **`.gitignore`** anidado: ignora artefactos de compilación
  (`*.aux`, `*.log`, `*.bbl`, ...).

## Cómo citar evidencia

Las figuras y tablas del proyecto están registradas en
`reports/INDEX.md`. Tanto en las notas Markdown como, finalmente, en
los `.tex`, las imágenes se referencian por su ruta:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\linewidth]{../../figures/o1_fig03_gap-distribution.png}
  \caption{Distribución de los intervalos entre registros GPS consecutivos.
    El umbral de 2h se sitúa en el percentil 95.}
  \label{fig:o1-gap-distribution}
\end{figure}
```

El texto del caption proviene de `reports/captions/o1_fig03_gap-distribution.md`.

## Bibliografía

Formato IEEE vía BibTeX. Fichero `latex/secciones/biblio.bib`. Citas
desde el texto con `\cite{clave}`. Compilación con `biber`.

## Anexo de uso de IA

Material de origen: `reports/ai-log/` (no versionado). El anexo
consolida en una tabla las entradas relevantes para la memoria
(filtradas por la regla descrita en `reports/ai-log/README.md`) y
selecciona los prompts más representativos como ejemplo.
