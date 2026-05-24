# o4_fig08_individual-vs-poblacional

- **Objetivo:** O4
- **Decisión justificada:** Comparar el modelo per-individuo de 91916A contra el poblacional evaluado sobre 91916A (poblacional@91916A) y las baselines, mismas filas
- **Figura:** `reports/figures/o4_fig08_individual-vs-poblacional.png`
- **Tabla:** `reports/tables/o4_tab08_individual-vs-poblacional.csv`

## Caption (memoria)

Comparación manzanas-con-manzanas sobre el test del ave 91916A: el modelo individual (RF/XGB entrenados solo con 91916A) frente al modelo poblacional restringido a 91916A (poblacional@91916A, RF/XGB) y las baselines persistencia y Markov sobre esas mismas filas. Replica la metodología de L3: si el individual no supera claramente a poblacional@91916A, se confirma que entrenar por-individuo no aporta frente al modelo global, reforzando la decisión 'global vs per-individuo' a favor del global.
