# o4_fig04_models-comparison

- **Objetivo:** O4
- **Decisión justificada:** Tabla comparativa de las 8 combinaciones (3 familias × 2 modos + 2 baselines)
- **Figura:** `reports/figures/o4_fig04_models-comparison.png`
- **Tabla:** `reports/tables/o4_tab04_models-comparison.csv`

## Caption (memoria)

Métricas globales sobre el conjunto de test temporal (último 20 % de días de cada ave). Las cuatro métricas se reportan en paneles separados para destacar al ganador en cada criterio. Persistencia trivial y Markov(1) se incluyen como baselines de referencia heredadas de O2. LightGBM queda visualmente diferenciado por sus valores anómalos en log-loss y distancia mediana, coherente con la divergencia observada en C2. Conviene leer este artefacto junto a C4, que argumenta la recomendación final del algoritmo ganador priorizando log-loss como criterio principal.
