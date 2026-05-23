# o4_fig04_models-comparison

- **Objetivo:** O4
- **Decisión justificada:** Tabla comparativa de las 8 combinaciones (3 familias × 2 modos + 2 baselines)
- **Figura:** `reports/figures/o4_fig04_models-comparison.png`
- **Tabla:** `reports/tables/o4_tab04_models-comparison.csv`

## Caption (memoria)

Métricas globales sobre el conjunto de test temporal (último 20 % de días de cada ave). Se reportan cinco paneles: top-1 global, top-3 global, log-loss, distancia mediana y top-1 restringido a los días de movimiento (true_cell ≠ celda predicha por persistencia, n=908, 22,5 % del test). Persistencia trivial y Markov(1) se incluyen como baselines heredadas de O2. En las métricas globales, persistencia domina sobre ML (73 % de días son self-loops, resultado estructural del dataset). En los días de movimiento —donde persistencia es trivialmente cero— los modelos ML (RF/XGB) alcanzan top-1 ~0,17-0,20 frente al 0,09 de Markov(1): este es el margen de valor real del aprendizaje supervisado. LightGBM queda visualmente diferenciado por sus valores anómalos en log-loss y distancia mediana, coherente con la divergencia observada en C2. Conviene leer este artefacto junto a C4, que argumenta la recomendación final del algoritmo ganador.
