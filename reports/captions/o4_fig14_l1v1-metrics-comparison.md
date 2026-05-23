# o4_fig14_l1v1-metrics-comparison

- **Objetivo:** O4
- **Decisión justificada:** Comparativa central del aporte del viento (L1-v0 vs L1-v1)
- **Figura:** `reports/figures/o4_fig14_l1v1-metrics-comparison.png`
- **Tabla:** `reports/tables/o4_tab14_l1v1-metrics-comparison.csv`

## Caption (memoria)

Comparativa side-by-side de las cuatro métricas globales en el test split (top-1, top-3, log-loss, distancia mediana km) para los cuatro modelos comunes a L1-v0 y L1-v1 (RF/XGB × personalizado/poblacional). LightGBM, persistencia y Markov(1) se excluyen para que la comparación sea simétrica. Es el entregable narrativo central de L1: cuantifica el aporte aislado del viento como predictor sin contaminar con otras decisiones.
