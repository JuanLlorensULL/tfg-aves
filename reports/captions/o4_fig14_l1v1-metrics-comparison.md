# o4_fig14_l1v1-metrics-comparison

- **Objetivo:** O4
- **Decisión justificada:** Comparativa central del aporte del viento (L1-v0 vs L1-v1)
- **Figura:** `reports/figures/o4_fig14_l1v1-metrics-comparison.png`
- **Tabla:** `reports/tables/o4_tab14_l1v1-metrics-comparison.csv`

## Caption (memoria)

Comparativa side-by-side de las cuatro métricas globales en el test split (top-1, top-3, log-loss, distancia mediana km) para los cuatro modelos comunes a L1-v0 y L1-v1 (RF/XGB × personalizado/poblacional). LightGBM, persistencia y Markov(1) se excluyen para que la comparación sea simétrica. Resultado: sólo el RF poblacional mejora en las cuatro métricas con viento (top-1 +4 pp, log-loss -0,12 a 5,52), cumpliendo el criterio primario (§9). El RF personalizado se degrada (top-1 -3,6 pp), consistente con que bird_id ya satura la señal individual y las tres features de viento introducen ruido relativo. XGBoost se mantiene plano en ambos modos (±0,02 en log-loss). El aporte del viento existe pero es modesto y arquitectura-dependiente.
