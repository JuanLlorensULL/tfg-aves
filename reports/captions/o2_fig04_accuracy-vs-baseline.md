# o2_fig04_accuracy-vs-baseline

- **Objetivo:** O2
- **Decisión justificada:** Comparación Markov vs persistencia en top-1 accuracy, distancia mediana y log-loss por mes
- **Figura:** `reports/figures/o2_fig04_accuracy-vs-baseline.png`
- **Tabla:** `reports/tables/o2_tab04_accuracy-vs-baseline.csv`

## Caption (memoria)

Comparación de las tres métricas principales (top-1 accuracy, distancia mediana en km, log-loss) entre el modelo Markov mensual con suavizado Laplace y la baseline de persistencia (predecir mañana = hoy), desglosada por mes. Markov bate persistencia cuando los movimientos día-a-día son predecibles más allá del simple 'quedarse donde estaba'.
