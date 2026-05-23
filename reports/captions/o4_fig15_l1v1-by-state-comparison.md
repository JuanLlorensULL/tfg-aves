# o4_fig15_l1v1-by-state-comparison

- **Objetivo:** O4
- **Decisión justificada:** Desglosar el aporte del viento por régimen biológico
- **Figura:** `reports/figures/o4_fig15_l1v1-by-state-comparison.png`
- **Tabla:** `reports/tables/o4_tab15_l1v1-by-state-comparison.csv`

## Caption (memoria)

Comparativa de top-1 por estado HMM (estacionario vs migración) entre L1-v0 y L1-v1 para los cuatro modelos comunes. La hipótesis es que el aporte del viento se concentra en los días de migración (state_b=1), donde el modelo se beneficia más de saber si el viento es favorable o no. Si la mejora en estacionario es nula y en migración es ≥ +3 pp absolutos en al menos uno de los ganadores, se cumple el criterio secundario de éxito de L1 (§9 del spec).
