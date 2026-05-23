# o4_fig15_l1v1-by-state-comparison

- **Objetivo:** O4
- **Decisión justificada:** Desglosar el aporte del viento por régimen biológico
- **Figura:** `reports/figures/o4_fig15_l1v1-by-state-comparison.png`
- **Tabla:** `reports/tables/o4_tab15_l1v1-by-state-comparison.csv`

## Caption (memoria)

Comparativa de top-1 por estado HMM (estacionario vs migración) entre L1-v0 y L1-v1 para los cuatro modelos comunes. La hipótesis del spec era que el aporte del viento se concentraría en los días de migración (state_b=1). Resultado: la hipótesis se refuta empíricamente. Ningún modelo mejora ≥ +3 pp absolutos en migración (criterio secundario §9 NO cumplido). La única ganancia neta proviene del RF poblacional, que mejora ~+4,7 pp en estacionario y queda casi plano en migración — patrón contraintuitivo que indica que el modelo explota el viento como pista climática general (estacionalidad + posición geográfica) más que como señal direccional de vuelo activo. La caída en migración (~0,12 top-1) persiste como límite estructural del enfoque de un día con features locales.
