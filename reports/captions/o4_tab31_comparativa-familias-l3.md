# o4_tab31_comparativa-familias-l3

- **Objetivo:** O4
- **Decisión justificada:** Comparativa de las tres familias supervisadas (XGBoost, LightGBM, Random Forest) en la regresión de cuantiles, modo poblacional.
- **Tabla:** `reports/tables/o4_tab31_comparativa-familias-l3.csv`

## Caption (memoria)

Comparativa maestra de las tres familias del proposal sobre la tarea de regresión de cuantiles del desplazamiento (modo poblacional, test completo), por régimen (global, estacionario, migración, días de movimiento). Métricas: top-1/top-3 mapeados, distancia vía centroide y nativa, pérdida pinball y cobertura [p10,p90] por eje, y nº total de cruces de cuantil corregidos (cero en Random Forest por construcción). Baseline: persistencia trivial. Cierra la comparativa de tres familias que O4 base hizo sobre clasificación, ahora sobre regresión.
