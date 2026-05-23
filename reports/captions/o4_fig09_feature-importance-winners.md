# o4_fig09_feature-importance-winners

- **Objetivo:** O4
- **Decisión justificada:** Verificar que state_b_causal y posterior_b_migracion_causal son usados; cuantificar el peso de bird_id en el modelo personalizado
- **Figura:** `reports/figures/o4_fig09_feature-importance-winners.png`
- **Tabla:** `reports/tables/o4_tab09_feature-importance-winners.csv`

## Caption (memoria)

Importancia relativa de las features para los dos modelos ganadores (pipeline causal: cinemática t-1→t y estado HMM forward-filtered). En el modelo personalizado (RF), lat y lon acaparan ~0,66 de la importancia total, seguidos de bird_id (~0,14) y posterior_b_migracion_causal (~0,05). En el modelo poblacional (XGB), lat+lon suman ~0,74, con posterior_b_migracion_causal (~0,05) y state_b_causal (~0,02) entre las features informativas, lo que valida que el aporte de O3 al pipeline supervisado es real aunque secundario. La cinemática causal (step_in_km, cos_turning_in, sin/cos_bearing_in) ocupa posiciones intermedias en ambos modelos, con importancias modestas pero consistentes.
