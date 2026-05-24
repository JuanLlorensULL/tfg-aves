# o4_fig09_feature-importance-winners

- **Objetivo:** O4
- **Decisión justificada:** Verificar que state_b_causal y posterior_b_migracion_causal se usan en ambos modelos
- **Figura:** `reports/figures/o4_fig09_feature-importance-winners.png`
- **Tabla:** `reports/tables/o4_tab09_feature-importance-winners.csv`

## Caption (memoria)

Importancia relativa de las features para el ganador poblacional y el modelo individual de 91916A (pipeline causal: cinemática t-1→t y estado HMM forward-filtered). En ambos modelos lat y lon acaparan la mayor parte de la importancia, seguidos de las features de O3 (posterior_b_migracion_causal y, en menor medida, state_b_causal), lo que valida que el aporte de O3 al pipeline supervisado es real aunque secundario. La cinemática causal (step_in_km, cos_turning_in, sin/cos_bearing_in) ocupa posiciones intermedias en ambos modelos, con importancias modestas pero consistentes. El modelo individual no incluye la identidad del ave como feature; se entrena sobre un único individuo.
