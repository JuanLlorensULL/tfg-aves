# o4_fig13_l1v1-feature-importance

- **Objetivo:** O4
- **Decisión justificada:** Confirmar que las features de viento son usadas por los modelos
- **Figura:** `reports/figures/o4_fig13_l1v1-feature-importance.png`
- **Tabla:** `reports/tables/o4_tab13_l1v1-feature-importance.csv`

## Caption (memoria)

Importancia relativa de las features para los dos modelos ganadores de L1-v0 (RF personalizado y XGBoost poblacional) tras reentrenarlos con las tres features de viento en L1-v1. Las barras en naranja corresponden a las nuevas features de viento (wind_u_850, wind_v_850, wind_speed_850). Si aparecen en el top-10 de al menos uno de los ganadores, se cumple el criterio diagnóstico de éxito (§9 del spec) — el modelo no ignora las nuevas señales.
