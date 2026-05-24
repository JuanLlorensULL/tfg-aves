# o4_fig29_quantile-crossing

- **Objetivo:** O4
- **Decisión justificada:** Cruces de cuantil por familia: boosting (XGB/LGBM) > 0 corregidos; RF = 0.
- **Figura:** `reports/figures/o4_fig29_quantile-crossing.png`
- **Tabla:** `reports/tables/o4_tab29_quantile-crossing.csv`

## Caption (memoria)

Porcentaje de filas donde los cuantiles predichos se cruzan (p10>p50 o p50>p90) antes de la corrección monótona post-hoc, por familia, modo y eje. XGBoost y LightGBM entrenan un modelo independiente por cuantil y pueden cruzarse (se corrige con ordenación); Random Forest (QRF) obtiene los tres cuantiles de la misma distribución de hojas y es monótono por construcción (cero cruces).
