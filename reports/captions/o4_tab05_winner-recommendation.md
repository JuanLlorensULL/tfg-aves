# o4_tab05_winner-recommendation

- **Objetivo:** O4
- **Decisión justificada:** Ganador poblacional = xgb; ganador individual 91916A = rf. Criterio: log-loss.
- **Tabla:** `reports/tables/o4_tab05_winner-recommendation.csv`

## Caption (memoria)

**Criterio principal: log-loss** (calibración probabilística, heredado de O2). **Ganador poblacional (82 aves):** `xgb` con log-loss = 5.827. **Ganador individual (91916A):** `rf` con log-loss = 4.340 (evaluado solo sobre 91916A). 

El poblacional es el modelo canónico (generaliza a las 82 aves). El individual se entrena exclusivamente con 91916A y solo es comparable contra el poblacional restringido a esa ave (poblacional@91916A); esa comparación se analiza en C7. LightGBM queda descartado por la divergencia de C2.
