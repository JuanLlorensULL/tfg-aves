# o4_tab05_winner-recommendation

- **Objetivo:** O4
- **Decisión justificada:** Ganador poblacional = xgb (canónico, empate con rf); ganador individual 91916A = rf. Criterio: log-loss.
- **Tabla:** `reports/tables/o4_tab05_winner-recommendation.csv`

## Caption (memoria)

**Criterio principal: log-loss** (calibración probabilística, heredado de O2). Random Forest (log-loss 5,49) y XGBoost (log-loss 5,60) empatan en la práctica (diferencia 0,11; top-1 0,576 frente a 0,582), mientras que LightGBM queda descartado por la divergencia de C2. Como la línea base sólo fija el techo estructural del problema, la elección entre los dos líderes no altera ninguna conclusión: se adopta **XGBoost** como modelo canónico poblacional (82 aves). **Ganador individual (91916A):** `rf` con log-loss = 4.327 (evaluado solo sobre 91916A). 

El poblacional es el modelo canónico (generaliza a las 82 aves). El individual se entrena exclusivamente con 91916A y solo es comparable contra el poblacional restringido a esa ave (poblacional@91916A); esa comparación se analiza en C7.
