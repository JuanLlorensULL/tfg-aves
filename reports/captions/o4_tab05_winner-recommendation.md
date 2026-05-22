# o4_tab05_winner-recommendation

- **Objetivo:** O4
- **Decisión justificada:** Algoritmo ganador personalizado = rf; ganador poblacional = xgb. Criterio: log-loss en test temporal.
- **Tabla:** `reports/tables/o4_tab05_winner-recommendation.csv`

## Caption (memoria)

**Criterio principal: log-loss** (consistente con la decisión metodológica heredada de O2). El log-loss penaliza la sobreconfianza en predicciones incorrectas y recompensa una buena calibración probabilística, lo que es esencial para alimentar O5 con distribuciones de probabilidad por celda. 

**Ganador del modo personalizado:** `rf` con log-loss = 5.224. **Ganador del modo poblacional:** `xgb` con log-loss = 5.518. 

LightGBM queda descartado en ambos modos por la divergencia documentada en C2 — su configuración conservadora de §8.6 no extrae señal generalizable de este dataset con ~849 clases activas. 

Ambos ganadores se utilizarán como base para los artefactos siguientes (C5/C6 análisis de error por estado, C7 comparativa memorización, C8 feature importance). En O5, se cargarán por defecto como combinación recomendada de los dos selectores; el usuario podrá cambiar a las otras combinaciones para comparar visualmente.
