# o4_tab05_winner-recommendation

- **Objetivo:** O4
- **Decisión justificada:** Algoritmo ganador personalizado = rf; ganador poblacional = xgb. Criterio: log-loss en test temporal.
- **Tabla:** `reports/tables/o4_tab05_winner-recommendation.csv`

## Caption (memoria)

**Criterio principal: log-loss** (calibración probabilística; heredado de O2 como criterio más informativo que el top-1 argmax). El log-loss penaliza la sobreconfianza en predicciones incorrectas y es la métrica más útil para alimentar O5 con distribuciones de probabilidad por celda. 

**Ganador del modo personalizado:** `rf` con log-loss = 5.666. **Ganador del modo poblacional:** `xgb` con log-loss = 5.827. 

Resultado global: ML bate a Markov(1) en todas las métricas (C3), pero pierde ante la persistencia trivial en top-1 global y distancia mediana — resultado estructural del dataset (73 % de días son self-loops, lo que favorece por construcción a la baseline trivial). El margen de valor real del ML se concentra en los días de movimiento (ver columna top1_dias_movimiento en C3): ML ~0,18 vs persistencia = 0 y Markov ~0,09. El error espacial absoluto es modesto: mediana ~24 km, ~80 % de predicciones dentro de 110 km. El límite es estructural (features locales de un único día, sin historial multi-paso ni información de viento). 

LightGBM queda descartado en ambos modos por la divergencia documentada en C2 — su configuración conservadora de §8.6 no extrae señal generalizable de este dataset con ~849 clases activas. 

Ambos ganadores se utilizarán como base para los artefactos siguientes (C5/C6 análisis de error por estado, C7 memorización, C8 feature importance). En O5, se cargarán por defecto; el usuario podrá cambiar a las otras combinaciones para comparar visualmente.
