# o4_fig09_feature-importance-winners

- **Objetivo:** O4
- **Decisión justificada:** Verificar que state_b y posterior_b_migracion son usados; cuantificar el peso de bird_id en el modelo personalizado
- **Figura:** `reports/figures/o4_fig09_feature-importance-winners.png`
- **Tabla:** `reports/tables/o4_tab09_feature-importance-winners.csv`

## Caption (memoria)

Importancia relativa de las features para los dos modelos ganadores. En el modelo personalizado, la importancia de bird_id indica el peso de la identidad individual frente al resto de señales (posición actual, ciclo anual, cinemática, régimen biológico). En el modelo poblacional, la ausencia de bird_id obliga al modelo a apoyarse completamente en lat/lon, doy cíclico, cinemática y state_b — confirmar que state_b y posterior_b_migracion ocupan posiciones altas valida que el aporte de O3 al pipeline supervisado es real.
