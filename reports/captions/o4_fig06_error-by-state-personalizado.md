# o4_fig06_error-by-state-personalizado

- **Objetivo:** O4
- **Decisión justificada:** Desglose de top-1 y top-3 por estado HMM para el ganador personalizado
- **Figura:** `reports/figures/o4_fig06_error-by-state-personalizado.png`
- **Tabla:** `reports/tables/o4_tab06_error-by-state-personalizado.csv`

## Caption (memoria)

Accuracy del modelo ganador personalizado (rf) desglosada por estado biológico inferido por O3 (Modelo B). La comparación global vs estacionario vs migración revela cuándo falla el modelo: una caída fuerte en migración indicaría que el modelo acierta sólo los días triviales (estacionarios, donde la persistencia ya lo hace bien). Esta es la pregunta que define el aporte real del ML supervisado frente a las baselines.
