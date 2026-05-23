# o4_fig06_error-by-state-personalizado

- **Objetivo:** O4
- **Decisión justificada:** Desglose de top-1 y top-3 por estado HMM causal para el ganador personalizado
- **Figura:** `reports/figures/o4_fig06_error-by-state-personalizado.png`
- **Tabla:** `reports/tables/o4_tab06_error-by-state-personalizado.csv`

## Caption (memoria)

Accuracy del modelo ganador personalizado (rf) desglosada por estado biológico inferido por O3 con cinemática causal (state_b_causal). La caída del top-1 entre estacionario (~0,64) y migración (~0,13) confirma el límite estructural del modelo: acierta los días estacionarios —donde la persistencia también acertaría— pero falla en los días de migración activa, que son biológicamente los más relevantes. Este resultado no es un artefacto del overfit sino el techo real de un modelo que dispone únicamente de features locales de un solo día sin información de viento ni historial multi-paso.
