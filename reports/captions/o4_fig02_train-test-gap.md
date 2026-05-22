# o4_fig02_train-test-gap

- **Objetivo:** O4
- **Decisión justificada:** Diagnóstico del gap train-test para detectar overfit
- **Figura:** `reports/figures/o4_fig02_train-test-gap.png`
- **Tabla:** `reports/tables/o4_tab02_train-test-gap.csv`

## Caption (memoria)

Comparativa de top-1 (panel izquierdo) y log-loss (panel derecho) evaluados sobre train y test para los seis modelos (tres familias × dos modos). La diferencia train→test es el indicador empírico del overfit: un gap pequeño sugiere que el modelo aprende patrones generalizables; un gap grande sugiere memorización. El modo personalizado (que incluye bird_id) es el más expuesto a memorización; su comparación contra el modo poblacional se analiza explícitamente en C7.
