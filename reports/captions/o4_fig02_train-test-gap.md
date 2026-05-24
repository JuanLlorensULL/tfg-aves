# o4_fig02_train-test-gap

- **Objetivo:** O4
- **Decisión justificada:** Diagnóstico del gap train-test (overfit) del individual frente al poblacional sobre 91916A
- **Figura:** `reports/figures/o4_fig02_train-test-gap.png`
- **Tabla:** `reports/tables/o4_tab02_train-test-gap.csv`

## Caption (memoria)

Top-1 (izquierda) y log-loss (derecha) evaluados sobre train y test, todo restringido a las filas del ave 91916A: el modelo individual (entrenado solo con 91916A) frente al poblacional evaluado sobre esas mismas filas (poblacional@91916A). La diferencia train→test es el indicador empírico del overfit. La comparación es manzanas-con-manzanas porque ambos se miden sobre el mismo conjunto de evaluación; el modo poblacional global se analiza aparte en C3.
