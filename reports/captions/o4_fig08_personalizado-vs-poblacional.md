# o4_fig08_personalizado-vs-poblacional

- **Objetivo:** O4
- **Decisión justificada:** bird_id aporta una mejora marginal en test (~+1 pp top-1) sin aumentar el gap de overfit respecto al modo poblacional
- **Figura:** `reports/figures/o4_fig08_personalizado-vs-poblacional.png`
- **Tabla:** `reports/tables/o4_tab08_personalizado-vs-poblacional.csv`

## Caption (memoria)

Análisis del papel de bird_id (identidad individual) en el pipeline. Panel izquierdo: gap train-test en top-1 para cada familia en modo personalizado (con bird_id) frente a poblacional (sin bird_id). Los gaps son prácticamente iguales entre modos (~0,20 para RF, ~0,21 para XGBoost): añadir bird_id NO aumenta el overfit, pero tampoco lo reduce. Panel derecho: diferencia de top-1 en test (pers − pob) para RF y XGBoost. La diferencia es positiva pero pequeña (~+1 pp), lo que indica que bird_id aporta señal generalizable real —probablemente rangos de hábitat individuales que el modelo captura— aunque el impacto cuantitativo es modesto. Interpretación para la memoria: el modo personalizado es marginalmente preferible en top-1 si el ave es conocida; el poblacional es igualmente válido y más generalizable a individuos nuevos.
