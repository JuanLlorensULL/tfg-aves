# o4_fig25_calibration-coverage

- **Objetivo:** O4
- **Decisión justificada:** Cobertura empírica de [p10,p90] cerca del 80% nominal en las tres familias.
- **Figura:** `reports/figures/o4_fig25_calibration-coverage.png`
- **Tabla:** `reports/tables/o4_tab25_calibration-coverage.csv`

## Caption (memoria)

Cobertura empírica del intervalo de predicción [p10, p90] frente al 80% nominal, por familia (XGBoost, LightGBM, Random Forest) y eje, en modo poblacional. Una cobertura próxima al 80% indica incertidumbre del desplazamiento bien calibrada; permite comparar la calibración del boosting (pinball) frente al QRF de Random Forest (cuantiles de hojas).
