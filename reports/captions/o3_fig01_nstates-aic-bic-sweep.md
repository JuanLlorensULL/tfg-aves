# o3_fig01_nstates-aic-bic-sweep

- **Objetivo:** O3
- **Decisión justificada:** n_components fijado en 2 (estacionario + migración) respaldado por AIC/BIC sweep
- **Figura:** `reports/figures/o3_fig01_nstates-aic-bic-sweep.png`
- **Tabla:** `reports/tables/o3_tab01_nstates-aic-bic-sweep.csv`

## Caption (memoria)

AIC y BIC para HMMs Modelo A con n_components en {2, 3, 4} entrenados sobre el conjunto de entrenamiento (80 por ciento de aves) con 5 restarts. La feature step_in_km se usa en escala cruda (km), por lo que los valores absolutos de LL/AIC/BIC son distintos a los de versiones previas con log-escala. Se mantiene n=2 por alineación con el objetivo del TFG (estacionario vs migración) y por interpretabilidad biológica de los estados. Si AIC/BIC muestran preferencia marcada por n>2, los estados adicionales no admiten etiquetado biológico claro y se documenta como seguimiento en lugar de adoptarse.
