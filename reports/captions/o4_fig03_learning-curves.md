# o4_fig03_learning-curves

- **Objetivo:** O4
- **Decisión justificada:** Curvas de log-loss en validación interna para XGBoost y LightGBM
- **Figura:** `reports/figures/o4_fig03_learning-curves.png`

## Caption (memoria)

Evolución del log-loss sobre la validación interna a lo largo de las iteraciones de boosting para XGBoost (izquierda) y LightGBM (derecha), para el poblacional (XGB y LightGBM) y el individual de 91916A (solo XGB). La línea vertical punteada marca la iteración óptima detectada por early stopping (paciencia 50). XGBoost presenta el comportamiento esperado: descenso monotónico del log-loss en validación hasta estabilizarse en torno a la iteración 330-340, con el early stopping deteniendo el entrenamiento poco después. LightGBM, en contraste, diverge desde la primera iteración: el log-loss en validación arranca ya por encima del valor de una distribución uniforme sobre las clases activas y sube monotónicamente hasta estabilizarse alrededor de 30. El early stopping actúa correctamente como salvaguarda y detiene el entrenamiento en la iteración 1, pero el resultado revela que la configuración conservadora de LightGBM (§8.6) no extrae señal generalizable de este dataset con un target de ~849 clases activas fuertemente desbalanceadas. Este hallazgo se interpreta en C7 y se discute en la memoria §6: LightGBM con la configuración fijada queda descartado como modelo candidato, mientras que RF y XGBoost ofrecen comparativas significativas (ver C3 y C4).
