# o4_tab01_hyperparameter-config

- **Objetivo:** O4
- **Decisión justificada:** Configuración fija de hiperparámetros por familia (sin rejilla)
- **Tabla:** `reports/tables/o4_tab01_hyperparameter-config.csv`

## Caption (memoria)

Hiperparámetros conservadores fijados por familia para O4. La elección se basa en el consenso de literatura aplicada para datasets de tamaño moderado y target multiclase de alta cardinalidad. Se descarta una rejilla amplia porque (i) controlar el overfit y optimizar rendimiento son objetivos distintos: el primero se logra con hiperparámetros conservadores + early stopping + diagnóstico train-test (ver C1), mientras que el segundo introduce el riesgo de cherry-picking sobre la validación interna; y (ii) el coste de un barrido por familia (8-12 combinaciones) multiplicaría por 4-6 el tiempo de entrenamiento total sin garantía de mejora significativa para los seis modelos del estudio.
