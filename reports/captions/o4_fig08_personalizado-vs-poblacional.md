# o4_fig08_personalizado-vs-poblacional

- **Objetivo:** O4
- **Decisión justificada:** Cuantificar cuánto del rendimiento se debe a memorizar bird_id frente a generalización
- **Figura:** `reports/figures/o4_fig08_personalizado-vs-poblacional.png`
- **Tabla:** `reports/tables/o4_tab08_personalizado-vs-poblacional.csv`

## Caption (memoria)

Diferencia train-test (gap) en top-1 para cada familia, contrastando el modo personalizado (con bird_id) frente al poblacional (sin bird_id). Un gap_pers >> gap_pob sería evidencia directa de que bird_id se está usando para memorizar el train más que para extraer patrones generalizables — es decir, el modo personalizado estaría sobreajustando a las trayectorias específicas de las 82 aves vistas. Si los gaps son similares y diff_test_pers_vs_pob > 0, entonces bird_id aporta señal generalizable real.
