# o4_fig03_learning-curves

- **Objetivo:** O4
- **Decisión justificada:** Curvas de log-loss en validación interna para XGBoost y LightGBM
- **Figura:** `reports/figures/o4_fig03_learning-curves.png`

## Caption (memoria)

Evolución del log-loss sobre la validación interna a lo largo de las iteraciones de boosting, para XGBoost (izquierda) y LightGBM (derecha), en ambos modos. La línea vertical punteada indica la iteración óptima detectada por early stopping (paciencia 50). El que el algoritmo se detenga claramente antes de las 1 000 iteraciones máximas confirma que el control automático de complejidad está operativo; un plateau muy temprano sugeriría subajuste y uno tardío, sobreajuste — ambos guían un eventual ajuste manual de la configuración de §8.6.
