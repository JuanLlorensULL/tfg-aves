# o4_tab30_comparativa-maestra-regimen

- **Objetivo:** O4
- **Decisión justificada:** Comparativa maestra L1/L2/L3 frente a baselines (persistencia y Markov(1)) por régimen biológico sobre el test poblacional de O4.
- **Tabla:** `reports/tables/o4_tab30_comparativa-maestra-regimen.csv`

## Caption (memoria)

Comparativa maestra de las tres líneas de mejora de O4 (L1 clasificación categórica, L2 dos etapas, L3 regresión de cuantiles) frente a las líneas base (persistencia trivial y Markov(1) reentrenado) sobre el conjunto de test poblacional (n=4037 transiciones; 3566 días estacionarios, 471 de migración, 908 con cambio efectivo de celda). Métricas: top-1 y top-3 accuracy sobre la celda 0,5° y distancia mediana haversine del centroide de la celda predicha a la posición real en t+1; para L3 se añade la distancia nativa del punto continuo p50 (no comparable con las celdas, sólo informativa). Lecturas clave: (1) ningún modelo aprendido bate a la persistencia en el top-1 del destino diario en migración —el cuello de botella estructural del horizonte a un día—; (2) L2 logra el mejor top-1 global del TFG (0,780) actuando como «persistencia inteligente», pero hereda su punto ciego y colapsa a 0,036 en los días de movimiento real; (3) L3 es el modelo más útil cuando el ave abandona su celda (mejor top-1, top-3 y distancia en el régimen «moves») y ofrece el mejor top-3 en todos los regímenes además de incertidumbre calibrada; (4) Markov(1) es dominado en todas las métricas y se descarrila en distancia durante la migración (1098 km). L1 RF es equivalente a L1 XGB y L2 soft XGB a L2 soft RF; se omiten para no recargar.
