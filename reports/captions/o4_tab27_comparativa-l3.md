# o4_tab27_comparativa-l3

- **Objetivo:** O4
- **Decisión justificada:** L3 cuantil vs categórico (A, target) y individual vs poblacional sobre 91916A (B, per-individuo), familia XGBoost.
- **Tabla:** `reports/tables/o4_tab27_comparativa-l3.csv`

## Caption (memoria)

Comparativas centrales de L3 (XGBoost). Bloque A: efecto de reformular el target (categórico L3-v0 vs cuantil L3-v1) en modo poblacional, por estado HMM y en días de movimiento. Bloque B: hipótesis per-individuo, modelo individual de 91916A frente al poblacional sobre las mismas filas. Sin columna log-loss: L3 no produce distribución categórica, su veredicto es geométrico (distancia + top-1 mapeado).
