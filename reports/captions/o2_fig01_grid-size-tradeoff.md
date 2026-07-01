# o2_fig01_grid-size-tradeoff

- **Objetivo:** O2
- **Decisión justificada:** Tamaño de celda fijado en 0.5° (justificado por trade-off de 4 candidatos)
- **Figura:** `reports/figures/o2_fig01_grid-size-tradeoff.png`
- **Tabla:** `reports/tables/o2_tab01_grid-size-tradeoff.csv`

## Caption (memoria)

Comparación de 4 candidatos de tamaño de celda para la discretización del espacio en O2: 0,25°, 0,5°, 1° y 2°. El panel A muestra el desplazamiento diario observado con líneas verticales en el tamaño físico aproximado de cada celda (cell_deg × 111 km). El panel B resume métricas de cobertura y densidad para cada candidato. El panel C superpone cada grid sobre la nube de fixes válidos. Se adopta cell_deg = 0.5° porque equilibra resolución espacial (transiciones cruzan varias celdas en periodos migratorios) y robustez estadística (mediana de transiciones por celda-mes y % de pares observados aceptables tras el suavizado Laplace α=1).
