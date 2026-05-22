# o3_fig02_features-by-state-a

- **Objetivo:** O3
- **Decisión justificada:** Modelo A separa estacionario/migración por cinemática: bajo desplazamiento+rumbo errático vs alto desplazamiento+rumbo sostenido
- **Figura:** `reports/figures/o3_fig02_features-by-state-a.png`

## Caption (memoria)

Distribución de las dos features cinemáticas del Modelo A condicionada al estado Viterbi (estacionario en azul, migración en rojo). El estado estacionario concentra masa en log_displacement_km bajo (ave estática) y abs_turning_angle_rad alto/aleatorio (direcciones erráticas, sin rumbo sostenido). El estado migración presenta el patrón contrario: log_displacement_km alto y abs_turning_angle_rad bajo (movimiento dirigido). La separación visual confirma que el HMM A descubre estados con semántica biológica clara.
