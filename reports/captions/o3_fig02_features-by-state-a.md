# o3_fig02_features-by-state-a

- **Objetivo:** O3
- **Decisión justificada:** Modelo A separa estacionario/migración por cinemática: bajo desplazamiento+rumbo errático vs alto desplazamiento+rumbo sostenido
- **Figura:** `reports/figures/o3_fig02_features-by-state-a.png`

## Caption (memoria)

Distribución de las dos features cinemáticas del Modelo A condicionada al estado Viterbi (estacionario en azul, migración en rojo). La feature step_length_km se muestra en escala logarítmica para hacer visible la bimodalidad entre pocos km (estado estacionario) y decenas-cientos de km (estado migración). El estado estacionario concentra masa en step_length_km bajo y cos_turning_angle cercano a 0 o negativo (giros erráticos, sin rumbo sostenido). El estado migración presenta el patrón contrario: step_length_km alto y cos_turning_angle cercano a +1 (vuelo rectilíneo). La separación visual confirma que el HMM A descubre estados con semántica biológica clara.
