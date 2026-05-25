# o3_fig02_features-by-state-a

- **Objetivo:** O3
- **Decisión justificada:** Modelo A separa estacionario/migración por cinemática entrante: bajo desplazamiento entrante y rumbo errático vs alto desplazamiento y rumbo sostenido
- **Figura:** `reports/figures/o3_fig02_features-by-state-a.png`

## Caption (memoria)

Distribución de las dos features cinemáticas entrantes del Modelo A condicionada al estado detectado mediante filtrado forward-only (estacionario en azul, migración en rojo). La feature step_in_km (desplazamiento del tramo t-1 a t) se muestra en escala logarítmica para hacer visible la bimodalidad entre pocos km (estado estacionario) y decenas-cientos de km (estado migración). El estado estacionario concentra masa en step_in_km bajo y cos_turning_in cercano a 0 o negativo (giros erráticos, sin rumbo sostenido). El estado migración presenta el patrón contrario: step_in_km alto y cos_turning_in cercano a +1 (vuelo rectilíneo). La separación visual confirma que el HMM A descubre estados con semántica biológica clara.
