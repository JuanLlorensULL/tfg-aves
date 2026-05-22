# o3_fig03_features-by-state-b

- **Objetivo:** O3
- **Decisión justificada:** Modelo B añade contexto ambiental (vegetación, fotoperiodo) a la cinemática; comparación con C1 detecta posible circularidad
- **Figura:** `reports/figures/o3_fig03_features-by-state-b.png`

## Caption (memoria)

Distribución de las cinco features del Modelo B condicionada al estado Viterbi. La primera feature (step_length_km) se muestra en escala logarítmica. Las dos primeras (step_length_km, abs_turning_angle_rad) replican el patrón del Modelo A: bimodalidad entre pocos km (estacionario) y decenas-cientos km (migración). Las tres adicionales (daylight_hours, veg_low, veg_high) muestran si los estados resultantes están condicionados también por contexto temporal y ambiental: comparar con C1 permite ver si el contexto refina la separación o si la domina (alarma de circularidad si los estados se reducen a 'verano vs invierno').
