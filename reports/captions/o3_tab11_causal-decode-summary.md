# o3_tab11_causal-decode-summary

- **Objetivo:** O3
- **Decisión justificada:** Decodificado causal (filtrado forward-only) produce estados biológicamente coherentes y sin look-ahead, aptos como features de los modelos supervisados
- **Tabla:** `reports/tables/o3_tab11_causal-decode-summary.csv`

## Caption (memoria)

Resumen de caracterización del decodificado causal para el Modelo B canónico. El estado se obtiene por filtrado forward-only, usando solo la emision en t y las observaciones anteriores, sin acceder al futuro. Sobre las observaciones validas, el porcentaje global de dias en migracion es 13.9 por ciento, con desplazamiento medio de 190.8 km en migracion frente a 6.2 km en estacionario. El minimo estacional de migracion se produce en el mes 6 (Jun) y el maximo en el mes 4 (Abr), en coherencia con la fenologia de Larus fuscus (reproduccion en verano, pasos en primavera y otono). El acuerdo entre Modelo A y Modelo B es del 98.1 por ciento. Estos valores confirman que el cambio de suavizado Viterbi a filtrado forward-only preserva la interpretabilidad biologica mientras elimina la dependencia temporal del futuro.
