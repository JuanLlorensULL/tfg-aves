# o3_fig08_bird-trajectory-by-state

- **Objetivo:** O3
- **Decisión justificada:** Validación visual individual: la trayectoria del ave con más observaciones muestra que los estados HMM se alinean con tramos geográficamente coherentes (roost vs paso migratorio)
- **Figura:** `reports/figures/o3_fig08_bird-trajectory-by-state.png`

## Caption (memoria)

Trayectoria del ave 91916A (la de mayor cobertura temporal del dataset, 2051 días válidos) sobre mapa de Europa y Africa con coastlines y fronteras nacionales. La línea gris conecta días consecutivos; cada punto se colorea según el estado detectado por filtrado forward-only (azul estacionario, rojo migración). Comparando Modelo A (cinemática entrante pura) y Modelo B (cinemática entrante más contexto), se aprecia visualmente que ambos modelos identifican como estado migración los tramos de mayor desplazamiento diario entre zonas geográficamente distantes, mientras que los puntos estacionarios se agrupan en zonas de roost o de cría. La concordancia visual entre A y B confirma el alto acuerdo cuantitativo (C4).
