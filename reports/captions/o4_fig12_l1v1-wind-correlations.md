# o4_fig12_l1v1-wind-correlations

- **Objetivo:** O4
- **Decisión justificada:** Detectar colinealidad entre las features de viento y las existentes
- **Figura:** `reports/figures/o4_fig12_l1v1-wind-correlations.png`
- **Tabla:** `reports/tables/o4_tab12_l1v1-wind-correlations.csv`

## Caption (memoria)

Matriz de correlaciones Pearson entre las tres features de viento (wind_u_850, wind_v_850, wind_speed_850) y las features ya presentes en O4 base relevantes para la predicción (state_b, posterior_b_migracion, step_length_km, lat, lon). Correlaciones absolutas pequeñas con state_b y posterior_b_migracion (|r| < 0,2 esperado) confirman que las features de viento aportan información independiente y no son redundantes con el régimen HMM que ya codifica el contexto biológico.
