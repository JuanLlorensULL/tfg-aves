# o3_fig07_feature-influence-cohens-d

- **Objetivo:** O3
- **Decisión justificada:** step_length domina la separación de estados (|d|≫1); cos_turning aporta señal secundaria; daylight y veg apenas discriminan (|d|<0,5) — confirma cuantitativamente el 93 % de acuerdo A-B
- **Figura:** `reports/figures/o3_fig07_feature-influence-cohens-d.png`
- **Tabla:** `reports/tables/o3_tab07_feature-influence-cohens-d.csv`

## Caption (memoria)

Cohen's d para cada feature del Modelo B, comparando observaciones del estado migración (state_b=1) contra estacionario (state_b=0), agrupado por la desviación típica conjunta. La magnitud absoluta indica la fuerza discriminativa de cada feature; el signo, el sentido (rojo = mayor en migración, azul = mayor en estacionario). El step_length domina por dos órdenes de magnitud relativos al contexto, lo que justifica cuantitativamente la decisión §9.2 (sin StandardScaler) y explica el 93 % de acuerdo entre Modelo A (sólo cinemática) y Modelo B (cinemática + contexto): las features contextuales aportan refinamiento marginal pero no son el motor de la clasificación. Escala interpretativa de Cohen: |d|<0,2 mínimo, 0,2-0,5 pequeño, 0,5-0,8 medio, >0,8 grande.
