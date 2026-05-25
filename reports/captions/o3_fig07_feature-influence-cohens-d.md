# o3_fig07_feature-influence-cohens-d

- **Objetivo:** O3
- **Decisión justificada:** step_in_km domina la separación de estados (|d|>>1); cos_turning_in aporta señal secundaria; daylight y veg apenas discriminan (|d|<0,5), confirma el alto acuerdo A-B
- **Figura:** `reports/figures/o3_fig07_feature-influence-cohens-d.png`
- **Tabla:** `reports/tables/o3_tab07_feature-influence-cohens-d.csv`

## Caption (memoria)

Cohen's d para cada feature del Modelo B, comparando observaciones del estado migración (state_b_causal=1) contra estacionario (state_b_causal=0), normalizado por la desviación típica conjunta. La magnitud absoluta indica la fuerza discriminativa de cada feature; el signo, el sentido (rojo = mayor en migración, azul = mayor en estacionario). El desplazamiento entrante step_in_km domina por dos órdenes de magnitud relativos al contexto, lo que justifica el uso de la cinemática cruda sin estandarización y explica el alto acuerdo entre Modelo A (solo cinemática) y Modelo B (cinemática más contexto): las features contextuales aportan refinamiento marginal pero no son el motor de la clasificación. Escala interpretativa de Cohen: |d|<0,2 minimo, 0,2-0,5 pequeno, 0,5-0,8 medio, >0,8 grande.
