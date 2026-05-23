# o4_fig16_l1v1-tailwind-effect

- **Objetivo:** O4
- **Decisión justificada:** Verificar si el modelo aprende a interpretar la dirección del viento
- **Figura:** `reports/figures/o4_fig16_l1v1-tailwind-effect.png`
- **Tabla:** `reports/tables/o4_tab16_l1v1-tailwind-effect.csv`

## Caption (memoria)

Análisis post-hoc del aprendizaje del viento en L1-v1: top-1 accuracy sobre los días de migración (state_b=1) desglosado por dirección del viento respecto a la dirección fenológica esperada de Larus fuscus (primavera: hacia el norte; otoño: hacia el sur; invernada y cría se excluyen al no tener dirección clara). Si delta = top1_favorable - top1_desfavorable es positivo y no trivial, el modelo está capturando la interacción viento×fenología — evidencia indirecta de que las features de viento aportan más que ruido. Es la prueba final para distinguir entre 'el modelo usa el viento como señal direccional' y 'el modelo lo usa como pista climática genérica'.
