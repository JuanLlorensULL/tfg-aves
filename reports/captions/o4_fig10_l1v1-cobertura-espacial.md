# o4_fig10_l1v1-cobertura-espacial

- **Objetivo:** O4
- **Decisión justificada:** Verificar que el bbox del viento cubre todos los fixes
- **Figura:** `reports/figures/o4_fig10_l1v1-cobertura-espacial.png`
- **Tabla:** `reports/tables/o4_tab10_l1v1-cobertura-espacial.csv`

## Caption (memoria)

Distribución espacial de los 24 444 fixes diarios del dataset Movebank superpuestos al bbox del viento reanalysis ECMWF a 850 hPa (lat ∈ [-3°, 66°] × lon ∈ [7°, 53°]). De los 21 823 fixes con coordenadas válidas, 21 803 (99,91 %) caen dentro del bbox; los 20 fixes restantes (0,09 %) corresponden a la migración postnupcial de 2009 sobre Bélgica y Países Bajos (lon < 7°) y reciben NaN en las features de viento, descartados automáticamente por el pipeline gap-aware. Cobertura espacial esencialmente completa pero documentada con honestidad.
