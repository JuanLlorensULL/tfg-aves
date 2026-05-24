# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # O5 — Mapas interactivos y análisis del error
#
# Construye los mapas folium (vista A por ave con slider, coropletas de error
# y calibración, vectores de fallo en migración y demo multi-paso) y las
# tablas de evidencia a partir de **L3 LightGBM poblacional**.
# Ver spec `docs/superpowers/specs/2026-05-24-o5-viz-design.md`.
#
# Los mapas se guardan como HTML en `reports/figures/`; las figuras de la
# memoria son screenshots manuales de esos HTML. Las tablas numéricas van por
# `save_artifact` (CSV + caption + INDEX).

# %%
from tfg_aves.viz import build

result = build.build_o5()
print("Mapas de predicción:", sorted(result["prediction_maps"]))
print("Mapas de error:", sorted(result["error_maps"]))
print("Demo multi-paso:", sorted(result["demo_maps"]))

# %% [markdown]
# ## Tablas de evidencia

# %% [markdown]
# ### Aves curadas (las 4 con más histórico)

# %%
result["tables"]["aves_curadas"]

# %% [markdown]
# ### Error por régimen HMM
#
# top-1 cae de ~0,84 (estacionario) a ~0,23 (migración): el modelo acierta el
# destino cuando el ave se queda, falla cuando migra. La cobertura conjunta
# (rectángulo) es menor que la marginal (~0,80) por construcción.

# %%
result["tables"]["por_regimen"]

# %% [markdown]
# ### Error por mes (cruce con la fenología)
#
# top-1 mínimo en los picos migratorios (abril; sep-oct), máximo en invierno
# (ene-feb). Coherente con la fenología de *Larus fuscus*.

# %%
result["tables"]["por_mes"]

# %% [markdown]
# ## Mapas interactivos
#
# Los HTML generados en `reports/figures/` se abren en el navegador:
#
# - `o5_fig01..04_prediccion-<ave>.html` — vista A por ave (slider temporal:
#   p50 + banda [p10,p90] + persistencia + real t+1).
# - `o5_fig20_error-por-celda.html` — coropleta de distancia mediana p50→real.
# - `o5_fig21_calibracion-por-celda.html` — coropleta de cobertura marginal.
# - `o5_fig22_vectores-fallo-migracion.html` — vectores de los mayores fallos
#   en días de migración.
# - `o5_fig30_demo-multipaso-91916A.html` — demo exploratoria multi-paso
#   (cono ilustrativo no calibrado).
