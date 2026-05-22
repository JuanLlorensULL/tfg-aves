# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # O4 — Predicción ML supervisada (RF, XGBoost, LightGBM)
#
# Spec: `docs/superpowers/specs/2026-05-22-o4-ml-design.md`.
#
# Cuatro fases:
#
# - **Fase A:** sanity-check del dataset + ejecutar `build_o4()` + D1
#   (configuración de hiperparámetros).
# - **Fase B:** C1 (gap train-test) + C2 (learning curves).
# - **Fase C:** C3 (comparativa global) + C4 (recomendación ganador).
# - **Fase D:** C5/C6 (error por estado), C7 (memorización), C8 (feature
#   importance).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.ml import build_o4
from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET
from tfg_aves.reporting import save_artifact

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Fase A.1 — Sanity-check de inputs

# %%
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
print(f"features_o3: {features_o3.shape}")
print(f"cells: {cells.shape}")
print(f"Aves: {features_o3['bird_id'].nunique()}")
print(f"Filas válidas (is_observation_valid=True): {features_o3['is_observation_valid'].sum()}")
print(
    "Cobertura state_b en válidas: "
    f"{(features_o3.loc[features_o3['is_observation_valid'], 'state_b'].notna()).mean():.3f}"
)

# %% [markdown]
# ## Fase A.2 — Ejecutar `build_o4()`

# %%
import time

t0 = time.time()
result = build_o4(seed=0)
elapsed = time.time() - t0
print(f"build_o4 terminado en {elapsed/60:.1f} min")
print(result)

# %%
metrics = pd.read_parquet(result.metrics_path)
preds = pd.read_parquet(result.predictions_path)
print(f"metrics: {metrics.shape}")
print(f"predictions: {preds.shape}")
metrics = metrics.sort_values(["split", "modelo", "modo"]).reset_index(drop=True)
metrics

# %% [markdown]
# ## D1 — Configuración fija de hiperparámetros (§8.6)

# %%
d1_rows = [
    ("Random Forest", "n_estimators", "300", "Suficiente para varianza estable; coste lineal"),
    ("Random Forest", "max_depth", "12", "Corta el crecimiento; evita memorización a hojas únicas"),
    ("Random Forest", "min_samples_leaf", "10", "Hojas con ≥10 muestras → no memoriza filas sueltas"),
    ("XGBoost", "learning_rate", "0,05", "Tasa baja + early stopping para no saltarse el óptimo"),
    ("XGBoost", "max_depth", "6", "Profundidad moderada; clave anti-overfit en boosting"),
    ("XGBoost", "min_child_weight", "10", "Equivalente a min_samples_leaf en boosting"),
    ("XGBoost", "subsample", "0,8", "Bagging estocástico de filas"),
    ("XGBoost", "colsample_bytree", "0,8", "Bagging estocástico de columnas"),
    ("XGBoost", "n_estimators", "1000 + early stopping 50", "Se ajusta automático al problema"),
    ("XGBoost", "tree_method", "hist", "Acelera multiclase de alta cardinalidad"),
    ("LightGBM", "learning_rate", "0,05", "Idem XGBoost"),
    ("LightGBM", "num_leaves", "31", "Equivalente a max_depth en LightGBM"),
    ("LightGBM", "min_child_samples", "20", "Hojas con suficiente soporte"),
    ("LightGBM", "subsample / colsample_bytree", "0,8 / 0,8", "Idem XGBoost"),
    ("LightGBM", "n_estimators", "1000 + early stopping 50", "Idem"),
]
d1 = pd.DataFrame(d1_rows, columns=["familia", "hiperparámetro", "valor", "motivo"])
d1

# %%
save_artifact(
    slug="hyperparameter-config",
    objective="o4",
    num=1,
    decision="Configuración fija de hiperparámetros por familia (sin rejilla)",
    caption_es=(
        "Hiperparámetros conservadores fijados por familia para O4. La elección "
        "se basa en el consenso de literatura aplicada para datasets de tamaño "
        "moderado y target multiclase de alta cardinalidad. Se descarta una "
        "rejilla amplia porque (i) controlar el overfit y optimizar rendimiento "
        "son objetivos distintos: el primero se logra con hiperparámetros "
        "conservadores + early stopping + diagnóstico train-test (ver C1), "
        "mientras que el segundo introduce el riesgo de cherry-picking sobre la "
        "validación interna; y (ii) el coste de un barrido por familia (8-12 "
        "combinaciones) multiplicaría por 4-6 el tiempo de entrenamiento total "
        "sin garantía de mejora significativa para los seis modelos del estudio."
    ),
    table=d1,
)
