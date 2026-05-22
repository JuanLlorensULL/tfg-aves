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
    overwrite=True,
)

# %% [markdown]
# ## Fase B — Diagnóstico del overfit
# ### C1 — Gap train-test por modelo

# %%
gap = (
    metrics
    .pivot_table(index=["modelo", "modo"], columns="split",
                 values=["top1", "log_loss", "dist_median_km"])
    .reset_index()
)
gap.columns = [
    "_".join([str(c) for c in col if c]).strip("_") for col in gap.columns
]
gap["gap_top1"] = gap["top1_train"] - gap["top1_test"]
gap["gap_log_loss"] = gap["log_loss_test"] - gap["log_loss_train"]
gap_view = gap[gap["modelo"].isin(["rf", "xgb", "lgbm"])].copy()
gap_view.sort_values(["modo", "modelo"])

# %%
fig_c1, axes = plt.subplots(1, 2, figsize=(13, 4))
fams_order = ["rf", "xgb", "lgbm"]
modes_order = ["personalizado", "poblacional"]
x = np.arange(len(fams_order))
width = 0.35

for ax_idx, (col_train, col_test, title) in enumerate([
    ("top1_train", "top1_test", "Top-1 (mayor mejor)"),
    ("log_loss_train", "log_loss_test", "Log-loss (menor mejor)"),
]):
    ax = axes[ax_idx]
    for i, mode in enumerate(modes_order):
        sub = gap_view[gap_view["modo"] == mode].set_index("modelo")
        train_v = sub.loc[fams_order, col_train].to_numpy()
        test_v = sub.loc[fams_order, col_test].to_numpy()
        offset = (i - 0.5) * width
        ax.bar(x + offset - width / 4, train_v, width / 2, label=f"{mode} train", alpha=0.6)
        ax.bar(x + offset + width / 4, test_v, width / 2, label=f"{mode} test")
    ax.set_xticks(x); ax.set_xticklabels(fams_order)
    ax.set_title(title); ax.legend(fontsize=8)
fig_c1.suptitle("C1 — Diagnóstico de overfit: métricas train vs test por modelo y modo")
fig_c1.tight_layout()

save_artifact(
    slug="train-test-gap",
    objective="o4",
    num=2,
    decision="Diagnóstico del gap train-test para detectar overfit",
    caption_es=(
        "Comparativa de top-1 (panel izquierdo) y log-loss (panel derecho) "
        "evaluados sobre train y test para los seis modelos (tres familias × "
        "dos modos). La diferencia train→test es el indicador empírico del "
        "overfit: un gap pequeño sugiere que el modelo aprende patrones "
        "generalizables; un gap grande sugiere memorización. El modo "
        "personalizado (que incluye bird_id) es el más expuesto a "
        "memorización; su comparación contra el modo poblacional se analiza "
        "explícitamente en C7."
    ),
    fig=fig_c1,
    table=gap_view,
)

# %% [markdown]
# ### C2 — Curvas de aprendizaje (XGBoost y LightGBM)

# %%
import joblib

curves = {}
for mode in ["personalizado", "poblacional"]:
    for family in ["xgb", "lgbm"]:
        bundle = joblib.load(result.model_paths[f"{mode}_{family}"])
        model = bundle["model"]
        if family == "xgb":
            # _XGBoostWrapper expone el modelo en _xgb
            inner = model._xgb
            history = inner.evals_result()
            losses = history["validation_0"]["mlogloss"]
            best_iter = getattr(inner, "best_iteration", None)
        else:
            # _LightGBMWrapper expone el modelo en _lgbm
            inner = model._lgbm
            history = inner.evals_result_
            losses = history["valid_0"]["multi_logloss"]
            best_iter = getattr(inner, "best_iteration_", None)
        curves[(mode, family)] = (losses, best_iter)

# %%
fig_c2, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, family in zip(axes, ["xgb", "lgbm"]):
    for mode in ["personalizado", "poblacional"]:
        losses, best_iter = curves[(mode, family)]
        ax.plot(losses, label=mode)
        if best_iter is not None:
            ax.axvline(best_iter, linestyle="--", alpha=0.4)
    ax.set_xlabel("Iteración"); ax.set_ylabel("log-loss (val)")
    ax.set_title(family.upper()); ax.legend()
fig_c2.suptitle("C2 — Curvas de aprendizaje (val) — la línea punteada marca early stopping")
fig_c2.tight_layout()

save_artifact(
    slug="learning-curves",
    objective="o4",
    num=3,
    decision="Curvas de log-loss en validación interna para XGBoost y LightGBM",
    caption_es=(
        "Evolución del log-loss sobre la validación interna a lo largo de las "
        "iteraciones de boosting para XGBoost (izquierda) y LightGBM "
        "(derecha), en ambos modos. La línea vertical punteada marca la "
        "iteración óptima detectada por early stopping (paciencia 50). "
        "XGBoost presenta el comportamiento esperado: descenso monotónico del "
        "log-loss en validación hasta estabilizarse en torno a la iteración "
        "330-340, con el early stopping deteniendo el entrenamiento poco "
        "después. LightGBM, en contraste, diverge desde la primera "
        "iteración: el log-loss en validación arranca ya por encima del "
        "valor de una distribución uniforme sobre las clases activas y sube "
        "monotónicamente hasta estabilizarse alrededor de 30. El early "
        "stopping actúa correctamente como salvaguarda y detiene el "
        "entrenamiento en la iteración 1, pero el resultado revela que la "
        "configuración conservadora de LightGBM (§8.6) no extrae señal "
        "generalizable de este dataset con un target de ~849 clases activas "
        "fuertemente desbalanceadas. Este hallazgo se interpreta en C7 y se "
        "discute en la memoria §6: LightGBM con la configuración fijada "
        "queda descartado como modelo candidato, mientras que RF y XGBoost "
        "ofrecen comparativas significativas (ver C3 y C4)."
    ),
    fig=fig_c2,
    overwrite=True,
)
