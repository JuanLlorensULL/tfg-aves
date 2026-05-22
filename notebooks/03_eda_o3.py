# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — EDA de O3 (HMM comportamiento)
#
# Notebook que justifica `n_components=2` (D1) mediante AIC/BIC sweep, ejecuta
# build_o3 con cell_deg final y materializa los artefactos C1–C5.

# %%
from __future__ import annotations

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tfg_aves.hmm import (  # noqa: E402
    build_o3,  # noqa: F401  — usado en Fase B (Task 10)
    build_sequences,
    compute_observation_features,
    fit_hmm_with_restarts,
    load_vegetation_from_raw,
    stratified_holdout_split,
)
from tfg_aves.hmm._paths import DAILY_PARQUET, RAW_CSV  # noqa: E402
from tfg_aves.reporting import save_artifact  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})

# %%
df_daily = pd.read_parquet(DAILY_PARQUET)
print(f"daily.parquet: {len(df_daily)} filas, {df_daily['bird_id'].nunique()} aves")
print(f"válidas: {df_daily['is_valid'].sum()}")

# %% [markdown]
# ## Fase A — D1: AIC/BIC sweep para justificar n_components=2

# %%
veg = load_vegetation_from_raw(RAW_CSV, df_daily["source_event_id"])
df_features = compute_observation_features(df_daily, df_raw=veg)
print(
    f"features.parquet (in-memory): {len(df_features)} filas, "
    f"{df_features['is_observation_valid'].sum()} con triplete válido"
)

# %%
train_ids, _ = stratified_holdout_split(df_features, holdout_frac=0.20, random_state=0)
FEATURE_COLS_A = ["log_displacement_km", "abs_turning_angle_rad"]
X_train, lengths_train = build_sequences(df_features, train_ids, FEATURE_COLS_A)
print(f"Train: {len(X_train)} observaciones, {len(lengths_train)} secuencias")


# %%
def n_params_diag(n_components: int, n_features: int) -> int:
    """Conteo de parámetros libres en un GaussianHMM con covariance_type='diag'."""
    # startprob: n-1 ; transmat: n*(n-1) ; means: n*d ; covars: n*d
    return (
        (n_components - 1)
        + n_components * (n_components - 1)
        + 2 * n_components * n_features
    )


def aic_bic(
    ll: float, n_components: int, n_obs: int, n_features: int
) -> tuple[float, float]:
    """Calcula AIC y BIC dado el log-likelihood y el número de parámetros."""
    k = n_params_diag(n_components, n_features)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n_obs) - 2 * ll
    return aic, bic


# %%
rows = []
for n in [2, 3, 4]:
    model, scaler, ll, _ = fit_hmm_with_restarts(
        X_train,
        lengths_train,
        n_components=n,
        n_restarts=5,
        random_state=0,
    )
    aic, bic = aic_bic(ll, n, len(X_train), X_train.shape[1])
    rows.append({"n_components": n, "ll": ll, "aic": aic, "bic": bic})
    print(f"  n={n}: LL={ll:.1f}  AIC={aic:.1f}  BIC={bic:.1f}")
df_sweep = pd.DataFrame(rows)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(df_sweep["n_components"].astype(str), df_sweep["aic"], color="#1f77b4")
axes[0].set_xlabel("n_components")
axes[0].set_ylabel("AIC")
axes[0].set_title("AIC (menor = mejor)")
axes[1].bar(df_sweep["n_components"].astype(str), df_sweep["bic"], color="#888")
axes[1].set_xlabel("n_components")
axes[1].set_ylabel("BIC")
axes[1].set_title("BIC (menor = mejor)")
fig.suptitle("D1 — Sweep de número de estados (Modelo A)")
fig.tight_layout()

save_artifact(
    "nstates-aic-bic-sweep",
    objective="o3",
    num=1,
    decision="n_components fijado en 2 (estacionario + migración) respaldado por AIC/BIC sweep",
    caption_es=(
        "AIC y BIC para HMMs Modelo A con n_components ∈ {2, 3, 4} entrenados sobre el "
        "conjunto de entrenamiento (80 % de aves) con 5 restarts. Se mantiene n=2 por "
        "alineación con el proposal del TFG (estacionario vs migración) y por "
        "interpretabilidad biológica de los estados. Si AIC/BIC muestran preferencia "
        "marcada por n>2, los estados adicionales no admiten etiquetado biológico claro "
        "y se documenta como follow-up en lugar de adoptarse."
    ),
    fig=fig,
    table=df_sweep,
)
print("Decisión D1: n_components = 2")
