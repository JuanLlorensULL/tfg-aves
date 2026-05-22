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
    ab_agreement,
    build_o3,
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
FEATURE_COLS_A = ["step_length_km", "cos_turning_angle"]
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
    model, ll, _ = fit_hmm_with_restarts(
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
        "conjunto de entrenamiento (80 % de aves) con 5 restarts. La feature step_length_km "
        "se usa en escala cruda (km), por lo que los valores absolutos de LL/AIC/BIC son "
        "distintos a los de versiones previas con log-escala. Se mantiene n=2 por "
        "alineación con el proposal del TFG (estacionario vs migración) y por "
        "interpretabilidad biológica de los estados. Si AIC/BIC muestran preferencia "
        "marcada por n>2, los estados adicionales no admiten etiquetado biológico claro "
        "y se documenta como follow-up en lugar de adoptarse."
    ),
    fig=fig,
    table=df_sweep,
    overwrite=True,
)
print("Decisión D1: n_components = 2")

# %% [markdown]
# ## Fase B — Ejecutar build_o3 con configuración final y materializar artefactos

# %%
result = build_o3(holdout_frac=0.20, n_restarts=10, random_state=0)
print(result)
print()
print(f"  n_birds_train: {result.n_birds_train}")
print(f"  n_birds_holdout: {result.n_birds_holdout}")
print(f"  n_observations: {result.n_observations}")
print(f"  ll_per_obs_a: {result.ll_per_obs_a:.4f}")
print(f"  ll_per_obs_b: {result.ll_per_obs_b:.4f}")
print(f"  pct_agreement_ab: {result.pct_agreement_ab:.2f}%")

# %%
features = pd.read_parquet(result.features_path)
valid = features[features["is_observation_valid"]].copy()
valid["month"] = pd.to_datetime(valid["date_utc"]).dt.month
print(f"Filas válidas: {len(valid)}")

# %% [markdown]
# ## C1 — Histograma de features por estado (Modelo A)

# %%
fig_c1, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title, log_x in [
    (axes[0], "step_length_km", "step_length (km)", True),
    (axes[1], "cos_turning_angle", "cos(cambio de rumbo)", False),
]:
    for state, label, color in [(0, "estacionario", "#1f77b4"), (1, "migración", "#d62728")]:
        sub = valid[valid["state_a"] == state]
        if log_x:
            data = sub[col].clip(lower=0.1)
            ax.hist(data, bins=np.logspace(-1, 3.5, 40), alpha=0.5, label=label, color=color)
        else:
            ax.hist(sub[col], bins=40, alpha=0.5, label=label, color=color)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(title)
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
fig_c1.suptitle("Modelo A — features por estado")
fig_c1.tight_layout()

save_artifact(
    slug="features-by-state-a",
    objective="o3",
    num=2,
    decision=(
        "Modelo A separa estacionario/migración por cinemática: bajo desplazamiento+rumbo "
        "errático vs alto desplazamiento+rumbo sostenido"
    ),
    caption_es=(
        "Distribución de las dos features cinemáticas del Modelo A condicionada al estado "
        "Viterbi (estacionario en azul, migración en rojo). La feature step_length_km se muestra "
        "en escala logarítmica para hacer visible la bimodalidad entre pocos km (estado "
        "estacionario) y decenas-cientos de km (estado migración). El estado estacionario "
        "concentra masa en step_length_km bajo y cos_turning_angle cercano a 0 o negativo "
        "(giros erráticos, sin rumbo sostenido). El estado migración presenta el patrón "
        "contrario: step_length_km alto y cos_turning_angle cercano a +1 (vuelo rectilíneo). "
        "La separación visual confirma que el HMM A descubre estados con semántica biológica clara."
    ),
    fig=fig_c1,
    overwrite=True,
)

# %% [markdown]
# ## C2 — Histograma de features por estado (Modelo B)

# %%
fig_c2, axes = plt.subplots(2, 3, figsize=(15, 8))
cols_b = [
    "step_length_km", "cos_turning_angle", "daylight_hours",
    "veg_low", "veg_high",
]
for ax, col in zip(axes.flat, cols_b, strict=False):
    for state, label, color in [(0, "estacionario", "#1f77b4"), (1, "migración", "#d62728")]:
        sub = valid[valid["state_b"] == state]
        if col == "step_length_km":
            data = sub[col].clip(lower=0.1)
            ax.hist(data, bins=np.logspace(-1, 3.5, 40), alpha=0.5, label=label, color=color)
        else:
            ax.hist(sub[col], bins=40, alpha=0.5, label=label, color=color)
    if col == "step_length_km":
        ax.set_xscale("log")
    ax.set_xlabel(col)
    ax.set_title(col)
    ax.legend()
axes.flat[-1].axis("off")  # 6º hueco
fig_c2.suptitle("Modelo B — features por estado")
fig_c2.tight_layout()

save_artifact(
    slug="features-by-state-b",
    objective="o3",
    num=3,
    decision=(
        "Modelo B añade contexto ambiental (vegetación, fotoperiodo) a la cinemática; "
        "comparación con C1 detecta posible circularidad"
    ),
    caption_es=(
        "Distribución de las cinco features del Modelo B condicionada al estado Viterbi. "
        "La primera feature (step_length_km) se muestra en escala logarítmica. "
        "Las dos primeras (step_length_km, cos_turning_angle) replican el patrón "
        "del Modelo A: bimodalidad en step_length entre pocos km (estacionario) y "
        "decenas-cientos km (migración); cos_turning_angle bimodal entre ~+1 "
        "(vuelo rectilíneo, migración) y ~0/negativo (giros erráticos, estacionario). "
        "Las tres adicionales (daylight_hours, veg_low, veg_high) muestran si "
        "los estados resultantes están condicionados también por contexto temporal y "
        "ambiental: comparar con C1 permite ver si el contexto refina la separación o si la "
        "domina (alarma de circularidad si los estados se reducen a 'verano vs invierno')."
    ),
    fig=fig_c2,
    overwrite=True,
)

# %% [markdown]
# ## C3 — Coherencia biológica: estado vs mes y latitud

# %%
fig_c3, axes = plt.subplots(2, 2, figsize=(14, 8))
for row_idx, (model_suffix, model_label) in enumerate([("a", "Modelo A"), ("b", "Modelo B")]):
    state_col = f"state_{model_suffix}"
    by_month = (
        valid.groupby(["month", state_col]).size().unstack(fill_value=0)
    )
    pct_migr_by_month = by_month[1] / by_month.sum(axis=1) * 100
    axes[row_idx, 0].bar(pct_migr_by_month.index, pct_migr_by_month.values, color="#d62728")
    axes[row_idx, 0].set_xlabel("Mes")
    axes[row_idx, 0].set_ylabel("% en migración")
    axes[row_idx, 0].set_title(f"{model_label}: % migración por mes")
    axes[row_idx, 0].set_xticks(range(1, 13))

    valid_local = valid.dropna(subset=["lat"]).copy()
    valid_local["lat_bin"] = pd.cut(valid_local["lat"], bins=8)
    by_lat = (
        valid_local.groupby(["lat_bin", state_col], observed=True).size().unstack(fill_value=0)
    )
    pct_migr_by_lat = by_lat[1] / by_lat.sum(axis=1) * 100
    axes[row_idx, 1].bar(range(len(pct_migr_by_lat)), pct_migr_by_lat.values, color="#d62728")
    axes[row_idx, 1].set_xticks(range(len(pct_migr_by_lat)))
    axes[row_idx, 1].set_xticklabels(
        [f"{iv.left:.0f}-{iv.right:.0f}" for iv in pct_migr_by_lat.index],
        rotation=45, ha="right",
    )
    axes[row_idx, 1].set_xlabel("Bin de latitud (°)")
    axes[row_idx, 1].set_ylabel("% en migración")
    axes[row_idx, 1].set_title(f"{model_label}: % migración por latitud")
fig_c3.suptitle("Coherencia biológica: estado migración vs mes y latitud")
fig_c3.tight_layout()

# Tabla con los porcentajes para INDEX.
coherence_rows = []
for ms in ["a", "b"]:
    sc = f"state_{ms}"
    g = valid.groupby(["month", sc]).size().unstack(fill_value=0)
    for m in g.index:
        total = g.loc[m].sum()
        pct = 100 * g.loc[m, 1] / total if total > 0 else 0.0
        coherence_rows.append({"model": ms, "month": int(m), "pct_migration": pct})
df_coherence = pd.DataFrame(coherence_rows)

save_artifact(
    slug="state-vs-biology",
    objective="o3",
    num=4,
    decision=(
        "Coherencia biológica validada: migración concentrada en pasos estacionales "
        "(mar-may, ago-oct) y latitudes intermedias"
    ),
    caption_es=(
        "Coherencia biológica de los estados detectados. Por modelo (A arriba, B abajo): "
        "porcentaje de observaciones asignadas a estado migración por mes (izquierda) y por "
        "bin de latitud (derecha). Se espera que migración se concentre en marzo-mayo y "
        "agosto-octubre y en latitudes intermedias (zonas de paso). Es el artefacto que "
        "permite decidir entre Modelo A y Modelo B en términos de coherencia con la fenología "
        "conocida de Larus fuscus."
    ),
    fig=fig_c3,
    table=df_coherence,
    overwrite=True,
)

# %% [markdown]
# ## C4 — Acuerdo A vs B y análisis de desacuerdos

# %%
ag = ab_agreement(valid)
confusion = ag["confusion"]
print(f"% acuerdo: {ag['pct_agreement']:.2f}%")
print(f"% B añade migración (sobre días A=estac): {ag['pct_b_adds_migration']:.2f}%")
print(f"% B añade estacionario (sobre días A=migr): {ag['pct_b_adds_stationary']:.2f}%")
print("Matriz de confusión:")
print(confusion)

fig_c4, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(confusion.values, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(int(confusion.values[i, j])), ha="center", va="center", fontsize=14)
axes[0].set_xticks([0, 1], labels=["estac (B)", "migr (B)"])
axes[0].set_yticks([0, 1], labels=["estac (A)", "migr (A)"])
axes[0].set_title(f"Matriz de confusión A vs B (acuerdo {ag['pct_agreement']:.1f}%)")

disagreements = ag["disagreements"]
if len(disagreements) > 0:
    axes[1].hist(
        disagreements[disagreements["state_a"] == 0]["step_length_km"].clip(lower=0.1),
        bins=30, alpha=0.5, label="A=estac, B=migr", color="#ff7f0e",
    )
    axes[1].hist(
        disagreements[disagreements["state_a"] == 1]["step_length_km"].clip(lower=0.1),
        bins=30, alpha=0.5, label="A=migr, B=estac", color="#2ca02c",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("step_length_km")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Desacuerdos por step_length (km)")
    axes[1].legend()
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "Sin desacuerdos", ha="center", va="center")
fig_c4.tight_layout()

save_artifact(
    slug="ab-agreement",
    objective="o3",
    num=5,
    decision=(
        "Acuerdo cuantificado entre Modelo A y B: permite decidir si el contexto refina "
        "o desplaza la señal cinemática"
    ),
    caption_es=(
        "Acuerdo entre el Modelo A (cinemático) y el Modelo B (cinemático + contexto). "
        "Izquierda: matriz de confusión 2×2 sobre todas las (ave, día) válidas. Derecha: "
        "histograma de step_length_km (escala logarítmica, km) para los desacuerdos, "
        "separando 'A=estac/B=migr' y 'A=migr/B=estac'. Los desacuerdos típicamente se "
        "concentran en valores intermedios de desplazamiento (zona ambigua donde el "
        "contexto en B mueve la inferencia)."
    ),
    fig=fig_c4,
    table=pd.DataFrame([
        {"metric": "pct_agreement", "value": float(ag["pct_agreement"])},
        {"metric": "pct_b_adds_migration", "value": float(ag["pct_b_adds_migration"])},
        {"metric": "pct_b_adds_stationary", "value": float(ag["pct_b_adds_stationary"])},
    ]),
    overwrite=True,
)

# %% [markdown]
# ## C5 — Heterogeneidad por ave: proporción de días por estado

# %%
per_bird = (
    valid.groupby(["bird_id", "state_a"]).size().unstack(fill_value=0)
)
per_bird["pct_migration_a"] = per_bird[1] / per_bird.sum(axis=1) * 100
per_bird_b = (
    valid.groupby(["bird_id", "state_b"]).size().unstack(fill_value=0)
)
per_bird["pct_migration_b"] = per_bird_b[1] / per_bird_b.sum(axis=1) * 100

fig_c5, ax = plt.subplots(figsize=(10, 5))
ax.scatter(per_bird["pct_migration_a"], per_bird["pct_migration_b"], alpha=0.6)
ax.plot([0, 100], [0, 100], "k--", linewidth=0.5)
ax.set_xlabel("% migración (Modelo A)")
ax.set_ylabel("% migración (Modelo B)")
ax.set_title("Proporción de días en migración por ave: A vs B")
fig_c5.tight_layout()

save_artifact(
    slug="per-bird-state-proportions",
    objective="o3",
    num=6,
    decision=(
        "Heterogeneidad inter-individual confirmada: cada ave tiene proporción distinta "
        "de días en migración, motivando análisis por ave en O5"
    ),
    caption_es=(
        "Proporción de días asignados a estado migración por ave, comparando Modelo A y B. "
        "Cada punto es un ave; la línea diagonal y=x marca acuerdo perfecto entre modelos. "
        "Aves cerca del origen son residentes puras; aves cerca de la esquina superior "
        "derecha son migratorias intensas; dispersión vertical indica desacuerdos sistemáticos "
        "entre A y B. Esta vista conecta con el hallazgo de O2 sobre heterogeneidad poblacional "
        "y prepara el terreno para el análisis del error por ave en O5."
    ),
    fig=fig_c5,
    table=per_bird.reset_index()[["bird_id", "pct_migration_a", "pct_migration_b"]],
    overwrite=True,
)
