# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # O4 · L3 — Regresión espacial con cuantiles
#
# Ejecuta `build_o4_l3` (modo poblacional + individual 91916A) y genera los
# 6 artefactos (D1, C1..C5). Ataca Mo1 (target categórico) y D3 (rutas
# individuales). Baseline categórico = O4 monolítico causal poblacional.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V1_DIR, O4_OUT_DIR
from tfg_aves.ml.build_l3 import _prepare_poblacional_split, build_o4_l3
from tfg_aves.ml.quantile import INDIVIDUAL_BIRD_ID, derive_displacement_target
from tfg_aves.reporting import save_artifact

result = build_o4_l3()
print("filas pob train/test:", result.n_rows_train_pob, result.n_rows_test_pob)
print("filas ind train/test:", result.n_rows_train_ind, result.n_rows_test_ind)
print("cobertura:", result.coverage)
print("cruces de cuantil:", result.n_crossings)

metrics = pd.read_parquet(O4_L3V1_DIR / "metrics.parquet")
preds = pd.read_parquet(O4_L3V1_DIR / "predictions_test.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")  # O4 causal (L3-v0)

# Splits reales para D1 (distribución del target y conteo de histórico por ave).
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
train_pob, _, _ = _prepare_poblacional_split(features_o3, cells, seed=0)

# %% [markdown]
# ## D1 — Distribución del target y selección del individuo
# %%
tgt = derive_displacement_target(train_pob)
hist = (
    train_pob.groupby("bird_id").size().sort_values(ascending=False)
    .head(8).rename("n_filas_train").reset_index()
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(tgt["y_dlat"], bins=80, alpha=0.7, label="Δlat")
axes[0].hist(tgt["y_dlon"], bins=80, alpha=0.7, label="Δlon")
axes[0].set_yscale("log")
axes[0].set_xlabel("desplazamiento (grados)"); axes[0].set_ylabel("frecuencia (log)")
axes[0].set_title("Distribución del target Δ"); axes[0].legend()
axes[1].barh(hist["bird_id"][::-1], hist["n_filas_train"][::-1], color="steelblue")
axes[1].set_xlabel("filas de entrenamiento")
axes[1].set_title(f"Histórico por ave (top 8) — individual: {INDIVIDUAL_BIRD_ID}")
fig.tight_layout()
save_artifact(
    "target-dist-bird-history", objective="o4", num=24,
    decision=("Target continuo concentrado en cero (justifica cuantiles + pinball); "
              f"{INDIVIDUAL_BIRD_ID} es el ave con más histórico (modo individual)."),
    caption_es=(
        "Izquierda: distribución del desplazamiento diario (Δlat, Δlon) en escala "
        "logarítmica, fuertemente concentrada en cero por el dominio de días "
        "estacionarios, con colas de migración. Justifica modelar tres cuantiles "
        "{p10, p50, p90} y evaluar con pérdida pinball. Derecha: número de filas de "
        f"entrenamiento por ave; {INDIVIDUAL_BIRD_ID} encabeza el histórico y se elige "
        "para el modelo individual de L3."),
    fig=fig, table=hist, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C1 — Calibración de la incertidumbre
# %%
cov_rows = []
for mode in ("poblacional", "individual"):
    sub = preds[preds["modo"] == mode]
    cov_rows.append({"modo": mode,
                     "cobertura_lat": float(sub["in_interval_lat"].mean()),
                     "cobertura_lon": float(sub["in_interval_lon"].mean())})
cov_df = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(cov_df)); w = 0.35
ax.bar(x - w / 2, cov_df["cobertura_lat"], w, label="Δlat")
ax.bar(x + w / 2, cov_df["cobertura_lon"], w, label="Δlon")
ax.axhline(0.80, color="red", ls="--", label="nominal 80%")
ax.set_xticks(x); ax.set_xticklabels(cov_df["modo"])
ax.set_ylabel("cobertura empírica de [p10, p90]"); ax.set_ylim(0, 1)
ax.set_title("Calibración del intervalo de incertidumbre"); ax.legend()
fig.tight_layout()
save_artifact(
    "calibration-coverage", objective="o4", num=25,
    decision="Cobertura empírica del intervalo [p10,p90] cercana al 80% nominal.",
    caption_es=(
        "Cobertura empírica del intervalo de predicción [p10, p90] frente al 80% "
        "nominal, por modo y eje. Una cobertura próxima al 80% indica que la "
        "incertidumbre del desplazamiento está bien calibrada, contribución "
        "diferencial de la regresión de cuantiles frente al clasificador."),
    fig=fig, table=cov_df, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C2 — Distribución de distancias (ablación de target, poblacional)
# %%
preds_pob = preds[preds["modo"] == "poblacional"]
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
v0_xgb_pob = preds_v0[(preds_v0["modelo"] == "xgb") & (preds_v0["modo"] == "poblacional")]
v0_pers = preds_v0[preds_v0["modelo"] == "persistencia"]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 300, 60)
ax.hist(preds_pob["pred_dist_km"], bins=bins, alpha=0.6, density=True, label="L3 cuantil")
ax.hist(v0_xgb_pob["pred_dist_km"], bins=bins, alpha=0.6, density=True, label="L3-v0 categórico")
ax.hist(v0_pers["pred_dist_km"], bins=bins, alpha=0.4, density=True, label="persistencia")
ax.set_xlabel("distancia vía centroide (km)"); ax.set_ylabel("densidad")
ax.set_title("Distribución de distancias en test (poblacional)"); ax.legend()
fig.tight_layout()
dist_tbl = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["modo", "scope", "dist_centroide_km", "dist_nativa_km"]]
save_artifact(
    "distance-distribution", objective="o4", num=26,
    decision="Comparación de la distancia del error: L3 cuantil vs categórico vs persistencia.",
    caption_es=(
        "Distribución de la distancia (vía centroide) entre la predicción y la "
        "posición real en el conjunto de test (modo poblacional), comparando la "
        "regresión de cuantiles de L3, el clasificador categórico L3-v0 y la "
        "persistencia trivial. La ventaja de la regresión, si existe, se concentra "
        "en los días de movimiento (cola de la distribución)."),
    fig=fig, table=dist_tbl, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C3 — Comparativas centrales (A: ablación de target; B: per-individuo)
# %%
tabla_a = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["modo", "scope", "n_obs", "top1", "top3", "dist_centroide_km",
     "dist_nativa_km", "pinball_lat", "pinball_lon", "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
ind_id = result.individual_bird_id
tabla_b = metrics[metrics["modo"].isin(
    ["individual", f"poblacional@{ind_id}", f"persistencia@{ind_id}"])][
    ["modo", "scope", "n_obs", "top1", "dist_centroide_km", "dist_nativa_km",
     "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
tabla_comparativa = pd.concat([
    tabla_a.assign(bloque="A_target_poblacional"),
    tabla_b.assign(bloque="B_per_individuo"),
], ignore_index=True)
print(tabla_comparativa.to_string())
save_artifact(
    "comparativa-l3", objective="o4", num=27,
    decision=("L3 cuantil vs categórico (A, target) y individual vs poblacional "
              "sobre 91916A (B, per-individuo)."),
    caption_es=(
        "Comparativas centrales de L3. Bloque A: efecto de reformular el target "
        "(categórico L3-v0 vs cuantil L3-v1) en el modo poblacional, sobre todo el "
        "test, por estado HMM y en días de movimiento. Bloque B: hipótesis "
        "per-individuo, modelo individual de 91916A frente al poblacional evaluado "
        "sobre las mismas filas. Sin columna log-loss: L3 no produce una "
        "distribución categórica, su veredicto es geométrico (distancia + top-1 "
        "mapeado)."),
    table=tabla_comparativa, overwrite=True,
)

# %% [markdown]
# ## C4 — Vectores de desplazamiento predichos vs reales (91916A)
# %%
ind = preds[preds["modo"] == "individual"].copy().sort_values("date_utc").reset_index(drop=True)
sample = ind.iloc[:: max(1, len(ind) // 40)].copy()  # ~40 días para legibilidad
fig, ax = plt.subplots(figsize=(7, 7))
# Real (gris) vs predicho p50 (azul) desde la posición de t.
true_lat = sample["pred_lat"] - (sample["dlat_p50"])  # = lat_t
true_lon = sample["pred_lon"] - (sample["dlon_p50"])  # = lon_t
ax.quiver(true_lon, true_lat,
          sample["pred_lon"] - true_lon, sample["pred_lat"] - true_lat,
          angles="xy", scale_units="xy", scale=1, color="steelblue",
          width=0.004, label="predicho p50")
# Banda de incertidumbre: rango p10-p90 como segmento en cada eje.
ax.errorbar(sample["pred_lon"], sample["pred_lat"],
            xerr=[(sample["dlon_p50"] - sample["dlon_p10"]).abs(),
                  (sample["dlon_p90"] - sample["dlon_p50"]).abs()],
            yerr=[(sample["dlat_p50"] - sample["dlat_p10"]).abs(),
                  (sample["dlat_p90"] - sample["dlat_p50"]).abs()],
            fmt="none", ecolor="orange", alpha=0.5, label="banda [p10,p90]")
ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
ax.set_title(f"Desplazamientos predichos de {result.individual_bird_id} (muestra)")
ax.legend()
fig.tight_layout()
save_artifact(
    "vectores-desplazamiento-91916a", objective="o4", num=28,
    decision="Vectores de desplazamiento p50 del modelo individual con banda de incertidumbre.",
    caption_es=(
        f"Vectores de desplazamiento diario predichos (mediana p50) por el modelo "
        f"individual de {result.individual_bird_id} sobre una muestra de su test, con "
        "la banda de incertidumbre [p10, p90] por eje. Ilustra la salida geométrica "
        "e interpretable de la regresión de cuantiles, no disponible en el "
        "clasificador categórico."),
    fig=fig, table=sample[["date_utc", "pred_lat", "pred_lon",
                           "dlat_p10", "dlat_p90", "dlon_p10", "dlon_p90"]],
    overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C5 — Incidencia de quantile crossing
# %%
cross_rows = []
for mode, d in result.n_crossings.items():
    n_test = result.n_rows_test_pob if mode == "poblacional" else result.n_rows_test_ind
    cross_rows.append({"modo": mode, "n_test": n_test,
                       "cruces_lat": d["lat"], "cruces_lon": d["lon"],
                       "pct_lat": 100 * d["lat"] / max(1, n_test),
                       "pct_lon": 100 * d["lon"] / max(1, n_test)})
cross_df = pd.DataFrame(cross_rows)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(cross_df)); w = 0.35
ax.bar(x - w / 2, cross_df["pct_lat"], w, label="Δlat")
ax.bar(x + w / 2, cross_df["pct_lon"], w, label="Δlon")
ax.set_xticks(x); ax.set_xticklabels(cross_df["modo"])
ax.set_ylabel("% de filas con cruce (antes de ordenar)")
ax.set_title("Incidencia de quantile crossing"); ax.legend()
fig.tight_layout()
save_artifact(
    "quantile-crossing", objective="o4", num=29,
    decision="Incidencia de quantile crossing corregida por ordenación post-hoc.",
    caption_es=(
        "Porcentaje de filas donde los cuantiles predichos se cruzan (p10>p50 o "
        "p50>p90) antes de la corrección monótona post-hoc, por modo y eje. Una "
        "incidencia baja confirma que los regresores independientes producen "
        "cuantiles coherentes; la ordenación garantiza monotonía en todo caso."),
    fig=fig, table=cross_df, overwrite=True,
)
plt.close(fig)
print("Artefactos L3 generados (D1, C1..C5).")
