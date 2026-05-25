# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # O4 · L3 — Regresión espacial con cuantiles (multi-familia)
#
# Ejecuta `build_o4_l3` con las tres familias del proposal (XGBoost, LightGBM y
# Random Forest). XGBoost conserva sus modos poblacional + individual 91916A;
# LightGBM y RF solo poblacional. Genera D1, C1..C5 (C1/C2/C5 extendidos a las
# tres familias) y la tabla maestra de familias (tab31).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.ml._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L3V2_DIR, O4_OUT_DIR
from tfg_aves.ml.build_l3 import _prepare_poblacional_split, build_o4_l3
from tfg_aves.ml.quantile import INDIVIDUAL_BIRD_ID, derive_displacement_target
from tfg_aves.reporting import save_artifact

_FAMILIES = ["xgb", "lgbm", "rf"]
_FAM_LABEL = {"xgb": "XGBoost", "lgbm": "LightGBM", "rf": "Random Forest"}

result = build_o4_l3()
print("filas pob train/test:", result.n_rows_train_pob, result.n_rows_test_pob)
print("filas ind train/test:", result.n_rows_train_ind, result.n_rows_test_ind)
print("cobertura:", result.coverage)
print("cruces de cuantil:", result.n_crossings)

metrics = pd.read_parquet(O4_L3V2_DIR / "metrics.parquet")
preds = pd.read_parquet(O4_L3V2_DIR / "predictions_test.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")  # O4 causal (L3-v0)

# Splits reales para D1 (distribución del target y conteo de histórico por ave).
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
train_pob, _, _ = _prepare_poblacional_split(features_o3, cells)

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
# ## C1 — Calibración de la incertidumbre (tres familias, poblacional)
# %%
cov_rows = []
for fam in _FAMILIES:
    sub = preds[(preds["familia"] == fam) & (preds["modo"] == "poblacional")]
    cov_rows.append({"familia": fam,
                     "cobertura_lat": float(sub["in_interval_lat"].mean()),
                     "cobertura_lon": float(sub["in_interval_lon"].mean())})
cov_df = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(cov_df)); w = 0.35
ax.bar(x - w / 2, cov_df["cobertura_lat"], w, label="Δlat")
ax.bar(x + w / 2, cov_df["cobertura_lon"], w, label="Δlon")
ax.axhline(0.80, color="red", ls="--", label="nominal 80%")
ax.set_xticks(x); ax.set_xticklabels([_FAM_LABEL[f] for f in cov_df["familia"]])
ax.set_ylabel("cobertura empírica de [p10, p90]"); ax.set_ylim(0, 1)
ax.set_title("Calibración del intervalo por familia (poblacional)"); ax.legend()
fig.tight_layout()
save_artifact(
    "calibration-coverage", objective="o4", num=25,
    decision="Cobertura empírica de [p10,p90] cerca del 80% nominal en las tres familias.",
    caption_es=(
        "Cobertura empírica del intervalo de predicción [p10, p90] frente al 80% "
        "nominal, por familia (XGBoost, LightGBM, Random Forest) y eje, en modo "
        "poblacional. Una cobertura próxima al 80% indica incertidumbre del "
        "desplazamiento bien calibrada; permite comparar la calibración del boosting "
        "(pinball) frente al QRF de Random Forest (cuantiles de hojas)."),
    fig=fig, table=cov_df, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C2 — Distribución de distancias (tres familias vs persistencia, poblacional)
# %%
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
v0_pers = preds_v0[preds_v0["modelo"] == "persistencia"]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 300, 60)
for fam in _FAMILIES:
    sub = preds[(preds["familia"] == fam) & (preds["modo"] == "poblacional")]
    ax.hist(sub["pred_dist_km"], bins=bins, density=True, histtype="step",
            linewidth=1.6, label=f"L3 {_FAM_LABEL[fam]}")
ax.hist(v0_pers["pred_dist_km"], bins=bins, alpha=0.35, density=True, label="persistencia")
ax.set_xlabel("distancia vía centroide (km)"); ax.set_ylabel("densidad")
ax.set_title("Distribución de distancias en test (poblacional)"); ax.legend()
fig.tight_layout()
dist_tbl = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["familia", "modo", "scope", "dist_centroide_km", "dist_nativa_km"]].reset_index(drop=True)
save_artifact(
    "distance-distribution", objective="o4", num=26,
    decision="Comparación de la distancia del error de las tres familias vs persistencia.",
    caption_es=(
        "Distribución de la distancia (vía centroide) entre la predicción puntual "
        "(p50) y la posición real en test (modo poblacional), para las tres familias "
        "de L3 y la persistencia trivial. Las diferencias, si existen, se concentran "
        "en los días de movimiento (cola de la distribución)."),
    fig=fig, table=dist_tbl, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C3 — Comparativas centrales XGBoost (A: ablación de target; B: per-individuo)
# %%
xgb_pob = (metrics["familia"] == "xgb") & (metrics["modo"] == "poblacional")
pers = metrics["modo"] == "persistencia"
tabla_a = metrics[xgb_pob | pers][
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
              "sobre 91916A (B, per-individuo), familia XGBoost."),
    caption_es=(
        "Comparativas centrales de L3 (XGBoost). Bloque A: efecto de reformular el "
        "target (categórico L3-v0 vs cuantil L3-v1) en modo poblacional, por estado "
        "HMM y en días de movimiento. Bloque B: hipótesis per-individuo, modelo "
        "individual de 91916A frente al poblacional sobre las mismas filas. Sin "
        "columna log-loss: L3 no produce distribución categórica, su veredicto es "
        "geométrico (distancia + top-1 mapeado)."),
    table=tabla_comparativa, overwrite=True,
)

# %% [markdown]
# ## C4 — Vectores de desplazamiento predichos vs reales (91916A, XGBoost)
# %%
ind = preds[(preds["familia"] == "xgb") & (preds["modo"] == "individual")].copy()
ind = ind.sort_values("date_utc").reset_index(drop=True)
sample = ind.iloc[:: max(1, len(ind) // 40)].copy()  # ~40 días para legibilidad
fig, ax = plt.subplots(figsize=(7, 7))
true_lat = sample["pred_lat"] - sample["dlat_p50"]  # = lat_t
true_lon = sample["pred_lon"] - sample["dlon_p50"]  # = lon_t
ax.quiver(true_lon, true_lat,
          sample["pred_lon"] - true_lon, sample["pred_lat"] - true_lat,
          angles="xy", scale_units="xy", scale=1, color="steelblue",
          width=0.004, label="predicho p50")
ax.errorbar(sample["pred_lon"], sample["pred_lat"],
            xerr=[(sample["dlon_p50"] - sample["dlon_p10"]).abs(),
                  (sample["dlon_p90"] - sample["dlon_p50"]).abs()],
            yerr=[(sample["dlat_p50"] - sample["dlat_p10"]).abs(),
                  (sample["dlat_p90"] - sample["dlat_p50"]).abs()],
            fmt="none", ecolor="orange", alpha=0.5, label="banda [p10,p90]")
ax.set_xlabel("longitud"); ax.set_ylabel("latitud")
ax.set_title(f"Desplazamientos predichos de {ind_id} (muestra, XGBoost)")
ax.legend()
fig.tight_layout()
save_artifact(
    "vectores-desplazamiento-91916a", objective="o4", num=28,
    decision="Vectores de desplazamiento p50 del modelo individual con banda de incertidumbre.",
    caption_es=(
        f"Vectores de desplazamiento diario predichos (mediana p50) por el modelo "
        f"individual XGBoost de {ind_id} sobre una muestra de su test, con la banda "
        "de incertidumbre [p10, p90] por eje. Ilustra la salida geométrica e "
        "interpretable de la regresión de cuantiles, no disponible en el clasificador."),
    fig=fig, table=sample[["date_utc", "pred_lat", "pred_lon",
                           "dlat_p10", "dlat_p90", "dlon_p10", "dlon_p90"]],
    overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## C5 — Incidencia de quantile crossing por familia (RF = 0 por construcción)
# %%
cross_rows = []
for key, d in result.n_crossings.items():       # key = "{familia}_{modo}"
    family, mode = key.rsplit("_", 1)
    n_test = result.n_rows_test_pob if mode == "poblacional" else result.n_rows_test_ind
    cross_rows.append({
        "familia_modo": f"{_FAM_LABEL[family]}·{mode}", "n_test": n_test,
        "cruces_lat": d["lat"], "cruces_lon": d["lon"],
        "pct_lat": 100 * d["lat"] / max(1, n_test),
        "pct_lon": 100 * d["lon"] / max(1, n_test),
    })
cross_df = pd.DataFrame(cross_rows)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(cross_df)); w = 0.35
ax.bar(x - w / 2, cross_df["pct_lat"], w, label="Δlat")
ax.bar(x + w / 2, cross_df["pct_lon"], w, label="Δlon")
ax.set_xticks(x); ax.set_xticklabels(cross_df["familia_modo"], rotation=30, ha="right")
ax.set_ylabel("% de filas con cruce (antes de ordenar)")
ax.set_title("Incidencia de quantile crossing por familia"); ax.legend()
fig.tight_layout()
save_artifact(
    "quantile-crossing", objective="o4", num=29,
    decision="Cruces de cuantil por familia: boosting (XGB/LGBM) > 0 corregidos; RF = 0.",
    caption_es=(
        "Porcentaje de filas donde los cuantiles predichos se cruzan (p10>p50 o "
        "p50>p90) antes de la corrección monótona post-hoc, por familia, modo y eje. "
        "XGBoost y LightGBM entrenan un modelo independiente por cuantil y pueden "
        "cruzarse (se corrige con ordenación); Random Forest (QRF) obtiene los tres "
        "cuantiles de la misma distribución de hojas y es monótono por construcción "
        "(cero cruces)."),
    fig=fig, table=cross_df, overwrite=True,
)
plt.close(fig)

# %% [markdown]
# ## tab31 — Tabla maestra: comparativa de las tres familias (poblacional)
# %%
maestra = metrics[metrics["modo"].isin(["poblacional", "persistencia"])][
    ["familia", "modo", "scope", "n_obs", "top1", "top3", "dist_centroide_km",
     "dist_nativa_km", "pinball_lat", "pinball_lon", "coverage_lat", "coverage_lon"]
].reset_index(drop=True)
cruces_pob = {key.rsplit("_", 1)[0]: d["lat"] + d["lon"]
              for key, d in result.n_crossings.items() if key.endswith("poblacional")}
maestra["cruces_total"] = [
    cruces_pob.get(r["familia"]) if (r["scope"] == "global" and r["modo"] == "poblacional")
    else np.nan
    for _, r in maestra.iterrows()
]
maestra = maestra.sort_values(["scope", "familia", "modo"]).reset_index(drop=True)
print(maestra.to_string())
save_artifact(
    "comparativa-familias-l3", objective="o4", num=31,
    decision=("Comparativa de las tres familias supervisadas (XGBoost, LightGBM, "
              "Random Forest) en la regresión de cuantiles, modo poblacional."),
    caption_es=(
        "Comparativa maestra de las tres familias del proposal sobre la tarea de "
        "regresión de cuantiles del desplazamiento (modo poblacional, test completo), "
        "por régimen (global, estacionario, migración, días de movimiento). Métricas: "
        "top-1/top-3 mapeados, distancia vía centroide y nativa, pérdida pinball y "
        "cobertura [p10,p90] por eje, y nº total de cruces de cuantil corregidos "
        "(cero en Random Forest por construcción). Baseline: persistencia trivial. "
        "Cierra la comparativa de tres familias que O4 base hizo sobre clasificación, "
        "ahora sobre regresión."),
    table=maestra, overwrite=True,
)
print("Artefactos L3 multi-familia generados (D1, C1..C5, tab31).")
