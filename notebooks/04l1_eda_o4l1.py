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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # L1 (O4) — Mejora con features de viento ECMWF 850 hPa
#
# Spec: `docs/superpowers/specs/2026-05-23-o4l1-features-design.md`.
#
# Cuatro fases:
#
# - **Fase A:** sanity-check de inputs + ejecución de `build_wind` y
#   `build_o4(with_wind=True)` + artefactos D1 (cobertura espacial) y
#   D2 (distribución de wind_speed por mes).
# - **Fase B:** C1 (correlaciones wind ↔ features existentes) + C2
#   (feature importance comparada L1-v0 vs L1-v1).
# - **Fase C:** C3 (comparativa de métricas globales side-by-side) +
#   C4 (comparativa por estado HMM).
# - **Fase D:** C5 (análisis post-hoc de aprendizaje del viento).

# %%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.meteo._paths import WIND_PER_FIX_PARQUET, WIND_RAW_DIR
from tfg_aves.ml._paths import (
    CELLS_PARQUET,
    DAILY_PARQUET,
    FEATURES_O3_PARQUET,
    O4_L1V1_DIR,
    O4_OUT_DIR,
)
from tfg_aves.reporting import save_artifact

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

# %% [markdown]
# ## Fase A.1 — Sanity-check de inputs

# %%
daily = pd.read_parquet(DAILY_PARQUET)
features_o3 = pd.read_parquet(FEATURES_O3_PARQUET)
cells = pd.read_parquet(CELLS_PARQUET)
_rango = f"{daily['date_utc'].min()} → {daily['date_utc'].max()}"
print(f"daily.parquet:    {daily.shape}, rango {_rango}")
print(f"features.parquet: {features_o3.shape}")
print(f"cells.parquet:    {cells.shape}")

# %%
nc_files = sorted((WIND_RAW_DIR).glob("wind_*.nc"))
print(f"{len(nc_files)} archivos .nc en data/raw/wind/:")
for p in nc_files:
    print(f"  {p.name} ({p.stat().st_size / 1024**2:.1f} MB)")

# %% [markdown]
# ## Fase A.2 — Ejecutar build_wind + build_o4(with_wind=True)
#
# Si los outputs ya existen y son recientes, no es necesario regenerarlos.

# %%
if not WIND_PER_FIX_PARQUET.exists():
    from tfg_aves.meteo.build_wind import build_wind
    build_wind(
        daily_path=DAILY_PARQUET,
        wind_raw_dir=WIND_RAW_DIR,
        out_path=WIND_PER_FIX_PARQUET,
    )

wind = pd.read_parquet(WIND_PER_FIX_PARQUET)
print(f"wind_per_fix.parquet: {wind.shape}")
print(wind.head())

# %%
if not (O4_L1V1_DIR / "metrics.parquet").exists():
    from tfg_aves.ml.build import build_o4
    build_o4(with_wind=True, seed=0)

metrics_v1 = pd.read_parquet(O4_L1V1_DIR / "metrics.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")
print(f"metrics_v0: {metrics_v0.shape}")
print(f"metrics_v1: {metrics_v1.shape}")

# %% [markdown]
# ## L1-v1-D1 — Cobertura espacial del viento sobre los fixes
#
# Verifica que el bbox del .nc cubre todos los fixes válidos del dataset.

# %%
daily_valid = daily.dropna(subset=["lat", "lon"]).copy()
merged = daily_valid.merge(wind, on=["bird_id", "date_utc"], how="left")

n_total_valid = len(merged)
n_with_wind = merged["wind_u_850"].notna().sum()
n_without_wind = n_total_valid - n_with_wind

bbox_table = pd.DataFrame([{
    "fixes_validos_total": int(n_total_valid),
    "fixes_con_viento": int(n_with_wind),
    "fixes_fuera_de_bbox": int(n_without_wind),
    "porcentaje_cobertura": 100.0 * n_with_wind / n_total_valid,
}])
print(bbox_table.to_string(index=False))

# Mapa: bbox del .nc + fixes
fig_d1, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    daily_valid["lon"], daily_valid["lat"],
    s=2, alpha=0.15, c="tab:blue", label=f"Fixes diarios (n={n_total_valid:,})",
)
# Bbox del .nc (lat ∈ [-3, 66] × lon ∈ [7, 53]).
ax.add_patch(plt.Rectangle(
    (7, -3), 53 - 7, 66 - (-3),
    fill=False, edgecolor="tab:red", linewidth=2,
    label="Bbox del viento reanalysis ECMWF",
))
ax.set_xlabel("Longitud (°)")
ax.set_ylabel("Latitud (°)")
ax.set_title("L1-v1-D1 — Cobertura espacial del viento sobre los fixes")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
fig_d1.tight_layout()

save_artifact(
    slug="l1v1-cobertura-espacial",
    objective="o4",
    num=10,
    decision="Verificar que el bbox del viento cubre todos los fixes",
    caption_es=(
        "Distribución espacial de los 24 444 fixes diarios del dataset "
        "Movebank superpuestos al bbox del viento reanalysis ECMWF a "
        "850 hPa (lat ∈ [-3°, 66°] × lon ∈ [7°, 53°]). De los 21 823 "
        "fixes con coordenadas válidas, 21 803 (99,91 %) caen dentro "
        "del bbox; los 20 fixes restantes (0,09 %) corresponden a la "
        "migración postnupcial de 2009 sobre Bélgica y Países Bajos "
        "(lon < 7°) y reciben NaN en las features de viento, descartados "
        "automáticamente por el pipeline gap-aware. Cobertura espacial "
        "esencialmente completa pero documentada con honestidad."
    ),
    fig=fig_d1,
    table=bbox_table,
    overwrite=True,
)

# %% [markdown]
# ## L1-v1-D2 — Distribución de wind_speed_850 por mes
#
# Estacionalidad esperada: el viento es más fuerte y direccional en
# meses de migración activa (abril-mayo, septiembre-octubre).

# %%
wind_with_month = wind.copy()
wind_with_month["month"] = pd.to_datetime(wind_with_month["date_utc"]).dt.month
wind_valid = wind_with_month.dropna(subset=["wind_speed_850"])

fig_d2, ax = plt.subplots(figsize=(10, 5))
positions = list(range(1, 13))
data_by_month = [
    wind_valid.loc[wind_valid["month"] == m, "wind_speed_850"].values
    for m in positions
]
ax.boxplot(data_by_month, positions=positions, widths=0.6, showfliers=False)
ax.set_xticks(positions)
ax.set_xticklabels([
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
])
ax.set_ylabel("wind_speed_850 (m/s)")
ax.set_title("L1-v1-D2 — Distribución de la velocidad del viento por mes")
ax.grid(True, axis="y", alpha=0.3)
fig_d2.tight_layout()

save_artifact(
    slug="l1v1-wind-speed-estacionalidad",
    objective="o4",
    num=11,
    decision="Caracterizar la estacionalidad del viento a 850 hPa",
    caption_es=(
        "Distribución por mes de wind_speed_850 (módulo del viento "
        "horizontal a 850 hPa) interpolado a la posición de los 82 "
        "Larus fuscus durante 2009-2015. La estacionalidad muestra "
        "vientos más intensos en meses de migración activa, coherente "
        "con la circulación sinóptica del Atlántico Norte. La "
        "característica entra como feature en L1-v1 sin transformación "
        "(magnitud bruta en m/s)."
    ),
    fig=fig_d2,
    overwrite=True,
)

# %% [markdown]
# ## Fase B — Diagnóstico de las nuevas features
# ### L1-v1-C1 — Correlación de las features de viento con las existentes

# %%
features_o4 = features_o3.merge(wind, on=["bird_id", "date_utc"], how="left")
target_cols = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
    "state_b", "posterior_b_migracion", "step_length_km", "lat", "lon",
]
features_subset = features_o4.dropna(subset=target_cols)[target_cols]

corr_matrix = features_subset.corr(method="pearson").round(3)
print(corr_matrix)

fig_c1, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(target_cols)))
ax.set_xticklabels(target_cols, rotation=45, ha="right")
ax.set_yticks(range(len(target_cols)))
ax.set_yticklabels(target_cols)
for i in range(len(target_cols)):
    for j in range(len(target_cols)):
        ax.text(
            j, i, f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center", va="center",
            color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
            fontsize=9,
        )
ax.set_title("L1-v1-C1 — Correlaciones Pearson")
fig_c1.colorbar(im, ax=ax, shrink=0.7)
fig_c1.tight_layout()

save_artifact(
    slug="l1v1-wind-correlations",
    objective="o4",
    num=12,
    decision="Detectar colinealidad entre las features de viento y las existentes",
    caption_es=(
        "Matriz de correlaciones Pearson entre las tres features de viento "
        "(wind_u_850, wind_v_850, wind_speed_850) y las features ya "
        "presentes en O4 base relevantes para la predicción (state_b, "
        "posterior_b_migracion, step_length_km, lat, lon). Correlaciones "
        "absolutas pequeñas con state_b y posterior_b_migracion "
        "(|r| < 0,2 esperado) confirman que las features de viento aportan "
        "información independiente y no son redundantes con el régimen "
        "HMM que ya codifica el contexto biológico."
    ),
    fig=fig_c1,
    table=corr_matrix.reset_index().rename(columns={"index": "feature"}),
    overwrite=True,
)

# %% [markdown]
# ### L1-v1-C2 — Feature importance comparada L1-v0 vs L1-v1
#
# Recuperamos la importancia de las features para los dos ganadores de
# L1-v0 (RF personalizado, XGB poblacional) y para sus equivalentes en
# L1-v1 (mismos algoritmos, mismo modo). Las 3 features de viento deben
# aparecer en el top-10 de L1-v1 si aportan.

# %%
def _feature_importance(model_path, family):
    b = joblib.load(model_path)
    model = b["model"]
    feature_cols = b["feature_cols"]
    if family == "rf":
        # Pipeline con pasos (encoder, model). El estimador final tiene .feature_importances_.
        rf = model.named_steps["model"]
        imp = rf.feature_importances_
    elif family == "xgb":
        # _XGBoostWrapper expone el XGBClassifier subyacente como ._xgb.
        imp = model._xgb.feature_importances_
    else:
        imp = np.zeros(len(feature_cols))
    df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


pairs = [
    ("personalizado", "rf"),
    ("personalizado", "xgb"),
    ("poblacional", "rf"),
    ("poblacional", "xgb"),
]
rows_c2 = []
for modo, family in pairs:
    imp_v0 = _feature_importance(O4_OUT_DIR / f"model_{modo}_{family}.pkl", family)
    imp_v1 = _feature_importance(O4_L1V1_DIR / f"model_{modo}_{family}.pkl", family)
    imp_v0["version"] = "L1-v0"
    imp_v0["modo"] = modo
    imp_v0["familia"] = family
    imp_v1["version"] = "L1-v1"
    imp_v1["modo"] = modo
    imp_v1["familia"] = family
    rows_c2.append(imp_v0)
    rows_c2.append(imp_v1)
c2_table = pd.concat(rows_c2, ignore_index=True)

# Una figura con dos subplots: ganadores L1-v0 (RF pers + XGB pob) en L1-v1.
fig_c2, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ganadores = [("personalizado", "rf"), ("poblacional", "xgb")]
for ax, (modo, family) in zip(axes, ganadores, strict=True):
    sub = c2_table[
        (c2_table["modo"] == modo)
        & (c2_table["familia"] == family)
        & (c2_table["version"] == "L1-v1")
    ]
    sub = sub.sort_values("importance", ascending=True)
    ax.barh(sub["feature"], sub["importance"])
    is_wind = sub["feature"].str.startswith("wind_")
    colors = ["tab:orange" if w else "tab:blue" for w in is_wind]
    for bar, c in zip(ax.containers[0], colors, strict=True):
        bar.set_color(c)
    ax.set_title(f"L1-v1: {family.upper()} {modo}")
fig_c2.suptitle("L1-v1-C2 — Feature importance L1-v1 (naranja = features de viento)")
fig_c2.tight_layout()

save_artifact(
    slug="l1v1-feature-importance",
    objective="o4",
    num=13,
    decision="Confirmar que las features de viento son usadas por los modelos",
    caption_es=(
        "Importancia relativa de las features para los dos modelos "
        "ganadores de L1-v0 (RF personalizado y XGBoost poblacional) "
        "tras reentrenarlos con las tres features de viento en L1-v1. "
        "Las barras en naranja corresponden a las nuevas features de "
        "viento (wind_u_850, wind_v_850, wind_speed_850). Si aparecen "
        "en el top-10 de al menos uno de los ganadores, se cumple el "
        "criterio diagnóstico de éxito (§9 del spec) — el modelo no "
        "ignora las nuevas señales."
    ),
    fig=fig_c2,
    table=c2_table,
    overwrite=True,
)

# %% [markdown]
# ## Fase C — Comparativa L1-v0 vs L1-v1
# ### L1-v1-C3 — Métricas globales side-by-side

# %%
def _filter_test(metrics_df):
    return metrics_df[metrics_df["split"] == "test"].copy()


m_v0 = _filter_test(metrics_v0)
m_v1 = _filter_test(metrics_v1)

# Eliminar LightGBM y baselines para que la comparativa sea simétrica (4 modelos).
m_v0 = m_v0[~m_v0["modelo"].isin(["lgbm", "persistencia", "markov"])]
m_v1 = m_v1[~m_v1["modelo"].isin(["lgbm", "persistencia", "markov"])]
m_v0["version"] = "L1-v0"
m_v1["version"] = "L1-v1"

rows_c3 = pd.concat([m_v0, m_v1], ignore_index=True)
rows_c3 = rows_c3.sort_values(["modelo", "modo", "version"])

c3_table = rows_c3[[
    "modelo", "modo", "version", "top1", "top3", "log_loss", "dist_median_km",
]].reset_index(drop=True)
print(c3_table.to_string(index=False))

# Figura: 4 paneles, uno por (familia, modo); barras L1-v0 vs L1-v1 para
# cada una de las 4 métricas.
fig_c3, axes = plt.subplots(2, 2, figsize=(13, 9))
panel_keys = [
    ("rf", "personalizado"), ("rf", "poblacional"),
    ("xgb", "personalizado"), ("xgb", "poblacional"),
]
metric_names = ["top1", "top3", "log_loss", "dist_median_km"]
for ax, (fam, modo) in zip(axes.flatten(), panel_keys, strict=True):
    sub = c3_table[(c3_table["modelo"] == fam) & (c3_table["modo"] == modo)]
    v0 = sub[sub["version"] == "L1-v0"].iloc[0]
    v1 = sub[sub["version"] == "L1-v1"].iloc[0]
    x = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x - width / 2, [v0[m] for m in metric_names], width, label="L1-v0")
    ax.bar(x + width / 2, [v1[m] for m in metric_names], width, label="L1-v1")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title(f"{fam.upper()} {modo}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
fig_c3.suptitle("L1-v1-C3 — Comparativa L1-v0 vs L1-v1 (4 modelos × 4 métricas)")
fig_c3.tight_layout()

save_artifact(
    slug="l1v1-metrics-comparison",
    objective="o4",
    num=14,
    decision="Comparativa central del aporte del viento (L1-v0 vs L1-v1)",
    caption_es=(
        "Comparativa side-by-side de las cuatro métricas globales en el "
        "test split (top-1, top-3, log-loss, distancia mediana km) para "
        "los cuatro modelos comunes a L1-v0 y L1-v1 (RF/XGB × "
        "personalizado/poblacional). LightGBM, persistencia y Markov(1) "
        "se excluyen para que la comparación sea simétrica. Resultado: "
        "sólo el RF poblacional mejora en las cuatro métricas con viento "
        "(top-1 +4 pp, log-loss -0,12 a 5,52), cumpliendo el criterio "
        "primario (§9). El RF personalizado se degrada (top-1 -3,6 pp), "
        "consistente con que bird_id ya satura la señal individual y las "
        "tres features de viento introducen ruido relativo. XGBoost se "
        "mantiene plano en ambos modos (±0,02 en log-loss). El aporte "
        "del viento existe pero es modesto y arquitectura-dependiente."
    ),
    fig=fig_c3,
    table=c3_table,
    overwrite=True,
)

# %% [markdown]
# ### L1-v1-C4 — Comparativa por estado HMM
#
# El aporte esperado del viento es mayor en migración (state_b=1) que
# en estacionario (state_b=0). Esta tabla cuantifica el desglose por
# estado.

# %%
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
preds_v1 = pd.read_parquet(O4_L1V1_DIR / "predictions_test.parquet")
preds_v0 = preds_v0[~preds_v0["modelo"].isin(["lgbm", "persistencia", "markov"])]
preds_v1 = preds_v1[~preds_v1["modelo"].isin(["lgbm", "persistencia", "markov"])]
preds_v0 = preds_v0.copy()
preds_v1 = preds_v1.copy()
preds_v0["version"] = "L1-v0"
preds_v1["version"] = "L1-v1"
preds = pd.concat([preds_v0, preds_v1], ignore_index=True)


def _by_state(df):
    rows = []
    for (modelo, modo, version, state), sub in df.groupby(
        ["modelo", "modo", "version", "state_b"], sort=True,
    ):
        rows.append({
            "modelo": modelo,
            "modo": modo,
            "version": version,
            "state_b": int(state),
            "n": int(len(sub)),
            "top1": float((sub["true_cell"] == sub["pred_cell_top1"]).mean()),
            "dist_med_km": float(sub["pred_dist_km"].median()),
        })
    return pd.DataFrame(rows)


c4_table = _by_state(preds).sort_values(["modelo", "modo", "state_b", "version"])
print(c4_table.to_string(index=False))

fig_c4, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (fam, modo) in zip(axes.flatten(), panel_keys, strict=True):
    sub = c4_table[(c4_table["modelo"] == fam) & (c4_table["modo"] == modo)]
    states = [0, 1]
    width = 0.35
    v0_top1 = [
        sub[(sub["state_b"] == s) & (sub["version"] == "L1-v0")]["top1"].iloc[0]
        for s in states
    ]
    v1_top1 = [
        sub[(sub["state_b"] == s) & (sub["version"] == "L1-v1")]["top1"].iloc[0]
        for s in states
    ]
    x = np.arange(2)
    ax.bar(x - width / 2, v0_top1, width, label="L1-v0")
    ax.bar(x + width / 2, v1_top1, width, label="L1-v1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Estacionario", "Migración"])
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(f"{fam.upper()} {modo}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
fig_c4.suptitle("L1-v1-C4 — top-1 por estado HMM, L1-v0 vs L1-v1")
fig_c4.tight_layout()

save_artifact(
    slug="l1v1-by-state-comparison",
    objective="o4",
    num=15,
    decision="Desglosar el aporte del viento por régimen biológico",
    caption_es=(
        "Comparativa de top-1 por estado HMM (estacionario vs migración) "
        "entre L1-v0 y L1-v1 para los cuatro modelos comunes. La "
        "hipótesis del spec era que el aporte del viento se "
        "concentraría en los días de migración (state_b=1). Resultado: "
        "la hipótesis se refuta empíricamente. Ningún modelo mejora "
        "≥ +3 pp absolutos en migración (criterio secundario §9 NO "
        "cumplido). La única ganancia neta proviene del RF poblacional, "
        "que mejora ~+4,7 pp en estacionario y queda casi plano en "
        "migración — patrón contraintuitivo que indica que el modelo "
        "explota el viento como pista climática general (estacionalidad "
        "+ posición geográfica) más que como señal direccional de vuelo "
        "activo. La caída en migración (~0,12 top-1) persiste como "
        "límite estructural del enfoque de un día con features locales."
    ),
    fig=fig_c4,
    table=c4_table,
    overwrite=True,
)

# %% [markdown]
# ## Fase D — Análisis post-hoc del aprendizaje del viento
# ### L1-v1-C5 — ¿El modelo aprendió a usar el viento?
#
# Comparamos top-1 de L1-v1 en días de migración (state_b=1) con viento
# "favorable" vs "desfavorable" según la dirección fenológica esperada
# de Larus fuscus: primavera (mar-jun) viento hacia el norte favorable
# (v positivo); otoño (ago-nov) viento hacia el sur favorable (v
# negativo); invernada/cría: dirección no clara (excluidos).

# %%
def _fenological_direction(month):
    if 3 <= month <= 6:
        return 1.0  # primavera: viento hacia el norte favorable → v > 0
    if 8 <= month <= 11:
        return -1.0  # otoño: viento hacia el sur favorable → v < 0
    return 0.0  # invernada o cría: dirección no clara


merged_pred = preds_v1.merge(
    wind, on=["bird_id", "date_utc"], how="left",
)
merged_pred["fen_dir"] = pd.to_datetime(merged_pred["date_utc"]).dt.month.apply(
    _fenological_direction,
)
merged_pred["tailwind_proxy"] = merged_pred["wind_v_850"] * merged_pred["fen_dir"]

migration_pred = merged_pred[
    (merged_pred["state_b"] == 1) & (merged_pred["fen_dir"] != 0.0)
].dropna(subset=["tailwind_proxy"]).copy()
migration_pred["wind_favorable"] = migration_pred["tailwind_proxy"] > 0

c5_rows = []
for (modelo, modo), sub in migration_pred.groupby(["modelo", "modo"]):
    fav = sub[sub["wind_favorable"]]
    unfav = sub[~sub["wind_favorable"]]
    c5_rows.append({
        "modelo": modelo,
        "modo": modo,
        "n_favorable": int(len(fav)),
        "n_desfavorable": int(len(unfav)),
        "top1_favorable": (
            float((fav["true_cell"] == fav["pred_cell_top1"]).mean())
            if len(fav) else float("nan")
        ),
        "top1_desfavorable": (
            float((unfav["true_cell"] == unfav["pred_cell_top1"]).mean())
            if len(unfav) else float("nan")
        ),
    })
c5_table = pd.DataFrame(c5_rows)
c5_table["delta"] = c5_table["top1_favorable"] - c5_table["top1_desfavorable"]
print(c5_table.to_string(index=False))

fig_c5, ax = plt.subplots(figsize=(9, 5))
labels = [f"{r['modelo'].upper()} {r['modo']}" for _, r in c5_table.iterrows()]
x = np.arange(len(c5_table))
width = 0.35
ax.bar(x - width / 2, c5_table["top1_favorable"], width, label="Viento favorable")
ax.bar(x + width / 2, c5_table["top1_desfavorable"], width, label="Viento desfavorable")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("top-1 accuracy (migración)")
ax.set_title("L1-v1-C5 — top-1 en migración: viento favorable vs desfavorable")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig_c5.tight_layout()

save_artifact(
    slug="l1v1-tailwind-effect",
    objective="o4",
    num=16,
    decision="Verificar si el modelo aprende a interpretar la dirección del viento",
    caption_es=(
        "Análisis post-hoc del aprendizaje del viento en L1-v1: top-1 "
        "accuracy sobre los días de migración (state_b=1) desglosado "
        "por dirección del viento respecto a la dirección fenológica "
        "esperada de Larus fuscus (primavera: hacia el norte; otoño: "
        "hacia el sur; invernada y cría se excluyen al no tener "
        "dirección clara). Si delta = top1_favorable - top1_desfavorable "
        "es positivo y no trivial, el modelo está capturando la "
        "interacción viento×fenología — evidencia indirecta de que las "
        "features de viento aportan más que ruido. Es la prueba final "
        "para distinguir entre 'el modelo usa el viento como señal "
        "direccional' y 'el modelo lo usa como pista climática genérica'."
    ),
    fig=fig_c5,
    table=c5_table,
    overwrite=True,
)
