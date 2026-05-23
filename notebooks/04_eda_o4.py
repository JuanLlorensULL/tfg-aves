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
    "Cobertura state_b en válidas (columna O3 de entrada): "
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
preds_all = preds  # alias usado en los bloques de análisis
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
    overwrite=True,
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

# %% [markdown]
# ## Fase C — Comparativa global y recomendación del ganador
# ### C3 — Tabla comparativa de los 8 modelos

# %%
test_metrics = metrics[metrics["split"] == "test"].copy()
test_metrics["combo"] = (
    test_metrics["modelo"] + "_" + test_metrics["modo"].fillna("—")
)
display_cols = ["modelo", "modo", "top1", "top3", "log_loss", "dist_median_km"]
c3_table = test_metrics[display_cols].sort_values(
    "log_loss", ascending=True,
).reset_index(drop=True)

# ── Días de movimiento (true_cell ≠ celda predicha por persistencia) ──────────
pers_key = preds_all[(preds_all["modelo"] == "persistencia") & (preds_all["modo"] == "—")][
    ["bird_id", "date_utc", "pred_cell_top1"]
].rename(columns={"pred_cell_top1": "_pers_pred"})

move_top1 = {}
for _, row in c3_table.iterrows():
    mdl, modo = row["modelo"], row["modo"]
    sub = preds_all[(preds_all["modelo"] == mdl) & (preds_all["modo"] == modo)]
    merged = sub.merge(pers_key, on=["bird_id", "date_utc"], how="inner")
    move = merged[merged["true_cell"] != merged["_pers_pred"]]
    move_top1[(mdl, modo)] = (
        (move["true_cell"] == move["pred_cell_top1"]).mean() if len(move) > 0 else float("nan")
    )

c3_table["top1_dias_movimiento"] = c3_table.apply(
    lambda r: move_top1.get((r["modelo"], r["modo"]), float("nan")), axis=1
)
c3_table

# %%
fig_c3, axes = plt.subplots(1, 5, figsize=(22, 4.5))
metrics_panels = [
    ("top1", "Top-1 global", False),
    ("top3", "Top-3 global", False),
    ("log_loss", "Log-loss (menor mejor)", True),
    ("dist_median_km", "Dist mediana km (menor mejor)", True),
    ("top1_dias_movimiento", "Top-1 días de movimiento", False),
]
for ax, (metric, title, ascending) in zip(axes, metrics_panels):
    sub = c3_table.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    labels = sub.apply(lambda r: f"{r['modelo']}/{r['modo']}", axis=1)
    ax.barh(range(len(sub)), sub[metric].values)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_title(title, fontsize=9)
fig_c3.suptitle("C3 — Comparativa global de los 8 modelos sobre el test temporal")
fig_c3.tight_layout()

save_artifact(
    slug="models-comparison",
    objective="o4",
    num=4,
    decision="Tabla comparativa de las 8 combinaciones (3 familias × 2 modos + 2 baselines)",
    caption_es=(
        "Métricas globales sobre el conjunto de test temporal (último 20 % de "
        "días de cada ave). Se reportan cinco paneles: top-1 global, top-3 "
        "global, log-loss, distancia mediana y top-1 restringido a los días de "
        "movimiento (true_cell ≠ celda predicha por persistencia, n=908, 22,5 % "
        "del test). Persistencia trivial y Markov(1) se incluyen como baselines "
        "heredadas de O2. En las métricas globales, persistencia domina sobre "
        "ML (73 % de días son self-loops, resultado estructural del dataset). "
        "En los días de movimiento —donde persistencia es trivialmente cero— "
        "los modelos ML (RF/XGB) alcanzan top-1 ~0,17-0,20 frente al 0,09 de "
        "Markov(1): este es el margen de valor real del aprendizaje supervisado. "
        "LightGBM queda visualmente diferenciado por sus valores anómalos en "
        "log-loss y distancia mediana, coherente con la divergencia observada "
        "en C2. Conviene leer este artefacto junto a C4, que argumenta la "
        "recomendación final del algoritmo ganador."
    ),
    fig=fig_c3,
    table=c3_table,
    overwrite=True,
)

# %% [markdown]
# ### C4 — Recomendación del algoritmo ganador (criterio principal: log-loss)

# %%
ml_only = c3_table[c3_table["modelo"].isin(["rf", "xgb", "lgbm"])].copy()

# Ganadores por criterio
winners = {
    "log_loss": ml_only.sort_values("log_loss").iloc[0],
    "top1": ml_only.sort_values("top1", ascending=False).iloc[0],
    "top3": ml_only.sort_values("top3", ascending=False).iloc[0],
    "dist_median_km": ml_only.sort_values("dist_median_km").iloc[0],
}
winners_table = pd.DataFrame([
    {"métrica": k,
     "ganador_modelo": v["modelo"],
     "ganador_modo": v["modo"],
     "valor": v[k]}
    for k, v in winners.items()
])

# Ganador absoluto por modo (log-loss)
ganador_pers = ml_only[ml_only["modo"] == "personalizado"].sort_values("log_loss").iloc[0]
ganador_pob = ml_only[ml_only["modo"] == "poblacional"].sort_values("log_loss").iloc[0]
print(f"Ganador personalizado: {ganador_pers['modelo']} (log-loss={ganador_pers['log_loss']:.3f})")
print(f"Ganador poblacional:   {ganador_pob['modelo']} (log-loss={ganador_pob['log_loss']:.3f})")
winners_table

# %%
recomendacion_es = (
    "**Criterio principal: log-loss** (calibración probabilística; heredado de "
    "O2 como criterio más informativo que el top-1 argmax). El log-loss "
    "penaliza la sobreconfianza en predicciones incorrectas y es la métrica "
    "más útil para alimentar O5 con distribuciones de probabilidad por celda. "
    "\n\n"
    f"**Ganador del modo personalizado:** `{ganador_pers['modelo']}` con "
    f"log-loss = {ganador_pers['log_loss']:.3f}. "
    f"**Ganador del modo poblacional:** `{ganador_pob['modelo']}` con "
    f"log-loss = {ganador_pob['log_loss']:.3f}. "
    "\n\n"
    "Resultado global: ML bate a Markov(1) en todas las métricas (C3), pero "
    "pierde ante la persistencia trivial en top-1 global y distancia mediana "
    "— resultado estructural del dataset (73 % de días son self-loops, lo que "
    "favorece por construcción a la baseline trivial). El margen de valor real "
    "del ML se concentra en los días de movimiento (ver columna "
    "top1_dias_movimiento en C3): ML ~0,18 vs persistencia = 0 y Markov ~0,09. "
    "El error espacial absoluto es modesto: mediana ~24 km, ~80 % de "
    "predicciones dentro de 110 km. El límite es estructural (features "
    "locales de un único día, sin historial multi-paso ni información de viento). "
    "\n\n"
    "LightGBM queda descartado en ambos modos por la divergencia documentada "
    "en C2 — su configuración conservadora de §8.6 no extrae señal "
    "generalizable de este dataset con ~849 clases activas. "
    "\n\n"
    "Ambos ganadores se utilizarán como base para los artefactos siguientes "
    "(C5/C6 análisis de error por estado, C7 memorización, C8 feature "
    "importance). En O5, se cargarán por defecto; el usuario podrá cambiar "
    "a las otras combinaciones para comparar visualmente."
)

# Tabla expandida: ganador por modo (log-loss) + comparativa con baselines
final_rows = []
for tag, model_row in [("personalizado", ganador_pers), ("poblacional", ganador_pob)]:
    final_rows.append({
        "rol": f"ganador {tag}",
        "modelo": model_row["modelo"], "modo": tag,
        "top1": model_row["top1"], "log_loss": model_row["log_loss"],
        "dist_median_km": model_row["dist_median_km"],
    })
for name in ["persistencia", "markov"]:
    row = c3_table[c3_table["modelo"] == name].iloc[0]
    final_rows.append({
        "rol": f"baseline {name}",
        "modelo": name, "modo": "—",
        "top1": row["top1"], "log_loss": row["log_loss"],
        "dist_median_km": row["dist_median_km"],
    })
c4_table = pd.DataFrame(final_rows)

save_artifact(
    slug="winner-recommendation",
    objective="o4",
    num=5,
    decision=(
        f"Algoritmo ganador personalizado = {ganador_pers['modelo']}; "
        f"ganador poblacional = {ganador_pob['modelo']}. "
        "Criterio: log-loss en test temporal."
    ),
    caption_es=recomendacion_es,
    table=c4_table,
    overwrite=True,
)

# %% [markdown]
# ## Fase D — Análisis del modelo ganador
# ### C5 — Error por estado HMM (ganador personalizado)
# ### C6 — Error por estado HMM (ganador poblacional)

# %%
from tfg_aves.ml.evaluate import evaluate_by_state


def _table_for_winner(family: str, mode: str) -> pd.DataFrame:
    sub = preds_all[(preds_all["modelo"] == family) & (preds_all["modo"] == mode)]
    table = evaluate_by_state(sub, state_col="state_b_causal")
    table["modelo"] = family
    table["modo"] = mode
    return table


c5_table = _table_for_winner(ganador_pers["modelo"], "personalizado")
c6_table = _table_for_winner(ganador_pob["modelo"], "poblacional")
display(c5_table)
display(c6_table)

# %%
fig_state, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (table, label) in zip(
    axes, [(c5_table, "personalizado"), (c6_table, "poblacional")],
):
    metrics_to_plot = ["top1", "top3"]
    states_order = ["global", "estacionario", "migración"]
    x = np.arange(len(states_order))
    width = 0.35
    for i, met in enumerate(metrics_to_plot):
        vals = [table[table["state"] == s][met].iloc[0] for s in states_order]
        ax.bar(x + (i - 0.5) * width, vals, width, label=met)
    ax.set_xticks(x)
    ax.set_xticklabels(states_order)
    ax.set_title(f"Ganador {label} ({table['modelo'].iloc[0]})")
    ax.set_ylim(0, 1)
    ax.legend()
fig_state.suptitle("C5 + C6 — Top-1 y top-3 por estado HMM (estacionario vs migración)")
fig_state.tight_layout()

save_artifact(
    slug="error-by-state-personalizado",
    objective="o4",
    num=6,
    decision="Desglose de top-1 y top-3 por estado HMM causal para el ganador personalizado",
    caption_es=(
        f"Accuracy del modelo ganador personalizado ({ganador_pers['modelo']}) "
        "desglosada por estado biológico inferido por O3 con cinemática causal "
        "(state_b_causal). La caída del top-1 entre estacionario (~0,64) y "
        "migración (~0,13) confirma el límite estructural del modelo: acierta "
        "los días estacionarios —donde la persistencia también acertaría— pero "
        "falla en los días de migración activa, que son biológicamente los más "
        "relevantes. Este resultado no es un artefacto del overfit sino el "
        "techo real de un modelo que dispone únicamente de features locales de "
        "un solo día sin información de viento ni historial multi-paso."
    ),
    fig=fig_state,
    table=c5_table,
    overwrite=True,
)
save_artifact(
    slug="error-by-state-poblacional",
    objective="o4",
    num=7,
    decision="Desglose de top-1 y top-3 por estado HMM causal para el ganador poblacional",
    caption_es=(
        f"Idem C5 para el ganador poblacional ({ganador_pob['modelo']}, sin "
        "bird_id). El patrón es prácticamente idéntico al personalizado: "
        "estacionario ~0,63 y migración ~0,13-0,14. La pequeña diferencia "
        "(~+1 pp en estacionario para el personalizado) refleja la señal "
        "generalizable que aporta bird_id sin aumentar el overfit (ver C7). "
        "La coherencia entre ambos modos refuerza que el colapso en migración "
        "es estructural y no atribuible a la presencia o ausencia de la "
        "identidad del ave. Fenología de referencia: estado 1 (migración) "
        "representa el 11,7 % del test, concentrado en abr-may y sep-oct, "
        "coherente con la fenología conocida de Larus fuscus."
    ),
    fig=fig_state,
    table=c6_table,
    overwrite=True,
)

# %% [markdown]
# ### C7 — Test de memorización: personalizado vs poblacional

# %%
c7_rows = []
for family in ["rf", "xgb", "lgbm"]:
    pers = metrics[(metrics["modelo"] == family) & (metrics["modo"] == "personalizado") & (metrics["split"] == "test")].iloc[0]
    pob = metrics[(metrics["modelo"] == family) & (metrics["modo"] == "poblacional") & (metrics["split"] == "test")].iloc[0]
    pers_tr = metrics[(metrics["modelo"] == family) & (metrics["modo"] == "personalizado") & (metrics["split"] == "train")].iloc[0]
    pob_tr = metrics[(metrics["modelo"] == family) & (metrics["modo"] == "poblacional") & (metrics["split"] == "train")].iloc[0]
    c7_rows.append({
        "familia": family,
        "top1_pers_train": pers_tr["top1"], "top1_pers_test": pers["top1"],
        "gap_pers": pers_tr["top1"] - pers["top1"],
        "top1_pob_train": pob_tr["top1"], "top1_pob_test": pob["top1"],
        "gap_pob": pob_tr["top1"] - pob["top1"],
        "diff_test_pers_vs_pob": pers["top1"] - pob["top1"],
    })
c7_table = pd.DataFrame(c7_rows)
print(c7_table.to_string())

fig_c7, axes_c7 = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel izquierdo: gap train-test por familia y modo
ax_gap = axes_c7[0]
x = np.arange(len(c7_table))
width = 0.4
ax_gap.bar(x - width / 2, c7_table["gap_pers"], width, label="gap personalizado")
ax_gap.bar(x + width / 2, c7_table["gap_pob"], width, label="gap poblacional")
ax_gap.set_xticks(x)
ax_gap.set_xticklabels(c7_table["familia"])
ax_gap.set_ylabel("gap top-1 (train - test)")
ax_gap.set_title("Gap train-test por familia y modo")
ax_gap.legend()

# Panel derecho: diferencia test pers vs pob
ax_diff = axes_c7[1]
rf_xgb = c7_table[c7_table["familia"].isin(["rf", "xgb"])].reset_index(drop=True)
x2 = np.arange(len(rf_xgb))
ax_diff.bar(x2, rf_xgb["diff_test_pers_vs_pob"], width=0.5, color=["steelblue", "darkorange"])
ax_diff.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_diff.set_xticks(x2)
ax_diff.set_xticklabels(rf_xgb["familia"])
ax_diff.set_ylabel("top1_pers_test − top1_pob_test")
ax_diff.set_title("Aportación neta de bird_id en test (pers − pob)")

fig_c7.suptitle("C7 — Análisis de la contribución de bird_id: gaps similares, aporte neto modesto")
fig_c7.tight_layout()

save_artifact(
    slug="personalizado-vs-poblacional",
    objective="o4",
    num=8,
    decision=(
        "bird_id aporta una mejora marginal en test (~+1 pp top-1) sin aumentar "
        "el gap de overfit respecto al modo poblacional"
    ),
    caption_es=(
        "Análisis del papel de bird_id (identidad individual) en el pipeline. "
        "Panel izquierdo: gap train-test en top-1 para cada familia en modo "
        "personalizado (con bird_id) frente a poblacional (sin bird_id). Los "
        "gaps son prácticamente iguales entre modos (~0,20 para RF, ~0,21 para "
        "XGBoost): añadir bird_id NO aumenta el overfit, pero tampoco lo "
        "reduce. Panel derecho: diferencia de top-1 en test (pers − pob) para "
        "RF y XGBoost. La diferencia es positiva pero pequeña (~+1 pp), lo que "
        "indica que bird_id aporta señal generalizable real —probablemente "
        "rangos de hábitat individuales que el modelo captura— aunque el impacto "
        "cuantitativo es modesto. Interpretación para la memoria: el modo "
        "personalizado es marginalmente preferible en top-1 si el ave es "
        "conocida; el poblacional es igualmente válido y más generalizable a "
        "individuos nuevos."
    ),
    fig=fig_c7,
    table=c7_table,
    overwrite=True,
)

# %% [markdown]
# ### C8 — Importancia de features de los dos ganadores

# %%
def _feature_importance(mode: str, family: str) -> pd.DataFrame:
    bundle = joblib.load(result.model_paths[f"{mode}_{family}"])
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    if family == "rf":
        inner = model.named_steps["model"]
        imp = inner.feature_importances_
    elif family == "xgb":
        imp = model._xgb.feature_importances_
    else:  # lgbm
        imp = model._lgbm.booster_.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["modo"] = mode
    df["modelo"] = family
    return df


imp_pers = _feature_importance("personalizado", ganador_pers["modelo"])
imp_pob = _feature_importance("poblacional", ganador_pob["modelo"])

fig_c8, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, df, title in zip(
    axes,
    [imp_pers, imp_pob],
    [
        f"Ganador personalizado ({ganador_pers['modelo']})",
        f"Ganador poblacional ({ganador_pob['modelo']})",
    ],
):
    ax.barh(df["feature"], df["importance"])
    ax.invert_yaxis()
    ax.set_title(title)
fig_c8.suptitle("C8 — Importancia de features de los dos ganadores")
fig_c8.tight_layout()

c8_table = pd.concat([imp_pers, imp_pob], ignore_index=True)
save_artifact(
    slug="feature-importance-winners",
    objective="o4",
    num=9,
    decision=(
        "Verificar que state_b_causal y posterior_b_migracion_causal son usados; "
        "cuantificar el peso de bird_id en el modelo personalizado"
    ),
    caption_es=(
        "Importancia relativa de las features para los dos modelos ganadores "
        "(pipeline causal: cinemática t-1→t y estado HMM forward-filtered). "
        "En el modelo personalizado (RF), lat y lon acaparan ~0,66 de la "
        "importancia total, seguidos de bird_id (~0,14) y "
        "posterior_b_migracion_causal (~0,05). En el modelo poblacional (XGB), "
        "lat+lon suman ~0,74, con posterior_b_migracion_causal (~0,05) y "
        "state_b_causal (~0,02) entre las features informativas, lo que "
        "valida que el aporte de O3 al pipeline supervisado es real aunque "
        "secundario. La cinemática causal (step_in_km, cos_turning_in, "
        "sin/cos_bearing_in) ocupa posiciones intermedias en ambos modelos, "
        "con importancias modestas pero consistentes."
    ),
    fig=fig_c8,
    table=c8_table,
    overwrite=True,
)
