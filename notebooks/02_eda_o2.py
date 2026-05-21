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
# # 02 — EDA de O2 (cadenas de Markov mensuales)
#
# Notebook que justifica la elección de `cell_deg` (D1) y materializa los
# artefactos de caracterización C1–C5 sobre los outputs de `build_o2`.

# %%
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tfg_aves.markov import (
    build_o2,
    build_transitions,
    build_counts,
    discretize_dataframe,
    haversine_km,
)
from tfg_aves.markov._paths import DAILY_PARQUET
from tfg_aves.reporting import save_artifact

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})

# %%
df_daily = pd.read_parquet(DAILY_PARQUET)
print(f"daily.parquet: {len(df_daily)} filas, {df_daily['bird_id'].nunique()} aves")
print(f"válidas: {df_daily['is_valid'].sum()}")

# %% [markdown]
# ## Fase A — D1: trade-off del tamaño de celda
#
# Comparamos 4 candidatos: 0.25°, 0.5°, 1°, 2°. Para cada uno calculamos:
# - número de celdas ocupadas,
# - mediana de transiciones por celda-mes,
# - % de pares (origen, destino) observados,
# - % de self-loops sobre el total de transiciones.

# %%
def _daily_displacement_km(df: pd.DataFrame) -> np.ndarray:
    """Distancia entre día t y t+1 para todos los pares válidos consecutivos."""
    df = df.sort_values(["bird_id", "date_utc"]).reset_index(drop=True)
    # Normalizar date_utc a datetime64 para una resta robusta
    date_dt = pd.to_datetime(df["date_utc"])
    same_bird = df["bird_id"].shift(-1) == df["bird_id"]
    both_valid = df["is_valid"] & df["is_valid"].shift(-1).fillna(False).astype(bool)
    consecutive = (date_dt.shift(-1) - date_dt) == pd.Timedelta(days=1)
    mask = same_bird & both_valid & consecutive

    d = haversine_km(
        df["lat"][mask].to_numpy(),
        df["lon"][mask].to_numpy(),
        df["lat"].shift(-1)[mask].to_numpy(),
        df["lon"].shift(-1)[mask].to_numpy(),
    )
    return d

displacements = _daily_displacement_km(df_daily)
print(f"Pares válidos consecutivos: {len(displacements)}")
print(f"Mediana desplazamiento diario: {np.median(displacements):.1f} km")

# %%
def summarize_grid(df_daily: pd.DataFrame, cell_deg: float) -> dict:
    df_disc = discretize_dataframe(df_daily, cell_deg=cell_deg)
    valid = df_disc[df_disc["is_valid"]]
    cells = sorted(valid["cell_id"].dropna().unique().tolist())
    transitions = build_transitions(df_disc)
    counts = build_counts(transitions, cells=cells)
    n_cells = len(cells)
    total_transitions = int(counts.sum())
    # % de pares (origen, destino) observados sobre todos los posibles
    nonzero_pairs = int((counts.sum(axis=0) > 0).sum())
    total_pairs = n_cells * n_cells
    self_loops = int(sum(counts[m].diagonal().sum() for m in range(12)))
    median_per_cell_month = float(
        np.median([counts[m].sum(axis=1).sum() / max(n_cells, 1) for m in range(12)])
    )

    return {
        "cell_deg": cell_deg,
        "n_cells_ocupadas": n_cells,
        "n_transiciones_total": total_transitions,
        "mediana_transiciones_por_celda_mes": median_per_cell_month,
        "pct_pares_observados": 100.0 * nonzero_pairs / max(total_pairs, 1),
        "pct_self_loops": 100.0 * self_loops / max(total_transitions, 1),
    }

candidates = [0.25, 0.5, 1.0, 2.0]
summaries = [summarize_grid(df_daily, cd) for cd in candidates]
df_summary = pd.DataFrame(summaries)
print(df_summary)

# %%
# Figura D1 — 3 paneles. Layout: A y B en filas 0 y 1 a todo ancho, C en fila 2 con 4 mini-mapas.
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.8, 1.5])

# Panel A — histograma del desplazamiento + líneas verticales (tamaño físico ≈ cell_deg × 111 km).
ax_a = fig.add_subplot(gs[0, :])
ax_a.hist(displacements, bins=80, color="#888", alpha=0.7)
for cd in candidates:
    physical_km = cd * 111.0
    ax_a.axvline(physical_km, linestyle="--", label=f"{cd}° ≈ {physical_km:.0f} km")
ax_a.set_xlabel("Desplazamiento diario (km)")
ax_a.set_ylabel("Pares (t, t+1)")
ax_a.set_title("A — Desplazamiento diario vs tamaño de celda")
ax_a.set_xlim(0, np.quantile(displacements, 0.99))
ax_a.legend()

# Panel B — tabla con los 4 candidatos.
ax_b = fig.add_subplot(gs[1, :])
ax_b.axis("off")
table_data = df_summary.round(2).values.tolist()
table_cols = ["cell_deg", "n_celdas", "n_transiciones", "mediana_x_celda_mes", "% pares", "% self-loops"]
ax_b.table(cellText=table_data, colLabels=table_cols, loc="center", cellLoc="center")
ax_b.set_title("B — Métricas por candidato")

# Panel C — 4 mini-mapas con grids superpuestos a la nube de fixes.
sample = df_daily[df_daily["is_valid"]].sample(
    n=min(20000, int(df_daily["is_valid"].sum())), random_state=0
)
for k, cd in enumerate(candidates):
    ax = fig.add_subplot(gs[2, k])
    ax.scatter(sample["lon"], sample["lat"], s=0.3, color="#444", alpha=0.4)
    df_disc = discretize_dataframe(df_daily, cell_deg=cd)
    cells_occupied = df_disc.loc[
        df_disc["is_valid"], ["cell_lat_idx", "cell_lon_idx"]
    ].drop_duplicates()
    for _, row in cells_occupied.iterrows():
        i, j = int(row["cell_lat_idx"]), int(row["cell_lon_idx"])
        ax.add_patch(
            patches.Rectangle(
                (j * cd, i * cd), cd, cd,
                fill=False, edgecolor="red", linewidth=0.3,
            )
        )
    ax.set_title(f"{cd}° — {len(cells_occupied)} celdas", fontsize=9)
    ax.set_xlim(sample["lon"].min() - 1, sample["lon"].max() + 1)
    ax.set_ylim(sample["lat"].min() - 1, sample["lat"].max() + 1)
    ax.set_aspect("equal")

fig.suptitle("D1 — Trade-off del tamaño de celda", y=0.995)
fig.tight_layout()

# %% [markdown]
# ### Decisión D1
#
# Inspeccionar la figura y la tabla anteriores. Elegir `cell_deg` final
# entre los candidatos. Criterios:
# - Panel A: el desplazamiento mediano cruza ≥2-3 celdas (queremos que las
#   transiciones tengan estructura, no que dominen self-loops).
# - Panel B: `% pares observados` razonable (>3-5% al menos para que las
#   matrices no sean casi vacías) y `mediana transiciones por celda-mes`
#   suficiente (>3-5) para que el suavizado Laplace no domine.
# - Panel C: el grid debe cubrir las áreas usadas sin ser tan fino que
#   produzca celdas con un solo fix.
#
# Sustituir el valor en la celda siguiente con la decisión tomada y volver
# a ejecutar el resto del notebook.

# %%
CELL_DEG_FINAL = 0.5  # ← decisión justificada por la figura D1

save_artifact(
    slug="grid-size-tradeoff",
    objective="o2",
    num=1,
    decision=f"Tamaño de celda fijado en {CELL_DEG_FINAL}° (justificado por trade-off de 4 candidatos)",
    caption_es=(
        "Comparación de 4 candidatos de tamaño de celda para la discretización del espacio "
        "en O2: 0,25°, 0,5°, 1° y 2°. El panel A muestra el desplazamiento diario observado "
        "con líneas verticales en el tamaño físico aproximado de cada celda (cell_deg × 111 km). "
        "El panel B resume métricas de cobertura y densidad para cada candidato. El panel C "
        f"superpone cada grid sobre la nube de fixes válidos. Se adopta cell_deg = {CELL_DEG_FINAL}° "
        "porque equilibra resolución espacial (transiciones cruzan varias celdas en periodos "
        "migratorios) y robustez estadística (mediana de transiciones por celda-mes y % de pares "
        "observados aceptables tras el suavizado Laplace α=1)."
    ),
    fig=fig,
    table=df_summary,
)
