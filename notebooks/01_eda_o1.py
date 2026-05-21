# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # O1 — EDA y decisiones de la pipeline de datos
#
# Cuaderno de exploración para fijar los cuatro umbrales que parametrizan
# `build_o1`: `max_speed_kmh`, `reference_hour_utc`, `tolerance_min` y
# `min_valid_days`. Cada decisión se justifica con figura/tabla vía
# `save_artifact` y queda registrada en `reports/INDEX.md`.

# %%
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.data import (
    build_daily,
    coverage_by_hour,
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
    filter_birds_by_validity,
    load_raw,
    pick_reference_hour,
)
from tfg_aves.data.clean import _compute_speed_kmh
from tfg_aves.reporting import save_artifact

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})

# %% [markdown]
# ## C1 — Visión general del dataset

# %%
df_raw = load_raw()
print(f"Filas iniciales: {len(df_raw):,}")
print(f"Individuos: {df_raw['bird_id'].nunique()}")
print(f"Rango temporal: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

per_bird = df_raw.groupby("bird_id").size()
overview = pd.DataFrame(
    {
        "metric": [
            "n_fixes_total",
            "n_birds",
            "fixes_per_bird_median",
            "fixes_per_bird_p10",
            "fixes_per_bird_p90",
            "first_timestamp_utc",
            "last_timestamp_utc",
        ],
        "value": [
            len(df_raw),
            df_raw["bird_id"].nunique(),
            int(per_bird.median()),
            int(per_bird.quantile(0.10)),
            int(per_bird.quantile(0.90)),
            str(df_raw["timestamp"].min()),
            str(df_raw["timestamp"].max()),
        ],
    }
)
save_artifact(
    "dataset-overview",
    objective="o1",
    num=1,
    decision="Caracterización general del dataset Movebank tras carga",
    caption_es=(
        "Métricas agregadas del dataset crudo de Movebank tras la carga "
        "y normalización: número total de fixes GPS, individuos "
        "identificados, mediana e intervalo intercuartílico extendido de "
        "fixes por individuo y rango temporal cubierto."
    ),
    table=overview,
)

# %% [markdown]
# ## C2 — Distribución de intervalos entre fixes consecutivos

# %%
df_raw_sorted = df_raw.sort_values(["bird_id", "timestamp"])
dt_min = (
    df_raw_sorted.groupby("bird_id")["timestamp"]
    .diff()
    .dt.total_seconds()
    .div(60)
)
dt_min = dt_min.dropna()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(dt_min.clip(upper=600), bins=80, color="#4682B4", edgecolor="white")
ax.set_xlabel("Δt entre fixes consecutivos (min, recortado a 600)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del intervalo nativo de muestreo Movebank")

dt_summary = pd.DataFrame(
    {
        "percentil": ["p10", "p25", "p50", "p75", "p90", "p99"],
        "delta_min": [
            float(dt_min.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
        ],
    }
)
save_artifact(
    "fix-interval-distribution",
    objective="o1",
    num=2,
    decision="Caracterización del muestreo nativo de Movebank",
    caption_es=(
        "Distribución de los intervalos temporales entre fixes consecutivos "
        "de un mismo individuo. Los percentiles asociados (p10–p99) "
        "describen la frecuencia efectiva de muestreo nativa del dataset "
        "tras la carga, antes de aplicar filtros de outliers."
    ),
    fig=fig,
    table=dt_summary,
)
plt.close(fig)

# %% [markdown]
# ## D1 — Umbral de velocidad para descartar outliers GPS

# %%
df_flag_clean, report_flags = drop_movebank_flags(df_raw)
df_coord_clean, report_coords = drop_invalid_coords_and_dupes(df_flag_clean)

speeds = _compute_speed_kmh(df_coord_clean)
speeds = speeds[~np.isnan(speeds)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(speeds, 0, 300), bins=100, color="#B22222", edgecolor="white")
ax.axvline(120, color="black", linestyle="--", label="Umbral candidato 120 km/h")
ax.set_xlabel("Velocidad entre fixes consecutivos (km/h, recortada a 300)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de velocidades — Larus fuscus")
ax.legend()

speed_summary = pd.DataFrame(
    {
        "percentil": ["p50", "p90", "p95", "p99", "p99.5", "p99.9"],
        "speed_kmh": [
            float(np.quantile(speeds, q))
            for q in (0.50, 0.90, 0.95, 0.99, 0.995, 0.999)
        ],
    }
)
# Anotamos el umbral elegido. El valor final lo decide el autor a la
# vista de la figura y la biología (Larus fuscus alcanza picos de
# ~70-80 km/h con viento favorable; margen 1.5x para errores GPS aislados).
MAX_SPEED_KMH = 120.0
save_artifact(
    "speed-distribution",
    objective="o1",
    num=3,
    decision=f"Umbral de outlier de velocidad fijado en {MAX_SPEED_KMH:.0f} km/h",
    caption_es=(
        "Distribución de velocidades entre fixes GPS consecutivos del "
        "mismo individuo tras descartar marcas de Movebank y coordenadas "
        "inválidas. El umbral elegido (línea discontinua) deja un margen "
        "amplio sobre la velocidad de crucero de Larus fuscus (~50 km/h) "
        "y por encima de su pico observado con viento favorable, evitando "
        "penalizar tracks legítimos."
    ),
    fig=fig,
    table=speed_summary,
)
plt.close(fig)

# %%
df_clean, report_speed = drop_speed_outliers(df_coord_clean, max_speed_kmh=MAX_SPEED_KMH)
print(f"Tras limpieza: {len(df_clean):,} fixes ({len(df_clean)/len(df_raw):.1%} del original)")

# %% [markdown]
# ## D2 — Hora UTC de referencia (anclaje a pico discreto del muestreo)

# %%
# El muestreo de Movebank no es continuo: concentra los fixes en cuatro
# ventanas programadas (≈ 05, 08, 14, 20 UTC). Mostramos la distribución
# horaria del dataset limpio y elegimos el pico de las 08:00 UTC.

REFERENCE_HOUR_UTC = 8
TOLERANCE_MIN = 60

hour_counts = (
    df_clean["timestamp"].dt.tz_convert("UTC").dt.hour
    .value_counts()
    .reindex(range(24), fill_value=0)
    .sort_index()
)

peak_hours = sorted(hour_counts.nlargest(4).index.tolist())
peaks_table = pd.DataFrame(
    {
        "hora_pico_utc": peak_hours,
        "n_fixes_pico": [int(hour_counts[h]) for h in peak_hours],
        "n_fixes_h_menos_1": [int(hour_counts[(h - 1) % 24]) for h in peak_hours],
        "n_fixes_h_mas_1": [int(hour_counts[(h + 1) % 24]) for h in peak_hours],
    }
)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(
    hour_counts.index, hour_counts.values,
    color="#B0C4DE", edgecolor="black", linewidth=0.6,
)
for h in peak_hours:
    bars[h].set_color("#C04040")
ax.axvspan(
    REFERENCE_HOUR_UTC - TOLERANCE_MIN / 60,
    REFERENCE_HOUR_UTC + TOLERANCE_MIN / 60,
    alpha=0.18, color="#C04040",
    label=f"Ventana elegida: {REFERENCE_HOUR_UTC:02d}:00 ± {TOLERANCE_MIN} min",
)
for h, v in hour_counts.items():
    if v > 0:
        ax.text(h, v + max(hour_counts) * 0.012, f"{int(v):,}",
                ha="center", fontsize=8)
ax.set_xticks(range(24))
ax.set_xlabel("Hora del día (UTC)")
ax.set_ylabel("Número de fixes")
ax.set_title("Distribución horaria de fixes — cuatro picos discretos de muestreo Movebank")
ax.legend(loc="upper left")

save_artifact(
    "hourly-coverage",
    objective="o1",
    num=4,
    decision=f"Hora UTC de referencia fijada en {REFERENCE_HOUR_UTC:02d}:00 (pico de muestreo Movebank, ±{TOLERANCE_MIN} min)",
    caption_es=(
        "Distribución horaria del número de fixes GPS del dataset Movebank "
        "tras la limpieza. El muestreo no se distribuye de forma continua: "
        "se concentra en cuatro ventanas programadas a las 05:00, 08:00, "
        "14:00 y 20:00 UTC, con un volumen muy similar entre ellas "
        "(~15 500 fixes brutos por pico, resaltados en rojo). "
        "La ventana sombreada indica la elegida (08:00 ± 60 min), que "
        "coincide con el inicio del ciclo diario de actividad de Larus "
        "fuscus en gran parte de su rango migratorio. La posición a esa "
        "hora capta el lugar de roost nocturno o el punto inmediatamente "
        "posterior al despegue matinal — un estado espacialmente estable "
        "y bien definido, adecuado para alimentar una cadena de Markov "
        "día-a-día."
    ),
    fig=fig,
    table=peaks_table,
)
plt.close(fig)
print(f"Hora de referencia elegida: {REFERENCE_HOUR_UTC:02d}:00 UTC ± {TOLERANCE_MIN} min")

# %% [markdown]
# ## D3 — Tolerancia ± min alrededor de la hora de referencia

# %%
# Evaluamos cobertura y desfase medio para distintas tolerancias alrededor
# de la hora de referencia ya fijada. La elegida (±60 min) no solapa con
# el pico vecino más cercano (05:00, a 3 h de distancia).

rows = []
for tol in [30, 60, 90, 120, 180]:
    cov_at_ref = float(
        coverage_by_hour(df_clean, tolerance_min=tol)
        .loc[lambda d: d["hour"] == REFERENCE_HOUR_UTC, "coverage"]
        .iloc[0]
    )
    work = df_clean.copy()
    work["date_utc"] = work["timestamp"].dt.tz_convert("UTC").dt.date
    target = pd.Timedelta(hours=REFERENCE_HOUR_UTC)
    work["delta_min"] = (
        work["timestamp"]
        - pd.to_datetime(work["date_utc"]).dt.tz_localize("UTC")
        - target
    ).dt.total_seconds().div(60).abs()
    within = work[work["delta_min"] <= tol]
    closest = within.loc[
        within.groupby(["bird_id", "date_utc"])["delta_min"].idxmin()
    ]
    rows.append(
        {
            "tolerance_min": tol,
            "coverage_pct": cov_at_ref * 100,
            "delta_mean_min": float(closest["delta_min"].mean()) if len(closest) else float("nan"),
            "delta_median_min": float(closest["delta_min"].median()) if len(closest) else float("nan"),
        }
    )
tol_table = pd.DataFrame(rows)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(tol_table["tolerance_min"], tol_table["coverage_pct"],
         marker="o", color="#1f77b4", label="Cobertura (%)")
ax1.set_xlabel("Tolerancia (min)")
ax1.set_ylabel("Cobertura (%)", color="#1f77b4")
ax1.axvline(TOLERANCE_MIN, color="#C04040", linestyle="--",
            alpha=0.6, label=f"Tolerancia elegida ({TOLERANCE_MIN} min)")
ax2 = ax1.twinx()
ax2.plot(tol_table["tolerance_min"], tol_table["delta_mean_min"],
         marker="s", color="#d62728", label="Δ medio (min)")
ax2.set_ylabel("Δ medio al objetivo (min)", color="#d62728")
fig.suptitle("Trade-off cobertura vs. precisión temporal en torno a 08:00 UTC")
ax1.legend(loc="upper left")
fig.tight_layout()

save_artifact(
    "tolerance-tradeoff",
    objective="o1",
    num=5,
    decision=f"Tolerancia alrededor de las {REFERENCE_HOUR_UTC:02d}:00 UTC fijada en ±{TOLERANCE_MIN} min",
    caption_es=(
        "Trade-off entre cobertura del dataset y precisión temporal del "
        "fix elegido a distintas tolerancias alrededor de las 08:00 UTC. "
        "La tolerancia de ±60 min recoge únicamente fixes del pico de "
        "muestreo de las 08:00 (incluyendo los vecinos minoritarios "
        "de 07:00 y 09:00) sin solapar con el pico vecino más cercano "
        "(05:00, a 3 h de distancia). Tolerancias mayores aumentan "
        "marginalmente la cobertura a costa de mezclar muestras de "
        "distintos picos del muestreo."
    ),
    fig=fig,
    table=tol_table,
)
plt.close(fig)
print(f"Tolerancia elegida: ±{TOLERANCE_MIN} min")

# %% [markdown]
# ## D4 — Mínimo de días válidos por individuo

# %%
daily_unfiltered = build_daily(
    df_clean,
    reference_hour_utc=REFERENCE_HOUR_UTC,
    tolerance_min=TOLERANCE_MIN,
)
valid_per_bird = (
    daily_unfiltered.groupby("bird_id")["is_valid"].sum().astype(int)
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(valid_per_bird, bins=40, color="#2E8B57", edgecolor="white")
ax.set_xlabel("Días válidos por individuo")
ax.set_ylabel("Número de aves")
ax.set_title("Distribución de días válidos por individuo")
ax.axvline(30, color="black", linestyle="--", label="Umbral candidato 30")
ax.legend()

cuts = [10, 20, 30, 50, 100]
cuts_table = pd.DataFrame(
    {
        "min_valid_days": cuts,
        "n_birds_kept": [int((valid_per_bird >= c).sum()) for c in cuts],
        "pct_birds_kept": [
            (valid_per_bird >= c).mean() * 100 for c in cuts
        ],
    }
)
MIN_VALID_DAYS = 30
save_artifact(
    "valid-days-per-bird",
    objective="o1",
    num=6,
    decision=f"Mínimo de días válidos por individuo fijado en {MIN_VALID_DAYS}",
    caption_es=(
        "Distribución del número de días válidos por individuo en la "
        "tabla diaria, junto al recuento de aves conservadas según "
        "distintos umbrales candidatos. El umbral elegido descarta "
        "individuos con seguimiento insuficiente para alimentar la "
        "cadena de Markov día a día sin penalizar a la mayoría del "
        "dataset."
    ),
    fig=fig,
    table=cuts_table,
)
plt.close(fig)
print(f"Mínimo de días válidos: {MIN_VALID_DAYS}")

# %% [markdown]
# ## C3 — Desglose acumulado de descartes

# %%
n_initial = len(df_raw)
discards = (
    {"n_initial": n_initial}
    | report_flags
    | report_coords
    | report_speed
    | {"n_clean": len(df_clean)}
)
discard_table = pd.DataFrame(
    [
        {
            "causa": k,
            "n": v,
            "pct_sobre_inicial": v / n_initial * 100 if isinstance(v, int) else float("nan"),
        }
        for k, v in discards.items()
    ]
)
save_artifact(
    "discard-breakdown",
    objective="o1",
    num=7,
    decision="Desglose acumulado de fixes descartados por causa",
    caption_es=(
        "Conteo de fixes descartados en cada fase de la limpieza: marcas "
        "de Movebank (`visible=false` y `manually_marked_outlier`), "
        "coordenadas fuera de rango, duplicados por (individuo, "
        "timestamp) y velocidad imposible. El porcentaje se refiere al "
        "total inicial de fixes leídos del CSV crudo."
    ),
    table=discard_table,
)

# %% [markdown]
# ## C4 — Visión geográfica de los fixes limpios

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sample = df_clean.sample(min(20000, len(df_clean)), random_state=42)
ax.scatter(sample["lon"], sample["lat"], s=1, alpha=0.3, color="#444444")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Distribución espacial de los fixes limpios (muestreo aleatorio)")
ax.grid(alpha=0.3)

save_artifact(
    "spatial-overview",
    objective="o1",
    num=8,
    decision="Caracterización espacial del dataset limpio",
    caption_es=(
        "Distribución geográfica de los fixes GPS supervivientes tras "
        "los filtros de O1 (muestra aleatoria de 20 000 puntos). "
        "Permite verificar el dominio espacial del dataset y la "
        "consistencia con las rutas migratorias conocidas de Larus "
        "fuscus entre Europa septentrional y África occidental."
    ),
    fig=fig,
)
plt.close(fig)

# %% [markdown]
# ## C5 — Longitud de rachas consecutivas (build_o1 se cierra en T7)

# %%
# Construimos la tabla diaria con los valores definidos arriba y aplicamos
# el filtro por días válidos. La materialización a parquet se hace en T7-T8.
daily_unfiltered_full = build_daily(
    df_clean,
    reference_hour_utc=REFERENCE_HOUR_UTC,
    tolerance_min=TOLERANCE_MIN,
)
daily_final = filter_birds_by_validity(
    daily_unfiltered_full, min_valid_days=MIN_VALID_DAYS
)

# Longitudes de rachas consecutivas de is_valid=True por ave.
def _streaks(series: pd.Series) -> list[int]:
    streaks = []
    current = 0
    for v in series:
        if v:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


streak_lens: list[int] = []
for _, group in daily_final.groupby("bird_id"):
    streak_lens.extend(_streaks(group["is_valid"].to_numpy()))

streak_table = pd.DataFrame(
    {
        "percentil": ["p10", "p25", "p50", "p75", "p90", "p95", "max"],
        "longitud_dias": [
            int(np.quantile(streak_lens, q)) if streak_lens else 0
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
        ],
    }
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(streak_lens, 0, 200), bins=60, color="#6A5ACD", edgecolor="white")
ax.set_xlabel("Longitud de la racha (días consecutivos válidos)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de rachas consecutivas de días válidos")
save_artifact(
    "streak-length-distribution",
    objective="o1",
    num=9,
    decision="Caracterización de la fragmentación de las series diarias",
    caption_es=(
        "Distribución de la longitud de las rachas de días consecutivos "
        "con fix válido en la tabla diaria final, agregada sobre todos "
        "los individuos supervivientes. Indica la cantidad de "
        "transiciones día-a-día observables sin saltar huecos y "
        "constituye una entrada relevante para el diseño de la cadena de "
        "Markov de O2."
    ),
    fig=fig,
    table=streak_table,
)
plt.close(fig)
