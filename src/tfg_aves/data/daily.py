"""Resample a una fila por (bird_id, date_utc) con huecos explícitos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_by_hour(df: pd.DataFrame, tolerance_min: float) -> pd.DataFrame:
    """Cobertura % de (ave, día) con al menos un fix en ``[h ± tol]``.

    Si la ventana cruza medianoche se considera dentro del día propio
    del fix (no envuelve a días vecinos).
    """
    if tolerance_min < 0:
        raise ValueError("tolerance_min debe ser >= 0")

    df = df.copy()
    df["date_utc"] = df["timestamp"].dt.tz_convert("UTC").dt.date
    bird_days = df[["bird_id", "date_utc"]].drop_duplicates()
    total = len(bird_days)
    if total == 0:
        return pd.DataFrame({"hour": list(range(24)), "coverage": [0.0] * 24})

    minute_of_day = (
        df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    ).to_numpy()
    bird = df["bird_id"].to_numpy()
    day = df["date_utc"].to_numpy()

    coverage = np.zeros(24, dtype=float)
    for hour in range(24):
        target = hour * 60
        within = np.abs(minute_of_day - target) <= tolerance_min
        if not within.any():
            coverage[hour] = 0.0
            continue
        covered = pd.DataFrame(
            {"bird_id": bird[within], "date_utc": day[within]}
        ).drop_duplicates()
        coverage[hour] = len(covered) / total

    return pd.DataFrame({"hour": list(range(24)), "coverage": coverage})


def pick_reference_hour(
    df: pd.DataFrame, tolerance_min: float
) -> tuple[int, pd.DataFrame]:
    """Hora UTC con mayor cobertura (empates: hora menor)."""
    cov = coverage_by_hour(df, tolerance_min)
    best = int(cov.loc[cov["coverage"].idxmax(), "hour"])
    return best, cov


def build_daily(
    df: pd.DataFrame,
    reference_hour_utc: int,
    tolerance_min: float,
) -> pd.DataFrame:
    """Colapsa fixes a una fila por (bird_id, date_utc) con huecos explícitos.

    Para cada ``bird_id`` expande el rango ``[first_date, last_date]`` día a
    día y elige el fix más cercano (en minutos) a ``reference_hour_utc``.
    Deja ``NaN`` si no hay fix en ``[reference_hour_utc ± tolerance_min]``.
    """
    if not 0 <= reference_hour_utc <= 23:
        raise ValueError("reference_hour_utc debe estar en [0, 23]")
    if tolerance_min < 0:
        raise ValueError("tolerance_min debe ser >= 0")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "bird_id",
                "date_utc",
                "lat",
                "lon",
                "is_valid",
                "source_event_id",
                "delta_minutes",
            ]
        )

    work = df.copy()
    # Normaliza el nombre del identificador de evento al esperado en la salida.
    if "event_id" in work.columns and "source_event_id" not in work.columns:
        work = work.rename(columns={"event_id": "source_event_id"})
    work["date_utc"] = work["timestamp"].dt.tz_convert("UTC").dt.date
    work["delta_minutes"] = (
        work["timestamp"]
        - pd.to_datetime(work["date_utc"]).dt.tz_localize("UTC")
        - pd.Timedelta(hours=reference_hour_utc)
    ).dt.total_seconds().div(60).abs()

    # Para cada (bird_id, date_utc) nos quedamos con el fix de menor delta.
    work = work.sort_values(
        ["bird_id", "date_utc", "delta_minutes"], kind="stable"
    )
    chosen = work.drop_duplicates(subset=["bird_id", "date_utc"], keep="first")

    # Reconstruimos el rango calendario completo por ave para introducir
    # huecos explícitos.
    frames = []
    for bird_id, group in chosen.groupby("bird_id", sort=True):
        first = group["date_utc"].min()
        last = group["date_utc"].max()
        all_days = pd.date_range(first, last, freq="D").date
        full = pd.DataFrame({"date_utc": all_days, "bird_id": bird_id})
        merged = full.merge(group, on=["bird_id", "date_utc"], how="left")
        frames.append(merged)
    daily = pd.concat(frames, ignore_index=True)

    within_tol = daily["delta_minutes"] <= tolerance_min
    daily["is_valid"] = within_tol.fillna(False)
    # Anulamos lat/lon/source/delta donde no haya fix válido.
    invalid = ~daily["is_valid"]
    for col in ("lat", "lon", "source_event_id", "delta_minutes"):
        daily.loc[invalid, col] = np.nan

    daily["source_event_id"] = daily["source_event_id"].astype("Int64")
    daily = daily[
        [
            "bird_id",
            "date_utc",
            "lat",
            "lon",
            "is_valid",
            "source_event_id",
            "delta_minutes",
        ]
    ].sort_values(["bird_id", "date_utc"]).reset_index(drop=True)
    return daily


def filter_birds_by_validity(
    df_daily: pd.DataFrame, min_valid_days: int
) -> pd.DataFrame:
    """Descarta individuos cuyo ``count(is_valid) < min_valid_days``."""
    if min_valid_days < 0:
        raise ValueError("min_valid_days debe ser >= 0")

    valid_counts = (
        df_daily.groupby("bird_id")["is_valid"].sum().astype(int)
    )
    kept = valid_counts[valid_counts >= min_valid_days].index
    out = df_daily[df_daily["bird_id"].isin(kept)].reset_index(drop=True)
    return out
