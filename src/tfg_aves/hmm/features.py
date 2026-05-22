"""Construcción de features observacionales (cinemáticas + contextuales)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.markov.discretize import haversine_km


def bearing_rad(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Rumbo inicial del trayecto (lat1, lon1) → (lat2, lon2), en radianes [-π, π]."""
    lat1_r = np.deg2rad(lat1)
    lat2_r = np.deg2rad(lat2)
    dlon_r = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dlon_r) * np.cos(lat2_r)
    x = (
        np.cos(lat1_r) * np.sin(lat2_r)
        - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
    )
    return np.arctan2(y, x)


def daylight_hours(lat: float, day_of_year: int) -> float:
    """Horas de luz al mediodía local, dada latitud y día del año (1-366)."""
    # Declinación solar aproximada (fórmula común de Cooper).
    declination = 0.4093 * np.sin(2.0 * np.pi * (day_of_year - 81) / 365.0)
    lat_r = np.deg2rad(lat)
    cos_ha = -np.tan(lat_r) * np.tan(declination)
    # Clip para evitar nan en latitudes árticas/antárticas.
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.arccos(cos_ha)
    return float(2.0 * ha * 24.0 / (2.0 * np.pi))


def load_vegetation_from_raw(
    raw_csv: Path, event_ids: pd.Series
) -> pd.DataFrame:
    """Lee el CSV crudo y extrae veg_low + veg_high para los event_ids solicitados."""
    cols_needed = {
        "event-id": "event_id",
        "ECMWF Interim Full Daily Invariant Low Vegetation Cover": "veg_low",
        "ECMWF Interim Full Daily Invariant High Vegetation Cover": "veg_high",
    }
    df = pd.read_csv(raw_csv, usecols=list(cols_needed.keys()))
    df = df.rename(columns=cols_needed)
    df["event_id"] = df["event_id"].astype("Int64")
    # Filtra al subset solicitado para ahorrar memoria.
    requested = pd.Series(event_ids).dropna().astype("Int64").unique()
    return df[df["event_id"].isin(requested)].reset_index(drop=True)


def compute_observation_features(
    df_daily: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula las 5 features observacionales por (bird_id, date_utc).

    Devuelve DataFrame con las columnas del esquema 6.1 del spec (más
    is_observation_valid; las columnas state/posterior se añadirán en
    Viterbi posterior).
    """
    df = df_daily.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    df["_date_dt"] = pd.to_datetime(df["date_utc"])
    df["_day_of_year"] = df["_date_dt"].dt.dayofyear

    # Posiciones de t-1 y t+1 dentro del mismo bird (por shift).
    same_prev = df["bird_id"].shift(1) == df["bird_id"]
    same_next = df["bird_id"].shift(-1) == df["bird_id"]
    prev_valid = df["is_valid"].shift(1).fillna(False).astype(bool)
    next_valid = df["is_valid"].shift(-1).fillna(False).astype(bool)
    # Consecutividad en calendario:
    delta_prev = df["_date_dt"] - df["_date_dt"].shift(1)
    delta_next = df["_date_dt"].shift(-1) - df["_date_dt"]
    consec_prev = delta_prev == pd.Timedelta(days=1)
    consec_next = delta_next == pd.Timedelta(days=1)

    triplet_mask = (
        df["is_valid"]
        & same_prev & prev_valid & consec_prev
        & same_next & next_valid & consec_next
    )

    # Calcular displacement (t → t+1) sólo para los que tienen triplete.
    lat_t = df["lat"].to_numpy()
    lon_t = df["lon"].to_numpy()
    lat_next = df["lat"].shift(-1).to_numpy()
    lon_next = df["lon"].shift(-1).to_numpy()
    lat_prev = df["lat"].shift(1).to_numpy()
    lon_prev = df["lon"].shift(1).to_numpy()

    disp_km = np.full(len(df), np.nan)
    disp_km_arr = haversine_km(lat_t, lon_t, lat_next, lon_next)
    disp_km[triplet_mask] = np.asarray(disp_km_arr)[triplet_mask]
    log_disp = np.log1p(disp_km)

    # Turning angle = bearing(t, t+1) − bearing(t-1, t).
    bearing_in = bearing_rad(lat_prev, lon_prev, lat_t, lon_t)
    bearing_out = bearing_rad(lat_t, lon_t, lat_next, lon_next)
    turning = np.asarray(bearing_out) - np.asarray(bearing_in)
    # Normalizar a [-π, π] y luego tomar valor absoluto → [0, π].
    turning = (turning + np.pi) % (2.0 * np.pi) - np.pi
    abs_turning = np.full(len(df), np.nan)
    abs_turning[triplet_mask] = np.abs(turning[triplet_mask])

    # Daylight hours sólo para días con lat válida.
    dl = np.full(len(df), np.nan)
    for i in np.where(df["is_valid"].to_numpy())[0]:
        dl[i] = daylight_hours(float(lat_t[i]), int(df["_day_of_year"].iloc[i]))

    out = df.copy()
    out["log_displacement_km"] = log_disp
    out["abs_turning_angle_rad"] = abs_turning
    out["daylight_hours"] = dl
    out["is_observation_valid"] = triplet_mask.to_numpy()

    # Join con vegetación si df_raw está disponible.
    if df_raw is not None and not df_raw.empty:
        veg = df_raw.set_index("event_id")[["veg_low", "veg_high"]]
        out["veg_low"] = out["source_event_id"].map(veg["veg_low"])
        out["veg_high"] = out["source_event_id"].map(veg["veg_high"])
        # Si veg falta, invalida.
        veg_missing = out["veg_low"].isna() | out["veg_high"].isna()
        out.loc[veg_missing, "is_observation_valid"] = False
    else:
        out["veg_low"] = np.nan
        out["veg_high"] = np.nan

    # Limpieza: invalida features para filas no-triplete.
    not_valid = ~out["is_observation_valid"]
    for col in ["log_displacement_km", "abs_turning_angle_rad"]:
        out.loc[not_valid, col] = np.nan

    cols_out = [
        "bird_id", "date_utc", "lat", "lon",
        "log_displacement_km", "abs_turning_angle_rad", "daylight_hours",
        "veg_low", "veg_high", "is_observation_valid",
    ]
    if "source_event_id" in out.columns:
        cols_out.insert(4, "source_event_id")
    return out[cols_out].reset_index(drop=True)
