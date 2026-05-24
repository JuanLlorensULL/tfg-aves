"""Encadenado multi-paso de L3 (demo B de O5).

Realimenta el punto p50 del regresor de cuantiles para construir un track
sintético de varios días. La cinemática entrante se recomputa en cada paso
desde las posiciones; las features del HMM causal se CONGELAN en su último
valor real observado (refinamiento del spec §5.2: la emisión del HMM exige
covariables ambientales veg/daylight atadas a posiciones reales, no
disponibles para posiciones sintéticas futuras). Demo exploratoria: la banda
encadenada NO es la banda calibrada del modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.hmm.features import bearing_rad
from tfg_aves.markov.discretize import haversine_km
from tfg_aves.ml.features import FEATURES_O4_CAUSAL
from tfg_aves.ml.quantile import predict_quantiles


def incoming_kinematics(history: list[tuple[float, float]]) -> dict[str, float]:
    """Cinemática ENTRANTE del último punto de ``history`` (lista de (lat,lon)).

    ``history`` termina en la posición del día actual. Necesita ≥2 puntos
    para step/rumbo; con ≥3 calcula el giro, si no ``cos_turning_in=1.0``
    (neutro). Espejo causal de ``ml.features.compute_causal_kinematics``.
    """
    if len(history) < 2:
        raise ValueError("history necesita al menos 2 posiciones (t-1, t).")
    (lat1, lon1), (lat_t, lon_t) = history[-2], history[-1]
    step_in = float(haversine_km(np.array([lat1]), np.array([lon1]),
                                 np.array([lat_t]), np.array([lon_t]))[0])
    bearing_in = float(bearing_rad(np.array([lat1]), np.array([lon1]),
                                   np.array([lat_t]), np.array([lon_t]))[0])
    cos_turn = 1.0
    if len(history) >= 3:
        (lat2, lon2) = history[-3]
        bearing_prev = float(bearing_rad(np.array([lat2]), np.array([lon2]),
                                         np.array([lat1]), np.array([lon1]))[0])
        turning = (bearing_in - bearing_prev + np.pi) % (2.0 * np.pi) - np.pi
        cos_turn = float(np.cos(turning))
    return {
        "step_in_km": step_in,
        "sin_bearing_in": float(np.sin(bearing_in)),
        "cos_bearing_in": float(np.cos(bearing_in)),
        "cos_turning_in": cos_turn,
    }


def _feature_row(
    lat: float, lon: float, date: pd.Timestamp,
    kin: dict[str, float], state_b: int, posterior_mig: float,
) -> pd.DataFrame:
    """Construye la fila de features (orden FEATURES_O4_CAUSAL) para un paso."""
    doy = date.dayofyear
    angle = 2.0 * np.pi * doy / 365.0
    values = {
        "lat": lat, "lon": lon,
        "sin_doy": np.sin(angle), "cos_doy": np.cos(angle),
        "step_in_km": kin["step_in_km"],
        "sin_bearing_in": kin["sin_bearing_in"],
        "cos_bearing_in": kin["cos_bearing_in"],
        "cos_turning_in": kin["cos_turning_in"],
        "state_b_causal": float(state_b),
        "posterior_b_migracion_causal": float(posterior_mig),
    }
    return pd.DataFrame([[values[c] for c in FEATURES_O4_CAUSAL]], columns=FEATURES_O4_CAUSAL)


def chain_trajectory(
    axis_lat,
    axis_lon,
    seed_history: list[tuple[float, float]],
    *,
    start_date: pd.Timestamp,
    frozen_state_b: int,
    frozen_posterior_mig: float,
    k: int = 7,
) -> pd.DataFrame:
    """Encadena k pasos realimentando el p50. Demo B (ilustrativa).

    Args:
        axis_lat, axis_lon: predictores de eje (``predict_raw`` → (n,3)).
        seed_history: posiciones reales recientes [(.., t-1), (.., t)] (≥2).
        start_date: fecha del día t (la del primer paso predicho t→t+1).
        frozen_state_b, frozen_posterior_mig: features HMM congeladas.
        k: número de pasos a encadenar.

    Devuelve DataFrame por paso: step, date, lat, lon, dlat_p10/50/90,
    dlon_p10/50/90, cone_halfwidth_lat, cone_halfwidth_lon (semianchos
    [p10,p90] acumulados; cono ilustrativo no calibrado).
    """
    history = list(seed_history)
    rows = []
    cum_hw_lat = 0.0
    cum_hw_lon = 0.0
    for step in range(1, k + 1):
        lat_t, lon_t = history[-1]
        # date_t es el día t (origen del paso); se predicen las condiciones de
        # t+1 a partir del estado en t (causal). La fila de salida se fecha en t+1.
        date_t = pd.Timestamp(start_date) + pd.Timedelta(days=step - 1)
        kin = incoming_kinematics(history)
        X = _feature_row(lat_t, lon_t, date_t, kin, frozen_state_b, frozen_posterior_mig)
        qp, _ = predict_quantiles(axis_lat, axis_lon, X)
        pred = qp.iloc[0]  # fila con los 6 cuantiles (dlat_p* y dlon_p*)
        lat_next = lat_t + float(pred["dlat_p50"])
        lon_next = lon_t + float(pred["dlon_p50"])
        cum_hw_lat += (float(pred["dlat_p90"]) - float(pred["dlat_p10"])) / 2.0
        cum_hw_lon += (float(pred["dlon_p90"]) - float(pred["dlon_p10"])) / 2.0
        rows.append({
            "step": step, "date": date_t + pd.Timedelta(days=1),
            "lat": lat_next, "lon": lon_next,
            "dlat_p10": float(pred["dlat_p10"]), "dlat_p50": float(pred["dlat_p50"]),
            "dlat_p90": float(pred["dlat_p90"]),
            "dlon_p10": float(pred["dlon_p10"]), "dlon_p50": float(pred["dlon_p50"]),
            "dlon_p90": float(pred["dlon_p90"]),
            "cone_halfwidth_lat": cum_hw_lat, "cone_halfwidth_lon": cum_hw_lon,
        })
        history.append((lat_next, lon_next))
    return pd.DataFrame(rows)
