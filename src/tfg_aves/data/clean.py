"""Filtros de outliers sobre fixes GPS de Movebank."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0088
_MAX_SPEED_ITERATIONS = 20


def drop_movebank_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes con ``visible=False`` o ``manually_marked_outlier=True``.

    Devuelve el DataFrame filtrado y un dict con el conteo de descartes
    por causa. Las dos causas se contabilizan con precedencia para
    ``manually_marked_outlier`` (las filas con ambos flags computan como
    descartes por outlier), de modo que la suma de ambas claves coincida
    con el número total de filas eliminadas.
    """
    outlier_mask = df["manually_marked_outlier"]
    visible_only_mask = (~df["visible"]) & (~outlier_mask)
    discard_mask = outlier_mask | (~df["visible"])
    out = df[~discard_mask].reset_index(drop=True)
    report = {
        "discarded_movebank_outlier": int(outlier_mask.sum()),
        "discarded_visible_false": int(visible_only_mask.sum()),
    }
    return out, report


def drop_invalid_coords_and_dupes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta coordenadas fuera de rango y duplicados (bird_id, timestamp).

    Se conserva la primera ocurrencia en caso de duplicado.
    """
    valid_coords = (
        df["lat"].between(-90, 90, inclusive="both")
        & df["lon"].between(-180, 180, inclusive="both")
    )
    discarded_invalid = int((~valid_coords).sum())
    df_coords = df[valid_coords]

    before = len(df_coords)
    df_dedup = df_coords.drop_duplicates(
        subset=["bird_id", "timestamp"], keep="first"
    )
    discarded_dupes = before - len(df_dedup)

    out = df_dedup.reset_index(drop=True)
    return out, {
        "discarded_invalid_coords": discarded_invalid,
        "discarded_duplicates": discarded_dupes,
    }


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Distancia haversine vectorizada en km entre pares de puntos."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _compute_speed_kmh(df: pd.DataFrame) -> np.ndarray:
    """Velocidad en km/h respecto al fix anterior del mismo ``bird_id``.

    Devuelve ``NaN`` en la primera observación de cada ave y donde
    ``Δt`` sea 0 (no debe ocurrir tras el deduplicado).
    """
    df = df.sort_values(["bird_id", "timestamp"])
    bird = df["bird_id"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    ts = df["timestamp"].to_numpy()

    same_bird = np.concatenate([[False], bird[1:] == bird[:-1]])
    dist = np.full(len(df), np.nan)
    dist[1:] = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_h = np.full(len(df), np.nan)
    dt_h[1:] = (
        (ts[1:] - ts[:-1]).astype("timedelta64[s]").astype(float) / 3600.0
    )

    speed = np.full(len(df), np.nan)
    valid = same_bird & (dt_h > 0)
    speed[valid] = dist[valid] / dt_h[valid]
    return speed


def drop_speed_outliers(
    df: pd.DataFrame, max_speed_kmh: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta iterativamente fixes con velocidad > ``max_speed_kmh``.

    Un único fix outlier "envenena" su salto de entrada y de salida; en
    cada iteración descartamos sólo el fix con velocidad máxima (el más
    culpable). Las velocidades de sus vecinos se recalculan en la
    siguiente iteración y, si eran víctimas inocentes, dejan de superar
    el umbral. Se itera hasta ``_MAX_SPEED_ITERATIONS`` veces o hasta
    que ningún fix supere el umbral.
    """
    if max_speed_kmh <= 0:
        raise ValueError("max_speed_kmh debe ser estrictamente positivo")

    current = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    total_discarded = 0
    converged = False
    for _ in range(_MAX_SPEED_ITERATIONS):
        speed = _compute_speed_kmh(current)
        bad = speed > max_speed_kmh
        if not bad.any():
            converged = True
            break
        # `nanargmax` sobre la máscara: usamos -inf para los no-bad de
        # forma que el argmax recaiga siempre en un fix culpable.
        masked = np.where(bad, speed, -np.inf)
        worst = int(np.argmax(masked))
        current = (
            current.drop(current.index[worst]).reset_index(drop=True)
        )
        total_discarded += 1

    if not converged:
        warnings.warn(
            "drop_speed_outliers alcanzó el cap de "
            f"{_MAX_SPEED_ITERATIONS} iteraciones sin converger; "
            "el DataFrame devuelto puede contener fixes con velocidad "
            f"> {max_speed_kmh:g} km/h.",
            stacklevel=2,
        )

    return current, {"discarded_speed": total_discarded}
