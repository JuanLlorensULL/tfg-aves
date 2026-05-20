"""Carga y normalización del CSV crudo de Movebank."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import RAW_CSV

# Columnas de Movebank → snake_case del proyecto.
_RENAME_MAP: dict[str, str] = {
    "event-id": "event_id",
    "timestamp": "timestamp",
    "location-long": "lon",
    "location-lat": "lat",
    "manually-marked-outlier": "manually_marked_outlier",
    "visible": "visible",
    "sensor-type": "sensor_type",
    "individual-local-identifier": "bird_id",
}


def _to_bool_movebank(series: pd.Series) -> pd.Series:
    """Convierte ``'true'`` / ``'false'`` / ``''`` / NaN a ``bool``.

    Movebank codifica los flags como cadenas; ``''`` y NaN se interpretan
    como ``False`` (no marcado).
    """
    return series.fillna("").astype(str).str.lower().eq("true")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV de Movebank y devuelve un DataFrame normalizado.

    - Renombra columnas a snake_case y descarta las covariables ambientales.
    - Convierte ``timestamp`` a ``datetime64[ns, UTC]``.
    - Tipa ``visible`` y ``manually_marked_outlier`` como ``bool``.
    - Ordena por ``(bird_id, timestamp)`` y reindexa.
    - El CSV de Movebank duplica la columna ``visible``; se conserva la
      primera ocurrencia.
    """
    df = pd.read_csv(path)
    # Algunos exports de Movebank repiten 'visible'; nos quedamos con la
    # primera ocurrencia.
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[list(_RENAME_MAP.keys())].rename(columns=_RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["visible"] = _to_bool_movebank(df["visible"])
    df["manually_marked_outlier"] = _to_bool_movebank(
        df["manually_marked_outlier"]
    )
    df = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    return df
