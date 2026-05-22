"""Construcción de features y splits temporales para O4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.markov.discretize import _format_cell_id, assign_cell

_FEATURES_BASE = [
    "lat", "lon", "sin_doy", "cos_doy",
    "step_length_km", "cos_turning_angle",
    "state_b", "posterior_b_migracion",
]


def add_cyclic_doy(df: pd.DataFrame, date_col: str = "date_utc") -> pd.DataFrame:
    """Añade columnas ``sin_doy`` y ``cos_doy`` a partir de ``date_col``.

    Devuelve una copia del DataFrame con las dos columnas nuevas.
    """
    out = df.copy()
    doy = pd.to_datetime(out[date_col]).dt.dayofyear
    angle = 2.0 * np.pi * doy / 365.0
    out["sin_doy"] = np.sin(angle)
    out["cos_doy"] = np.cos(angle)
    return out


def assign_cells_to_features(
    df_features: pd.DataFrame,
    cells: pd.DataFrame,
    cell_deg: float = 0.5,
) -> pd.DataFrame:
    """Asigna ``cell_id_t`` y ``cell_id_t_next`` a cada fila de features.

    - ``cell_id_t``: celda de la posición del día t.
    - ``cell_id_t_next``: celda de la posición del día t+1 (mismo bird_id,
      siguiente día calendario). NaN si no existe o no es válido.

    Sólo se mantienen celdas presentes en ``cells.parquet`` (1 217 activas).
    """
    active = set(cells["cell_id"].tolist())

    def _to_cell(lat: float, lon: float) -> str | float:
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        i, j = assign_cell(lat, lon, cell_deg)
        cid = _format_cell_id(i, j)
        return cid if cid in active else np.nan

    out = df_features.copy()
    out = out.sort_values(["bird_id", "date_utc"]).reset_index(drop=True)
    out["cell_id_t"] = [_to_cell(lt, ln) for lt, ln in zip(out["lat"], out["lon"], strict=True)]

    out["lat_t_next"] = out.groupby("bird_id")["lat"].shift(-1)
    out["lon_t_next"] = out.groupby("bird_id")["lon"].shift(-1)
    out["date_t_next"] = out.groupby("bird_id")["date_utc"].shift(-1)
    expected_next = pd.to_datetime(out["date_utc"]) + pd.Timedelta(days=1)
    consecutive = pd.to_datetime(out["date_t_next"]) == expected_next
    out.loc[~consecutive, ["lat_t_next", "lon_t_next"]] = np.nan
    out["cell_id_t_next"] = [
        _to_cell(lt, ln) for lt, ln in zip(out["lat_t_next"], out["lon_t_next"], strict=True)
    ]
    out = out.drop(columns=["date_t_next"])
    return out


def build_feature_matrix(
    features_o3: pd.DataFrame,
    cells: pd.DataFrame,
    include_bird_id: bool,
) -> pd.DataFrame:
    """Construye matriz de features para O4 a partir de ``features.parquet`` de O3.

    Pasos:
        1. Filtra filas con ``is_observation_valid=True`` (heredado de O3).
        2. Asigna ``cell_id_t`` y ``cell_id_t_next`` vía ``cells``.
        3. Filtra filas con ``cell_id_t_next`` no nulo (gap-aware, §8.11).
        4. Añade ``sin_doy``, ``cos_doy``.
        5. Devuelve DataFrame con columnas:
            - clave: ``bird_id``, ``date_utc``
            - features: ``lat``, ``lon``, ``sin_doy``, ``cos_doy``,
              ``step_length_km``, ``cos_turning_angle``, ``state_b``,
              ``posterior_b_migracion`` (+ ``bird_id`` si ``include_bird_id``)
            - target: ``cell_id_t_next``
            - meta para evaluación: ``lat_t_next``, ``lon_t_next``.
    """
    df = features_o3[features_o3["is_observation_valid"]].copy()
    df = assign_cells_to_features(df, cells)
    df = add_cyclic_doy(df)

    df = df[df["cell_id_t_next"].notna()].copy()

    feature_cols = list(_FEATURES_BASE)
    if include_bird_id:
        feature_cols = ["bird_id", *feature_cols]

    keep_cols = list(dict.fromkeys([
        "bird_id", "date_utc",
        *_FEATURES_BASE,
        "cell_id_t", "cell_id_t_next",
        "lat_t_next", "lon_t_next",
    ]))
    out = df[keep_cols].reset_index(drop=True)
    out.attrs["_features"] = feature_cols
    return out


def split_temporal_per_bird(
    matrix: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac_of_train: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal por ave (§F4, F5 del spec).

    Por cada ``bird_id`` ordenado cronológicamente:
        - Primeros ``train_frac`` → bloque (train + val).
        - Últimos ``1 - train_frac`` → test.
        - Dentro de (train + val), los últimos ``val_frac_of_train``
          (proporción del bloque, no del total) → val.

    Defaults: 72 % / 8 % / 20 %.
    """
    train_parts, val_parts, test_parts = [], [], []
    for _bird, sub in matrix.sort_values(["bird_id", "date_utc"]).groupby(
        "bird_id", sort=False,
    ):
        n = len(sub)
        n_train_val = int(round(n * train_frac))
        train_val = sub.iloc[:n_train_val]
        test = sub.iloc[n_train_val:]
        n_val = int(round(len(train_val) * val_frac_of_train))
        train = train_val.iloc[: len(train_val) - n_val]
        val = train_val.iloc[len(train_val) - n_val :]
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    for df in (train_df, val_df, test_df):
        df.attrs["_features"] = matrix.attrs.get("_features", [])
    return train_df, val_df, test_df
