"""Construcción de features y splits temporales para O4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.markov.discretize import _format_cell_id, assign_cell

# Features cinemáticas causales (conocidas el día t). Ver R6 del spec.
FEATURES_KINEMATIC = [
    "lat", "lon", "sin_doy", "cos_doy",
    "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
]
# Features del HMM causal de O3, pegadas por attach_o3_state_and_split (no se recalculan).
FEATURES_HMM = ["state_b_causal", "posterior_b_migracion_causal"]
# Conjunto supervisado completo de O4 (10 features) + bird_id en personalizado.
FEATURES_O4_CAUSAL = [*FEATURES_KINEMATIC, *FEATURES_HMM]


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
    kin: pd.DataFrame,
    cells: pd.DataFrame,
    include_bird_id: bool,
) -> pd.DataFrame:
    """Construye la matriz de filas candidatas de O4 a partir de la cinemática causal.

    ``kin`` es el ``features.parquet`` de O3 (una fila por (bird_id, date_utc) con
    la cinemática entrante e ``is_hmm_obs_valid``). Pasos:
        1. Asigna ``cell_id_t`` y ``cell_id_t_next`` (target) vía ``cells``.
        2. Añade ``sin_doy``/``cos_doy``.
        3. Filtra a filas candidatas: ``is_hmm_obs_valid`` y target válido
           (``cell_id_t_next`` no nulo ⟺ t+1 consecutivo y celda activa). Es la
           máscara de racha de 4 días (t-2, t-1, t, t+1).

    Las columnas del HMM causal (``state_b_causal``, ``posterior_b_migracion_causal``)
    NO se añaden aquí: las pega ``attach_o3_state_and_split`` desde O3.
    El atributo ``_features`` contiene de momento sólo las cinemáticas (+bird_id).
    """
    df = assign_cells_to_features(kin, cells)
    df = add_cyclic_doy(df)

    df = df[df["is_hmm_obs_valid"] & df["cell_id_t_next"].notna()].copy()

    base = list(FEATURES_KINEMATIC)
    feature_cols = ["bird_id", *base] if include_bird_id else list(base)

    keep_cols = list(dict.fromkeys([
        "bird_id", "date_utc",
        *base,
        "cell_id_t", "cell_id_t_next",
        "lat_t_next", "lon_t_next",
    ]))
    out = df[keep_cols].reset_index(drop=True)
    out.attrs["_features"] = feature_cols
    return out


def attach_o3_state_and_split(
    matrix: pd.DataFrame, features_o3: pd.DataFrame,
) -> pd.DataFrame:
    """Pega state_b_causal/posterior_b_migracion_causal y split desde O3 (merge m:1).

    O3 nombra el posterior ``posterior_b_migracion``; aquí se renombra al
    nombre que consume O4 (``posterior_b_migracion_causal``). Lanza si alguna
    fila candidata queda sin estado o sin etiqueta de split.
    """
    cols = features_o3[[
        "bird_id", "date_utc", "state_b_causal", "posterior_b_migracion", "split",
    ]].rename(columns={"posterior_b_migracion": "posterior_b_migracion_causal"})
    merged = matrix.merge(cols, on=["bird_id", "date_utc"], how="left", validate="m:1")
    missing = (
        merged[["state_b_causal", "posterior_b_migracion_causal", "split"]]
        .isna().any().any()
    )
    if missing:
        raise ValueError("Filas candidatas sin estado HMM causal o sin split tras el merge.")
    return merged


