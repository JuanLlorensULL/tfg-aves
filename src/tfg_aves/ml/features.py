"""Construcción de features y splits temporales para O4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.hmm.features import bearing_rad
from tfg_aves.markov.discretize import _format_cell_id, assign_cell, haversine_km

# Features cinemáticas causales (conocidas el día t). Ver R6 del spec.
FEATURES_KINEMATIC = [
    "lat", "lon", "sin_doy", "cos_doy",
    "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
]
# Features derivadas del HMM causal, añadidas por build_o4 tras el filtrado.
FEATURES_HMM = ["state_b_causal", "posterior_b_migracion_causal"]
# Conjunto supervisado completo de O4 (10 features) + bird_id en personalizado.
FEATURES_O4_CAUSAL = [*FEATURES_KINEMATIC, *FEATURES_HMM]
# Emisión del HMM causal (5, paralela al Modelo B de O3, sin rumbo absoluto).
HMM_EMISSION_COLS = [
    "step_in_km", "cos_turning_in", "veg_low", "veg_high", "daylight_hours",
]

_FEATURES_WIND = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
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


def compute_causal_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Añade cinemática ENTRANTE (tramo t-1 → t) y la máscara is_hmm_obs_valid.

    - ``step_in_km``      = haversine(pos(t-1), pos(t)).
    - ``sin/cos_bearing_in`` = sin/cos del rumbo del tramo t-1 → t (dirección).
    - ``cos_turning_in``  = cos(rumbo(t-1→t) − rumbo(t-2→t-1)) (variabilidad del rumbo).
    - ``is_hmm_obs_valid``: día t con (t-2, t-1, t) válidos y consecutivos en
      calendario y con veg_low/veg_high/daylight_hours presentes (lo que exige
      la emisión del HMM causal).

    Toda la cinemática es función exclusiva de posiciones hasta t inclusive:
    no hay look-ahead. Las filas sin la racha necesaria reciben NaN.
    """
    out = df.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    out["_date_dt"] = pd.to_datetime(out["date_utc"])

    valid = out["lat"].notna() & out["lon"].notna()
    same1 = out["bird_id"].shift(1) == out["bird_id"]
    same2 = out["bird_id"].shift(2) == out["bird_id"]
    consec1 = (out["_date_dt"] - out["_date_dt"].shift(1)) == pd.Timedelta(days=1)
    consec2 = (out["_date_dt"].shift(1) - out["_date_dt"].shift(2)) == pd.Timedelta(days=1)
    valid1 = valid.shift(1).fillna(False).astype(bool)
    valid2 = valid.shift(2).fillna(False).astype(bool)

    lat_t, lon_t = out["lat"].to_numpy(), out["lon"].to_numpy()
    lat_1, lon_1 = out["lat"].shift(1).to_numpy(), out["lon"].shift(1).to_numpy()
    lat_2, lon_2 = out["lat"].shift(2).to_numpy(), out["lon"].shift(2).to_numpy()

    # Tramo entrante t-1 → t: necesita t-1 válido y consecutivo.
    mask_in = (valid & valid1 & same1 & consec1).to_numpy()
    step_in = np.full(len(out), np.nan)
    step_in[mask_in] = np.asarray(haversine_km(lat_1, lon_1, lat_t, lon_t))[mask_in]

    # Los rumbos/distancias se calculan sobre el array completo (coords NaN
    # producen NaN); sólo las posiciones enmascaradas se escriben en la salida.
    # Patrón deliberado, paralelo a tfg_aves.hmm.features.compute_observation_features.
    bearing_in = np.asarray(bearing_rad(lat_1, lon_1, lat_t, lon_t))
    sin_b = np.full(len(out), np.nan)
    cos_b = np.full(len(out), np.nan)
    sin_b[mask_in] = np.sin(bearing_in)[mask_in]
    cos_b[mask_in] = np.cos(bearing_in)[mask_in]

    # Giro causal: necesita además t-2 válido y consecutivo.
    mask_turn = mask_in & (valid2 & same2 & consec2).to_numpy()
    bearing_prev = np.asarray(bearing_rad(lat_2, lon_2, lat_1, lon_1))
    turning = (bearing_in - bearing_prev + np.pi) % (2.0 * np.pi) - np.pi
    cos_turn = np.full(len(out), np.nan)
    cos_turn[mask_turn] = np.cos(turning)[mask_turn]

    out["step_in_km"] = step_in
    out["sin_bearing_in"] = sin_b
    out["cos_bearing_in"] = cos_b
    out["cos_turning_in"] = cos_turn

    veg_ok = (
        out["veg_low"].notna() & out["veg_high"].notna() & out["daylight_hours"].notna()
    ).to_numpy()
    out["is_hmm_obs_valid"] = mask_turn & veg_ok

    return out.drop(columns=["_date_dt"])


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
    *,
    wind_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye la matriz de filas candidatas de O4 a partir de la cinemática causal.

    ``kin`` es la salida de ``compute_causal_kinematics`` (todas las filas con la
    cinemática entrante e ``is_hmm_obs_valid``). Pasos:
        1. Asigna ``cell_id_t`` y ``cell_id_t_next`` (target) vía ``cells``.
        2. Añade ``sin_doy``/``cos_doy``.
        3. (Opcional) fusiona viento — rama L1, no ejercitada en el rework.
        4. Filtra a filas candidatas: ``is_hmm_obs_valid`` y target válido
           (``cell_id_t_next`` no nulo ⟺ t+1 consecutivo y celda activa). Es la
           máscara de racha de 4 días (t-2, t-1, t, t+1).

    Las columnas del HMM causal (``state_b_causal``, ``posterior_b_migracion_causal``)
    NO se añaden aquí: las inserta ``build_o4`` tras ajustar y filtrar el HMM.
    El atributo ``_features`` contiene de momento sólo las cinemáticas (+bird_id).
    """
    df = assign_cells_to_features(kin, cells)
    df = add_cyclic_doy(df)

    if wind_df is not None:
        df = merge_wind_features(df, wind_df)

    df = df[df["is_hmm_obs_valid"] & df["cell_id_t_next"].notna()].copy()

    # RandomForest no tolera NaN; XGBoost sí. Para coherencia entre familias
    # descartamos las pocas filas con NaN en features de viento (20 fixes,
    # 0,09 %, fuera del bbox del .nc: Bélgica/Países Bajos, otoño 2009).
    if wind_df is not None:
        df = df.dropna(subset=_FEATURES_WIND).copy()

    base = list(FEATURES_KINEMATIC)
    if wind_df is not None:
        base = [*base, *_FEATURES_WIND]
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


def merge_wind_features(
    matrix: pd.DataFrame,
    wind_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une matrix con las 3 features de viento por (bird_id, date_utc).

    LEFT JOIN: el número de filas de matrix se preserva. Filas sin
    contrapartida en wind_df reciben NaN en wind_u_850, wind_v_850,
    wind_speed_850 (esperado raro, son fixes fuera del bbox del .nc).

    Args:
        matrix: salida parcial de build_feature_matrix antes del filtro
            de cell_id_t_next, con columnas bird_id, date_utc.
        wind_df: salida de build_wind (bird_id, date_utc + 3 features).

    Returns:
        DataFrame con todas las columnas de matrix + las 3 de viento.
    """
    if not {"bird_id", "date_utc"}.issubset(matrix.columns):
        raise ValueError("matrix necesita columnas bird_id y date_utc.")
    if not {"bird_id", "date_utc", *_FEATURES_WIND}.issubset(wind_df.columns):
        raise ValueError(
            "wind_df necesita bird_id, date_utc y las 3 features de viento.",
        )

    # Asegurar tipos compatibles para el join (date_utc como objeto Python date).
    m = matrix.copy()
    w = wind_df[["bird_id", "date_utc", *_FEATURES_WIND]].copy()

    n_before = len(m)
    out = m.merge(w, on=["bird_id", "date_utc"], how="left", validate="m:1")
    if len(out) != n_before:
        raise AssertionError(
            f"merge cambió número de filas: {n_before} → {len(out)} "
            "(¿wind_df tiene claves duplicadas?)",
        )
    return out
