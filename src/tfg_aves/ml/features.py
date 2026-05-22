"""Construcción de features y splits temporales para O4."""
from __future__ import annotations

import pandas as pd


def add_cyclic_doy(df: pd.DataFrame, date_col: str = "date_utc") -> pd.DataFrame:
    """Añade columnas ``sin_doy`` y ``cos_doy`` a partir de ``date_col``.

    Devuelve una copia del DataFrame con las dos columnas nuevas.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
