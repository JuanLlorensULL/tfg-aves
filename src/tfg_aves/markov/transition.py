"""Construcción de pares de transición y matrices de counts mensuales."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_transitions(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Convierte un DataFrame diario discretizado en pares válidos (t, t+1).

    Una transición se emite si y sólo si: misma ave, ``is_valid`` en ambos
    días, fecha consecutiva (``date_{t+1} - date_t == 1 día``).
    Mes de referencia: el del **origen**.
    """
    raise NotImplementedError


def build_counts(
    df_transitions: pd.DataFrame, cells: list[str]
) -> np.ndarray:
    """Tensor de counts ``(12, n_cells, n_cells)`` por mes y par de celdas."""
    raise NotImplementedError
