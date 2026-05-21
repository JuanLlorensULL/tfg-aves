"""Evaluación LOBO + baseline persistencia."""
from __future__ import annotations

import pandas as pd


def lobo_predictions(
    df_transitions: pd.DataFrame,
    cells: list[str],
    cell_deg: float,
    alpha: float,
) -> pd.DataFrame:
    """Predicciones LOBO: 1 fold por ave; matrices re-entrenadas en cada fold."""
    raise NotImplementedError


def persistence_predictions(
    df_transitions: pd.DataFrame, cell_deg: float, n_cells: int
) -> pd.DataFrame:
    """Baseline: ``cell_pred = cell_from``. Distribución casi-delta con ε."""
    raise NotImplementedError


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas en 3 niveles (bird_month, month, global). Formato long."""
    raise NotImplementedError
