"""Métricas, baselines y comparativas para O4."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def predict_with_meta(
    model: ClassifierMixin,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    label_encoder_y,
    top_k: int = 3,
) -> pd.DataFrame:
    """Devuelve DataFrame con columnas:

    ``bird_id, date_utc, true_cell, pred_cell_top1, pred_cell_topk,
    pred_prob_top1, pred_dist_km, state_b``.

    ``pred_dist_km`` es la distancia geodésica entre el centroide de la
    celda predicha (top-1) y la posición real ``(lat_t_next, lon_t_next)``.
    """
    raise NotImplementedError


def top_k_accuracy(predictions: pd.DataFrame, k: int = 1) -> float:
    """Fracción de filas donde ``true_cell`` está en el top-k."""
    raise NotImplementedError


def dist_median_km(predictions: pd.DataFrame) -> float:
    """Mediana de ``pred_dist_km`` sobre las predicciones."""
    raise NotImplementedError


def evaluate_global(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    label_encoder_y,
) -> dict[str, float]:
    """Devuelve dict con ``top1``, ``top3``, ``log_loss``, ``dist_median_km``."""
    raise NotImplementedError


def evaluate_by_state(
    predictions: pd.DataFrame, state_col: str = "state_b",
) -> pd.DataFrame:
    """Devuelve tabla con filas [global, estacionario, migración] y columnas
    [top1, top3, dist_median_km, n_obs]."""
    raise NotImplementedError


def compute_persistence_baseline(
    matrix_test: pd.DataFrame, *, cells: pd.DataFrame,
) -> pd.DataFrame:
    """Predicción trivial: ``pred_cell_top1 = cell_id_t``. Devuelve el mismo
    esquema que ``predict_with_meta`` para uniformidad."""
    raise NotImplementedError


def compute_markov_baseline(
    matrix_train: pd.DataFrame,
    matrix_test: pd.DataFrame,
    *,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Reentrena Markov(1) sobre el train temporal de O4 (no se reutiliza
    el LOBO de O2) y predice sobre el test temporal. Devuelve el mismo
    esquema que ``predict_with_meta``."""
    raise NotImplementedError


def compare_models(
    metrics_per_model: dict[str, dict], *, baselines: dict[str, dict],
) -> pd.DataFrame:
    """Tabla unificada para la comparativa final (§7 C3 del spec)."""
    raise NotImplementedError
