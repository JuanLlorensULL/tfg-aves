"""Entrenamiento de los tres modelos supervisados con configuración fija (§8.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena RF con configuración conservadora (§8.6 del spec).

    Aplica label encoding a ``categorical_cols`` internamente (RF no tiene
    soporte nativo para categóricas). Devuelve el modelo entrenado con un
    atributo ``_label_encoders`` adjunto para que ``predict_proba`` sea
    reusable sobre nuevos datos.
    """
    raise NotImplementedError


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena XGBoost (multi:softprob, hist) con early stopping (§8.6).

    Aplica label encoding a ``categorical_cols``. Early stopping
    ``early_stopping_rounds=50`` sobre log-loss en val.
    """
    raise NotImplementedError


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    categorical_cols: list[str],
    seed: int = 0,
) -> ClassifierMixin:
    """Entrena LightGBM con categóricas nativas y early stopping (§8.6)."""
    raise NotImplementedError
