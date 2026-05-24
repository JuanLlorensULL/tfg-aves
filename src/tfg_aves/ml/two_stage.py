"""Funciones puras del pipeline L2 (modelo de dos etapas) de O4 causal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from tfg_aves.markov.discretize import haversine_km


def derive_y_move(df: pd.DataFrame) -> pd.Series:
    """Devuelve y_move = (cell_id_t_next != cell_id_t) como Series booleana."""
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).rename("y_move")
