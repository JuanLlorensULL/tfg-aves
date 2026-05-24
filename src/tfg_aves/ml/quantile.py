"""Regresión de cuantiles del desplazamiento (L3 de O4).

Modela el target continuo (Δlat, Δlon) en grados con XGBoost
(`objective='reg:quantileerror'`), tres cuantiles {p10,p50,p90} por eje.
Las predicciones imitan el esquema de las de clasificación para reutilizar
``tfg_aves.ml.evaluate`` sin cambios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from tfg_aves.markov.discretize import _format_cell_id, assign_cell, haversine_km

QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
INDIVIDUAL_BIRD_ID: str = "91916A"   # ave con más histórico (rank 1, 2025 filas)
CELL_DEG: float = 0.5


def derive_displacement_target(matrix: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con y_dlat, y_dlon en grados.

    y_dlat = lat_t_next − lat ; y_dlon = lon_t_next − lon. No tiene fuga:
    ambas definen la etiqueta, no son features de entrada.
    """
    return pd.DataFrame({
        "y_dlat": matrix["lat_t_next"].to_numpy() - matrix["lat"].to_numpy(),
        "y_dlon": matrix["lon_t_next"].to_numpy() - matrix["lon"].to_numpy(),
    })


class _XGBQuantileRegressor(BaseEstimator, RegressorMixin):
    """Wrapper fino sobre XGBRegressor(objective='reg:quantileerror').

    Config conservadora §8.6 de O4 adaptada a regresión (F8 del spec),
    idéntica para todos los cuantiles y modos; sólo varía quantile_alpha.
    Early stopping sobre val con la pérdida pinball por defecto del objetivo.
    Ningún modo usa bird_id, así que no hace falta codificar categóricas.
    """

    def __init__(self, quantile: float, seed: int = 0) -> None:
        self.quantile = quantile
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "_XGBQuantileRegressor":
        self._reg = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.quantile,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
        self._reg.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._reg.predict(X), dtype=np.float64)

    @property
    def best_iteration(self) -> int | None:
        return getattr(self._reg, "best_iteration", None)


def fit_quantile_axis(
    X_train: pd.DataFrame,
    y_train_axis: np.ndarray,
    X_val: pd.DataFrame,
    y_val_axis: np.ndarray,
    *,
    seed: int = 0,
) -> dict[float, _XGBQuantileRegressor]:
    """Entrena los 3 regresores {p10,p50,p90} para UN eje (Δlat o Δlon)."""
    models: dict[float, _XGBQuantileRegressor] = {}
    for q in QUANTILES:
        models[q] = _XGBQuantileRegressor(quantile=q, seed=seed).fit(
            X_train, np.asarray(y_train_axis, dtype=np.float64),
            X_val, np.asarray(y_val_axis, dtype=np.float64),
        )
    return models
