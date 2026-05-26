"""Regresión de cuantiles del desplazamiento (L3 de O4).

Modela el target continuo (Δlat, Δlon) en grados con tres familias:
XGBoost (`reg:quantileerror`) y LightGBM (`objective='quantile'`) con pérdida
pinball nativa, y Random Forest vía Quantile Regression Forest. Tres cuantiles
{p10, p50, p90} por eje; las predicciones son compatibles con ``tfg_aves.ml.evaluate``.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

from tfg_aves.markov.discretize import _format_cell_id, haversine_km

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
    ) -> _XGBQuantileRegressor:
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": self.quantile,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": -1,
        }
        if has_val:
            params["early_stopping_rounds"] = 50
        self._reg = xgb.XGBRegressor(**params)
        eval_set = [(X_val, y_val)] if has_val else None
        self._reg.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._reg.predict(X), dtype=np.float64)

    @property
    def best_iteration(self) -> int | None:
        return getattr(self._reg, "best_iteration", None)


class _LGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """Espejo de _XGBQuantileRegressor con la API de LightGBM
    (objective='quantile', alpha=q). Pérdida pinball nativa, un cuantil por
    instancia. Config conservadora análoga a §8.6 de O4 (G4 del spec).
    Early stopping sobre val con metric='quantile' si val no está vacío."""

    def __init__(self, quantile: float, seed: int = 0) -> None:
        self.quantile = quantile
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> _LGBMQuantileRegressor:
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        self._reg = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.quantile,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,      # necesario para que subsample<1 actúe en LGBM
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(50, verbose=False)] if has_val else None
        eval_set = [(X_val, y_val)] if has_val else None
        self._reg.fit(
            X, y, eval_set=eval_set, eval_metric="quantile", callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._reg.predict(X), dtype=np.float64)


class _PerQuantileAxis:
    """Predictor de eje que envuelve un regresor independiente por cuantil
    (XGBoost o LightGBM). ``predict_raw`` apila los 3 cuantiles en orden
    QUANTILES → matriz (n, 3)."""

    def __init__(self, models: dict[float, object]) -> None:
        self.models = models

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [np.asarray(self.models[q].predict(X), dtype=np.float64) for q in QUANTILES]
        )


class _QRFAxis:
    """Predictor de eje basado en Quantile Regression Forest: un único bosque
    estima los tres cuantiles desde la distribución empírica de cada hoja
    (Meinshausen 2006). Los cuantiles son monótonos por construcción."""

    def __init__(self, forest: object) -> None:
        self.forest = forest

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(
            self.forest.predict(X, quantiles=list(QUANTILES)), dtype=np.float64
        )
        return out.reshape(len(X), len(QUANTILES))


def fit_quantile_axis(
    X_train: pd.DataFrame,
    y_train_axis: np.ndarray,
    X_val: pd.DataFrame,
    y_val_axis: np.ndarray,
    *,
    family: str = "xgb",
    seed: int = 0,
) -> _PerQuantileAxis | _QRFAxis:
    """Entrena el predictor de cuantiles de UN eje (Δlat o Δlon) para la
    familia indicada y lo devuelve tras una interfaz uniforme ``predict_raw``.

    - ``xgb``/``lgbm``: un regresor single-quantile por cuantil (pérdida
      pinball nativa), con early stopping sobre val.
    - ``rf``: un único Quantile Regression Forest (ignora val: no hay early
      stopping en bagging).
    """
    y_tr = np.asarray(y_train_axis, dtype=np.float64)
    if family in ("xgb", "lgbm"):
        cls = _XGBQuantileRegressor if family == "xgb" else _LGBMQuantileRegressor
        y_va = np.asarray(y_val_axis, dtype=np.float64)
        models = {
            q: cls(quantile=q, seed=seed).fit(X_train, y_tr, X_val, y_va)
            for q in QUANTILES
        }
        return _PerQuantileAxis(models)
    if family == "rf":
        from quantile_forest import RandomForestQuantileRegressor
        forest = RandomForestQuantileRegressor(
            n_estimators=300,
            min_samples_leaf=20,   # CRÍTICO en QRF: hojas con muestras suficientes
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        ).fit(X_train, y_tr)
        return _QRFAxis(forest)
    raise ValueError(f"familia de regresión desconocida: {family!r}")


def _axis_quantiles_sorted(
    axis: _PerQuantileAxis | _QRFAxis, X: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    """Devuelve (matriz (n,3) ordenada por fila, nº de filas con cruce)."""
    raw = np.asarray(axis.predict_raw(X), dtype=np.float64)
    ordered = (raw[:, 0] <= raw[:, 1]) & (raw[:, 1] <= raw[:, 2])
    n_crossings = int(np.sum(~ordered))
    sorted_q = np.sort(raw, axis=1)
    return sorted_q, n_crossings


def predict_quantiles(
    axis_lat: _PerQuantileAxis | _QRFAxis,
    axis_lon: _PerQuantileAxis | _QRFAxis,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Predice los 6 cuantiles, fuerza monotonía por eje (np.sort) y devuelve
    (DataFrame con dlat_p10/50/90, dlon_p10/50/90; dict de cruces ANTES de
    ordenar, para C5). Acepta cualquier predictor de eje con predict_raw."""
    lat_q, n_cross_lat = _axis_quantiles_sorted(axis_lat, X)
    lon_q, n_cross_lon = _axis_quantiles_sorted(axis_lon, X)
    out = pd.DataFrame({
        "dlat_p10": lat_q[:, 0], "dlat_p50": lat_q[:, 1], "dlat_p90": lat_q[:, 2],
        "dlon_p10": lon_q[:, 0], "dlon_p50": lon_q[:, 1], "dlon_p90": lon_q[:, 2],
    })
    return out, {"lat": n_cross_lat, "lon": n_cross_lon}


def point_to_cell(
    lat_pred: np.ndarray, lon_pred: np.ndarray, cells: pd.DataFrame,
    cell_deg: float = CELL_DEG,
) -> pd.DataFrame:
    """Mapea cada punto a su celda contenedora (discretización al grid 0,5°).

    Devuelve un DataFrame con: cell_id (str del grid, activo o no), cent_lat,
    cent_lon (centroide analítico de la celda contenedora) e is_active (bool,
    si la celda está en ``cells``). El centroide es analítico para que la
    distancia vía centroide quede definida también fuera del grid activo.
    lat_pred/lon_pred no deben contener NaN (el floor de NaN produce un cell_id sin sentido).
    """
    lat_pred = np.asarray(lat_pred, dtype=np.float64)
    lon_pred = np.asarray(lon_pred, dtype=np.float64)
    i = np.floor(lat_pred / cell_deg).astype(int)
    j = np.floor(lon_pred / cell_deg).astype(int)
    cell_ids = [_format_cell_id(int(a), int(b)) for a, b in zip(i, j, strict=True)]
    cent_lat = (i + 0.5) * cell_deg
    cent_lon = (j + 0.5) * cell_deg
    active = set(cells["cell_id"].astype(str))
    is_active = np.array([c in active for c in cell_ids])
    return pd.DataFrame({
        "cell_id": cell_ids, "cent_lat": cent_lat, "cent_lon": cent_lon,
        "is_active": is_active,
    })


def nearest_cells(
    lat_pred: float, lon_pred: float, cells: pd.DataFrame, k: int = 3,
) -> list[str]:
    """k celdas activas cuyo centroide está más cerca del punto (proximidad)."""
    d = np.asarray(haversine_km(
        cells["lat_c"].to_numpy(), cells["lon_c"].to_numpy(),
        float(lat_pred), float(lon_pred),
    ))
    order = np.argsort(d)[:k]
    return [str(c) for c in cells["cell_id"].to_numpy()[order]]


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pérdida de cuantil (pinball) media para el cuantil q."""
    d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred_q, dtype=np.float64)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def interval_coverage(
    y_true: np.ndarray, y_p10: np.ndarray, y_p90: np.ndarray,
) -> float:
    """Fracción de y_true dentro de [p10, p90]. Ideal ≈ 0.80."""
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y >= np.asarray(y_p10)) & (y <= np.asarray(y_p90))))


def build_regression_predictions(
    quantile_preds: pd.DataFrame,
    meta: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    k_top: int = 3,
    state_col: str = "state_b_causal",
) -> pd.DataFrame:
    """Ensambla predicciones con el MISMO esquema que la clasificación
    (true_cell, pred_cell_top1, pred_cell_topk, pred_prob_top1, pred_dist_km,
    la columna de estado) MÁS columnas de regresión (pred_lat, pred_lon,
    dist_native_km, dlat_p*/dlon_p*, in_interval_lat, in_interval_lon).

    - pred_cell_top1 = celda contenedora del punto p50 (discretización).
    - pred_cell_topk = k celdas activas más cercanas (proximidad).
    - pred_dist_km   = haversine(centroide de la celda contenedora, verdad t+1)
                       → comparable con dist_median_km de L1/L2.
    - dist_native_km = haversine(punto p50, verdad t+1) → métrica nativa.
    - in_interval_*  = el desplazamiento verdadero cae en [p10, p90] (por eje).
    """
    meta = meta.reset_index(drop=True)
    qp = quantile_preds.reset_index(drop=True)

    lat_t = meta["lat"].to_numpy(dtype=np.float64)
    lon_t = meta["lon"].to_numpy(dtype=np.float64)
    lat_next = meta["lat_t_next"].to_numpy(dtype=np.float64)
    lon_next = meta["lon_t_next"].to_numpy(dtype=np.float64)

    pred_lat = lat_t + qp["dlat_p50"].to_numpy()
    pred_lon = lon_t + qp["dlon_p50"].to_numpy()

    mapped = point_to_cell(pred_lat, pred_lon, cells)
    pred_dist_km = np.asarray(haversine_km(
        mapped["cent_lat"].to_numpy(), mapped["cent_lon"].to_numpy(),
        lat_next, lon_next,
    ))
    dist_native_km = np.asarray(haversine_km(pred_lat, pred_lon, lat_next, lon_next))

    topk = [
        nearest_cells(pl, pn, cells, k=k_top)
        for pl, pn in zip(pred_lat, pred_lon, strict=True)
    ]

    y_dlat_true = lat_next - lat_t
    y_dlon_true = lon_next - lon_t
    in_lat = (y_dlat_true >= qp["dlat_p10"].to_numpy()) & (
        y_dlat_true <= qp["dlat_p90"].to_numpy())
    in_lon = (y_dlon_true >= qp["dlon_p10"].to_numpy()) & (
        y_dlon_true <= qp["dlon_p90"].to_numpy())

    out = pd.DataFrame({
        "bird_id": meta["bird_id"].to_numpy(),
        "date_utc": meta["date_utc"].to_numpy(),
        "true_cell": meta["cell_id_t_next"].to_numpy(),
        "pred_cell_top1": mapped["cell_id"].to_numpy(),
        "pred_cell_topk": topk,
        "pred_prob_top1": np.nan,
        "pred_dist_km": pred_dist_km,
        state_col: meta[state_col].to_numpy(),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "dist_native_km": dist_native_km,
        "dlat_p10": qp["dlat_p10"].to_numpy(),
        "dlat_p50": qp["dlat_p50"].to_numpy(),
        "dlat_p90": qp["dlat_p90"].to_numpy(),
        "dlon_p10": qp["dlon_p10"].to_numpy(),
        "dlon_p50": qp["dlon_p50"].to_numpy(),
        "dlon_p90": qp["dlon_p90"].to_numpy(),
        "in_interval_lat": in_lat,
        "in_interval_lon": in_lon,
        "pred_cell_in_active_grid": mapped["is_active"].to_numpy(),
    })
    return out
