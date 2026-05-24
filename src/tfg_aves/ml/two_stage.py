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


def combine_soft(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    n_classes: int,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla soft canónica con renormalización (§1 del spec).

    Para cada fila i:
        p_final[i, cell_t]      = 1 - p_move[i]
        p_final[i, cell≠cell_t] = p_move[i] * p_2b[i, cell] / (1 - p_2b[i, cell_t])

    El denominador se evalúa como max(1 - p_2b[i, cell_t], eps). Se aplica
    una normalización final por fila para garantizar suma 1.0 incluso en
    el caso degenerado en que clf_dest colapsa sobre cell_t.
    """
    n = p_move.shape[0]
    rows = np.arange(n)
    p2b_cellt = p_2b[rows, cell_t_idx]
    denom = np.maximum(1.0 - p2b_cellt, eps)

    out = p_move[:, None] * p_2b / denom[:, None]
    out[rows, cell_t_idx] = 1.0 - p_move  # sobrescribe la masa de movimiento en cell_t

    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return out / row_sums


def combine_hard(
    p_move: np.ndarray,
    p_2b: np.ndarray,
    cell_t_idx: np.ndarray,
    tau: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Regla hard: cell_t si p_move<tau, si no argmax(p_2b).

    Devuelve (n, n_classes) con masa 1-eps en la celda predicha y
    eps/(n_classes-1) repartida en el resto. El clipping existe sólo por
    compatibilidad con sklearn.metrics.log_loss; el log-loss numérico de
    hard NO es una métrica honesta (cada fallo de argmax suma ~22.86).
    """
    n, n_classes = p_2b.shape
    pred_idx = np.where(p_move < tau, cell_t_idx, p_2b.argmax(axis=1))
    out = np.full((n, n_classes), eps / (n_classes - 1), dtype=np.float64)
    out[np.arange(n), pred_idx] = 1.0 - eps
    return out


def sweep_tau(
    p_move_val: np.ndarray,
    p_2b_val: np.ndarray,
    cell_t_idx_val: np.ndarray,
    y_true_idx_val: np.ndarray,
    n_classes: int,
    taus: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> tuple[float, pd.DataFrame]:
    """Barre tau y devuelve (tau*, tabla) maximizando top-1 sobre val.

    Se usa top-1 (NO log-loss) porque hard devuelve one-hot y su log-loss
    quedaría dominado por el clipping. Empates: gana el tau menor (primero).
    """
    records = []
    for tau in taus:
        out = combine_hard(p_move_val, p_2b_val, cell_t_idx_val, tau)
        top1 = float((out.argmax(axis=1) == y_true_idx_val).mean())
        records.append({"tau": tau, "top1": top1})
    table = pd.DataFrame(records)
    tau_star = float(table.loc[table["top1"].idxmax(), "tau"])
    return tau_star, table


def expand_proba_to_full(
    proba: np.ndarray,
    classes_str: np.ndarray,
    classes_full: np.ndarray,
) -> np.ndarray:
    """Proyecta proba (n, k) al espacio completo (n, m) de classes_full.

    Las columnas de classes_full ausentes en classes_str quedan a 0.
    classes_full debe ser un superconjunto ordenado de classes_str.
    """
    n = proba.shape[0]
    m = len(classes_full)
    out = np.zeros((n, m), dtype=np.float64)
    full_index = {c: j for j, c in enumerate(classes_full)}
    for src_col, c in enumerate(classes_str):
        out[:, full_index[c]] = proba[:, src_col]
    return out


def _centroid_lookup(cells: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {row.cell_id: (row.lat_c, row.lon_c) for row in cells.itertuples()}


def _dist_to_centroid(
    pred_cell: str, lat_next: float, lon_next: float,
    centroids: dict[str, tuple[float, float]],
) -> float:
    if pd.isna(lat_next) or pd.isna(lon_next) or pred_cell not in centroids:
        return np.nan
    c_lat, c_lon = centroids[pred_cell]
    return haversine_km(c_lat, c_lon, lat_next, lon_next)


def predictions_from_proba(
    proba_full: np.ndarray,
    classes_full: np.ndarray,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Construye el DataFrame de predicciones estándar desde proba_full.

    Hermano de evaluate.predict_with_meta pero parte de una matriz de
    probabilidad ya combinada (no de un modelo). Columnas: bird_id,
    date_utc, true_cell, pred_cell_top1, pred_cell_topk, pred_prob_top1,
    pred_dist_km, state_b_causal. Adjunta proba_full y classes_full en attrs.
    """
    top1_idx = np.argmax(proba_full, axis=1)
    topk_idx = np.argsort(proba_full, axis=1)[:, -top_k:][:, ::-1]
    pred_top1 = classes_full[top1_idx]
    pred_topk = [classes_full[row].tolist() for row in topk_idx]
    prob_top1 = proba_full[np.arange(len(proba_full)), top1_idx]

    centroids = _centroid_lookup(cells)
    dists = [
        _dist_to_centroid(pc, lt, ln, centroids) for pc, lt, ln in
        zip(pred_top1, meta["lat_t_next"], meta["lon_t_next"], strict=True)
    ]

    out = pd.DataFrame({
        "bird_id": meta["bird_id"].values,
        "date_utc": meta["date_utc"].values,
        "true_cell": meta["cell_id_t_next"].values,
        "pred_cell_top1": pred_top1,
        "pred_cell_topk": pred_topk,
        "pred_prob_top1": prob_top1,
        "pred_dist_km": dists,
        "state_b_causal": meta["state_b_causal"].values,
    })
    out.attrs["_proba"] = proba_full
    out.attrs["_classes"] = classes_full
    return out


def train_move_rf(
    X_train: pd.DataFrame, y_move: np.ndarray, *, seed: int = 0,
) -> RandomForestClassifier:
    """Etapa 1 RF binaria. Config §8.6 + class_weight='balanced'.

    Poblacional: X_train sólo tiene columnas numéricas (las 8 cinemáticas
    + las 2 del HMM causal), sin bird_id, así que no necesita encoder.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, np.asarray(y_move).astype(int))
    return rf


def train_move_xgb(
    X_train: pd.DataFrame, y_move: np.ndarray, *, seed: int = 0,
) -> xgb.XGBClassifier:
    """Etapa 1 XGBoost binaria. Config §8.6 adaptada a binario.

    n_estimators=300 FIJOS (sin early stopping) para reservar X_val
    exclusivamente a la calibración (F8). objective='binary:logistic',
    scale_pos_weight = n_neg / n_pos calculado sobre y_move.
    """
    y = np.asarray(y_move).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y)
    return model


def calibrate_prefit(
    base: ClassifierMixin, X_val: pd.DataFrame, y_val_move: np.ndarray,
) -> CalibratedClassifierCV:
    """Calibra isotónicamente un modelo ya entrenado usando el val temporal.

    Usa FrozenEstimator (sklearn 1.8; reemplaza cv='prefit'). El base NO se
    reentrena: sólo se ajusta el calibrador sobre (X_val, y_val_move).
    """
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(X_val, np.asarray(y_val_move).astype(int))
    return cal
