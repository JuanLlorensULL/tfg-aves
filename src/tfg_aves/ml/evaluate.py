"""Métricas, baselines y comparativas para O4."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss

from tfg_aves.markov.discretize import haversine_km


def _cell_centroid_lookup(cells: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {row.cell_id: (row.lat_c, row.lon_c) for row in cells.itertuples()}


def _haversine_to_centroid(
    pred_cell: str,
    lat_next: float,
    lon_next: float,
    centroids: dict[str, tuple[float, float]],
) -> float:
    """Distancia haversine del centroide de ``pred_cell`` a (lat_next, lon_next)."""
    if pd.isna(lat_next) or pd.isna(lon_next) or pred_cell not in centroids:
        return np.nan
    c_lat, c_lon = centroids[pred_cell]
    return haversine_km(c_lat, c_lon, lat_next, lon_next)


def top_k_accuracy(predictions: pd.DataFrame, k: int = 1) -> float:
    """Fracción de filas donde ``true_cell`` está en el top-k."""
    if k == 1:
        return float((predictions["true_cell"] == predictions["pred_cell_top1"]).mean())
    return float(
        predictions.apply(
            lambda r: r["true_cell"] in (
                list(r["pred_cell_topk"]) if r["pred_cell_topk"] is not None else []
            ),
            axis=1,
        ).mean()
    )


def dist_median_km(predictions: pd.DataFrame) -> float:
    """Mediana de la distancia geodésica predicha."""
    return float(np.median(predictions["pred_dist_km"]))


def evaluate_by_state(
    predictions: pd.DataFrame, state_col: str = "state_b_causal",
) -> pd.DataFrame:
    """Tabla con métricas global / estacionario / migración."""
    rows = [{
        "state": "global",
        "n_obs": len(predictions),
        "top1": top_k_accuracy(predictions, k=1),
        "top3": top_k_accuracy(predictions, k=3),
        "dist_median_km": dist_median_km(predictions),
    }]
    for state_val, label in [(0, "estacionario"), (1, "migración")]:
        sub = predictions[predictions[state_col] == state_val]
        if len(sub) == 0:
            rows.append({
                "state": label, "n_obs": 0,
                "top1": np.nan, "top3": np.nan, "dist_median_km": np.nan,
            })
            continue
        rows.append({
            "state": label,
            "n_obs": len(sub),
            "top1": top_k_accuracy(sub, k=1),
            "top3": top_k_accuracy(sub, k=3),
            "dist_median_km": dist_median_km(sub),
        })
    return pd.DataFrame(rows)


def predict_with_meta(
    model: ClassifierMixin,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    label_encoder_y,
    top_k: int = 3,
) -> pd.DataFrame:
    """Devuelve DataFrame con predicciones + meta para evaluación.

    Columnas: bird_id, date_utc, true_cell, pred_cell_top1, pred_cell_topk,
    pred_prob_top1, pred_dist_km, state_b_causal. Adjunta proba y classes en attrs.
    """
    proba = model.predict_proba(X)
    classes_enc = model.classes_  # ints (codificados)
    classes_str = label_encoder_y.inverse_transform(classes_enc)

    top1_idx = np.argmax(proba, axis=1)
    topk_idx = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]
    pred_top1 = classes_str[top1_idx]
    pred_topk = [classes_str[row].tolist() for row in topk_idx]
    prob_top1 = proba[np.arange(len(proba)), top1_idx]

    centroids = _cell_centroid_lookup(cells)

    dists = [
        _haversine_to_centroid(pc, lt, ln, centroids) for pc, lt, ln in
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
    out.attrs["_proba"] = proba
    out.attrs["_classes"] = classes_str
    return out


def evaluate_global(
    model: ClassifierMixin,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    cells: pd.DataFrame,
    label_encoder_y,
) -> dict[str, float]:
    """Métricas globales: top1, top3, log_loss, dist_median_km.

    Las etiquetas verdaderas se obtienen directamente de ``meta["cell_id_t_next"]``
    para evitar inconsistencias de codificación cuando el conjunto de evaluación
    contiene celdas no vistas en entrenamiento. La probabilidad se expande al
    espacio completo de clases observadas (train + eval), asignando 0 a las
    clases ausentes en el modelo.
    """
    preds = predict_with_meta(
        model, X, meta, cells=cells, label_encoder_y=label_encoder_y,
    )
    proba_model = preds.attrs["_proba"]
    classes_model = preds.attrs["_classes"]  # strings, sólo las vistas en train

    # Verdaderas etiquetas de clase (strings) tomadas directamente del meta
    y_str = np.asarray(meta["cell_id_t_next"].astype(str))

    # Espacio completo de clases: unión de las del modelo y las del eval set
    all_classes_set = sorted(set(classes_model) | set(y_str))
    all_classes = np.array(all_classes_set)

    if len(all_classes) > len(classes_model):
        # Expandir la matriz de probabilidades: columnas de clases ausentes = 0
        n = len(proba_model)
        proba_full = np.zeros((n, len(all_classes)), dtype=np.float64)
        model_col_idx = np.where(np.isin(all_classes, classes_model))[0]
        proba_full[:, model_col_idx] = proba_model
    else:
        proba_full = proba_model

    ll = log_loss(y_str, proba_full, labels=list(all_classes))
    return {
        "top1": top_k_accuracy(preds, k=1),
        "top3": top_k_accuracy(preds, k=3),
        "log_loss": float(ll),
        "dist_median_km": dist_median_km(preds),
    }


def compute_persistence_baseline(
    matrix_test: pd.DataFrame, *, cells: pd.DataFrame,
) -> pd.DataFrame:
    """Persistencia trivial: mañana = hoy."""
    centroids = _cell_centroid_lookup(cells)

    return pd.DataFrame({
        "bird_id": matrix_test["bird_id"].values,
        "date_utc": matrix_test["date_utc"].values,
        "true_cell": matrix_test["cell_id_t_next"].values,
        "pred_cell_top1": matrix_test["cell_id_t"].values,
        "pred_cell_topk": [[c] for c in matrix_test["cell_id_t"].values],
        "pred_prob_top1": [1.0] * len(matrix_test),
        "pred_dist_km": [
            _haversine_to_centroid(c, lt, ln, centroids) for c, lt, ln in zip(
                matrix_test["cell_id_t"],
                matrix_test["lat_t_next"],
                matrix_test["lon_t_next"],
                strict=True,
            )
        ],
        "state_b_causal": matrix_test["state_b_causal"].values,
    })


def compute_markov_baseline(
    matrix_train: pd.DataFrame,
    matrix_test: pd.DataFrame,
    *,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Reentrena Markov(1) mensual sobre el train temporal de O4 y predice
    sobre el test temporal (mismo modelo que O2, distinto split)."""
    from tfg_aves.markov.smooth import laplace_smooth
    from tfg_aves.markov.transition import build_counts, build_transitions

    train_t = matrix_train.rename(columns={"cell_id_t": "cell_id"})[
        ["bird_id", "date_utc", "lat", "lon", "cell_id"]
    ].copy()
    train_t["is_valid"] = True
    train_next = (
        matrix_train
        .assign(
            cell_id=matrix_train["cell_id_t_next"],
            lat=matrix_train["lat_t_next"],
            lon=matrix_train["lon_t_next"],
            date_utc=matrix_train["date_utc"] + pd.Timedelta(days=1),
            is_valid=True,
        )[["bird_id", "date_utc", "lat", "lon", "cell_id", "is_valid"]]
        .copy()
    )
    train_input = (
        pd.concat([train_t, train_next])
        .drop_duplicates(subset=["bird_id", "date_utc"])
        .sort_values(["bird_id", "date_utc"])
        .reset_index(drop=True)
    )

    transitions = build_transitions(train_input)
    cell_list = cells["cell_id"].tolist()
    counts = build_counts(transitions, cell_list)
    proba_by_month = laplace_smooth(counts, alpha=1.0)  # (12, n_cells, n_cells)

    centroids = _cell_centroid_lookup(cells)
    cell_to_idx = {c: i for i, c in enumerate(cell_list)}
    n_cells = len(cell_list)

    rows = []
    for r in matrix_test.itertuples():
        month_idx = pd.Timestamp(r.date_utc).month - 1
        idx_from = cell_to_idx.get(r.cell_id_t)
        if idx_from is None:
            pred_top1 = r.cell_id_t  # fallback persistencia
            topk = [pred_top1]
            prob = 1.0 / n_cells
        else:
            row_probs = proba_by_month[month_idx, idx_from]
            order = np.argsort(row_probs)[::-1]
            pred_top1 = cell_list[order[0]]
            topk = [cell_list[i] for i in order[:3]]
            prob = float(row_probs[order[0]])
        rows.append({
            "bird_id": r.bird_id, "date_utc": r.date_utc,
            "true_cell": r.cell_id_t_next,
            "pred_cell_top1": pred_top1,
            "pred_cell_topk": topk,
            "pred_prob_top1": prob,
            "pred_dist_km": _haversine_to_centroid(
                pred_top1, r.lat_t_next, r.lon_t_next, centroids
            ),
            "state_b_causal": r.state_b_causal,
        })
    return pd.DataFrame(rows)


def compare_models(
    metrics_per_model: dict[str, dict[str, Any]],
    *, baselines: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Tabla comparativa final (§7 C3 del spec).

    metrics_per_model: {"personalizado_rf": {"top1": .., "log_loss": .., ...}, ...}
    baselines: {"persistencia": {...}, "markov": {...}}
    """
    rows = []
    for name, m in baselines.items():
        rows.append({
            "modelo": name, "modo": "—", "split": m.get("split", "test"),
            "top1": m["top1"], "top3": m["top3"],
            "log_loss": m.get("log_loss", np.nan),
            "dist_median_km": m["dist_median_km"],
        })
    for full_name, m in metrics_per_model.items():
        modo, familia = full_name.split("_", 1)
        rows.append({
            "modelo": familia, "modo": modo, "split": m.get("split", "test"),
            "top1": m["top1"], "top3": m["top3"],
            "log_loss": m["log_loss"], "dist_median_km": m["dist_median_km"],
        })
    return pd.DataFrame(rows)
