"""Evaluación LOBO + baseline persistencia + agregados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .predict import predict_distribution, prediction_distance_km, topk_from_distribution
from .smooth import laplace_smooth, marginal_distribution
from .transition import build_counts

_EPSILON = 1e-9


def _row_from_distribution(
    distribution: np.ndarray,
    cells: list[str],
    cell_from: str,
    cell_real: str,
    lat_real: float,
    lon_real: float,
    cell_deg: float,
    bird_id: str,
    date_t,
    month_int: int,
    lat_t: float,
    lon_t: float,
    model: str,
    alpha: float | None = None,
) -> dict:
    """Construye una fila del DataFrame de predicciones."""
    top3 = topk_from_distribution(distribution, k=3, cells=cells)
    cell_pred_top1 = top3[0]
    prob_top1 = float(distribution[cells.index(cell_pred_top1)])

    if cell_real in cells:
        prob_assigned_real = float(distribution[cells.index(cell_real)])
    else:
        # Probabilidad Laplace de celda no observada en el fold.
        if alpha is not None:
            n_cells = len(cells)
            prob_assigned_real = alpha / (1.0 + alpha * n_cells)
        else:
            prob_assigned_real = _EPSILON

    dist_km = prediction_distance_km(
        cell_pred=cell_pred_top1, lat_real=lat_real, lon_real=lon_real, cell_deg=cell_deg
    )

    return {
        "bird_id": bird_id,
        "date_t": date_t,
        "month_int": month_int,
        "lat_t": lat_t,
        "lon_t": lon_t,
        "cell_t": cell_from,
        "lat_real": lat_real,
        "lon_real": lon_real,
        "cell_real": cell_real,
        "cell_pred_top1": cell_pred_top1,
        "cell_pred_top3": ",".join(top3),
        "prob_top1": prob_top1,
        "prob_assigned_real": prob_assigned_real,
        "dist_km": dist_km,
        "is_top1_hit": cell_pred_top1 == cell_real,
        "is_top3_hit": cell_real in top3,
        "model": model,
    }


def lobo_predictions(
    df_transitions: pd.DataFrame,
    cells: list[str],
    cell_deg: float,
    alpha: float,
) -> pd.DataFrame:
    """Predicciones LOBO: 1 fold por ave; matrices re-entrenadas en cada fold."""
    rows: list[dict] = []
    bird_ids = sorted(df_transitions["bird_id"].unique())

    for held_out in bird_ids:
        train = df_transitions[df_transitions["bird_id"] != held_out]
        test = df_transitions[df_transitions["bird_id"] == held_out]
        if len(test) == 0:
            continue

        counts = build_counts(train, cells=cells)
        P = laplace_smooth(counts, alpha=alpha) if counts.sum() > 0 else None
        marginal = marginal_distribution(counts)

        for _, t in test.iterrows():
            if P is None:
                # Training vacío: usa marginal pura.
                distribution = marginal[t["month_int"] - 1]
            else:
                try:
                    distribution = predict_distribution(
                        P, cell_from=t["cell_from"], month_int=t["month_int"],
                        cells=cells, marginal=marginal,
                    )
                except KeyError:
                    distribution = marginal[t["month_int"] - 1]

            rows.append(
                _row_from_distribution(
                    distribution=distribution,
                    cells=cells,
                    cell_from=t["cell_from"],
                    cell_real=t["cell_to"],
                    lat_real=t["lat_to"],
                    lon_real=t["lon_to"],
                    cell_deg=cell_deg,
                    bird_id=held_out,
                    date_t=t["date_t"],
                    month_int=t["month_int"],
                    lat_t=t["lat_from"],
                    lon_t=t["lon_from"],
                    model="markov",
                    alpha=alpha,
                )
            )
    return pd.DataFrame(rows)


def persistence_predictions(
    df_transitions: pd.DataFrame, cell_deg: float, n_cells: int
) -> pd.DataFrame:
    """Baseline: ``cell_pred = cell_from``. Distribución casi-delta con ε."""
    rows: list[dict] = []
    delta_prob = 1.0 - _EPSILON * (n_cells - 1) if n_cells > 1 else 1.0
    for _, t in df_transitions.iterrows():
        cell_from = t["cell_from"]
        cell_real = t["cell_to"]

        dist_km = prediction_distance_km(
            cell_pred=cell_from, lat_real=t["lat_to"], lon_real=t["lon_to"], cell_deg=cell_deg
        )

        rows.append(
            {
                "bird_id": t["bird_id"],
                "date_t": t["date_t"],
                "month_int": t["month_int"],
                "lat_t": t["lat_from"],
                "lon_t": t["lon_from"],
                "cell_t": cell_from,
                "lat_real": t["lat_to"],
                "lon_real": t["lon_to"],
                "cell_real": cell_real,
                "cell_pred_top1": cell_from,
                "cell_pred_top3": cell_from,
                "prob_top1": delta_prob,
                "prob_assigned_real": delta_prob if cell_real == cell_from else _EPSILON,
                "dist_km": dist_km,
                "is_top1_hit": cell_from == cell_real,
                "is_top3_hit": cell_from == cell_real,
                "model": "persistence",
            }
        )
    return pd.DataFrame(rows)


def _metrics_block(group: pd.DataFrame) -> dict:
    probs = group["prob_assigned_real"].clip(lower=_EPSILON)
    return {
        "n_predictions": int(len(group)),
        "top1_acc": float(group["is_top1_hit"].mean()),
        "top3_acc": float(group["is_top3_hit"].mean()),
        "dist_km_median": float(group["dist_km"].median()),
        "dist_km_p90": float(group["dist_km"].quantile(0.9)),
        "log_loss": float(-np.log(probs).mean()),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas en 3 niveles (bird_month, month, global). Formato long."""
    rows: list[dict] = []
    for (bird_id, month_int, model), g in predictions.groupby(
        ["bird_id", "month_int", "model"]
    ):
        rows.append(
            {"scope": "bird_month", "bird_id": bird_id, "month_int": int(month_int),
             "model": model, **_metrics_block(g)}
        )
    for (month_int, model), g in predictions.groupby(["month_int", "model"]):
        rows.append(
            {"scope": "month", "bird_id": None, "month_int": int(month_int),
             "model": model, **_metrics_block(g)}
        )
    for model, g in predictions.groupby("model"):
        rows.append(
            {"scope": "global", "bird_id": None, "month_int": None,
             "model": model, **_metrics_block(g)}
        )
    return pd.DataFrame(rows)
