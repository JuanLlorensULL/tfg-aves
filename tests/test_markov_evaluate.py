"""Tests de tfg_aves.markov.evaluate (LOBO + persistencia + agregados)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tfg_aves.markov.discretize import discretize_dataframe
from tfg_aves.markov.evaluate import (
    aggregate_metrics,
    lobo_predictions,
    persistence_predictions,
)
from tfg_aves.markov.transition import build_transitions


def _synthetic_daily(n_birds: int = 3, n_days: int = 10) -> pd.DataFrame:
    """Tres aves con 10 días válidos cada una, en celdas conocidas."""
    rows: list[dict] = []
    for b in range(n_birds):
        bird_id = f"BIRD{b}"
        # Cada ave se mueve entre dos celdas vecinas determinísticamente.
        for d in range(n_days):
            base_lat = 50.0 + b * 1.0  # ave 0: lat ~50, ave 1: ~51, ave 2: ~52
            lat = base_lat + (0.5 if d % 2 == 0 else 0.0)
            lon = 5.0
            rows.append(
                {
                    "bird_id": bird_id,
                    "date_utc": dt.date(2010, 1, 1) + dt.timedelta(days=d),
                    "lat": lat,
                    "lon": lon,
                    "is_valid": True,
                }
            )
    return pd.DataFrame(rows)


def test_lobo_predictions_shape_correcta() -> None:
    df = _synthetic_daily(n_birds=3, n_days=10)
    df = discretize_dataframe(df, cell_deg=0.5)
    transitions = build_transitions(df)
    cells = sorted(set(transitions["cell_from"]) | set(transitions["cell_to"]))

    preds = lobo_predictions(transitions, cells=cells, cell_deg=0.5, alpha=1.0)
    # 3 aves × 9 pares por ave = 27 filas.
    assert len(preds) == 27
    assert "is_top1_hit" in preds.columns
    assert "dist_km" in preds.columns
    assert (preds["model"] == "markov").all()


def test_lobo_ave_sin_pares_no_rompe() -> None:
    # Construir una transición sintética donde un ave no aparece en transitions.
    transitions = pd.DataFrame(
        [
            {"bird_id": "A", "date_t": dt.date(2010, 1, 1), "month_int": 1,
             "cell_from": "0_0", "cell_to": "1_0",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.75, "lon_to": 0.25},
        ]
    )
    cells = ["0_0", "1_0"]
    # Sólo aparece ave A; LOBO no produce filas pero no debe romper.
    preds = lobo_predictions(transitions, cells=cells, cell_deg=0.5, alpha=1.0)
    # Cuando se omite el único ave del fold, el training queda vacío y el
    # ave held-out (A) genera sus predicciones con la marginal uniforme.
    # Aceptamos 1 fila (la del único par).
    assert len(preds) == 1


def test_persistence_predictions_cell_pred_igual_cell_from() -> None:
    transitions = pd.DataFrame(
        [
            {"bird_id": "A", "date_t": dt.date(2010, 1, 1), "month_int": 1,
             "cell_from": "0_0", "cell_to": "1_0",
             "lat_from": 0.25, "lon_from": 0.25, "lat_to": 0.75, "lon_to": 0.25},
            {"bird_id": "A", "date_t": dt.date(2010, 1, 2), "month_int": 1,
             "cell_from": "1_0", "cell_to": "0_0",
             "lat_from": 0.75, "lon_from": 0.25, "lat_to": 0.25, "lon_to": 0.25},
        ]
    )
    preds = persistence_predictions(transitions, cell_deg=0.5, n_cells=2)
    assert (preds["cell_pred_top1"] == preds["cell_t"]).all()
    assert (preds["model"] == "persistence").all()


def test_aggregate_metrics_top1_acc_coincide_con_media() -> None:
    preds = pd.DataFrame(
        {
            "bird_id": ["A"] * 4,
            "month_int": [1, 1, 2, 2],
            "model": ["markov"] * 4,
            "is_top1_hit": [True, False, True, True],
            "is_top3_hit": [True, True, True, True],
            "dist_km": [0.0, 10.0, 0.0, 5.0],
            "prob_assigned_real": [0.5, 0.1, 0.7, 0.4],
        }
    )
    metrics = aggregate_metrics(preds)
    # Nivel "global": top1_acc = 3/4 = 0.75.
    global_row = metrics[(metrics["scope"] == "global") & (metrics["model"] == "markov")]
    assert len(global_row) == 1
    assert global_row["top1_acc"].iloc[0] == pytest.approx(0.75)


def test_aggregate_metrics_log_loss_perfecto_es_cero() -> None:
    preds = pd.DataFrame(
        {
            "bird_id": ["A", "A"],
            "month_int": [1, 1],
            "model": ["markov", "markov"],
            "is_top1_hit": [True, True],
            "is_top3_hit": [True, True],
            "dist_km": [0.0, 0.0],
            "prob_assigned_real": [1.0, 1.0],
        }
    )
    metrics = aggregate_metrics(preds)
    global_row = metrics[(metrics["scope"] == "global") & (metrics["model"] == "markov")]
    assert global_row["log_loss"].iloc[0] == pytest.approx(0.0, abs=1e-8)
