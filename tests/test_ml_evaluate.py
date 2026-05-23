"""Tests unitarios de tfg_aves.ml.evaluate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.ml.evaluate import (
    compute_persistence_baseline,
    dist_median_km,
    evaluate_by_state,
    top_k_accuracy,
)


def _synthetic_predictions() -> pd.DataFrame:
    """20 predicciones, mezcla de aciertos top-1 y top-3 por estado."""
    return pd.DataFrame({
        "bird_id": [f"b{i // 5}" for i in range(20)],
        "date_utc": pd.date_range("2020-01-01", periods=20),
        "true_cell": ["c1"] * 10 + ["c2"] * 10,
        "pred_cell_top1": (
            ["c1"] * 6 + ["c9"] * 4   # estado migración: 6/10 aciertos
            + ["c2"] * 8 + ["c9"] * 2  # estado estacionario: 8/10 aciertos
        ),
        "pred_cell_topk": (
            [["c1", "c3"]] * 8 + [["c9", "c8"]] * 2
            + [["c2", "c5"]] * 9 + [["c7", "c8"]] * 1
        ),
        "pred_prob_top1": [0.7] * 20,
        "pred_dist_km": (
            [10.0] * 6 + [200.0] * 4
            + [5.0] * 8 + [180.0] * 2
        ),
        "state_b_causal": [1] * 10 + [0] * 10,  # 1=migración, 0=estacionario
    })


def test_top_k_accuracy_top1() -> None:
    preds = _synthetic_predictions()
    assert top_k_accuracy(preds, k=1) == 14 / 20


def test_top_k_accuracy_top3_uses_topk_list() -> None:
    preds = _synthetic_predictions()
    assert top_k_accuracy(preds, k=3) == 17 / 20


def test_dist_median_km() -> None:
    preds = _synthetic_predictions()
    expected = float(np.median(preds["pred_dist_km"]))
    assert dist_median_km(preds) == expected


def test_evaluate_by_state_partitions_correctly() -> None:
    preds = _synthetic_predictions()
    table = evaluate_by_state(preds, state_col="state_b_causal")
    assert set(table["state"].tolist()) == {"global", "estacionario", "migración"}
    row_m = table[table["state"] == "migración"].iloc[0]
    row_e = table[table["state"] == "estacionario"].iloc[0]
    assert row_m["top1"] == 0.6
    assert row_e["top1"] == 0.8
    assert row_m["n_obs"] == 10
    assert row_e["n_obs"] == 10


def test_top_k_consistency_global_equals_weighted_per_state() -> None:
    preds = _synthetic_predictions()
    table = evaluate_by_state(preds, state_col="state_b_causal")
    row_g = table[table["state"] == "global"].iloc[0]
    weighted = (
        table[table["state"] != "global"]
        .assign(weighted=lambda d: d["top1"] * d["n_obs"])
        ["weighted"].sum() / 20
    )
    np.testing.assert_allclose(row_g["top1"], weighted, atol=1e-9)


def _matrix_test_for_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """3 filas con cell_id_t conocida y posiciones t+1 conocidas."""
    matrix_test = pd.DataFrame({
        "bird_id": ["A", "B", "C"],
        "date_utc": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "cell_id_t": ["40_-6", "41_-5", "78_-8"],
        "cell_id_t_next": ["40_-6", "42_-5", "78_-8"],
        "lat_t_next": [20.25, 21.25, 39.25],
        "lon_t_next": [-2.75, -2.25, -3.75],
        "state_b_causal": [0, 1, 0],
    })
    cells = pd.DataFrame({
        "cell_id": ["40_-6", "41_-5", "42_-5", "78_-8"],
        "cell_lat_idx": [40, 41, 42, 78],
        "cell_lon_idx": [-6, -5, -5, -8],
        "lat_c": [20.25, 20.75, 21.25, 39.25],
        "lon_c": [-2.75, -2.25, -2.25, -3.75],
        "n_obs_total": [10, 10, 10, 10],
    })
    return matrix_test, cells


def test_persistence_predicts_cell_id_t() -> None:
    matrix_test, cells = _matrix_test_for_baseline()
    out = compute_persistence_baseline(matrix_test, cells=cells)
    # La predicción debe ser exactamente cell_id_t en todas las filas.
    assert list(out["pred_cell_top1"]) == list(matrix_test["cell_id_t"])
    # En las filas donde true_cell == pred (filas 1 y 3) la distancia debe ser 0.
    assert out.loc[0, "pred_dist_km"] == 0.0
    assert out.loc[2, "pred_dist_km"] == 0.0
    # En la fila 2 (true=42_-5 vs pred=41_-5) la distancia debe ser > 0.
    assert out.loc[1, "pred_dist_km"] > 0.0
    # Esquema: mismas columnas que predict_with_meta.
    expected_cols = {
        "bird_id", "date_utc", "true_cell",
        "pred_cell_top1", "pred_cell_topk",
        "pred_prob_top1", "pred_dist_km", "state_b_causal",
    }
    assert expected_cols.issubset(out.columns)
