"""Tests de integración de build_o2."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from tfg_aves.markov.build import build_o2


def _write_synthetic_daily(path: Path, n_birds: int = 3, n_days: int = 60) -> None:
    """Sintetiza un daily.parquet con 3 aves × 60 días determinísticos."""
    rows: list[dict] = []
    for b in range(n_birds):
        bird_id = f"BIRD{b}"
        for d in range(n_days):
            base_lat = 50.0 + b * 1.0
            lat = base_lat + (0.5 if d % 2 == 0 else 0.0)
            lon = 5.0
            rows.append(
                {
                    "bird_id": bird_id,
                    "date_utc": dt.date(2010, 1, 1) + dt.timedelta(days=d),
                    "lat": lat,
                    "lon": lon,
                    "is_valid": True,
                    "source_event_id": d,
                    "delta_minutes": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_build_o2_produce_ficheros_esperados(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    _write_synthetic_daily(daily_path)
    out_dir = tmp_path / "o2"

    result = build_o2(
        cell_deg=0.5,
        alpha=1.0,
        do_lobo=True,
        daily_path=daily_path,
        out_dir=out_dir,
    )

    for p in [
        result.cells_path,
        result.counts_path,
        result.matrices_path,
        result.predictions_path,
        result.metrics_path,
    ]:
        assert p.is_file(), f"Falta {p}"

    assert result.n_cells > 0
    assert result.n_transitions > 0
    expected_keys = {
        "top1_acc_markov",
        "top1_acc_persistence",
        "dist_km_median_markov",
        "dist_km_median_persistence",
        "log_loss_markov",
        "log_loss_persistence",
        "n_predictions",
    }
    assert expected_keys.issubset(result.summary.keys())


def test_build_o2_esquemas_releen(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    _write_synthetic_daily(daily_path)
    out_dir = tmp_path / "o2"

    result = build_o2(
        cell_deg=0.5,
        alpha=1.0,
        do_lobo=True,
        daily_path=daily_path,
        out_dir=out_dir,
    )

    cells = pd.read_parquet(result.cells_path)
    expected_cell_cols = {
        "cell_id", "cell_lat_idx", "cell_lon_idx", "lat_c", "lon_c", "n_obs_total"
    }
    assert expected_cell_cols.issubset(cells.columns)

    preds = pd.read_parquet(result.predictions_path)
    expected_cols = {
        "bird_id", "date_t", "month_int", "lat_t", "lon_t", "cell_t",
        "lat_real", "lon_real", "cell_real", "cell_pred_top1", "cell_pred_top3",
        "prob_top1", "prob_assigned_real", "dist_km", "is_top1_hit", "is_top3_hit", "model",
    }
    assert expected_cols.issubset(preds.columns)
    # Hay tanto markov como persistence.
    assert set(preds["model"].unique()) == {"markov", "persistence"}

    metrics = pd.read_parquet(result.metrics_path)
    assert set(metrics["scope"].unique()) == {"bird_month", "month", "global"}

    # Carga de matrices y verificación de shape (12, n_cells, n_cells).
    npz_matrices = np.load(result.matrices_path)
    P = npz_matrices["matrices"]
    assert P.shape == (12, result.n_cells, result.n_cells)
    # Las filas con suma de counts > 0 deben sumar a 1.
    np.testing.assert_allclose(P.sum(axis=2), 1.0, atol=1e-9)
