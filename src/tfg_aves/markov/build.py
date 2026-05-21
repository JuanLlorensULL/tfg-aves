"""Orquestación end-to-end de O2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import DAILY_PARQUET, O2_OUT_DIR
from .discretize import _parse_cell_id, cell_centroid, discretize_dataframe
from .evaluate import aggregate_metrics, lobo_predictions, persistence_predictions
from .smooth import laplace_smooth
from .transition import build_counts, build_transitions


@dataclass
class BuildO2Result:
    cells_path: Path
    counts_path: Path
    matrices_path: Path
    predictions_path: Path
    metrics_path: Path
    n_cells: int
    n_transitions: int
    summary: dict[str, float]


def _build_cells_dataframe(
    df_daily: pd.DataFrame, cell_deg: float
) -> pd.DataFrame:
    """Diccionario de celdas activas: todas las que tienen ≥1 fix válido."""
    valid = df_daily[df_daily["is_valid"]]
    counts = valid.groupby("cell_id").size().reset_index(name="n_obs_total")
    counts = counts.sort_values("cell_id").reset_index(drop=True)
    lat_idx: list[int] = []
    lon_idx: list[int] = []
    lat_c: list[float] = []
    lon_c: list[float] = []
    for cid in counts["cell_id"]:
        i, j = _parse_cell_id(cid)
        ci, cj = cell_centroid((i, j), cell_deg)
        lat_idx.append(i)
        lon_idx.append(j)
        lat_c.append(ci)
        lon_c.append(cj)
    counts["cell_lat_idx"] = lat_idx
    counts["cell_lon_idx"] = lon_idx
    counts["lat_c"] = lat_c
    counts["lon_c"] = lon_c
    return counts[["cell_id", "cell_lat_idx", "cell_lon_idx", "lat_c", "lon_c", "n_obs_total"]]


def build_o2(
    *,
    cell_deg: float,
    alpha: float = 1.0,
    do_lobo: bool = True,
    daily_path: Path = DAILY_PARQUET,
    out_dir: Path = O2_OUT_DIR,
    seed: int = 0,
) -> BuildO2Result:
    """Pipeline completa de O2: discretiza → counts → suavizado → LOBO → escribe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_daily = pd.read_parquet(daily_path)
    df_daily_disc = discretize_dataframe(df_daily, cell_deg=cell_deg)

    cells_df = _build_cells_dataframe(df_daily_disc, cell_deg=cell_deg)
    cells: list[str] = cells_df["cell_id"].tolist()

    transitions = build_transitions(df_daily_disc)
    counts = build_counts(transitions, cells=cells)
    P = laplace_smooth(counts, alpha=alpha)

    # Escribir cells, counts, matrices.
    cells_path = out_dir / "cells.parquet"
    counts_path = out_dir / "transitions_counts.npz"
    matrices_path = out_dir / "transition_matrices.npz"
    cells_df.to_parquet(cells_path, index=False)
    np.savez(counts_path, counts=counts, cells=np.array(cells, dtype=object))
    np.savez(matrices_path, matrices=P, cells=np.array(cells, dtype=object))

    if not do_lobo:
        predictions_path = out_dir / "predictions_lobo.parquet"
        metrics_path = out_dir / "metrics.parquet"
        pd.DataFrame().to_parquet(predictions_path, index=False)
        pd.DataFrame().to_parquet(metrics_path, index=False)
        return BuildO2Result(
            cells_path=cells_path,
            counts_path=counts_path,
            matrices_path=matrices_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            n_cells=len(cells),
            n_transitions=len(transitions),
            summary={"do_lobo": False},
        )

    # LOBO + persistencia.
    preds_markov = lobo_predictions(
        transitions, cells=cells, cell_deg=cell_deg, alpha=alpha
    )
    preds_persistence = persistence_predictions(
        transitions, cell_deg=cell_deg, n_cells=len(cells)
    )
    predictions = pd.concat([preds_markov, preds_persistence], ignore_index=True)
    metrics = aggregate_metrics(predictions)

    predictions_path = out_dir / "predictions_lobo.parquet"
    metrics_path = out_dir / "metrics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_parquet(metrics_path, index=False)

    global_metrics = metrics[metrics["scope"] == "global"].set_index("model")
    summary = {
        "top1_acc_markov": float(global_metrics.loc["markov", "top1_acc"]),
        "top1_acc_persistence": float(global_metrics.loc["persistence", "top1_acc"]),
        "dist_km_median_markov": float(global_metrics.loc["markov", "dist_km_median"]),
        "dist_km_median_persistence": float(global_metrics.loc["persistence", "dist_km_median"]),
        "log_loss_markov": float(global_metrics.loc["markov", "log_loss"]),
        "log_loss_persistence": float(global_metrics.loc["persistence", "log_loss"]),
        "n_predictions": int(len(preds_markov)),
    }

    return BuildO2Result(
        cells_path=cells_path,
        counts_path=counts_path,
        matrices_path=matrices_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        n_cells=len(cells),
        n_transitions=len(transitions),
        summary=summary,
    )
