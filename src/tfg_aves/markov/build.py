"""Orquestación end-to-end de O2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._paths import DAILY_PARQUET, O2_OUT_DIR


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


def build_o2(
    *,
    cell_deg: float,
    alpha: float = 1.0,
    do_lobo: bool = True,
    daily_path: Path = DAILY_PARQUET,
    out_dir: Path = O2_OUT_DIR,
    seed: int = 0,
) -> BuildO2Result:
    """Pipeline completa de O2: discretiza → counts → suavizado → LOBO → escribe.

    Devuelve ``BuildO2Result`` con paths, ``n_cells``, ``n_transitions``
    y un dict ``summary`` con las claves: ``top1_acc_markov``,
    ``top1_acc_persistence``, ``dist_km_median_markov``,
    ``dist_km_median_persistence``, ``log_loss_markov``,
    ``log_loss_persistence``, ``n_predictions``.
    """
    raise NotImplementedError
