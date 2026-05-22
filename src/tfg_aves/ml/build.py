"""Orquestador único de O4."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfg_aves.ml._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
    O4_OUT_DIR,
)


@dataclass
class BuildO4Result:
    """Resumen serializable de la ejecución de ``build_o4``."""

    n_birds: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    model_paths: dict[str, Path]  # {"personalizado_rf": Path, ...}
    predictions_path: Path
    metrics_path: Path


def build_o4(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    output_dir: Path = O4_OUT_DIR,
    seed: int = 0,
) -> BuildO4Result:
    """Pipeline completa de O4 (§5.4 del spec).

    Pasos:
        1. Carga ``features.parquet`` y ``cells.parquet``.
        2. Construye matriz para ambos modos.
        3. Split temporal por ave (72 % / 8 % / 20 %).
        4. Entrena las 6 combinaciones (3 familias × 2 modos).
        5. Computa métricas globales y por estado HMM.
        6. Computa baselines (persistencia + Markov(1)) sobre el mismo
           split temporal.
        7. Guarda artefactos en ``output_dir``.
    """
    raise NotImplementedError
