"""Rutas estables del paquete viz (O5). Reusa las del paquete ml."""
from __future__ import annotations

from pathlib import Path

from tfg_aves.ml._paths import (  # noqa: F401  (reexport intencional)
    CELLS_PARQUET,
    DAILY_PARQUET,
    FEATURES_O3_PARQUET,
    O4_L3V2_DIR,
    ROOT,
)

PREDICTIONS_L3V2_PARQUET: Path = O4_L3V2_DIR / "predictions_test.parquet"
MODEL_LGBM_DLAT: Path = O4_L3V2_DIR / "model_lgbm_poblacional_dlat.pkl"
MODEL_LGBM_DLON: Path = O4_L3V2_DIR / "model_lgbm_poblacional_dlon.pkl"
FIGURES_DIR: Path = ROOT / "reports" / "figures"

# Aves curadas de O5: las 4 con más histórico (V6 del spec).
CURATED_BIRDS: tuple[str, ...] = ("91916A", "91752A", "91823A", "91763A")
