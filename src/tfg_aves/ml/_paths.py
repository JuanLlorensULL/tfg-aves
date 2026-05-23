"""Rutas estables del paquete ml. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
FEATURES_O3_PARQUET: Path = ROOT / "data" / "processed" / "o3" / "features.parquet"
CELLS_PARQUET: Path = ROOT / "data" / "processed" / "o2" / "cells.parquet"
DAILY_PARQUET: Path = ROOT / "data" / "processed" / "daily.parquet"
O4_OUT_DIR: Path = ROOT / "data" / "processed" / "o4"
O4_L1V1_DIR: Path = ROOT / "data" / "processed" / "o4" / "l1_v1"
