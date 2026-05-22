"""Rutas estables del paquete hmm. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
DAILY_PARQUET: Path = ROOT / "data" / "processed" / "daily.parquet"
RAW_CSV: Path = ROOT / "data" / "raw" / "migration_original.csv"
O3_OUT_DIR: Path = ROOT / "data" / "processed" / "o3"
