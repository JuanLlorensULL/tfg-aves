"""Rutas estables del paquete meteo. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
WIND_RAW_DIR: Path = ROOT / "data" / "raw" / "wind"
WIND_PROCESSED_DIR: Path = ROOT / "data" / "processed" / "wind"
WIND_PER_FIX_PARQUET: Path = WIND_PROCESSED_DIR / "wind_per_fix.parquet"
