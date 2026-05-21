"""Rutas estables del paquete markov. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
DAILY_PARQUET: Path = ROOT / "data" / "processed" / "daily.parquet"
O2_OUT_DIR: Path = ROOT / "data" / "processed" / "o2"
