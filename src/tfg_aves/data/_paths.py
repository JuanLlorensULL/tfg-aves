"""Rutas estables del proyecto. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
RAW_CSV: Path = ROOT / "data" / "raw" / "migration_original.csv"
INTERIM: Path = ROOT / "data" / "interim"
PROCESSED: Path = ROOT / "data" / "processed"
