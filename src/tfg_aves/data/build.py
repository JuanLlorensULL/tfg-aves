"""Orquestación end-to-end de la pipeline de O1."""
from __future__ import annotations

from pathlib import Path

from ._paths import PROCESSED


def build_o1(
    *,
    max_speed_kmh: float,
    reference_hour_utc: int,
    tolerance_min: float,
    min_valid_days: int,
    out_dir: Path = PROCESSED,
) -> dict[str, int]:
    """Compone load → clean → build_daily → filter y materializa parquets."""
    raise NotImplementedError
