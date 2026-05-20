"""Carga y normalización del CSV crudo de Movebank."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import RAW_CSV


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV de Movebank y devuelve un DataFrame normalizado."""
    raise NotImplementedError
