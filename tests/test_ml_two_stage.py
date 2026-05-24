"""Tests de las funciones puras de L2 (two_stage)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.ml import two_stage as ts


def test_derive_y_move_basic():
    df = pd.DataFrame({
        "cell_id_t":      ["A", "A", "B", "B", "A"],
        "cell_id_t_next": ["A", "B", "B", "A", "A"],
    })
    result = ts.derive_y_move(df).tolist()
    assert result == [False, True, False, True, False]
