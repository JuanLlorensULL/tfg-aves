"""Tests unitarios del módulo de regresión de cuantiles (L3)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_o4_l3v1_dir_exists():
    from tfg_aves.ml._paths import O4_L3V1_DIR, O4_OUT_DIR
    assert O4_L3V1_DIR == O4_OUT_DIR / "l3_v1"
