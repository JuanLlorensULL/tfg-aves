"""Tests unitarios del módulo de regresión de cuantiles (L3)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_o4_l3v1_dir_exists():
    from tfg_aves.ml._paths import O4_L3V1_DIR, O4_OUT_DIR
    assert O4_L3V1_DIR == O4_OUT_DIR / "l3_v1"


def test_derive_displacement_target():
    from tfg_aves.ml.quantile import derive_displacement_target
    m = pd.DataFrame({
        "lat": [40.0, 41.0], "lon": [-3.0, -2.5],
        "lat_t_next": [40.5, 41.2], "lon_t_next": [-2.0, -2.7],
    })
    out = derive_displacement_target(m)
    assert np.allclose(out["y_dlat"], [0.5, 0.2])
    assert np.allclose(out["y_dlon"], [1.0, -0.2])
