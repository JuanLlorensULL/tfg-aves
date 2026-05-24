"""Tests de tfg_aves.viz.chain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.viz.chain import incoming_kinematics


def test_incoming_kinematics_straight_north():
    history = [(40.0, -3.0), (40.1, -3.0), (40.2, -3.0)]
    kin = incoming_kinematics(history)
    assert 10.0 < kin["step_in_km"] < 12.5
    assert abs(kin["sin_bearing_in"]) < 1e-6
    assert kin["cos_bearing_in"] > 0.999
    assert kin["cos_turning_in"] > 0.999


def test_incoming_kinematics_needs_three_points_for_turning():
    kin = incoming_kinematics([(40.1, -3.0), (40.2, -3.0)])
    assert "step_in_km" in kin and not np.isnan(kin["step_in_km"])
    assert kin["cos_turning_in"] == 1.0
