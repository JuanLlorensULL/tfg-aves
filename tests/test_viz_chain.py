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


class _FakeAxis:
    """Predictor de eje que devuelve un desplazamiento constante (p10,p50,p90)."""
    def __init__(self, p10, p50, p90):
        self._q = np.array([p10, p50, p90], dtype=float)

    def predict_raw(self, X):
        return np.tile(self._q, (len(X), 1))


def test_chain_trajectory_straight_line_and_growing_cone():
    from tfg_aves.viz.chain import chain_trajectory
    axis_lat = _FakeAxis(0.05, 0.10, 0.15)
    axis_lon = _FakeAxis(-0.02, 0.0, 0.02)
    seed_history = [(40.0, -3.0), (40.1, -3.0)]
    out = chain_trajectory(
        axis_lat, axis_lon, seed_history,
        start_date=pd.Timestamp("2020-04-01"),
        frozen_state_b=0, frozen_posterior_mig=0.1, k=3,
    )
    assert list(out["step"]) == [1, 2, 3]
    assert np.allclose(out["lat"], [40.2, 40.3, 40.4])
    assert np.allclose(out["lon"], [-3.0, -3.0, -3.0])
    assert np.allclose(out["cone_halfwidth_lat"], [0.05, 0.10, 0.15])
    assert (out["cone_halfwidth_lat"].diff().dropna() > 0).all()


def test_chain_trajectory_feature_order_matches_model():
    from tfg_aves.ml.features import FEATURES_O4_CAUSAL
    from tfg_aves.viz.chain import chain_trajectory

    class _CheckingAxis:
        def predict_raw(self, X):
            assert list(X.columns) == FEATURES_O4_CAUSAL
            return np.tile(np.array([0.0, 0.1, 0.2]), (len(X), 1))

    out = chain_trajectory(
        _CheckingAxis(), _CheckingAxis(), [(40.0, -3.0), (40.1, -3.0)],
        start_date=pd.Timestamp("2020-04-01"),
        frozen_state_b=1, frozen_posterior_mig=0.8, k=2,
    )
    assert len(out) == 2
