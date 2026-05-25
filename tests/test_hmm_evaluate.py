"""Tests de tfg_aves.hmm.evaluate."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from tfg_aves.hmm.evaluate import log_likelihood_per_obs
from tfg_aves.hmm.fit import build_sequences, fit_hmm_with_restarts


def _synthetic_features_for_eval(n_birds: int = 5, days: int = 40) -> pd.DataFrame:
    """Features sintéticos: dos clusters claramente separables en km crudos."""
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for b in range(n_birds):
        for d in range(days):
            state = 0 if d < days // 2 else 1
            step_km = abs(
                rng.normal(3.0 if state == 0 else 150.0, 1.5 if state == 0 else 50.0)
            )
            cos_turn = (
                rng.uniform(-1.0, 0.0)   # giro errático → cos ∈ [-1, 0]
                if state == 0
                else rng.uniform(0.7, 1.0)  # vuelo recto → cos ∈ [0.7, 1]
            )
            rows.append({
                "bird_id": f"BIRD{b:02d}",
                "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=d),
                "lat": 50.0,
                "lon": 0.0,
                "step_length_km": step_km,
                "cos_turning_angle": cos_turn,
                "is_observation_valid": True,
            })
    return pd.DataFrame(rows)


def test_log_likelihood_per_obs_finito() -> None:
    df = _synthetic_features_for_eval()
    X, lengths = build_sequences(
        df,
        bird_ids=df["bird_id"].unique().tolist(),
        feature_cols=["step_length_km", "cos_turning_angle"],
    )
    model, _, _ = fit_hmm_with_restarts(X, lengths, n_restarts=2, random_state=0)
    ll = log_likelihood_per_obs(model, X, lengths)
    assert np.isfinite(ll)
    # Cota laxa: con features en km crudos (std ~50 km en migración) la densidad
    # gaussiana por observación puede ser muy baja → LL negativa y escala-dependiente.
    assert ll < 50.0


