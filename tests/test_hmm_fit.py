"""Tests de tfg_aves.hmm.fit."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from tfg_aves.hmm.fit import (
    build_sequences,
    fit_hmm_with_restarts,
)


def _synthetic_features_df(
    n_birds: int = 10,
    days_per_bird: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Construye un DataFrame de features sintético con n aves y días variables."""
    if days_per_bird is None:
        days_per_bird = tuple([50] * n_birds)
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for b in range(n_birds):
        bird_id = f"BIRD{b:02d}"
        n_days = days_per_bird[b]
        # Mitad de días en estado 0 (estacionario: step ~pocos km, turning alto),
        # mitad en estado 1 (migración: step ~cientos km, turning bajo).
        for d in range(n_days):
            state = 0 if d < n_days // 2 else 1
            if state == 0:
                step_km = abs(rng.normal(3.0, 1.5))
                cos_turn = rng.uniform(-1.0, 0.0)   # giro errático → cos ∈ [-1, 0]
            else:
                step_km = abs(rng.normal(150.0, 50.0))
                cos_turn = rng.uniform(0.7, 1.0)    # vuelo recto → cos ∈ [0.7, 1]
            rows.append({
                "bird_id": bird_id,
                "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=d),
                "step_length_km": step_km,
                "cos_turning_angle": cos_turn,
                "is_observation_valid": True,
            })
    return pd.DataFrame(rows)


def test_build_sequences_concatena_aves() -> None:
    """build_sequences concatena observaciones de varias aves con lengths correctos."""
    df = _synthetic_features_df(n_birds=3, days_per_bird=(20, 30, 25))
    X, lengths = build_sequences(
        df,
        bird_ids=["BIRD00", "BIRD01", "BIRD02"],
        feature_cols=["step_length_km", "cos_turning_angle"],
    )
    assert X.shape == (75, 2)
    assert lengths == [20, 30, 25]


def test_fit_converge_sobre_datos_sinteticos() -> None:
    """El HMM ajusta medias en km crudos: estacionario ~3 km, migración ~150 km."""
    df = _synthetic_features_df(n_birds=5, days_per_bird=(50,) * 5)
    X, lengths = build_sequences(
        df,
        bird_ids=[f"BIRD{i:02d}" for i in range(5)],
        feature_cols=["step_length_km", "cos_turning_angle"],
    )
    model, best_ll, all_lls = fit_hmm_with_restarts(
        X, lengths, n_components=2, n_restarts=3, random_state=0,
    )
    # Las medias del HMM están en km crudos; deben separar estacionario y migración.
    step_means = sorted(model.means_[:, 0])
    assert step_means[0] < 50.0   # estacionario: pocos km
    assert step_means[1] > 50.0   # migración: > 50 km


def test_multiple_restarts_no_decrecen_ll() -> None:
    """La mejor LL de 3 restarts es ≥ la LL de 1 restart."""
    df = _synthetic_features_df(n_birds=3, days_per_bird=(40,) * 3)
    X, lengths = build_sequences(
        df,
        bird_ids=[f"BIRD{i:02d}" for i in range(3)],
        feature_cols=["step_length_km", "cos_turning_angle"],
    )
    _, ll_1, _ = fit_hmm_with_restarts(X, lengths, n_restarts=1, random_state=42)
    _, ll_3, _ = fit_hmm_with_restarts(X, lengths, n_restarts=3, random_state=42)
    assert ll_3 >= ll_1 - 1e-6


