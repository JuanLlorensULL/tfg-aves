"""Tests de tfg_aves.hmm.evaluate."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tfg_aves.hmm.evaluate import ab_agreement, biological_coherence_table, log_likelihood_per_obs
from tfg_aves.hmm.fit import build_sequences, fit_hmm_with_restarts


def _synthetic_features_for_eval(n_birds: int = 5, days: int = 40) -> pd.DataFrame:
    """Features sintéticos: dos clusters claramente separables."""
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for b in range(n_birds):
        for d in range(days):
            state = 0 if d < days // 2 else 1
            log_dist = rng.normal(0.5 if state == 0 else 4.5, 0.3)
            turning = (
                rng.uniform(np.pi / 2, np.pi)
                if state == 0
                else rng.uniform(0, np.pi / 4)
            )
            rows.append({
                "bird_id": f"BIRD{b:02d}",
                "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=d),
                "lat": 50.0,
                "lon": 0.0,
                "log_displacement_km": log_dist,
                "abs_turning_angle_rad": turning,
                "is_observation_valid": True,
            })
    return pd.DataFrame(rows)


def test_log_likelihood_per_obs_finito() -> None:
    df = _synthetic_features_for_eval()
    X, lengths = build_sequences(
        df,
        bird_ids=df["bird_id"].unique().tolist(),
        feature_cols=["log_displacement_km", "abs_turning_angle_rad"],
    )
    model, scaler, _, _ = fit_hmm_with_restarts(X, lengths, n_restarts=2, random_state=0)
    ll = log_likelihood_per_obs(model, scaler, X, lengths)
    assert np.isfinite(ll)
    # LL por observación es negativa pero acotada para datos bien separados.
    assert -10.0 < ll < 0.0


def test_ab_agreement_total() -> None:
    """Si state_a == state_b siempre, acuerdo = 100%."""
    df = pd.DataFrame({
        "bird_id": ["A"] * 10 + ["B"] * 10,
        "date_utc": [dt.date(2010, 1, 1) + dt.timedelta(days=i) for i in range(10)] * 2,
        "state_a": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 2,
        "state_b": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 2,
        "is_observation_valid": [True] * 20,
    })
    out = ab_agreement(df)
    assert out["pct_agreement"] == pytest.approx(100.0)


def test_ab_agreement_mitad() -> None:
    """Si A y B difieren en la mitad de las observaciones, acuerdo = 50%."""
    df = pd.DataFrame({
        "bird_id": ["A"] * 10,
        "date_utc": [dt.date(2010, 1, 1) + dt.timedelta(days=i) for i in range(10)],
        "state_a": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "state_b": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "is_observation_valid": [True] * 10,
    })
    out = ab_agreement(df)
    assert out["pct_agreement"] == pytest.approx(50.0)


def test_biological_coherence_table_concentracion_mes() -> None:
    """Estado migración concentrado en mar y oct → la tabla lo refleja."""
    rows: list[dict] = []
    # 100 días en marzo, 100 en octubre, todos estado=1; 100 en junio, todos estado=0.
    for i in range(100):
        rows.append({
            "bird_id": "A",
            "date_utc": dt.date(2010, 3, 1) + dt.timedelta(days=i % 30),
            "state_a": 1,
            "lat": 45.0,
            "is_observation_valid": True,
        })
    for i in range(100):
        rows.append({
            "bird_id": "A",
            "date_utc": dt.date(2010, 10, 1) + dt.timedelta(days=i % 30),
            "state_a": 1,
            "lat": 45.0,
            "is_observation_valid": True,
        })
    for i in range(100):
        rows.append({
            "bird_id": "A",
            "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=i % 30),
            "state_a": 0,
            "lat": 50.0,
            "is_observation_valid": True,
        })
    df = pd.DataFrame(rows)
    tbl = biological_coherence_table(df, state_col="state_a")
    # La tabla debe tener una fila por (mes, estado) con conteo.
    assert "month" in tbl.columns
    assert "state" in tbl.columns
    assert "count" in tbl.columns
    # Mes 3 y mes 10 deben tener estado=1 con conteo alto (~100).
    march_migr = tbl[(tbl["month"] == 3) & (tbl["state"] == 1)]["count"].sum()
    oct_migr = tbl[(tbl["month"] == 10) & (tbl["state"] == 1)]["count"].sum()
    june_stat = tbl[(tbl["month"] == 6) & (tbl["state"] == 0)]["count"].sum()
    assert march_migr >= 80
    assert oct_migr >= 80
    assert june_stat >= 80
