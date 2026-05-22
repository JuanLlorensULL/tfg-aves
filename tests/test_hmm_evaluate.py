"""Tests de tfg_aves.hmm.evaluate."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tfg_aves.hmm.evaluate import ab_agreement, biological_coherence_table, log_likelihood_per_obs
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
        "state_b": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
