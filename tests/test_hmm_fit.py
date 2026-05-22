"""Tests de tfg_aves.hmm.fit."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from tfg_aves.hmm.fit import (
    build_sequences,
    fit_hmm_with_restarts,
    relabel_states,
    stratified_holdout_split,
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
        # Mitad de días en estado 0 (estacionario: log_dist bajo, turning alto),
        # mitad en estado 1 (migración: log_dist alto, turning bajo).
        for d in range(n_days):
            state = 0 if d < n_days // 2 else 1
            if state == 0:
                log_dist = rng.normal(0.5, 0.3)
                turning = rng.uniform(np.pi / 2, np.pi)
            else:
                log_dist = rng.normal(4.5, 0.3)
                turning = rng.uniform(0.0, np.pi / 4)
            rows.append({
                "bird_id": bird_id,
                "date_utc": dt.date(2010, 6, 1) + dt.timedelta(days=d),
                "log_displacement_km": log_dist,
                "abs_turning_angle_rad": turning,
                "is_observation_valid": True,
            })
    return pd.DataFrame(rows)


def test_stratified_split_proporcional() -> None:
    """Aves ricas (>200 días) y pobres (<50) se reparten proporcionalmente."""
    # 6 aves ricas, 4 pobres.
    days = (250, 240, 230, 220, 210, 205, 30, 25, 20, 15)
    df = _synthetic_features_df(n_birds=10, days_per_bird=days)
    train, holdout = stratified_holdout_split(df, holdout_frac=0.30, random_state=0)
    assert len(train) + len(holdout) == 10
    assert len(set(train) & set(holdout)) == 0
    # El holdout debe tener al menos un ave de cada extremo (rica y pobre).
    rich_birds = {f"BIRD{i:02d}" for i in range(6)}
    poor_birds = {f"BIRD{i:02d}" for i in range(6, 10)}
    assert len(set(holdout) & rich_birds) >= 1
    assert len(set(holdout) & poor_birds) >= 1


def test_build_sequences_concatena_aves() -> None:
    """build_sequences concatena observaciones de varias aves con lengths correctos."""
    df = _synthetic_features_df(n_birds=3, days_per_bird=(20, 30, 25))
    X, lengths = build_sequences(
        df,
        bird_ids=["BIRD00", "BIRD01", "BIRD02"],
        feature_cols=["log_displacement_km", "abs_turning_angle_rad"],
    )
    assert X.shape == (75, 2)
    assert lengths == [20, 30, 25]


def test_fit_converge_sobre_datos_sinteticos() -> None:
    """El HMM ajusta medias razonablemente cercanas a (0.5, π/2-π) y (4.5, 0-π/4)."""
    df = _synthetic_features_df(n_birds=5, days_per_bird=(50,) * 5)
    X, lengths = build_sequences(
        df,
        bird_ids=[f"BIRD{i:02d}" for i in range(5)],
        feature_cols=["log_displacement_km", "abs_turning_angle_rad"],
    )
    model, scaler, best_ll, all_lls = fit_hmm_with_restarts(
        X, lengths, n_components=2, n_restarts=3, random_state=0,
    )
    # Las medias del HMM, des-estandarizadas, deben separar log_dist bajo y alto.
    means_raw = scaler.inverse_transform(model.means_)
    log_dist_means = sorted(means_raw[:, 0])
    assert log_dist_means[0] < 2.0  # estacionario
    assert log_dist_means[1] > 3.0  # migración


def test_multiple_restarts_no_decrecen_ll() -> None:
    """La mejor LL de 3 restarts es ≥ la LL de 1 restart."""
    df = _synthetic_features_df(n_birds=3, days_per_bird=(40,) * 3)
    X, lengths = build_sequences(
        df,
        bird_ids=[f"BIRD{i:02d}" for i in range(3)],
        feature_cols=["log_displacement_km", "abs_turning_angle_rad"],
    )
    _, _, ll_1, _ = fit_hmm_with_restarts(X, lengths, n_restarts=1, random_state=42)
    _, _, ll_3, _ = fit_hmm_with_restarts(X, lengths, n_restarts=3, random_state=42)
    assert ll_3 >= ll_1 - 1e-6


def test_relabel_states_pone_estacionario_en_0() -> None:
    """El estado con menor μ[log_displacement] recibe etiqueta 'estacionario' (0)."""
    df = _synthetic_features_df(n_birds=5, days_per_bird=(50,) * 5)
    X, lengths = build_sequences(
        df,
        bird_ids=[f"BIRD{i:02d}" for i in range(5)],
        feature_cols=["log_displacement_km", "abs_turning_angle_rad"],
    )
    model, scaler, _, _ = fit_hmm_with_restarts(X, lengths, n_restarts=2, random_state=0)
    label_map = relabel_states(
        model, scaler, ["log_displacement_km", "abs_turning_angle_rad"]
    )
    # Debe haber exactamente 2 entradas, valores son las etiquetas.
    assert set(label_map.values()) == {"estacionario", "migración"}
    # El estado con menor media (en log_dist desescalado) → estacionario.
    means_raw = scaler.inverse_transform(model.means_)
    estac_idx = int(np.argmin(means_raw[:, 0]))
    assert label_map[estac_idx] == "estacionario"
