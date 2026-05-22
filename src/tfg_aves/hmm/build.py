"""Orquestación end-to-end de O3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV
from .evaluate import ab_agreement, log_likelihood_per_obs, viterbi_per_bird
from .features import compute_observation_features, load_vegetation_from_raw
from .fit import (
    build_sequences,
    fit_hmm_with_restarts,
    relabel_states,
    stratified_holdout_split,
)


@dataclass
class BuildO3Result:
    features_path: Path
    models_path: Path
    metrics_path: Path
    n_birds_train: int
    n_birds_holdout: int
    n_observations: int
    ll_per_obs_a: float
    ll_per_obs_b: float
    pct_agreement_ab: float


FEATURE_COLS_A = ["step_length_km", "abs_turning_angle_rad"]
FEATURE_COLS_B = [
    "step_length_km", "abs_turning_angle_rad",
    "veg_low", "veg_high", "daylight_hours",
]


def build_o3(
    *,
    holdout_frac: float = 0.20,
    n_restarts: int = 10,
    random_state: int = 0,
    daily_path: Path = DAILY_PARQUET,
    raw_csv: Path = RAW_CSV,
    out_dir: Path = O3_OUT_DIR,
) -> BuildO3Result:
    """Pipeline completa de O3: features → split → fit A y B → evaluar → escribir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_daily = pd.read_parquet(daily_path)
    veg = load_vegetation_from_raw(raw_csv, df_daily["source_event_id"])
    df_features = compute_observation_features(df_daily, df_raw=veg)

    train_ids, holdout_ids = stratified_holdout_split(
        df_features, holdout_frac=holdout_frac, random_state=random_state,
    )
    df_features["in_holdout"] = df_features["bird_id"].isin(holdout_ids)

    # Modelo A — features cinemáticas.
    X_train_a, lengths_train_a = build_sequences(df_features, train_ids, FEATURE_COLS_A)
    model_a, _, _ = fit_hmm_with_restarts(
        X_train_a, lengths_train_a, n_restarts=n_restarts, random_state=random_state,
    )
    label_map_a = relabel_states(model_a, FEATURE_COLS_A)

    # Modelo B — A + contexto.
    X_train_b, lengths_train_b = build_sequences(df_features, train_ids, FEATURE_COLS_B)
    model_b, _, _ = fit_hmm_with_restarts(
        X_train_b, lengths_train_b, n_restarts=n_restarts, random_state=random_state,
    )
    label_map_b = relabel_states(model_b, FEATURE_COLS_B)

    # Viterbi sobre TODAS las aves (train + holdout) — el modelo no las ha visto
    # como inputs de fit en el caso de holdout, pero les puede asignar estado.
    df_features = viterbi_per_bird(
        model_a, df_features, FEATURE_COLS_A, label_map_a, model_suffix="a",
    )
    df_features = viterbi_per_bird(
        model_b, df_features, FEATURE_COLS_B, label_map_b, model_suffix="b",
    )

    # LL en holdout.
    X_hold_a, lengths_hold_a = build_sequences(df_features, holdout_ids, FEATURE_COLS_A)
    X_hold_b, lengths_hold_b = build_sequences(df_features, holdout_ids, FEATURE_COLS_B)
    ll_a = log_likelihood_per_obs(model_a, X_hold_a, lengths_hold_a)
    ll_b = log_likelihood_per_obs(model_b, X_hold_b, lengths_hold_b)

    # Acuerdo A-B sobre todo el dataset válido.
    agreement = ab_agreement(df_features)
    pct_agree = float(agreement["pct_agreement"])

    # Reordenar columnas según esquema del spec (sección 6.1).
    cols_final = [
        "bird_id", "date_utc", "lat", "lon",
        "step_length_km", "abs_turning_angle_rad", "daylight_hours",
        "veg_low", "veg_high",
        "state_a", "state_b",
        "posterior_a_estacionario", "posterior_a_migracion",
        "posterior_b_estacionario", "posterior_b_migracion",
        "is_observation_valid", "in_holdout",
    ]
    out_cols = [c for c in cols_final if c in df_features.columns]

    features_path = out_dir / "features.parquet"
    models_path = out_dir / "models_a_b.pkl"
    metrics_path = out_dir / "metrics.parquet"

    df_features[out_cols].to_parquet(features_path, index=False)

    joblib.dump(
        {
            "model_a": model_a, "model_b": model_b,
            "feature_cols_a": FEATURE_COLS_A, "feature_cols_b": FEATURE_COLS_B,
            "label_map_a": label_map_a, "label_map_b": label_map_b,
            "train_bird_ids": train_ids, "holdout_bird_ids": holdout_ids,
            "random_state": random_state,
        },
        models_path,
    )

    metrics_rows = [
        {"model": "a", "scope": "holdout", "metric": "ll_per_obs", "value": ll_a},
        {"model": "b", "scope": "holdout", "metric": "ll_per_obs", "value": ll_b},
        {"model": "agreement", "scope": "both", "metric": "pct_agreement", "value": pct_agree},
        {
            "model": "agreement", "scope": "both",
            "metric": "pct_b_adds_migration",
            "value": float(agreement["pct_b_adds_migration"]),
        },
        {
            "model": "agreement", "scope": "both",
            "metric": "pct_b_adds_stationary",
            "value": float(agreement["pct_b_adds_stationary"]),
        },
    ]
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)

    return BuildO3Result(
        features_path=features_path,
        models_path=models_path,
        metrics_path=metrics_path,
        n_birds_train=len(train_ids),
        n_birds_holdout=len(holdout_ids),
        n_observations=int(df_features["is_observation_valid"].sum()),
        ll_per_obs_a=ll_a,
        ll_per_obs_b=ll_b,
        pct_agreement_ab=pct_agree,
    )
