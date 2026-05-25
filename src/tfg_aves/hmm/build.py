"""Orquestación end-to-end de O3 (causal)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from tfg_aves.data.split import assign_temporal_split

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV
from .causal import (
    HMM_EMISSION_COLS_A,
    HMM_EMISSION_COLS_B,
    build_hmm_sequences,
    compute_causal_kinematics,
    decode_causal_states,
    fit_causal_hmm,
)
from .evaluate import ab_agreement_causal, log_likelihood_per_obs
from .features import compute_observation_features, load_vegetation_from_raw


@dataclass
class BuildO3Result:
    features_path: Path
    models_path: Path
    metrics_path: Path
    n_observations: int
    ll_per_obs_a: float
    ll_per_obs_b: float
    pct_agreement_ab: float


def build_o3(
    *,
    n_restarts: int = 10,
    random_state: int = 0,
    daily_path: Path = DAILY_PARQUET,
    raw_csv: Path = RAW_CSV,
    out_dir: Path = O3_OUT_DIR,
) -> BuildO3Result:
    """Pipeline causal de O3: features entrantes, split temporal, fit A y B, filtrado, escritura."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_daily = pd.read_parquet(daily_path)
    veg = load_vegetation_from_raw(raw_csv, df_daily["source_event_id"])
    # compute_observation_features devuelve lat, lon, veg_low, veg_high,
    # daylight_hours por (bird_id, date_utc) — todo lo que necesita la
    # cinemática causal. Las columnas salientes que también trae son inocuas.
    base = compute_observation_features(df_daily, df_raw=veg)
    kin = compute_causal_kinematics(base)

    # Split temporal propio de O3, calculado sobre los días HMM-válidos.
    valid = kin[kin["is_hmm_obs_valid"]].copy()
    valid = assign_temporal_split(valid)
    kin = kin.merge(
        valid[["bird_id", "date_utc", "split"]],
        on=["bird_id", "date_utc"], how="left",
    )
    train_valid = valid[valid["split"] == "train"]
    cutoff_by_bird = (
        train_valid.assign(_d=pd.to_datetime(train_valid["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )

    # Ajuste y decodificado causal de A y B.
    model_a, labels_a = fit_causal_hmm(
        kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_A,
        n_restarts=n_restarts, seed=random_state,
    )
    model_b, labels_b = fit_causal_hmm(
        kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_B,
        n_restarts=n_restarts, seed=random_state,
    )
    states_a = decode_causal_states(
        model_a, labels_a, kin, emission_cols=HMM_EMISSION_COLS_A, suffix="a",
    )
    states_b = decode_causal_states(
        model_b, labels_b, kin, emission_cols=HMM_EMISSION_COLS_B, suffix="b",
    )
    df = (
        kin.merge(states_a, on=["bird_id", "date_utc"], how="left")
           .merge(states_b, on=["bird_id", "date_utc"], how="left")
    )

    # LL holdout temporal (días test) por modelo y acuerdo A-B.
    test_kin = kin[kin["split"] == "test"]
    xa, la, _ = build_hmm_sequences(
        test_kin, cutoff_by_bird=None, emission_cols=HMM_EMISSION_COLS_A,
    )
    xb, lb, _ = build_hmm_sequences(
        test_kin, cutoff_by_bird=None, emission_cols=HMM_EMISSION_COLS_B,
    )
    ll_a = log_likelihood_per_obs(model_a, xa, la)
    ll_b = log_likelihood_per_obs(model_b, xb, lb)
    agreement = ab_agreement_causal(df)
    pct_agree = float(agreement["pct_agreement"])

    cols_final = [
        "bird_id", "date_utc", "lat", "lon",
        "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
        "daylight_hours", "veg_low", "veg_high",
        "is_hmm_obs_valid", "split",
        "state_a_causal", "state_b_causal",
        "posterior_a_estacionario", "posterior_a_migracion",
        "posterior_b_estacionario", "posterior_b_migracion",
    ]
    out_cols = [c for c in cols_final if c in df.columns]
    features_path = out_dir / "features.parquet"
    models_path = out_dir / "models_a_b.pkl"
    metrics_path = out_dir / "metrics.parquet"

    df[out_cols].to_parquet(features_path, index=False)
    joblib.dump(
        {
            "model_a": model_a, "model_b": model_b,
            "emission_cols_a": HMM_EMISSION_COLS_A,
            "emission_cols_b": HMM_EMISSION_COLS_B,
            "label_map_a": labels_a, "label_map_b": labels_b,
            "cutoff_by_bird": cutoff_by_bird, "random_state": random_state,
        },
        models_path,
    )
    pd.DataFrame([
        {"model": "a", "scope": "test", "metric": "ll_per_obs", "value": ll_a},
        {"model": "b", "scope": "test", "metric": "ll_per_obs", "value": ll_b},
        {"model": "agreement", "scope": "both", "metric": "pct_agreement", "value": pct_agree},
    ]).to_parquet(metrics_path, index=False)

    return BuildO3Result(
        features_path=features_path,
        models_path=models_path,
        metrics_path=metrics_path,
        n_observations=int(df["is_hmm_obs_valid"].sum()),
        ll_per_obs_a=ll_a,
        ll_per_obs_b=ll_b,
        pct_agreement_ab=pct_agree,
    )
