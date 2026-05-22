"""Evaluación de los HMM: LL en holdout, coherencia biológica, acuerdo A-B."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def log_likelihood_per_obs(
    hmm: GaussianHMM,
    X_raw: np.ndarray,
    lengths: list[int],
) -> float:
    """Devuelve hmm.score(X_raw) / len(X_raw) sin estandarización previa."""
    if len(X_raw) == 0:
        return float("nan")
    return float(hmm.score(X_raw, lengths) / len(X_raw))


def viterbi_per_bird(
    hmm: GaussianHMM,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    label_map: dict[int, str],
    model_suffix: str,
) -> pd.DataFrame:
    """Añade state_<suffix>, posterior_<suffix>_estacionario, posterior_<suffix>_migracion."""
    df = df_features.copy()
    state_col = f"state_{model_suffix}"
    post_estac_col = f"posterior_{model_suffix}_estacionario"
    post_migr_col = f"posterior_{model_suffix}_migracion"

    df[state_col] = pd.array([pd.NA] * len(df), dtype="Int8")
    df[post_estac_col] = np.nan
    df[post_migr_col] = np.nan

    # Encuentra índice de estado para cada etiqueta (inverso del label_map).
    estac_idx = next(i for i, lab in label_map.items() if lab == "estacionario")
    migr_idx = next(i for i, lab in label_map.items() if lab == "migración")

    for _bird_id, sub in df.groupby("bird_id"):
        valid = sub[sub["is_observation_valid"]].sort_values("date_utc")
        if len(valid) == 0:
            continue
        dates = pd.to_datetime(valid["date_utc"]).reset_index(drop=True)
        gap = (dates.diff() != pd.Timedelta(days=1)).cumsum()
        for _, segment in valid.groupby(gap.values):
            X_seg = segment[feature_cols].to_numpy(dtype=np.float64)
            raw_states = hmm.predict(X_seg)
            posteriors = hmm.predict_proba(X_seg)
            mapped_states = np.array(
                [0 if s == estac_idx else 1 for s in raw_states], dtype=np.int8
            )
            df.loc[segment.index, state_col] = mapped_states
            df.loc[segment.index, post_estac_col] = posteriors[:, estac_idx]
            df.loc[segment.index, post_migr_col] = posteriors[:, migr_idx]
    return df


def biological_coherence_table(
    df_features: pd.DataFrame, state_col: str
) -> pd.DataFrame:
    """Cross-tabula estado vs mes + latitud (cuartil).

    Devuelve un DataFrame long con columnas: month, state, count, lat_q (opcional).
    """
    valid = df_features[
        df_features["is_observation_valid"] & df_features[state_col].notna()
    ].copy()
    valid["month"] = pd.to_datetime(valid["date_utc"]).dt.month
    valid["state"] = valid[state_col].astype(int)
    by_month_state = valid.groupby(["month", "state"]).size().reset_index(name="count")
    return by_month_state


def ab_agreement(df_features: pd.DataFrame) -> dict[str, float | pd.DataFrame]:
    """Matriz confusion 2×2 entre state_a y state_b + sub-DataFrame de desacuerdos."""
    valid = df_features[
        df_features["is_observation_valid"]
        & df_features["state_a"].notna()
        & df_features["state_b"].notna()
    ].copy()
    n = len(valid)
    if n == 0:
        return {
            "pct_agreement": float("nan"),
            "pct_b_adds_migration": float("nan"),
            "pct_b_adds_stationary": float("nan"),
            "disagreements": pd.DataFrame(),
            "confusion": pd.DataFrame(),
        }
    state_a = valid["state_a"].astype(int)
    state_b = valid["state_b"].astype(int)
    n_agree = int((state_a == state_b).sum())
    pct_agree = 100.0 * n_agree / n

    # Desagregados: A=0, B=1 → "B añade migración"; A=1, B=0 → "B añade estacionario".
    n_a_estac = int((state_a == 0).sum())
    n_a_migr = int((state_a == 1).sum())
    n_b_adds_migr = int(((state_a == 0) & (state_b == 1)).sum())
    n_b_adds_stat = int(((state_a == 1) & (state_b == 0)).sum())
    pct_b_adds_migr = (100.0 * n_b_adds_migr / n_a_estac) if n_a_estac > 0 else 0.0
    pct_b_adds_stat = (100.0 * n_b_adds_stat / n_a_migr) if n_a_migr > 0 else 0.0

    confusion = pd.crosstab(state_a, state_b, rownames=["state_a"], colnames=["state_b"])
    disagreements = valid[state_a != state_b]

    return {
        "pct_agreement": pct_agree,
        "pct_b_adds_migration": pct_b_adds_migr,
        "pct_b_adds_stationary": pct_b_adds_stat,
        "disagreements": disagreements,
        "confusion": confusion,
    }
