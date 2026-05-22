"""Tests unitarios de tfg_aves.ml.train."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_aves.ml.train import (
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)


def _synthetic_train_val(n_classes: int = 5, n_train: int = 200, n_val: int = 50):
    """Dataset sintético separable. Cada clase tiene un offset en lat/lon."""
    rng = np.random.default_rng(0)
    rows_train, rows_val = [], []
    for _split, n, target in [("train", n_train, rows_train), ("val", n_val, rows_val)]:
        for _ in range(n):
            cls = int(rng.integers(0, n_classes))
            lat = cls * 5.0 + rng.normal(0, 0.3)
            lon = cls * 5.0 + rng.normal(0, 0.3)
            bird = f"bird_{int(rng.integers(0, 3))}"
            target.append({
                "lat": lat, "lon": lon, "bird_id": bird,
                "step_length_km": rng.uniform(0, 100),
                "y": cls,
            })
    df_train = pd.DataFrame(rows_train)
    df_val = pd.DataFrame(rows_val)
    X_train = df_train[["lat", "lon", "step_length_km", "bird_id"]]
    y_train = df_train["y"].to_numpy()
    X_val = df_val[["lat", "lon", "step_length_km", "bird_id"]]
    y_val = df_val["y"].to_numpy()
    return X_train, y_train, X_val, y_val


def test_random_forest_trains_and_predicts() -> None:
    X_train, y_train, X_val, y_val = _synthetic_train_val()
    model = train_random_forest(
        X_train, y_train, categorical_cols=["bird_id"], seed=0,
    )
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_xgboost_trains_with_early_stopping() -> None:
    X_train, y_train, X_val, y_val = _synthetic_train_val()
    model = train_xgboost(
        X_train, y_train, X_val, y_val,
        categorical_cols=["bird_id"], seed=0,
    )
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 5)
    n_boosted = getattr(model, "best_iteration", None)
    assert n_boosted is None or n_boosted < 1000


def test_lightgbm_trains_and_predicts() -> None:
    X_train, y_train, X_val, y_val = _synthetic_train_val()
    model = train_lightgbm(
        X_train, y_train, X_val, y_val,
        categorical_cols=["bird_id"], seed=0,
    )
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 5)


def test_three_models_reach_high_accuracy_on_separable_data() -> None:
    """Sanity check: top-1 ≈ 1 sobre dataset trivialmente separable."""
    X_train, y_train, X_val, y_val = _synthetic_train_val(n_train=400, n_val=100)
    for fn in (
        lambda: train_random_forest(X_train, y_train, categorical_cols=["bird_id"], seed=0),
        lambda: train_xgboost(X_train, y_train, X_val, y_val,
                              categorical_cols=["bird_id"], seed=0),
        lambda: train_lightgbm(X_train, y_train, X_val, y_val,
                               categorical_cols=["bird_id"], seed=0),
    ):
        m = fn()
        preds = m.predict(X_val)
        acc = (preds == y_val).mean()
        assert acc >= 0.95, f"{m.__class__.__name__} acc={acc:.3f}"
