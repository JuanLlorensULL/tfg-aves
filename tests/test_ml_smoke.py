"""Smoke test: el paquete tfg_aves.ml se importa sin errores."""
from __future__ import annotations


def test_import_ml() -> None:
    import tfg_aves.ml as ml

    assert hasattr(ml, "build_o4")
    assert hasattr(ml, "build_feature_matrix")
    assert hasattr(ml, "train_random_forest")
    assert hasattr(ml, "train_xgboost")
    assert hasattr(ml, "train_lightgbm")
    assert hasattr(ml, "evaluate_global")
    assert hasattr(ml, "compute_persistence_baseline")
    assert hasattr(ml, "compute_markov_baseline")
