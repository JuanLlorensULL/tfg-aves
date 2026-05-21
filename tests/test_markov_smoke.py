"""Smoke test: el paquete tfg_aves.markov se importa sin errores."""
from __future__ import annotations


def test_import_markov() -> None:
    import tfg_aves.markov as m

    # Las firmas existen como callables.
    assert callable(m.assign_cell)
    assert callable(m.cell_centroid)
    assert callable(m.discretize_dataframe)
    assert callable(m.haversine_km)
    assert callable(m.build_transitions)
    assert callable(m.build_counts)
    assert callable(m.laplace_smooth)
    assert callable(m.marginal_distribution)
    assert callable(m.predict_distribution)
    assert callable(m.topk_from_distribution)
    assert callable(m.prediction_distance_km)
    assert callable(m.lobo_predictions)
    assert callable(m.persistence_predictions)
    assert callable(m.aggregate_metrics)
    assert callable(m.build_o2)
