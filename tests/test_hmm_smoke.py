"""Smoke test: el paquete tfg_aves.hmm se importa sin errores."""
from __future__ import annotations


def test_import_hmm() -> None:
    import tfg_aves.hmm as m

    # Funciones públicas.
    assert callable(m.bearing_rad)
    assert callable(m.daylight_hours)
    assert callable(m.load_vegetation_from_raw)
    assert callable(m.compute_observation_features)
    assert callable(m.stratified_holdout_split)
    assert callable(m.build_sequences)
    assert callable(m.fit_hmm_with_restarts)
    assert callable(m.relabel_states)
    assert callable(m.log_likelihood_per_obs)
    assert callable(m.viterbi_per_bird)
    assert callable(m.biological_coherence_table)
    assert callable(m.ab_agreement)
    assert callable(m.build_o3)
    assert callable(m.BuildO3Result)
