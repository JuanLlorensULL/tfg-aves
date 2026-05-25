"""Reexport temporal: el HMM causal se ha movido a tfg_aves.hmm.causal.

Preserva el esquema de salida antiguo de O4 (columna
``posterior_b_migracion_causal`` y sin el posterior estacionario) para que las
líneas L1/L2/L3 sigan funcionando sin cambios hasta su rewire posterior.
"""
from tfg_aves.hmm.causal import (  # noqa: F401
    build_hmm_sequences,
    fit_causal_hmm,
    forward_filtered_posteriors,
)
from tfg_aves.hmm.causal import decode_causal_states as _decode_causal_states


def decode_causal_states(model, label_map, kin):
    """Decodifica el Modelo B y devuelve el esquema antiguo que consume O4.

    El decode generalizado de ``hmm.causal`` devuelve
    ``posterior_b_estacionario`` y ``posterior_b_migracion``; aquí se
    renombra a ``posterior_b_migracion_causal`` y se descarta el estacionario
    para reproducir exactamente la salida previa.
    """
    out = _decode_causal_states(model, label_map, kin)
    return out.rename(
        columns={"posterior_b_migracion": "posterior_b_migracion_causal"},
    )[["bird_id", "date_utc", "state_b_causal", "posterior_b_migracion_causal"]]
