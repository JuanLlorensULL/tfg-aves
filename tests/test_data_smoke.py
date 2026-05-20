"""Smoke test del paquete tfg_aves.data: API pública importable."""
from __future__ import annotations


def test_public_api_imports():
    from tfg_aves.data import (
        INTERIM,
        PROCESSED,
        RAW_CSV,
        ROOT,
        build_daily,
        build_o1,
        coverage_by_hour,
        drop_invalid_coords_and_dupes,
        drop_movebank_flags,
        drop_speed_outliers,
        filter_birds_by_validity,
        load_raw,
        pick_reference_hour,
    )

    # Las rutas son objetos Path con sufijos esperados.
    assert RAW_CSV.name == "migration_original.csv"
    assert PROCESSED.name == "processed"
    assert INTERIM.name == "interim"
    assert (ROOT / "pyproject.toml").is_file()

    # Las funciones son callables.
    for fn in (
        load_raw,
        drop_movebank_flags,
        drop_invalid_coords_and_dupes,
        drop_speed_outliers,
        coverage_by_hour,
        pick_reference_hour,
        build_daily,
        filter_birds_by_validity,
        build_o1,
    ):
        assert callable(fn)
