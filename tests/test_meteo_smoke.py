"""Smoke tests: el paquete meteo importa y expone su API."""
from __future__ import annotations


def test_meteo_imports():
    from tfg_aves import meteo

    assert hasattr(meteo, "load_wind_dataset")
    assert hasattr(meteo, "interpolate_wind_to_fixes")
    assert hasattr(meteo, "build_wind")
