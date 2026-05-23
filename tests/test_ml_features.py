"""Tests unitarios de tfg_aves.ml.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfg_aves.ml.features import (
    add_cyclic_doy,
    assign_cells_to_features,
    build_feature_matrix,
    split_temporal_per_bird,
)


def _build_synthetic_o3_features() -> pd.DataFrame:
    """3 aves × 20 días con valid=True y posiciones que cubren varias celdas."""
    rng = np.random.default_rng(0)
    rows = []
    for bird in ["A", "B", "C"]:
        dates = pd.date_range("2020-01-01", periods=20)
        lats = 40.0 + rng.normal(0, 0.5, size=20).cumsum() * 0.1
        lons = -3.0 + rng.normal(0, 0.5, size=20).cumsum() * 0.1
        for i, d in enumerate(dates):
            rows.append({
                "bird_id": bird,
                "date_utc": d,
                "lat": lats[i], "lon": lons[i],
                "step_length_km": rng.uniform(1, 100),
                "cos_turning_angle": rng.uniform(-1, 1),
                "daylight_hours": 12.0,
                "veg_low": 0.5, "veg_high": 0.5,
                "state_a": int(rng.integers(0, 2)),
                "state_b": int(rng.integers(0, 2)),
                "posterior_a_estacionario": 0.5, "posterior_a_migracion": 0.5,
                "posterior_b_estacionario": 0.5, "posterior_b_migracion": 0.5,
                "is_observation_valid": True,
                "in_holdout": False,
            })
    return pd.DataFrame(rows)


def _build_synthetic_cells() -> pd.DataFrame:
    """Grid 0,5° de 40 celdas alrededor de (40, -3)."""
    rows = []
    for i in range(78, 82):  # cell_lat_idx
        for j in range(-8, -4):  # cell_lon_idx
            rows.append({
                "cell_id": f"{i}_{j}",
                "cell_lat_idx": i,
                "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5,
                "lon_c": (j + 0.5) * 0.5,
                "n_obs_total": 10,
            })
    return pd.DataFrame(rows)


def test_add_cyclic_doy_identity() -> None:
    df = pd.DataFrame({"date_utc": pd.date_range("2020-03-21", periods=4)})
    out = add_cyclic_doy(df)
    np.testing.assert_allclose(out["sin_doy"] ** 2 + out["cos_doy"] ** 2, 1.0, atol=1e-10)
    assert -1.0 <= out["sin_doy"].iloc[0] <= 1.0


def test_assign_cells_adds_columns() -> None:
    df = _build_synthetic_o3_features()
    cells = _build_synthetic_cells()
    out = assign_cells_to_features(df, cells, cell_deg=0.5)
    assert "cell_id_t" in out.columns
    assert "cell_id_t_next" in out.columns


def test_build_feature_matrix_filters_invalid_and_gap_aware() -> None:
    """Filas sin t+1 calendario válido deben quedar fuera (§8.11 del spec)."""
    df = _build_synthetic_o3_features()
    df.loc[df.groupby("bird_id")["date_utc"].idxmax(), "is_observation_valid"] = False
    cells = _build_synthetic_cells()
    out = build_feature_matrix(df, cells, include_bird_id=True)
    last_days = df.groupby("bird_id")["date_utc"].max()
    for bird, last_day in last_days.items():
        assert not ((out["bird_id"] == bird) & (out["date_utc"] == last_day)).any()


def test_build_feature_matrix_include_bird_id_flag() -> None:
    df = _build_synthetic_o3_features()
    cells = _build_synthetic_cells()
    out_pers = build_feature_matrix(df, cells, include_bird_id=True)
    out_pob = build_feature_matrix(df, cells, include_bird_id=False)
    assert "bird_id" in out_pers.columns
    assert "_features" in out_pob.attrs
    assert "bird_id" not in out_pob.attrs["_features"]
    assert "bird_id" in out_pers.attrs["_features"]


def test_build_feature_matrix_target_is_next_day_cell() -> None:
    df = _build_synthetic_o3_features()
    cells = _build_synthetic_cells()
    out = build_feature_matrix(df, cells, include_bird_id=True)
    assert (out["cell_id_t_next"].notna()).all()
    assert set(out["bird_id"].unique()) == {"A", "B", "C"}


def test_split_temporal_per_bird_no_leakage() -> None:
    df = _build_synthetic_o3_features()
    cells = _build_synthetic_cells()
    matrix = build_feature_matrix(df, cells, include_bird_id=True)
    train, val, test = split_temporal_per_bird(matrix)
    for bird in matrix["bird_id"].unique():  # noqa: B007
        train_max = pd.concat([train, val]).query("bird_id == @bird")["date_utc"].max()
        test_min = test.query("bird_id == @bird")["date_utc"].min()
        assert pd.isna(train_max) or pd.isna(test_min) or train_max < test_min


def test_split_temporal_per_bird_fractions() -> None:
    df = _build_synthetic_o3_features()
    cells = _build_synthetic_cells()
    matrix = build_feature_matrix(df, cells, include_bird_id=True)
    train, val, test = split_temporal_per_bird(matrix, train_frac=0.8, val_frac_of_train=0.1)
    n_total = len(matrix)
    n_train, n_val, n_test = len(train), len(val), len(test)
    assert abs(n_test / n_total - 0.20) <= 0.05
    assert abs(n_val / n_total - 0.08) <= 0.05
    assert abs(n_train / n_total - 0.72) <= 0.05


def test_build_feature_matrix_excludes_rows_around_calendar_gap() -> None:
    """Una fila cuyo día t+1 (calendario) no existe queda fuera (§8.11)."""
    cells = _build_synthetic_cells()
    # Ave única con un gap real: días 1, 2, 4, 5 (falta el día 3).
    rows = []
    for i, d in enumerate(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]):
        rows.append({
            "bird_id": "G",
            "date_utc": pd.Timestamp(d),
            "lat": 40.0 + 0.05 * i, "lon": -3.0 + 0.05 * i,
            "step_length_km": 10.0,
            "cos_turning_angle": 0.0,
            "daylight_hours": 12.0,
            "veg_low": 0.5, "veg_high": 0.5,
            "state_a": 0, "state_b": 0,
            "posterior_a_estacionario": 0.5, "posterior_a_migracion": 0.5,
            "posterior_b_estacionario": 0.5, "posterior_b_migracion": 0.5,
            "is_observation_valid": True,
            "in_holdout": False,
        })
    df = pd.DataFrame(rows)
    out = build_feature_matrix(df, cells, include_bird_id=True)
    # Días esperados en la salida:
    #   - día 1 → t+1 es día 2 (válido)         → SÍ entra
    #   - día 2 → t+1 sería día 3 (no existe)   → NO entra (gap)
    #   - día 4 → t+1 es día 5 (válido)         → SÍ entra
    #   - día 5 → t+1 sería día 6 (no existe)   → NO entra (último día)
    out_dates = set(out["date_utc"].dt.date.astype(str))
    assert out_dates == {"2020-01-01", "2020-01-04"}


def test_merge_wind_features_preserves_rows():
    """LEFT JOIN por (bird_id, date_utc) no cambia el número de filas."""
    from tfg_aves.ml.features import merge_wind_features

    matrix = pd.DataFrame({
        "bird_id": ["A", "A", "B"],
        "date_utc": [
            pd.Timestamp("2010-03-15").date(),
            pd.Timestamp("2010-03-16").date(),
            pd.Timestamp("2010-04-10").date(),
        ],
        "lat": [59.5, 59.6, 60.0],
        "lon": [10.5, 10.6, 10.9],
    })
    wind = pd.DataFrame({
        "bird_id": ["A", "A", "B", "C"],
        "date_utc": [
            pd.Timestamp("2010-03-15").date(),
            pd.Timestamp("2010-03-16").date(),
            pd.Timestamp("2010-04-10").date(),
            pd.Timestamp("2010-04-10").date(),
        ],
        "wind_u_850": [1.0, 2.0, 3.0, 99.0],
        "wind_v_850": [-1.0, -2.0, -3.0, -99.0],
        "wind_speed_850": [1.4, 2.8, 4.2, 140.0],
    })

    out = merge_wind_features(matrix, wind)

    assert len(out) == len(matrix)
    assert set(out.columns) == {
        "bird_id", "date_utc", "lat", "lon",
        "wind_u_850", "wind_v_850", "wind_speed_850",
    }
    # La fila de C en wind no debe aparecer (LEFT JOIN sobre matrix).
    assert "C" not in out["bird_id"].tolist()


def test_merge_wind_features_propagates_nan_when_no_match():
    """Filas de matrix sin contrapartida en wind → NaN en columnas de viento."""
    from tfg_aves.ml.features import merge_wind_features

    matrix = pd.DataFrame({
        "bird_id": ["A", "B"],
        "date_utc": [
            pd.Timestamp("2010-03-15").date(),
            pd.Timestamp("2010-03-16").date(),
        ],
        "lat": [59.5, 59.6],
        "lon": [10.5, 10.6],
    })
    wind = pd.DataFrame({
        "bird_id": ["A"],
        "date_utc": [pd.Timestamp("2010-03-15").date()],
        "wind_u_850": [1.0],
        "wind_v_850": [-1.0],
        "wind_speed_850": [1.4],
    })

    out = merge_wind_features(matrix, wind)

    a_row = out[out["bird_id"] == "A"].iloc[0]
    b_row = out[out["bird_id"] == "B"].iloc[0]
    assert a_row["wind_u_850"] == pytest.approx(1.0)
    assert np.isnan(b_row["wind_u_850"])
    assert np.isnan(b_row["wind_v_850"])
    assert np.isnan(b_row["wind_speed_850"])


from tfg_aves.ml.features import compute_causal_kinematics
from tfg_aves.markov.discretize import haversine_km


def _linear_bird(bird="A", n=10, lat0=40.0, lon0=-3.0, dlat=0.1, dlon=0.0):
    """Un ave con n días contiguos, posiciones en línea recta hacia el norte."""
    dates = pd.date_range("2020-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "bird_id": bird, "date_utc": d,
            "lat": lat0 + dlat * i, "lon": lon0 + dlon * i,
            "veg_low": 0.5, "veg_high": 0.5, "daylight_hours": 12.0,
            "is_observation_valid": True,
        })
    return pd.DataFrame(rows)


def test_step_in_km_es_haversine_entrante():
    df = _linear_bird(n=5, dlat=0.1)
    out = compute_causal_kinematics(df)
    # step_in_km en t = dist(pos(t-1), pos(t)); fila 0 es NaN (no hay t-1).
    esperado = haversine_km(40.0, -3.0, 40.1, -3.0)
    assert np.isnan(out.loc[0, "step_in_km"])
    assert out.loc[1, "step_in_km"] == pytest.approx(esperado, rel=1e-6)


def test_bearing_in_es_unitario():
    df = _linear_bird(n=5, dlat=0.1, dlon=0.1)
    out = compute_causal_kinematics(df)
    s = out.loc[2, "sin_bearing_in"]
    c = out.loc[2, "cos_bearing_in"]
    assert s**2 + c**2 == pytest.approx(1.0, rel=1e-6)


def test_cos_turning_in_es_causal():
    """Cambiar pos(t+1) no debe alterar cos_turning_in(t): sólo usa t-2,t-1,t."""
    df = _linear_bird(n=6, dlat=0.1)
    out1 = compute_causal_kinematics(df)
    df2 = df.copy()
    df2.loc[4, "lat"] = 99.0  # altera pos en t=4
    out2 = compute_causal_kinematics(df2)
    # cos_turning_in en t=3 sólo depende de t=1,2,3 → no cambia.
    assert out1.loc[3, "cos_turning_in"] == pytest.approx(out2.loc[3, "cos_turning_in"])


def test_vuelo_recto_da_cos_turning_uno():
    df = _linear_bird(n=5, dlat=0.1)  # recto hacia el norte
    out = compute_causal_kinematics(df)
    # primeras dos filas NaN (necesita t-2); de la 2 en adelante recto → +1.
    assert out.loc[2, "cos_turning_in"] == pytest.approx(1.0, abs=1e-6)


def test_is_hmm_obs_valid_requiere_racha_de_3():
    df = _linear_bird(n=5, dlat=0.1)
    out = compute_causal_kinematics(df)
    # Filas 0 y 1 no tienen t-2 → inválidas; 2,3,4 válidas.
    assert not out.loc[0, "is_hmm_obs_valid"]
    assert not out.loc[1, "is_hmm_obs_valid"]
    assert out.loc[2, "is_hmm_obs_valid"]
    assert out.loc[4, "is_hmm_obs_valid"]
