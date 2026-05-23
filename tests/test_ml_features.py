"""Tests unitarios de tfg_aves.ml.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfg_aves.markov.discretize import haversine_km
from tfg_aves.ml.features import (
    FEATURES_KINEMATIC,
    add_cyclic_doy,
    assign_cells_to_features,
    build_feature_matrix,
    compute_causal_kinematics,
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


def test_build_feature_matrix_filters_gap_aware() -> None:
    """El último día de cada ave queda fuera: no tiene t+1 → cell_id_t_next nulo."""
    # 3 aves × 10 días contiguos procesados con cinemática causal.
    birds = [_linear_bird(bird=b, n=10, dlat=0.1) for b in ["A", "B", "C"]]
    df = pd.concat(birds, ignore_index=True)
    kin = compute_causal_kinematics(df)
    cells = _cells_grid()
    out = build_feature_matrix(kin, cells, include_bird_id=True)
    last_days = df.groupby("bird_id")["date_utc"].max()
    for bird, last_day in last_days.items():
        assert not ((out["bird_id"] == bird) & (out["date_utc"] == last_day)).any()


def test_build_feature_matrix_include_bird_id_flag() -> None:
    """El flag include_bird_id controla si bird_id aparece en _features y columnas."""
    kin = compute_causal_kinematics(_linear_bird(n=10, dlat=0.1))
    cells = _cells_grid()
    out_pers = build_feature_matrix(kin, cells, include_bird_id=True)
    out_pob = build_feature_matrix(kin, cells, include_bird_id=False)
    assert "bird_id" in out_pers.columns
    assert "_features" in out_pob.attrs
    assert "bird_id" not in out_pob.attrs["_features"]
    assert "bird_id" in out_pers.attrs["_features"]


def test_build_feature_matrix_target_is_next_day_cell() -> None:
    """Todas las filas de la salida tienen cell_id_t_next no nulo."""
    birds = [_linear_bird(bird=b, n=10, dlat=0.1) for b in ["A", "B", "C"]]
    df = pd.concat(birds, ignore_index=True)
    kin = compute_causal_kinematics(df)
    cells = _cells_grid()
    out = build_feature_matrix(kin, cells, include_bird_id=True)
    assert (out["cell_id_t_next"].notna()).all()
    assert set(out["bird_id"].unique()) == {"A", "B", "C"}


def test_split_temporal_per_bird_no_leakage() -> None:
    birds = [_linear_bird(bird=b, n=20, dlat=0.05) for b in ["A", "B", "C"]]
    df = pd.concat(birds, ignore_index=True)
    kin = compute_causal_kinematics(df)
    matrix = build_feature_matrix(kin, _cells_grid(), include_bird_id=True)
    train, val, test = split_temporal_per_bird(matrix)
    for bird in matrix["bird_id"].unique():  # noqa: B007
        train_max = pd.concat([train, val]).query("bird_id == @bird")["date_utc"].max()
        test_min = test.query("bird_id == @bird")["date_utc"].min()
        assert pd.isna(train_max) or pd.isna(test_min) or train_max < test_min


def test_split_temporal_per_bird_fractions() -> None:
    birds = [_linear_bird(bird=b, n=20, dlat=0.05) for b in ["A", "B", "C"]]
    df = pd.concat(birds, ignore_index=True)
    kin = compute_causal_kinematics(df)
    matrix = build_feature_matrix(kin, _cells_grid(), include_bird_id=True)
    train, val, test = split_temporal_per_bird(matrix, train_frac=0.8, val_frac_of_train=0.1)
    n_total = len(matrix)
    n_train, n_val, n_test = len(train), len(val), len(test)
    assert abs(n_test / n_total - 0.20) <= 0.05
    assert abs(n_val / n_total - 0.08) <= 0.05
    assert abs(n_train / n_total - 0.72) <= 0.05


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


def _cells_grid():
    cells = []
    for i in range(78, 84):
        for j in range(-9, -3):
            cells.append({"cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j})
    return pd.DataFrame(cells)


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


def test_no_contaminacion_entre_aves():
    """Verifica que las filas de inicio de ave B no usan posiciones de ave A.

    Ave A: lat ~40, ave B: lat ~50. Si hubiera contaminación, step_in_km en la
    primera fila de B sería ~1 100 km (haversine 40°→50°) en vez de NaN.
    """
    bird_a = _linear_bird(bird="A", n=5, lat0=40.0, lon0=-3.0, dlat=0.1, dlon=0.0)
    bird_b = _linear_bird(bird="B", n=5, lat0=50.0, lon0=10.0, dlat=0.1, dlon=0.0)

    df = pd.concat([bird_a, bird_b], ignore_index=True)
    out = compute_causal_kinematics(df)

    b_rows = out[out["bird_id"] == "B"].reset_index(drop=True)

    # Primera fila de B: no existe t-1 dentro de B → step_in_km debe ser NaN.
    assert np.isnan(b_rows.loc[0, "step_in_km"]), (
        f"Contaminación detectada: step_in_km fila 0 de B = {b_rows.loc[0, 'step_in_km']:.1f} km "
        "(esperado NaN; si fuera ~1100 km hay contaminación con posiciones de A)"
    )
    # Primera fila de B: sin t-1 propio → is_hmm_obs_valid debe ser False.
    assert not b_rows.loc[0, "is_hmm_obs_valid"]

    # Segunda fila de B: tiene t-1 dentro de B pero no t-2 → cos_turning_in NaN.
    assert np.isnan(b_rows.loc[1, "cos_turning_in"]), (
        f"Contaminación detectada: cos_turning_in fila 1 de B = {b_rows.loc[1, 'cos_turning_in']} "
        "(esperado NaN; t-2 de B no existe, no debe tomarse de A)"
    )
    # Segunda fila de B: sin t-2 propio → is_hmm_obs_valid debe ser False.
    assert not b_rows.loc[1, "is_hmm_obs_valid"]

    # A partir de la tercera fila de B (t-2 y t-1 dentro de B) sí hay valores válidos.
    assert not np.isnan(b_rows.loc[2, "step_in_km"])
    assert not np.isnan(b_rows.loc[2, "cos_turning_in"])
    assert b_rows.loc[2, "is_hmm_obs_valid"]


def test_build_feature_matrix_solo_filas_candidatas():
    kin = compute_causal_kinematics(_linear_bird(n=8, dlat=0.1))
    cells = _cells_grid()
    m = build_feature_matrix(kin, cells, include_bird_id=False)
    # Candidata = is_hmm_obs_valid(t) (t>=2) y cell_id_t_next no nulo (t+1 existe).
    # Con 8 días contiguos: t en {2..6} cumplen ambas (t=7 no tiene t+1).
    assert set(m["date_utc"]) == set(pd.date_range("2020-01-01", periods=8)[2:7])
    # Las 8 features cinemáticas presentes y sin NaN.
    assert FEATURES_KINEMATIC == m.attrs["_features"]
    assert not m[FEATURES_KINEMATIC].isna().any().any()


def test_build_feature_matrix_no_cruza_gap():
    df = _linear_bird(n=8, dlat=0.1)
    # Inserta un gap: día 4 inválido (lat NaN).
    df.loc[4, ["lat", "lon"]] = [np.nan, np.nan]
    kin = compute_causal_kinematics(df)
    m = build_feature_matrix(kin, _cells_grid(), include_bird_id=False)
    # Ninguna fila candidata puede tener t, t-1, t-2 o t+1 tocando el día 4.
    fechas = set(m["date_utc"])
    base = pd.date_range("2020-01-01", periods=8)
    assert base[4] not in fechas  # el propio gap
    assert base[3] not in fechas  # su t+1 cae en gap
    assert base[5] not in fechas and base[6] not in fechas  # necesitan t-1/t-2 en gap


def test_build_feature_matrix_incluye_bird_id():
    kin = compute_causal_kinematics(_linear_bird(n=8, dlat=0.1))
    m = build_feature_matrix(kin, _cells_grid(), include_bird_id=True)
    assert m.attrs["_features"][0] == "bird_id"
    assert "bird_id" in m.columns
