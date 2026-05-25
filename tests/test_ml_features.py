"""Tests unitarios de tfg_aves.ml.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfg_aves.data.split import split_temporal_per_bird
from tfg_aves.hmm.causal import compute_causal_kinematics
from tfg_aves.markov.discretize import haversine_km
from tfg_aves.ml.features import (
    FEATURES_KINEMATIC,
    add_cyclic_doy,
    assign_cells_to_features,
    attach_o3_state_and_split,
    build_feature_matrix,
)

# ---------------------------------------------------------------------------
# Helpers — preceden a sus usuarios según la convención del fichero.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests de add_cyclic_doy
# ---------------------------------------------------------------------------

def test_add_cyclic_doy_identity() -> None:
    df = pd.DataFrame({"date_utc": pd.date_range("2020-03-21", periods=4)})
    out = add_cyclic_doy(df)
    np.testing.assert_allclose(out["sin_doy"] ** 2 + out["cos_doy"] ** 2, 1.0, atol=1e-10)
    assert -1.0 <= out["sin_doy"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# Tests de assign_cells_to_features
# ---------------------------------------------------------------------------

def test_assign_cells_adds_columns() -> None:
    """assign_cells_to_features añade cell_id_t y cell_id_t_next."""
    df = _linear_bird(n=10, dlat=0.1)
    cells = _cells_grid()
    out = assign_cells_to_features(df, cells, cell_deg=0.5)
    assert "cell_id_t" in out.columns
    assert "cell_id_t_next" in out.columns


# ---------------------------------------------------------------------------
# Tests de build_feature_matrix
# ---------------------------------------------------------------------------

def test_build_feature_matrix_filters_gap_aware() -> None:
    """Los dos primeros y el último día de cada ave quedan fuera.

    - Primeros dos días: no tienen t-2 → is_hmm_obs_valid = False.
    - Último día: no tiene t+1 → cell_id_t_next nulo.
    """
    # 3 aves × 10 días contiguos procesados con cinemática causal.
    birds = [_linear_bird(bird=b, n=10, dlat=0.1) for b in ["A", "B", "C"]]
    df = pd.concat(birds, ignore_index=True)
    kin = compute_causal_kinematics(df)
    cells = _cells_grid()
    out = build_feature_matrix(kin, cells, include_bird_id=True)
    # Último día de cada ave ausente (sin t+1).
    last_days = df.groupby("bird_id")["date_utc"].max()
    for bird, last_day in last_days.items():
        assert not ((out["bird_id"] == bird) & (out["date_utc"] == last_day)).any()
    # Primeros dos días de cada ave ausentes (sin t-2 → is_hmm_obs_valid False).
    first_days = df.groupby("bird_id")["date_utc"].min()
    for bird, first_day in first_days.items():
        second_day = first_day + pd.Timedelta(days=1)
        assert not ((out["bird_id"] == bird) & (out["date_utc"] == first_day)).any()
        assert not ((out["bird_id"] == bird) & (out["date_utc"] == second_day)).any()


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


def test_build_feature_matrix_no_cruza_gap():
    """Ninguna fila candidata toca el día 4 (gap) ni los días adyacentes."""
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
    # Aserción positiva: día índice 2 (2020-01-03) tiene t-2,t-1,t,t+1 = días 0,1,2,3 válidos.
    assert pd.Timestamp("2020-01-03") in fechas
    # El último día (índice 7) no tiene t+1 → ausente.
    assert base[7] not in fechas


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


def test_build_feature_matrix_incluye_bird_id():
    kin = compute_causal_kinematics(_linear_bird(n=8, dlat=0.1))
    m = build_feature_matrix(kin, _cells_grid(), include_bird_id=True)
    assert m.attrs["_features"][0] == "bird_id"
    assert "bird_id" in m.columns


# ---------------------------------------------------------------------------
# Tests de attach_o3_state_and_split
# ---------------------------------------------------------------------------

def _features_o3_for(matrix, *, split="train", state=0, posterior=0.3):
    """Construye un features.parquet sintético de O3 que cubre las filas de matrix."""
    keys = matrix[["bird_id", "date_utc"]].drop_duplicates().reset_index(drop=True)
    keys["state_b_causal"] = state
    keys["posterior_b_migracion"] = posterior
    keys["split"] = split
    return keys


def test_attach_o3_state_and_split_renombra_y_pega() -> None:
    kin = compute_causal_kinematics(_linear_bird(n=10, dlat=0.1))
    cells = _cells_grid()
    matrix = build_feature_matrix(kin, cells, include_bird_id=False)
    features_o3 = _features_o3_for(matrix, split="train", state=1, posterior=0.7)
    out = attach_o3_state_and_split(matrix, features_o3)
    # El posterior se renombra al nombre que consume O4.
    assert "posterior_b_migracion_causal" in out.columns
    assert "posterior_b_migracion" not in out.columns
    assert (out["state_b_causal"] == 1).all()
    assert (out["posterior_b_migracion_causal"] == 0.7).all()
    assert (out["split"] == "train").all()
    assert len(out) == len(matrix)


def test_attach_o3_state_and_split_lanza_si_falta() -> None:
    kin = compute_causal_kinematics(_linear_bird(n=10, dlat=0.1))
    cells = _cells_grid()
    matrix = build_feature_matrix(kin, cells, include_bird_id=False)
    features_o3 = _features_o3_for(matrix)
    # Quita una fila candidata de O3 → su estado/split queda NaN tras el merge.
    features_o3 = features_o3.iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="sin estado HMM causal o sin split"):
        attach_o3_state_and_split(matrix, features_o3)


# ---------------------------------------------------------------------------
# Tests de split_temporal_per_bird
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests de compute_causal_kinematics
# ---------------------------------------------------------------------------

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
