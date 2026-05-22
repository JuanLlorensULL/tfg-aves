"""Tests unitarios de tfg_aves.ml.features."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
