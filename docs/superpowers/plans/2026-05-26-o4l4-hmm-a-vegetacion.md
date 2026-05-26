# L4 (HMM A + vegetación cruda) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir L4, una línea exploratoria de O4 idéntica a L3 salvo que el estado latente lo aporta el HMM A (cinemático puro) y la vegetación (`veg_low`, `veg_high`) entra como feature cruda del regresor en vez de dentro de la emisión del HMM.

**Architecture:** L4 solo lee de O3 (no reentrena nada): `features.parquet` ya persiste `state_a_causal`, `posterior_a_migracion`, `veg_low` y `veg_high`. Se parametrizan dos puntos compartidos de L3 (`attach_o3_state_and_split` por `suffix`, y `build_regression_predictions`/`compute_persistence_baseline` por `state_col`), ambos retrocompatibles, y se añade un orquestador `build_l4.py` que es copia fina de `build_l3.py` (mismo patrón que `build_l2`/`build_l3`).

**Tech Stack:** Python 3.12, pandas/numpy, xgboost/lightgbm/quantile-forest, pytest, uv.

**Convención de commits (override del usuario):** NO se commitea por tarea. Todo el trabajo se integra en un **único commit experimental** al final (Task 5), confirmado por el autor (rama vs main a decidir entonces). Esto prevalece sobre el "commit por tarea" de la skill.

---

### Task 1: Parametrizar `attach_o3_state_and_split` por `suffix`

Permite pegar el estado del HMM A (`state_a_causal` + `posterior_a_migracion`) además del B. Default `suffix="b"` reproduce el comportamiento actual de L3.

**Files:**
- Modify: `src/tfg_aves/ml/features.py:110-129`
- Test: `tests/test_ml_features.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir al final de `tests/test_ml_features.py`:

```python
def test_attach_o3_state_and_split_suffix_a():
    """Con suffix='a' pega el estado del HMM A renombrando el posterior."""
    import pandas as pd

    from tfg_aves.ml.features import attach_o3_state_and_split

    matrix = pd.DataFrame({
        "bird_id": ["A", "A"],
        "date_utc": pd.to_datetime(["2020-01-03", "2020-01-04"]),
        "lat": [40.0, 40.1], "lon": [-3.0, -2.9],
    })
    features_o3 = pd.DataFrame({
        "bird_id": ["A", "A"],
        "date_utc": pd.to_datetime(["2020-01-03", "2020-01-04"]),
        "state_a_causal": [0, 1],
        "posterior_a_migracion": [0.2, 0.8],
        "state_b_causal": [1, 0],
        "posterior_b_migracion": [0.9, 0.1],
        "split": ["train", "test"],
    })
    out = attach_o3_state_and_split(matrix, features_o3, suffix="a")
    assert "state_a_causal" in out.columns
    assert "posterior_a_migracion_causal" in out.columns
    assert "state_b_causal" not in out.columns
    assert out["state_a_causal"].tolist() == [0, 1]
    assert out["posterior_a_migracion_causal"].tolist() == [0.2, 0.8]
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `uv run pytest tests/test_ml_features.py::test_attach_o3_state_and_split_suffix_a -v`
Expected: FAIL (`attach_o3_state_and_split() got an unexpected keyword argument 'suffix'`).

- [ ] **Step 3: Implementar la parametrización**

Reemplazar la función completa en `src/tfg_aves/ml/features.py` (líneas 110-129) por:

```python
def attach_o3_state_and_split(
    matrix: pd.DataFrame, features_o3: pd.DataFrame, *, suffix: str = "b",
) -> pd.DataFrame:
    """Pega state_<suffix>_causal/posterior_<suffix>_migracion_causal y split desde O3 (merge m:1).

    O3 nombra el posterior ``posterior_<suffix>_migracion``; aquí se renombra al
    nombre que consume O4 (``posterior_<suffix>_migracion_causal``). ``suffix``
    selecciona el modelo de la ablación de O3 ("b" = L3, "a" = L4). Lanza si
    alguna fila candidata queda sin estado o sin etiqueta de split.
    """
    state_col = f"state_{suffix}_causal"
    post_src = f"posterior_{suffix}_migracion"
    post_dst = f"{post_src}_causal"
    cols = features_o3[[
        "bird_id", "date_utc", state_col, post_src, "split",
    ]].rename(columns={post_src: post_dst})
    merged = matrix.merge(cols, on=["bird_id", "date_utc"], how="left", validate="m:1")
    missing = merged[[state_col, post_dst, "split"]].isna().any().any()
    if missing:
        raise ValueError("Filas candidatas sin estado HMM causal o sin split tras el merge.")
    return merged
```

- [ ] **Step 4: Ejecutar el test nuevo y la regresión de L3/features**

Run: `uv run pytest tests/test_ml_features.py tests/test_ml_build_l3.py -q`
Expected: PASS (el nuevo test pasa; los de L3 siguen verdes porque el default `suffix="b"` no altera nada).

---

### Task 2: Parametrizar el estado de salida por `state_col` en `evaluate.py`

`build_regression_predictions` y `compute_persistence_baseline` cablean `state_b_causal`. Se añade `state_col="state_b_causal"` (retrocompatible) para que L4 pueda emitir/agrupar por `state_a_causal`.

**Files:**
- Modify: `src/tfg_aves/ml/evaluate.py:172-194` (`compute_persistence_baseline`)
- Modify: `src/tfg_aves/ml/quantile.py:279-350` (`build_regression_predictions`)
- Test: `tests/test_ml_quantile.py`

- [ ] **Step 1: Escribir el test que falla**

Añadir al final de `tests/test_ml_quantile.py`:

```python
def test_build_regression_predictions_state_col_a():
    """state_col='state_a_causal' emite esa columna en vez de state_b_causal."""
    import numpy as np
    import pandas as pd

    from tfg_aves.ml.quantile import build_regression_predictions

    cells = pd.DataFrame({
        "cell_id": ["80_-6", "80_-5"],
        "lat_c": [40.25, 40.25], "lon_c": [-2.75, -2.25],
    })
    qp = pd.DataFrame({
        "dlat_p10": [0.0], "dlat_p50": [0.05], "dlat_p90": [0.1],
        "dlon_p10": [0.0], "dlon_p50": [0.05], "dlon_p90": [0.1],
    })
    meta = pd.DataFrame({
        "bird_id": ["A"], "date_utc": pd.to_datetime(["2020-01-03"]),
        "lat": [40.2], "lon": [-2.8],
        "lat_t_next": [40.26], "lon_t_next": [-2.74],
        "cell_id_t_next": ["80_-6"],
        "state_a_causal": [1],
    })
    out = build_regression_predictions(qp, meta, cells, state_col="state_a_causal")
    assert "state_a_causal" in out.columns
    assert "state_b_causal" not in out.columns
    assert out["state_a_causal"].tolist() == [1]
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `uv run pytest tests/test_ml_quantile.py::test_build_regression_predictions_state_col_a -v`
Expected: FAIL (`build_regression_predictions() got an unexpected keyword argument 'state_col'`).

- [ ] **Step 3: Implementar `state_col` en las dos funciones**

En `src/tfg_aves/ml/evaluate.py`, en `compute_persistence_baseline` cambiar la firma y la columna de estado:

```python
def compute_persistence_baseline(
    matrix_test: pd.DataFrame, *, cells: pd.DataFrame,
    state_col: str = "state_b_causal",
) -> pd.DataFrame:
    """Persistencia trivial: mañana = hoy."""
    centroids = _cell_centroid_lookup(cells)

    return pd.DataFrame({
        "bird_id": matrix_test["bird_id"].values,
        "date_utc": matrix_test["date_utc"].values,
        "true_cell": matrix_test["cell_id_t_next"].values,
        "pred_cell_top1": matrix_test["cell_id_t"].values,
        "pred_cell_topk": [[c] for c in matrix_test["cell_id_t"].values],
        "pred_prob_top1": [1.0] * len(matrix_test),
        "pred_dist_km": [
            _haversine_to_centroid(c, lt, ln, centroids) for c, lt, ln in zip(
                matrix_test["cell_id_t"],
                matrix_test["lat_t_next"],
                matrix_test["lon_t_next"],
                strict=True,
            )
        ],
        state_col: matrix_test[state_col].values,
    })
```

En `build_regression_predictions` (en `src/tfg_aves/ml/quantile.py`, NO en evaluate.py — corregir ruta: la función vive en `quantile.py:279`), cambiar la firma y la fila del estado del dict de salida:

```python
def build_regression_predictions(
    quantile_preds: pd.DataFrame,
    meta: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    k_top: int = 3,
    state_col: str = "state_b_causal",
) -> pd.DataFrame:
```

y dentro del `pd.DataFrame({...})` de salida, sustituir la línea

```python
        "state_b_causal": meta["state_b_causal"].to_numpy(),
```

por

```python
        state_col: meta[state_col].to_numpy(),
```

- [ ] **Step 4: Ejecutar el test nuevo y la regresión**

Run: `uv run pytest tests/test_ml_quantile.py tests/test_ml_evaluate.py tests/test_ml_build_l3.py -q`
Expected: PASS (default `state_col="state_b_causal"` mantiene el comportamiento de L3).

> **Nota para el ejecutor:** `build_regression_predictions` está definida en `src/tfg_aves/ml/quantile.py` (línea ~279), aunque `compute_persistence_baseline` está en `src/tfg_aves/ml/evaluate.py` (línea ~172). Editar cada una en su fichero.

---

### Task 3: Fixture sintético con el HMM A + ruta `O4_L4_DIR`

El helper de tests `write_synthetic_o3_features` solo ajusta el Modelo B. L4 necesita `state_a_causal`/`posterior_a_migracion` en el `features.parquet` sintético. El cambio es aditivo (las columnas B no se tocan, los tests de L3 no se ven afectados).

**Files:**
- Modify: `tests/conftest.py:10-54` (`write_synthetic_o3_features`)
- Modify: `src/tfg_aves/ml/_paths.py:13`
- Test: `tests/test_ml_build_l3.py` (regresión) + verificación inline

- [ ] **Step 1: Añadir la ruta de salida de L4**

En `src/tfg_aves/ml/_paths.py`, tras la línea de `O4_L3V2_DIR` (línea 13) añadir:

```python
O4_L4_DIR: Path = ROOT / "data" / "processed" / "o4" / "l4"
```

- [ ] **Step 2: Extender el helper de conftest para emitir el Modelo A**

Reemplazar el cuerpo de `write_synthetic_o3_features` en `tests/conftest.py` (importes + ajuste) por la versión que ajusta y decodifica A y B:

```python
def write_synthetic_o3_features(raw: pd.DataFrame, out_path: Path, *, seed: int = 0) -> Path:
    """Escribe un features.parquet con el esquema de salida de O3 a partir de
    filas crudas (bird_id, date_utc, lat, lon, veg_low, veg_high, daylight_hours).

    Reproduce el ensamblaje causal de build_o3: cinemática entrante, split
    temporal sobre los días HMM-válidos, ajuste y decodificado filtrado de los
    Modelos A y B. Lo consumen los tests de integración de O4 (L3 lee el estado
    B; L4 lee el estado A), que no recalculan el HMM.
    """
    from tfg_aves.data.split import assign_temporal_split
    from tfg_aves.hmm.causal import (
        HMM_EMISSION_COLS_A,
        HMM_EMISSION_COLS_B,
        compute_causal_kinematics,
        decode_causal_states,
        fit_causal_hmm,
    )

    kin = compute_causal_kinematics(raw)
    valid = kin[kin["is_hmm_obs_valid"]].copy()
    valid = assign_temporal_split(valid)
    kin = kin.merge(
        valid[["bird_id", "date_utc", "split"]], on=["bird_id", "date_utc"], how="left",
    )
    train_valid = valid[valid["split"] == "train"]
    cutoff_by_bird = (
        train_valid.assign(_d=pd.to_datetime(train_valid["date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )
    model_a, labels_a = fit_causal_hmm(
        kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_A, n_restarts=4, seed=seed,
    )
    model_b, labels_b = fit_causal_hmm(
        kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_B, n_restarts=4, seed=seed,
    )
    states_a = decode_causal_states(
        model_a, labels_a, kin, emission_cols=HMM_EMISSION_COLS_A, suffix="a",
    )
    states_b = decode_causal_states(
        model_b, labels_b, kin, emission_cols=HMM_EMISSION_COLS_B, suffix="b",
    )
    df = (
        kin.merge(states_a, on=["bird_id", "date_utc"], how="left")
           .merge(states_b, on=["bird_id", "date_utc"], how="left")
    )
    cols = [
        "bird_id", "date_utc", "lat", "lon",
        "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
        "daylight_hours", "veg_low", "veg_high",
        "is_hmm_obs_valid", "split",
        "state_a_causal", "posterior_a_estacionario", "posterior_a_migracion",
        "state_b_causal", "posterior_b_estacionario", "posterior_b_migracion",
    ]
    out_cols = [c for c in cols if c in df.columns]
    df[out_cols].to_parquet(out_path, index=False)
    return out_path
```

- [ ] **Step 3: Verificar que el fixture emite columnas A y que L3 sigue verde**

Run: `uv run pytest tests/test_ml_build_l3.py tests/test_ml_build_l2.py -q`
Expected: PASS (cambio aditivo; L2/L3 leen solo sus columnas).

Verificación inline de que las columnas A existen ahora:

Run:
```bash
uv run python -c "
import pandas as pd, tempfile, os
from pathlib import Path
from tests.conftest import write_synthetic_o3_features
import numpy as np
rng=np.random.default_rng(0)
rows=[]
for b in ['A','B','C','D','E']:
    lat0=40+rng.uniform(-1,1); lon0=-3+rng.uniform(-1,1)
    for i,d in enumerate(pd.date_range('2020-01-01',periods=90)):
        rows.append({'bird_id':b,'date_utc':d,'lat':lat0+0.05*i+rng.normal(0,0.05),'lon':lon0+0.05*i+rng.normal(0,0.05),'daylight_hours':12.0,'veg_low':0.5,'veg_high':0.5})
p=Path(tempfile.mkdtemp())/'f.parquet'
write_synthetic_o3_features(pd.DataFrame(rows),p)
cols=set(pd.read_parquet(p).columns)
assert {'state_a_causal','posterior_a_migracion','state_b_causal','veg_low','veg_high'} <= cols, cols
print('OK columnas A presentes')
"
```
Expected: imprime `OK columnas A presentes`.

---

### Task 4: Orquestador `build_l4.py` + export + test de integración

Copia fina de `build_l3.py` (mismo patrón que `build_l2`/`build_l3`): mismas 3 familias + individual xgb, feature set = cinemáticas + estado A + posterior A + vegetación.

**Files:**
- Create: `src/tfg_aves/ml/build_l4.py`
- Modify: `src/tfg_aves/ml/__init__.py`
- Test: `tests/test_ml_build_l4.py`

- [ ] **Step 1: Escribir el test de integración que falla**

Crear `tests/test_ml_build_l4.py`:

```python
"""Tests de integración de build_o4_l4 (HMM A + vegetación cruda)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import write_synthetic_o3_features


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """5 aves × 90 días válidos (espejo de test_ml_build_l3.py)."""
    rng = np.random.default_rng(0)
    birds = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=90)
    rows = []
    for b in birds:
        lat0 = 40.0 + rng.uniform(-1, 1)
        lon0 = -3.0 + rng.uniform(-1, 1)
        for i, d in enumerate(dates):
            rows.append({
                "bird_id": b, "date_utc": d,
                "lat": lat0 + 0.05 * i + rng.normal(0, 0.05),
                "lon": lon0 + 0.05 * i + rng.normal(0, 0.05),
                "daylight_hours": 12.0, "veg_low": 0.5, "veg_high": 0.5,
            })
    feat_path = tmp_path / "features.parquet"
    write_synthetic_o3_features(pd.DataFrame(rows), feat_path)

    cells = []
    for i in range(76, 92):
        for j in range(-11, 7):
            cells.append({
                "cell_id": f"{i}_{j}", "cell_lat_idx": i, "cell_lon_idx": j,
                "lat_c": (i + 0.5) * 0.5, "lon_c": (j + 0.5) * 0.5, "n_obs_total": 30,
            })
    cells_path = tmp_path / "cells.parquet"
    pd.DataFrame(cells).to_parquet(cells_path)
    return feat_path, cells_path


def test_l4_feature_set():
    """L4 usa estado A + posterior A + vegetación; nunca B ni daylight."""
    from tfg_aves.ml.build_l4 import _FEATURES
    assert "state_a_causal" in _FEATURES
    assert "posterior_a_migracion_causal" in _FEATURES
    assert "veg_low" in _FEATURES and "veg_high" in _FEATURES
    assert "state_b_causal" not in _FEATURES
    assert "posterior_b_migracion_causal" not in _FEATURES
    assert "daylight_hours" not in _FEATURES


def test_prepare_poblacional_split_attaches_state_a(tmp_path):
    from tfg_aves.ml.build_l4 import _prepare_poblacional_split
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    features_o3 = pd.read_parquet(feat_path)
    cells = pd.read_parquet(cells_path)
    train, val, test = _prepare_poblacional_split(features_o3, cells)
    for df in (train, val, test):
        assert "state_a_causal" in df.columns
        assert "posterior_a_migracion_causal" in df.columns
        assert "veg_low" in df.columns and "veg_high" in df.columns
        assert "state_b_causal" not in df.columns
    assert len(train) > 0 and len(test) > 0


def test_build_o4_l4_artifacts(tmp_path):
    from tfg_aves.ml.build_l4 import build_o4_l4
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l4"
    result = build_o4_l4(
        features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
        seed=0, individual_bird_id="A",
    )
    for family in ("xgb", "lgbm", "rf"):
        for axis in ("dlat", "dlon"):
            assert (out_dir / f"model_{family}_poblacional_{axis}.pkl").exists()
    for axis in ("dlat", "dlon"):
        assert (out_dir / f"model_xgb_individual_{axis}.pkl").exists()
    assert (out_dir / "predictions_test.parquet").exists()
    assert (out_dir / "metrics.parquet").exists()

    metrics = pd.read_parquet(out_dir / "metrics.parquet")
    assert {"familia", "modo", "scope", "top1", "dist_centroide_km",
            "coverage_lat"}.issubset(metrics.columns)
    assert {"xgb", "lgbm", "rf"}.issubset(set(metrics["familia"]))
    assert "persistencia" in set(metrics["modo"])
    assert result.n_rows_test_pob > 0


def test_build_o4_l4_no_leakage(tmp_path):
    """feature_cols == las 12 features de L4, sin columnas de t+1 ni de B/daylight."""
    import joblib

    from tfg_aves.ml.build_l4 import _FEATURES, build_o4_l4
    feat_path, cells_path = _write_synthetic_inputs(tmp_path)
    out_dir = tmp_path / "l4"
    build_o4_l4(features_path=feat_path, cells_path=cells_path, out_dir=out_dir,
                seed=0, individual_bird_id="A")
    forbidden = {"lat_t_next", "lon_t_next", "cell_id_t_next", "y_dlat", "y_dlon",
                 "state_b_causal", "posterior_b_migracion_causal", "daylight_hours"}
    for family in ("xgb", "lgbm", "rf"):
        payload = joblib.load(out_dir / f"model_{family}_poblacional_dlat.pkl")
        assert payload["feature_cols"] == _FEATURES
        assert forbidden.isdisjoint(set(payload["feature_cols"]))


def test_build_o4_l4_exported():
    import tfg_aves.ml as ml
    assert hasattr(ml, "build_o4_l4")
    assert hasattr(ml, "BuildO4L4Result")
```

- [ ] **Step 2: Ejecutar el test para ver que falla**

Run: `uv run pytest tests/test_ml_build_l4.py -q`
Expected: FAIL (`No module named 'tfg_aves.ml.build_l4'`).

- [ ] **Step 3: Crear `src/tfg_aves/ml/build_l4.py`**

```python
"""Orquestador del pipeline L4 (exploratorio): L3 con estado del HMM A +
vegetación cruda como feature del regresor (sin daylight, redundante con
lat + día del año). Línea de curiosidad, fuera del trabajo fundamental."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ._paths import CELLS_PARQUET, FEATURES_O3_PARQUET, O4_L4_DIR
from .evaluate import (
    compute_persistence_baseline,
    evaluate_by_state,
    evaluate_moves_only,
)
from .features import (
    FEATURES_KINEMATIC,
    attach_o3_state_and_split,
    build_feature_matrix,
)
from .quantile import (
    INDIVIDUAL_BIRD_ID,
    QUANTILES,
    build_regression_predictions,
    derive_displacement_target,
    fit_quantile_axis,
    interval_coverage,
    pinball_loss,
    predict_quantiles,
)

# Estado del HMM A (cinemático) + vegetación cruda. Sin daylight (redundante
# con lat + sin/cos_doy, ya presentes en FEATURES_KINEMATIC).
_FEATURES_HMM_A = ["state_a_causal", "posterior_a_migracion_causal"]
_FEATURES_VEG = ["veg_low", "veg_high"]
_FEATURES = [*FEATURES_KINEMATIC, *_FEATURES_HMM_A, *_FEATURES_VEG]
_STATE_COL = "state_a_causal"
_FAMILIES_DEFAULT = ("xgb", "lgbm", "rf")


def _modes_for(family: str) -> tuple[str, ...]:
    """XGBoost conserva poblacional + individual; RF/LGBM solo poblacional."""
    return ("poblacional", "individual") if family == "xgb" else ("poblacional",)


@dataclass
class BuildO4L4Result:
    """Resumen serializable de build_o4_l4."""

    n_rows_train_pob: int
    n_rows_val_pob: int
    n_rows_test_pob: int
    n_rows_train_ind: int
    n_rows_val_ind: int
    n_rows_test_ind: int
    individual_bird_id: str
    n_crossings: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    model_paths: dict[str, Path] = field(default_factory=dict)
    predictions_path: Path = Path()
    metrics_path: Path = Path()


def _prepare_poblacional_split(
    features_o3: pd.DataFrame, cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) poblacional con estado del HMM A + vegetación cruda.

    El estado A (``state_a_causal``, ``posterior_a_migracion_causal``) y el split
    se leen de O3 vía attach(suffix="a"); ``veg_low``/``veg_high`` se pegan con
    un merge m:1 adicional desde O3.
    """
    matrix = build_feature_matrix(features_o3, cells, include_bird_id=False)
    matrix = attach_o3_state_and_split(matrix, features_o3, suffix="a")
    veg = features_o3[["bird_id", "date_utc", "veg_low", "veg_high"]]
    matrix = matrix.merge(veg, on=["bird_id", "date_utc"], how="left", validate="m:1")
    if matrix[_FEATURES_VEG].isna().any().any():
        raise ValueError("Filas candidatas sin vegetación tras el merge.")
    return tuple(
        matrix[matrix["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
    )


def _y_move(df: pd.DataFrame) -> np.ndarray:
    return (df["cell_id_t_next"].astype(str) != df["cell_id_t"].astype(str)).to_numpy()


def _metric_rows(
    preds: pd.DataFrame, y_move: np.ndarray, modo_label: str, familia: str,
    *, pinball_lat: float, pinball_lon: float, cov_lat: float, cov_lon: float,
) -> list[dict]:
    """Filas de métrica por estado (global/estacionario/migración) + moves."""
    rows: list[dict] = []
    by = evaluate_by_state(preds, state_col=_STATE_COL)
    for _, r in by.iterrows():
        scope = r["state"]
        if scope == "estacionario":
            sub = preds[preds[_STATE_COL] == 0]
        elif scope == "migración":
            sub = preds[preds[_STATE_COL] == 1]
        else:
            sub = preds
        native = float(np.median(sub["dist_native_km"])) if len(sub) else np.nan
        is_global = scope == "global"
        rows.append({
            "familia": familia, "modo": modo_label, "scope": scope,
            "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": native,
            "pinball_lat": pinball_lat if is_global else np.nan,
            "pinball_lon": pinball_lon if is_global else np.nan,
            "coverage_lat": cov_lat if is_global else np.nan,
            "coverage_lon": cov_lon if is_global else np.nan,
        })
    mask = np.asarray(y_move).astype(bool)
    mo = evaluate_moves_only(preds, mask)
    native_moves = (
        float(np.median(preds[mask]["dist_native_km"])) if mask.any() else np.nan
    )
    rows.append({
        "familia": familia, "modo": modo_label, "scope": "moves",
        "n_obs": mo["n_obs"],
        "top1": mo["top1"], "top3": mo["top3"],
        "dist_centroide_km": mo["dist_median_km"], "dist_nativa_km": native_moves,
        "pinball_lat": np.nan, "pinball_lon": np.nan,
        "coverage_lat": np.nan, "coverage_lon": np.nan,
    })
    return rows


def _baseline_rows(preds: pd.DataFrame, modo_label: str, familia: str) -> list[dict]:
    by = evaluate_by_state(preds, state_col=_STATE_COL)
    rows = []
    for _, r in by.iterrows():
        rows.append({
            "familia": familia, "modo": modo_label, "scope": r["state"],
            "n_obs": int(r["n_obs"]),
            "top1": r["top1"], "top3": r["top3"],
            "dist_centroide_km": r["dist_median_km"], "dist_nativa_km": np.nan,
            "pinball_lat": np.nan, "pinball_lon": np.nan,
            "coverage_lat": np.nan, "coverage_lon": np.nan,
        })
    return rows


def build_o4_l4(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    out_dir: Path = O4_L4_DIR,
    seed: int = 0,
    *,
    families: tuple[str, ...] = _FAMILIES_DEFAULT,
    individual_bird_id: str | None = None,
) -> BuildO4L4Result:
    """Pipeline L4 (exploratorio): regresor de cuantiles multi-familia con
    estado del HMM A + vegetación cruda. Espejo de build_o4_l3."""
    if individual_bird_id is None:
        individual_bird_id = INDIVIDUAL_BIRD_ID

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    train_pob, val_pob, test_pob = _prepare_poblacional_split(features_o3, cells)

    model_paths: dict[str, Path] = {}
    n_crossings: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict[str, float]] = {}
    metric_rows: list[dict] = []
    preds_frames: list[pd.DataFrame] = []

    splits_by_mode = {
        "poblacional": (train_pob, val_pob, test_pob),
        "individual": tuple(
            d[d["bird_id"] == individual_bird_id].reset_index(drop=True)
            for d in (train_pob, val_pob, test_pob)
        ),
    }

    xgb_pob_full: pd.DataFrame | None = None

    for family in families:
        for mode in _modes_for(family):
            train_m, val_m, test_m = splits_by_mode[mode]
            tgt_tr = derive_displacement_target(train_m)
            tgt_va = derive_displacement_target(val_m)

            axis_lat = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlat"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlat"].to_numpy(),
                family=family, seed=seed,
            )
            axis_lon = fit_quantile_axis(
                train_m[_FEATURES], tgt_tr["y_dlon"].to_numpy(),
                val_m[_FEATURES], tgt_va["y_dlon"].to_numpy(),
                family=family, seed=seed,
            )

            qp_test, crossings = predict_quantiles(axis_lat, axis_lon, test_m[_FEATURES])
            key = f"{family}_{mode}"
            n_crossings[key] = crossings
            preds = build_regression_predictions(
                qp_test, test_m, cells, state_col=_STATE_COL,
            )

            tgt_te = derive_displacement_target(test_m)
            pin_lat = float(np.mean([
                pinball_loss(tgt_te["y_dlat"].to_numpy(),
                             qp_test[f"dlat_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            pin_lon = float(np.mean([
                pinball_loss(tgt_te["y_dlon"].to_numpy(),
                             qp_test[f"dlon_p{int(q*100):02d}"].to_numpy(), q)
                for q in QUANTILES
            ]))
            cov_lat = interval_coverage(
                tgt_te["y_dlat"].to_numpy(),
                qp_test["dlat_p10"].to_numpy(), qp_test["dlat_p90"].to_numpy(),
            )
            cov_lon = interval_coverage(
                tgt_te["y_dlon"].to_numpy(),
                qp_test["dlon_p10"].to_numpy(), qp_test["dlon_p90"].to_numpy(),
            )
            coverage[key] = {"lat": cov_lat, "lon": cov_lon}

            metric_rows.extend(_metric_rows(
                preds, _y_move(test_m), mode, family,
                pinball_lat=pin_lat, pinball_lon=pin_lon,
                cov_lat=cov_lat, cov_lon=cov_lon,
            ))

            for axis_name, axis_model in (("dlat", axis_lat), ("dlon", axis_lon)):
                mp = out_dir / f"model_{family}_{mode}_{axis_name}.pkl"
                joblib.dump({
                    "model": axis_model, "feature_cols": _FEATURES,
                    "familia": family, "mode": mode, "axis": axis_name,
                    "individual_bird_id": (
                        individual_bird_id if mode == "individual" else None
                    ),
                }, mp)
                model_paths[f"{family}_{mode}_{axis_name}"] = mp

            preds_out = preds.copy()
            preds_out["familia"] = family
            preds_out["modo"] = mode
            preds_out["pred_cell_topk"] = preds_out["pred_cell_topk"].apply(list)
            preds_frames.append(preds_out)
            if family == "xgb" and mode == "poblacional":
                xgb_pob_full = preds.copy()

    if xgb_pob_full is not None:
        pob_at_ind = xgb_pob_full[
            xgb_pob_full["bird_id"] == individual_bird_id
        ].reset_index(drop=True)
        if len(pob_at_ind) > 0:
            test_ind = splits_by_mode["individual"][2]
            metric_rows.extend(_metric_rows(
                pob_at_ind, _y_move(test_ind), f"poblacional@{individual_bird_id}", "xgb",
                pinball_lat=np.nan, pinball_lon=np.nan, cov_lat=np.nan, cov_lon=np.nan,
            ))

    persistence = compute_persistence_baseline(
        test_pob, cells=cells, state_col=_STATE_COL,
    )
    metric_rows.extend(_baseline_rows(persistence, "persistencia", "—"))
    pers_ind = persistence[persistence["bird_id"] == individual_bird_id]
    if len(pers_ind) > 0:
        metric_rows.extend(_baseline_rows(
            pers_ind.reset_index(drop=True), f"persistencia@{individual_bird_id}", "—"))

    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.parquet"
    metrics.to_parquet(metrics_path)

    predictions_df = pd.concat(preds_frames, ignore_index=True)
    predictions_path = out_dir / "predictions_test.parquet"
    predictions_df.to_parquet(predictions_path)

    ind_train = splits_by_mode["individual"][0]
    ind_test = splits_by_mode["individual"][2]
    return BuildO4L4Result(
        n_rows_train_pob=len(train_pob),
        n_rows_val_pob=len(val_pob),
        n_rows_test_pob=len(test_pob),
        n_rows_train_ind=len(ind_train),
        n_rows_val_ind=len(splits_by_mode["individual"][1]),
        n_rows_test_ind=len(ind_test),
        individual_bird_id=individual_bird_id,
        n_crossings=n_crossings,
        coverage=coverage,
        model_paths=model_paths,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
    )
```

- [ ] **Step 4: Exportar L4 en `__init__.py`**

En `src/tfg_aves/ml/__init__.py`, añadir tras la línea de import de `build_l3` (línea 14):

```python
from tfg_aves.ml.build_l4 import BuildO4L4Result, build_o4_l4
```

y añadir al `__all__` (orden alfabético) las entradas `"BuildO4L4Result",` (tras `"BuildO4L3Result",`) y `"build_o4_l4",` (tras `"build_o4_l3",`).

- [ ] **Step 5: Ejecutar los tests de L4 y la regresión completa de ml**

Run: `uv run pytest tests/test_ml_build_l4.py -q`
Expected: PASS (los 5 tests).

---

### Task 5: Verificación, ejecución sobre datos reales y reporte

No es TDD: corre la suite completa + linter, ejecuta L4 sobre los datos reales y compara contra L3. El commit único experimental se decide aquí con el autor.

**Files:** ninguno nuevo (ejecución + verificación).

- [ ] **Step 1: Suite completa + linter**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: toda la suite verde, ruff sin errores.

- [ ] **Step 2: Ejecutar L4 sobre los datos reales de O3**

Run:
```bash
uv run python -c "
from tfg_aves.ml.build_l4 import build_o4_l4
r = build_o4_l4()
print('test_pob', r.n_rows_test_pob, 'test_ind', r.n_rows_test_ind)
print('coverage', r.coverage)
"
```
Expected: corre sin error y escribe `data/processed/o4/l4/{metrics,predictions_test}.parquet` + modelos.

- [ ] **Step 3: Comparar L4 vs L3 (poblacional, las 3 familias)**

Run:
```bash
uv run python -c "
import pandas as pd
l3 = pd.read_parquet('data/processed/o4/l3_v2/metrics.parquet')
l4 = pd.read_parquet('data/processed/o4/l4/metrics.parquet')
keep = ['familia','modo','scope','top1','top3','dist_centroide_km','dist_nativa_km','coverage_lat','coverage_lon']
q = \"modo=='poblacional' and scope in ['global','estacionario','migración','moves']\"
print('--- L3 ---'); print(l3.query(q)[keep].to_string(index=False))
print('--- L4 ---'); print(l4.query(q)[keep].to_string(index=False))
"
```
Expected: dos tablas comparables. Reportar al autor el contraste (¿el estado cinemático A + vegetación cruda mejora, empata o empeora frente a B?).

- [ ] **Step 4: Reportar y decidir el commit experimental**

Presentar al autor el contraste L4 vs L3 y proponer el commit único experimental (mensaje en castellano, estilo imperativo, p. ej. `L4 (exploratorio): HMM A + vegetación cruda en el regresor de cuantiles`). Confirmar con el autor rama vs main antes de commitear. Sin tag, sin entrada de ai-log, sin notas de memoria (curiosidad, fuera del trabajo fundamental).

---

## Notas de implementación

- **`data/processed/o4/l4/` es artefacto, no se versiona** salvo que el autor lo decida; el commit experimental versiona código + tests + spec + plan.
- **Sanity-check biológico:** si en datos reales L4 marcara migración en pleno invierno o estacionario en plena migración de forma masiva, es bug (el estado A es puramente cinemático y debería seguir separando step corto vs largo).
