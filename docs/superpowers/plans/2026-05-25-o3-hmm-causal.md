# Plan de implementación — Conversión de O3 a HMM causal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que O3 produzca el HMM causal (cinemática entrante + filtrado forward-only + split temporal propio), conservando la ablación Modelo A vs B, y que L1/L2/L3 (y por tanto O5) lo consuman en lugar de recalcularlo cada una.

**Architecture:** La maquinaria causal sube de `tfg_aves.ml` a `tfg_aves.hmm`; el split temporal se mueve a `tfg_aves.data`. `build_o3` escribe en `features.parquet` la cinemática entrante, los estados causales A/B y una columna `split`. Las tres líneas de O4 leen esas columnas y particionan por `split`, eliminando la cuádruple duplicación del HMM. Se prioriza arquitectura limpia sobre neutralidad exacta: los números de L1/L2/L3/O5 pueden moverse y se re-verifican.

**Tech Stack:** Python 3.12, pandas, numpy, hmmlearn 0.3.3, scikit-learn, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-05-25-o3-hmm-causal-design.md`

---

## Mapa de ficheros

**Crear:**
- `src/tfg_aves/data/split.py` — `split_temporal_per_bird` (movido) + `assign_temporal_split` (nuevo, añade columna `split`).
- `src/tfg_aves/hmm/causal.py` — cinemática entrante + HMM causal generalizado a A/B (movido desde `ml/`).
- `tests/test_data_split.py`, `tests/test_hmm_causal.py` (movido desde `test_ml_hmm_causal.py`).
- `data/processed/_snapshot_pre_o3causal/` — baselines para el gate de re-verificación.

**Modificar:**
- `src/tfg_aves/hmm/build.py` — reescritura causal de `build_o3`.
- `src/tfg_aves/hmm/__init__.py`, `src/tfg_aves/data/__init__.py`, `src/tfg_aves/ml/__init__.py` — exports.
- `src/tfg_aves/ml/build.py`, `build_l2.py`, `build_l3.py` — consumir O3 en vez de recalcular.
- `src/tfg_aves/ml/features.py` — quitar `compute_causal_kinematics`, `HMM_EMISSION_COLS`, `split_temporal_per_bird`.
- Tests: `test_hmm_build.py`, `test_hmm_fit.py`, `test_hmm_evaluate.py`, `test_hmm_features.py`, `test_hmm_smoke.py`, `test_ml_build.py`, `test_ml_build_l2.py`, `test_ml_build_l3.py`, `test_ml_features.py`, `test_ml_smoke.py`.
- `notebooks/03_eda_o3.py` — regenerar artefactos C1–C9/D1.
- `reports/memoria/05_o3_hmm.md` — notas causales.

**Eliminar:**
- `src/tfg_aves/ml/hmm_causal.py` (y `tests/test_ml_hmm_causal.py`, movido).

---

## Fase 0 — Snapshot de baselines (gate de re-verificación)

### Task 0: Congelar entregables actuales

**Files:**
- Create: `data/processed/_snapshot_pre_o3causal/`

- [ ] **Step 1: Regenerar el estado actual desde cero para tener un baseline limpio**

Run:
```bash
uv run python -c "from tfg_aves.hmm import build_o3; build_o3()"
uv run python -c "from tfg_aves.ml import build_o4; build_o4()"
uv run python -c "from tfg_aves.ml.build_l2 import build_o4_l2; build_o4_l2()"
uv run python -c "from tfg_aves.ml.build_l3 import build_o4_l3; build_o4_l3()"
```
Expected: cuatro ejecuciones sin error; parquets en `data/processed/{o3,o4,o4/l2_v1,o4/l3_v2}`.

- [ ] **Step 2: Copiar las métricas y predicciones a un snapshot**

Run:
```bash
mkdir -p data/processed/_snapshot_pre_o3causal
cp data/processed/o3/metrics.parquet data/processed/_snapshot_pre_o3causal/o3_metrics.parquet
cp data/processed/o3/features.parquet data/processed/_snapshot_pre_o3causal/o3_features.parquet
cp data/processed/o4/metrics.parquet data/processed/_snapshot_pre_o3causal/o4_metrics.parquet
cp data/processed/o4/l2_v1/metrics.parquet data/processed/_snapshot_pre_o3causal/l2_metrics.parquet
cp data/processed/o4/l3_v2/metrics.parquet data/processed/_snapshot_pre_o3causal/l3_metrics.parquet
```
Expected: 5 ficheros copiados. (El snapshot está bajo `data/processed/`, gitignored: no se versiona, es de trabajo.)

- [ ] **Step 3: Commit (solo doc, sin datos)**

No hay nada que commitear en git (todo es gitignored). Anotar en el cuaderno de trabajo que el snapshot existe en disco. Continuar.

---

## Fase 1 — Mover el split a `tfg_aves.data`

### Task 1: Crear `data/split.py` con el split y el etiquetado

**Files:**
- Create: `src/tfg_aves/data/split.py`
- Test: `tests/test_data_split.py`

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_data_split.py
"""Tests del split temporal por ave y el etiquetado de split."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data.split import assign_temporal_split, split_temporal_per_bird


def _frame(bird: str, n: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"bird_id": bird, "date_utc": dates, "v": range(n)})


def test_split_fracciones_por_ave() -> None:
    df = pd.concat([_frame("A", 100), _frame("B", 50)], ignore_index=True)
    train, val, test = split_temporal_per_bird(df)
    # Por ave: 72% train, 8% val, 20% test (redondeo).
    assert len(train[train["bird_id"] == "A"]) == 72
    assert len(val[val["bird_id"] == "A"]) == 8
    assert len(test[test["bird_id"] == "A"]) == 20
    # El test es siempre posterior al train.
    assert test[test["bird_id"] == "A"]["date_utc"].min() > train[train["bird_id"] == "A"]["date_utc"].max()


def test_assign_temporal_split_etiqueta_coherente() -> None:
    df = pd.concat([_frame("A", 100), _frame("B", 50)], ignore_index=True)
    labelled = assign_temporal_split(df)
    assert set(labelled["split"].unique()) <= {"train", "val", "test"}
    # La partición por etiqueta coincide con split_temporal_per_bird.
    train, val, test = split_temporal_per_bird(df)
    assert (labelled["split"] == "train").sum() == len(train)
    assert (labelled["split"] == "test").sum() == len(test)
    # Orden temporal por ave: train < val < test en fechas.
    a = labelled[labelled["bird_id"] == "A"]
    assert a[a["split"] == "train"]["date_utc"].max() < a[a["split"] == "val"]["date_utc"].min()
    assert a[a["split"] == "val"]["date_utc"].max() < a[a["split"] == "test"]["date_utc"].min()
```

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `uv run pytest tests/test_data_split.py -q`
Expected: FAIL (`ModuleNotFoundError: tfg_aves.data.split`).

- [ ] **Step 3: Implementar `data/split.py`**

```python
# src/tfg_aves/data/split.py
"""Split temporal por ave, compartido por O3 (HMM) y O4 (clasificadores).

El split se calcula por ``bird_id`` ordenando por fecha: primeros
``train_frac`` → bloque (train+val); últimos → test; dentro del bloque, el
último ``val_frac_of_train`` → val. Vive en ``tfg_aves.data`` (upstream de
O3 y O4) para que O3 lo posea sin depender de la matriz del clasificador.
"""
from __future__ import annotations

import pandas as pd


def split_temporal_per_bird(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac_of_train: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, val, test) particionando por tiempo dentro de cada ave.

    Defaults: 72% / 8% / 20% por ave (igual que el split de O4 previo).
    """
    train_parts, val_parts, test_parts = [], [], []
    for _bird, sub in df.sort_values(["bird_id", "date_utc"]).groupby(
        "bird_id", sort=False,
    ):
        n = len(sub)
        n_train_val = int(round(n * train_frac))
        train_val = sub.iloc[:n_train_val]
        test = sub.iloc[n_train_val:]
        n_val = int(round(len(train_val) * val_frac_of_train))
        train = train_val.iloc[: len(train_val) - n_val]
        val = train_val.iloc[len(train_val) - n_val :]
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    for out in (train_df, val_df, test_df):
        out.attrs["_features"] = df.attrs.get("_features", [])
    return train_df, val_df, test_df


def assign_temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac_of_train: float = 0.1,
) -> pd.DataFrame:
    """Devuelve una copia de ``df`` con una columna ``split`` ∈ {train,val,test}.

    Asigna la etiqueta a cada fila según el mismo criterio que
    ``split_temporal_per_bird``. Útil para persistir el split en parquet y que
    los consumidores aguas abajo particionen sin recalcularlo.
    """
    out = df.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    out["split"] = pd.Series(["train"] * len(out), dtype="object")
    for _bird, sub in out.groupby("bird_id", sort=False):
        n = len(sub)
        n_train_val = int(round(n * train_frac))
        n_val = int(round(n_train_val * val_frac_of_train))
        idx = sub.index
        out.loc[idx[: n_train_val - n_val], "split"] = "train"
        out.loc[idx[n_train_val - n_val : n_train_val], "split"] = "val"
        out.loc[idx[n_train_val:], "split"] = "test"
    return out
```

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `uv run pytest tests/test_data_split.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Exportar desde `data/__init__.py`**

En `src/tfg_aves/data/__init__.py`, añadir el import y los nombres a `__all__`:
```python
from .split import assign_temporal_split, split_temporal_per_bird
```
Y en `__all__` añadir `"assign_temporal_split"`, `"split_temporal_per_bird"` (orden alfabético).

- [ ] **Step 6: Shim temporal en `ml/features.py` para no romper imports**

En `src/tfg_aves/ml/features.py`, sustituir la definición de `split_temporal_per_bird` (líneas 176-210) por una reexportación:
```python
from tfg_aves.data.split import split_temporal_per_bird  # reexport (movido a data)
```
Colocar este import junto a los demás imports al principio del fichero y borrar el cuerpo de la función antigua. Los consumidores de `ml/features.split_temporal_per_bird` siguen funcionando.

- [ ] **Step 7: Ejecutar la suite de ml para confirmar que sigue verde**

Run: `uv run pytest tests/test_ml_features.py tests/test_ml_build.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/tfg_aves/data/split.py src/tfg_aves/data/__init__.py src/tfg_aves/ml/features.py tests/test_data_split.py
git commit -m "datos: mueve el split temporal por ave a tfg_aves.data y añade assign_temporal_split"
```

---

## Fase 2 — Mover y generalizar el HMM causal a `tfg_aves.hmm`

### Task 2: Crear `hmm/causal.py` (cinemática entrante + HMM causal A/B)

**Files:**
- Create: `src/tfg_aves/hmm/causal.py`
- Test: `tests/test_hmm_causal.py` (movido desde `tests/test_ml_hmm_causal.py`)

- [ ] **Step 1: Mover el test y adaptarlo a la nueva ubicación y firma generalizada**

```bash
git mv tests/test_ml_hmm_causal.py tests/test_hmm_causal.py
```
Editar en `tests/test_hmm_causal.py` los imports (líneas 9-15) a:
```python
from tfg_aves.hmm.causal import (
    HMM_EMISSION_COLS_B,
    build_hmm_sequences,
    compute_causal_kinematics,
    decode_causal_states,
    fit_causal_hmm,
    forward_filtered_posteriors,
)
```
Y actualizar las referencias a `HMM_EMISSION_COLS` → `HMM_EMISSION_COLS_B`. En `test_decode_causal_states_columnas_y_rango`, generalizar la firma de decode: `decode_causal_states(model, label_map, kin, emission_cols=HMM_EMISSION_COLS_B, suffix="b")` y comprobar columnas `{"bird_id","date_utc","state_b_causal","posterior_b_estacionario","posterior_b_migracion"}`. Añadir un test nuevo para el suffix "a":
```python
def test_decode_acepta_suffix_a():
    kin = compute_causal_kinematics(_two_regime_bird(n=60))
    emission_a = ["step_in_km", "cos_turning_in"]
    model, label_map = fit_causal_hmm(kin, cutoff_by_bird=None, emission_cols=emission_a, n_restarts=4, seed=0)
    states = decode_causal_states(model, label_map, kin, emission_cols=emission_a, suffix="a")
    assert "state_a_causal" in states.columns
    assert "posterior_a_migracion" in states.columns
```
En `_two_regime_bird`, las filas ya traen `veg_low/veg_high/daylight_hours`, válido para el Modelo B; el Modelo A ignora esas columnas.

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `uv run pytest tests/test_hmm_causal.py -q`
Expected: FAIL (`ModuleNotFoundError: tfg_aves.hmm.causal`).

- [ ] **Step 3: Implementar `hmm/causal.py`**

Copiar `compute_causal_kinematics` desde `ml/features.py:37-96` tal cual (ya importa `bearing_rad` de `hmm.features` y `haversine_km` de `markov.discretize`; al estar ahora en `hmm/`, importar `from tfg_aves.hmm.features import bearing_rad` y `from tfg_aves.markov.discretize import haversine_km`). Definir las columnas de emisión y generalizar el fit/decode:

```python
# src/tfg_aves/hmm/causal.py
"""HMM causal: cinemática entrante + decodificado por filtrado forward-only.

Propiedad de O3. La emisión entrante (t-1 → t) y el filtrado
``P(estado_t | obs_1..t)`` garantizan que el estado no observa el futuro,
condición para que alimente a O4 como feature predictiva sin fuga.
Generalizado a los dos modelos de la ablación de O3:
  - Modelo A (cinemático): emisión ``HMM_EMISSION_COLS_A``.
  - Modelo B (cinemático + contexto): emisión ``HMM_EMISSION_COLS_B`` (= L3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from tfg_aves.hmm.features import bearing_rad
from tfg_aves.hmm.fit import fit_hmm_with_restarts
from tfg_aves.markov.discretize import haversine_km

HMM_EMISSION_COLS_A = ["step_in_km", "cos_turning_in"]
HMM_EMISSION_COLS_B = ["step_in_km", "cos_turning_in", "veg_low", "veg_high", "daylight_hours"]


def compute_causal_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    # CUERPO IDÉNTICO a ml/features.compute_causal_kinematics (líneas 37-96),
    # con los imports de bearing_rad y haversine_km ya resueltos arriba.
    ...


def _diag_log_emission(X, means, covars):
    # IDÉNTICO a ml/hmm_causal._diag_log_emission (líneas 19-38).
    ...


def forward_filtered_posteriors(model, X, lengths):
    # IDÉNTICO a ml/hmm_causal.forward_filtered_posteriors (líneas 41-78).
    ...


def build_hmm_sequences(kin, cutoff_by_bird=None, emission_cols=HMM_EMISSION_COLS_B):
    # IDÉNTICO a ml/hmm_causal.build_hmm_sequences (líneas 81-115) PERO usando
    # ``emission_cols`` en lugar de la constante global HMM_EMISSION_COLS.
    ...


def _relabel_by_step(model, emission_cols):
    step_idx = emission_cols.index("step_in_km")
    estac = int(np.argmin(model.means_[:, step_idx]))
    return {i: ("estacionario" if i == estac else "migración") for i in range(model.n_components)}


def fit_causal_hmm(kin, cutoff_by_bird, *, emission_cols=HMM_EMISSION_COLS_B, n_restarts=10, seed=0):
    X, lengths, _ = build_hmm_sequences(kin, cutoff_by_bird=cutoff_by_bird, emission_cols=emission_cols)
    model, _, _ = fit_hmm_with_restarts(X, lengths, n_components=2, n_restarts=n_restarts, random_state=seed)
    return model, _relabel_by_step(model, emission_cols)


def decode_causal_states(model, label_map, kin, *, emission_cols=HMM_EMISSION_COLS_B, suffix="b"):
    """Decodifica todos los días HMM-válidos por filtrado forward-only.

    Devuelve (bird_id, date_utc, state_<suffix>_causal,
    posterior_<suffix>_estacionario, posterior_<suffix>_migracion).
    """
    X, lengths, row_index = build_hmm_sequences(kin, cutoff_by_bird=None, emission_cols=emission_cols)
    migr_idx = next(i for i, lab in label_map.items() if lab == "migración")
    estac_idx = next(i for i, lab in label_map.items() if lab == "estacionario")
    post = forward_filtered_posteriors(model, X, lengths)
    raw_state = np.argmax(post, axis=1)
    res = kin.loc[row_index, ["bird_id", "date_utc"]].copy()
    res[f"state_{suffix}_causal"] = np.where(raw_state == migr_idx, 1, 0).astype(np.int8)
    res[f"posterior_{suffix}_estacionario"] = post[:, estac_idx]
    res[f"posterior_{suffix}_migracion"] = post[:, migr_idx]
    return res.reset_index(drop=True)
```
(Rellenar los cuerpos `...` copiando literalmente desde los ficheros origen indicados.)

- [ ] **Step 4: Ejecutar y ver pasar**

Run: `uv run pytest tests/test_hmm_causal.py -q`
Expected: PASS (todos, incluido el nuevo `test_decode_acepta_suffix_a`).

- [ ] **Step 5: Shims temporales para no romper O4 todavía**

En `src/tfg_aves/ml/hmm_causal.py`, sustituir todo el contenido por reexportaciones (mantiene la firma vieja Modelo B por defecto):
```python
"""Reexport temporal: el HMM causal se ha movido a tfg_aves.hmm.causal."""
from tfg_aves.hmm.causal import (  # noqa: F401
    build_hmm_sequences,
    decode_causal_states,
    fit_causal_hmm,
    forward_filtered_posteriors,
)
```
En `src/tfg_aves/ml/features.py`, sustituir `compute_causal_kinematics` (líneas 37-96) y `HMM_EMISSION_COLS` (líneas 20-22) por:
```python
from tfg_aves.hmm.causal import HMM_EMISSION_COLS_B as HMM_EMISSION_COLS  # reexport
from tfg_aves.hmm.causal import compute_causal_kinematics  # reexport  # noqa: F401
```

- [ ] **Step 6: Exportar desde `hmm/__init__.py`**

Añadir a `src/tfg_aves/hmm/__init__.py` el import y los `__all__`:
```python
from .causal import (
    HMM_EMISSION_COLS_A,
    HMM_EMISSION_COLS_B,
    build_hmm_sequences,
    compute_causal_kinematics,
    decode_causal_states,
    fit_causal_hmm,
    forward_filtered_posteriors,
)
```

- [ ] **Step 7: Confirmar que la suite completa de ml y hmm sigue verde con shims**

Run: `uv run pytest tests/test_ml_hmm_causal.py 2>/dev/null; uv run pytest tests/ -q -k "ml or hmm"`
Expected: PASS (los shims preservan el comportamiento; nota: `test_ml_hmm_causal.py` ya no existe, se movió).

- [ ] **Step 8: Commit**

```bash
git add src/tfg_aves/hmm/causal.py src/tfg_aves/hmm/__init__.py src/tfg_aves/ml/hmm_causal.py src/tfg_aves/ml/features.py tests/test_hmm_causal.py
git commit -m "O3: mueve el HMM causal a tfg_aves.hmm y lo generaliza a los modelos A y B"
```

---

## Fase 3 — Reescribir `build_o3` a causal

### Task 3: Pipeline causal de O3 con split propio y estados A/B filtrados

**Files:**
- Modify: `src/tfg_aves/hmm/build.py` (reescritura de `build_o3`)
- Test: `tests/test_hmm_build.py`

- [ ] **Step 1: Actualizar el test de columnas de `build_o3` al esquema causal**

En `tests/test_hmm_build.py` (líneas ~88-94), sustituir la lista esperada de columnas por:
```python
expected_cols = {
    "bird_id", "date_utc", "lat", "lon",
    "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
    "daylight_hours", "veg_low", "veg_high",
    "is_hmm_obs_valid", "split",
    "state_a_causal", "state_b_causal",
    "posterior_a_estacionario", "posterior_a_migracion",
    "posterior_b_estacionario", "posterior_b_migracion",
}
assert expected_cols.issubset(set(df.columns))
```
**Nota crítica:** `sin_bearing_in` y `cos_bearing_in` DEBEN salir en `features.parquet` aunque el HMM no los use en su emisión, porque el clasificador de O4 los consume vía `FEATURES_KINEMATIC`. Olvidarlos rompe `build_feature_matrix` aguas abajo.
Y donde el test compruebe `in_holdout` o `stratified`, sustituir por aserciones sobre `split` (∈ {train,val,test}) y sobre que el LL de holdout se mide en `split=='test'`.

- [ ] **Step 2: Ejecutar y ver fallar**

Run: `uv run pytest tests/test_hmm_build.py -q`
Expected: FAIL (columnas viejas ausentes).

- [ ] **Step 3: Reescribir `build_o3`**

Reescribir `src/tfg_aves/hmm/build.py` con esta estructura (sustituye el split por aves y el Viterbi suavizado por split temporal + filtrado causal A y B):
```python
"""Orquestación end-to-end de O3 (causal)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from tfg_aves.data.split import assign_temporal_split, split_temporal_per_bird

from ._paths import DAILY_PARQUET, O3_OUT_DIR, RAW_CSV
from .causal import (
    HMM_EMISSION_COLS_A,
    HMM_EMISSION_COLS_B,
    compute_causal_kinematics,
    decode_causal_states,
    fit_causal_hmm,
)
from .evaluate import ab_agreement_causal, log_likelihood_per_obs
from .features import compute_observation_features, load_vegetation_from_raw


@dataclass
class BuildO3Result:
    features_path: Path
    models_path: Path
    metrics_path: Path
    n_observations: int
    ll_per_obs_a: float
    ll_per_obs_b: float
    pct_agreement_ab: float


def build_o3(*, n_restarts=10, random_state=0,
            daily_path=DAILY_PARQUET, raw_csv=RAW_CSV, out_dir=O3_OUT_DIR) -> BuildO3Result:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_daily = pd.read_parquet(daily_path)
    veg = load_vegetation_from_raw(raw_csv, df_daily["source_event_id"])
    # compute_observation_features ya devuelve lat, lon, veg_low, veg_high,
    # daylight_hours por (bird_id, date_utc) — todo lo que necesita la
    # cinemática causal. Las columnas salientes que también trae (step_length_km,
    # cos_turning_angle, is_observation_valid) son inocuas y no se exportan.
    base = compute_observation_features(df_daily, df_raw=veg)
    kin = compute_causal_kinematics(base)  # añade step_in_km, *_bearing_in, cos_turning_in, is_hmm_obs_valid

    # Split temporal propio sobre días HMM-válidos.
    valid = kin[kin["is_hmm_obs_valid"]].copy()
    valid = assign_temporal_split(valid)
    kin = kin.merge(valid[["bird_id", "date_utc", "split"]], on=["bird_id", "date_utc"], how="left")
    cutoff_by_bird = (
        valid[valid["split"] == "train"]
        .assign(_d=pd.to_datetime(valid.loc[valid["split"] == "train", "date_utc"]))
        .groupby("bird_id")["_d"].max().to_dict()
    )

    # Ajuste y decodificado causal de A y B.
    model_a, labels_a = fit_causal_hmm(kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_A,
                                       n_restarts=n_restarts, seed=random_state)
    model_b, labels_b = fit_causal_hmm(kin, cutoff_by_bird, emission_cols=HMM_EMISSION_COLS_B,
                                       n_restarts=n_restarts, seed=random_state)
    states_a = decode_causal_states(model_a, labels_a, kin, emission_cols=HMM_EMISSION_COLS_A, suffix="a")
    states_b = decode_causal_states(model_b, labels_b, kin, emission_cols=HMM_EMISSION_COLS_B, suffix="b")

    df = (kin.merge(states_a, on=["bird_id", "date_utc"], how="left")
              .merge(states_b, on=["bird_id", "date_utc"], how="left"))

    # LL holdout temporal (días test) y acuerdo A-B.
    from .causal import build_hmm_sequences
    test_kin = kin[kin["split"] == "test"]
    Xa, la, _ = build_hmm_sequences(test_kin, cutoff_by_bird=None, emission_cols=HMM_EMISSION_COLS_A)
    Xb, lb, _ = build_hmm_sequences(test_kin, cutoff_by_bird=None, emission_cols=HMM_EMISSION_COLS_B)
    ll_a = log_likelihood_per_obs(model_a, Xa, la)
    ll_b = log_likelihood_per_obs(model_b, Xb, lb)
    agreement = ab_agreement_causal(df)  # nuevo helper sobre state_a_causal/state_b_causal
    pct_agree = float(agreement["pct_agreement"])

    cols_final = [
        "bird_id", "date_utc", "lat", "lon",
        "step_in_km", "sin_bearing_in", "cos_bearing_in", "cos_turning_in",
        "daylight_hours", "veg_low", "veg_high",
        "is_hmm_obs_valid", "split",
        "state_a_causal", "state_b_causal",
        "posterior_a_estacionario", "posterior_a_migracion",
        "posterior_b_estacionario", "posterior_b_migracion",
    ]
    out_cols = [c for c in cols_final if c in df.columns]
    features_path = out_dir / "features.parquet"
    models_path = out_dir / "models_a_b.pkl"
    metrics_path = out_dir / "metrics.parquet"

    df[out_cols].to_parquet(features_path, index=False)
    joblib.dump({"model_a": model_a, "model_b": model_b,
                 "emission_cols_a": HMM_EMISSION_COLS_A, "emission_cols_b": HMM_EMISSION_COLS_B,
                 "label_map_a": labels_a, "label_map_b": labels_b,
                 "cutoff_by_bird": cutoff_by_bird, "random_state": random_state}, models_path)
    pd.DataFrame([
        {"model": "a", "scope": "test", "metric": "ll_per_obs", "value": ll_a},
        {"model": "b", "scope": "test", "metric": "ll_per_obs", "value": ll_b},
        {"model": "agreement", "scope": "both", "metric": "pct_agreement", "value": pct_agree},
    ]).to_parquet(metrics_path, index=False)

    return BuildO3Result(features_path, models_path, metrics_path,
                         int(df["is_hmm_obs_valid"].sum()), ll_a, ll_b, pct_agree)
```
Nota de implementación: confirmar al ejecutar que `compute_observation_features` devuelve `veg_low/veg_high/daylight_hours` por `(bird_id, date_utc)`; si no, calcular `daylight_hours` directamente y mapear veg como hace hoy. Mantener el cálculo equivalente.

- [ ] **Step 4: Añadir `ab_agreement_causal` a `evaluate.py`**

En `src/tfg_aves/hmm/evaluate.py`, añadir un helper paralelo a `ab_agreement` pero sobre las columnas causales:
```python
def ab_agreement_causal(df: pd.DataFrame) -> dict[str, float]:
    """Acuerdo entre state_a_causal y state_b_causal sobre días HMM-válidos."""
    valid = df[df["is_hmm_obs_valid"] & df["state_a_causal"].notna() & df["state_b_causal"].notna()]
    a = valid["state_a_causal"].astype(int)
    b = valid["state_b_causal"].astype(int)
    n = len(valid)
    return {"pct_agreement": float((a == b).mean() * 100.0) if n else 0.0, "n": n}
```

- [ ] **Step 5: Ejecutar y ver pasar**

Run: `uv run pytest tests/test_hmm_build.py tests/test_hmm_evaluate.py -q`
Expected: PASS. (Si `test_hmm_evaluate.py` aún prueba `ab_agreement`/`viterbi_per_bird` con columnas viejas, mantener esos tests del modo suavizado tal cual: las funciones suavizadas siguen existiendo en `evaluate.py`/`fit.py` aunque `build_o3` ya no las use. No se borran en esta tarea; ver Task 4 Step 7.)

- [ ] **Step 6: Commit**

```bash
git add src/tfg_aves/hmm/build.py src/tfg_aves/hmm/evaluate.py tests/test_hmm_build.py
git commit -m "O3: reescribe build_o3 a causal (split propio, estados A/B filtrados)"
```

---

## Fase 4 — Rewire de L1/L2/L3 para consumir O3 y limpieza de shims

### Task 4: L1/L2/L3 leen estado + split de O3; eliminar duplicación

**Files:**
- Modify: `src/tfg_aves/ml/build.py`, `build_l2.py`, `build_l3.py`, `features.py`, `__init__.py`
- Delete: `src/tfg_aves/ml/hmm_causal.py`
- Test: `tests/test_ml_build.py`, `test_ml_build_l2.py`, `test_ml_build_l3.py`, `test_ml_features.py`, `test_ml_smoke.py`

- [ ] **Step 1: Crear un helper de consumo en `ml/features.py`**

Añadir a `src/tfg_aves/ml/features.py`:
```python
def attach_o3_state_and_split(matrix: pd.DataFrame, features_o3: pd.DataFrame) -> pd.DataFrame:
    """Pega state_b_causal/posterior_b_migracion_causal y split desde O3 (merge m:1).

    O3 nombra los posteriores ``posterior_b_migracion``; aquí se renombra al
    nombre que consume O4 (``posterior_b_migracion_causal``) por compatibilidad.
    Lanza si alguna fila candidata queda sin estado o sin split.
    """
    cols = features_o3[[
        "bird_id", "date_utc", "state_b_causal", "posterior_b_migracion", "split",
    ]].rename(columns={"posterior_b_migracion": "posterior_b_migracion_causal"})
    merged = matrix.merge(cols, on=["bird_id", "date_utc"], how="left", validate="m:1")
    if merged[["state_b_causal", "posterior_b_migracion_causal", "split"]].isna().any().any():
        raise ValueError("Filas candidatas sin estado HMM causal o sin split tras el merge.")
    return merged
```

- [ ] **Step 2: Reescribir el ensamblaje en `build.py` (L1)**

En `src/tfg_aves/ml/build.py`, sustituir las líneas 123-143 por:
```python
    kin = features_o3  # O3 ya trae cinemática entrante + estado + split
    matrix = build_feature_matrix(kin, cells, include_bird_id=False)
    matrix = attach_o3_state_and_split(matrix, features_o3)

    def _by_split(label: str) -> pd.DataFrame:
        return matrix[matrix["split"] == label].reset_index(drop=True)

    train_s, val_s, test_s = _by_split("train"), _by_split("val"), _by_split("test")
```
Eliminar de los imports (líneas 26-33) `compute_causal_kinematics`, `split_temporal_per_bird`, y la línea `from .hmm_causal import ...`. Añadir `attach_o3_state_and_split` al import de `.features`. **Importante:** `build_feature_matrix` espera la cinemática entrante; como `features_o3` ya la trae (Fase 3), pasarla directamente.

- [ ] **Step 3: Reescribir `_prepare_causal_splits` en `build_l2.py`**

Sustituir el cuerpo de `_prepare_causal_splits` (`build_l2.py:60-94`) por:
```python
def _prepare_causal_splits(features_o3, cells, seed):
    matrices = {
        "personalizado": build_feature_matrix(features_o3, cells, include_bird_id=True),
        "poblacional": build_feature_matrix(features_o3, cells, include_bird_id=False),
    }
    out = {}
    for mode, mat in matrices.items():
        mat = attach_o3_state_and_split(mat, features_o3)
        out[mode] = tuple(
            mat[mat["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")
        )
    return out
```
Ajustar imports igual que en Step 2 (quitar `compute_causal_kinematics`, `split_temporal_per_bird`, `from .hmm_causal import ...`; añadir `attach_o3_state_and_split`). El parámetro `seed` queda sin uso aquí (el HMM ya no se ajusta en L2); eliminarlo de la firma y de la llamada.

- [ ] **Step 4: Reescribir `_prepare_poblacional_split` en `build_l3.py`**

Sustituir el cuerpo (`build_l3.py:63-90`) por:
```python
def _prepare_poblacional_split(features_o3, cells, seed):
    matrix = build_feature_matrix(features_o3, cells, include_bird_id=False)
    matrix = attach_o3_state_and_split(matrix, features_o3)
    return tuple(matrix[matrix["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
```
Mismos ajustes de import; `seed` sin uso → eliminar de la firma y la llamada.

- [ ] **Step 5: Eliminar el módulo `hmm_causal.py` y limpiar `features.py` e `__init__.py`**

```bash
git rm src/tfg_aves/ml/hmm_causal.py
```
En `src/tfg_aves/ml/features.py`, eliminar los reexports temporales de `compute_causal_kinematics`, `HMM_EMISSION_COLS` y `split_temporal_per_bird` (los consumidores ya no los usan; comprobar con grep). En `src/tfg_aves/ml/__init__.py`, eliminar de imports y `__all__`: `compute_causal_kinematics`, `split_temporal_per_bird`, `decode_causal_states`, `fit_causal_hmm`. Verificar:
```bash
grep -rn "from .hmm_causal\|hmm_causal\|compute_causal_kinematics\|split_temporal_per_bird" src/tfg_aves/ml/
```
Expected: sin resultados (salvo, si acaso, `build_feature_matrix` que sigue en features).

- [ ] **Step 6: Actualizar los tests de ml afectados**

En `tests/test_ml_features.py`, `test_ml_build.py`, `test_ml_build_l2.py`, `test_ml_build_l3.py`, `test_ml_smoke.py`: cualquier test que llame a `compute_causal_kinematics`, `fit_causal_hmm`, `decode_causal_states` o `split_temporal_per_bird` desde `tfg_aves.ml` debe (a) importarlos de su nueva ubicación (`tfg_aves.hmm.causal`, `tfg_aves.data.split`) si prueban esa unidad, o (b) construir un `features_o3` de juguete que ya traiga `step_in_km`, `cos_turning_in`, `is_hmm_obs_valid`, `state_b_causal`, `posterior_b_migracion`, `split` y verificar el consumo. Ejecutar para localizar:
```bash
uv run pytest tests/ -q -k "ml" 2>&1 | head -40
```
Adaptar los fallos uno a uno hasta verde.

- [ ] **Step 7: Decidir el destino del modo suavizado de O3 (limpieza)**

El spec (C1) descarta el modo suavizado. Las funciones `viterbi_per_bird`, `stratified_holdout_split` y las features salientes (`compute_observation_features` produce `step_length_km`/`cos_turning_angle`) ya no las usa `build_o3`. Opciones acordadas: **conservar `compute_observation_features`** (build_o3 la usa para veg/daylight) pero **eliminar `viterbi_per_bird` y `stratified_holdout_split`** y sus tests, ya que nada los llama. Verificar con grep antes de borrar; borrar los tests `test_hmm_fit.py::test_stratified_*` y `test_hmm_evaluate.py::test_*viterbi*` correspondientes. Mantener `ab_agreement`/`biological_coherence_table` si el notebook los usa con columnas causales (renombrar uso a `*_causal`).

- [ ] **Step 8: Ejecutar la suite completa**

Run: `uv run pytest -q`
Expected: PASS (toda la suite). Resolver fallos residuales.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "O4: L1/L2/L3 consumen el HMM y el split de O3; elimina la duplicación causal"
```

---

## Fase 5 — Verificación estática

### Task 5: Lint y suite verde

- [ ] **Step 1: Ruff**

Run: `uv run ruff check src tests`
Expected: `All checks passed!` (corregir lo que salga).

- [ ] **Step 2: Suite completa**

Run: `uv run pytest -q`
Expected: PASS, sin tests omitidos inesperados.

- [ ] **Step 3: Commit si hubo correcciones**

```bash
git add -A && git commit -m "O3/O4: limpieza de ruff tras el rework causal" || echo "nada que commitear"
```

---

## Fase 6 — Regeneración y gate de re-verificación

### Task 6: Regenerar la cadena y cuantificar el desplazamiento

**Files:**
- Create: `notebooks/_reverify_o3causal.py` (script de comparación, temporal)

- [ ] **Step 1: Regenerar la cadena completa**

Run:
```bash
uv run python -c "from tfg_aves.hmm import build_o3; print(build_o3())"
uv run python -c "from tfg_aves.ml import build_o4; build_o4()"
uv run python -c "from tfg_aves.ml.build_l2 import build_o4_l2; build_o4_l2()"
uv run python -c "from tfg_aves.ml.build_l3 import build_o4_l3; build_o4_l3()"
```
Expected: sin errores; ningún `ValueError` de "filas sin estado/split".

- [ ] **Step 2: Comparar métricas antes/después**

Escribir y ejecutar un script que cargue `data/processed/_snapshot_pre_o3causal/*_metrics.parquet` y los nuevos `metrics.parquet`, y tabule las diferencias en top-1, top-3, log-loss, dist mediana (L1/L2) y pinball/cobertura (L3). Imprimir un resumen legible.

- [ ] **Step 3: Evaluar el gate**

Criterio (spec §5): el desplazamiento debe ser pequeño y no cambiar ninguna conclusión cualitativa (ML pierde a persistencia en top-1; gana en log-loss donde aplica; calibración ~80% en L3; caída en migración). **Si alguna conclusión cambia, PARAR y avisar al autor** con la tabla de diferencias antes de continuar.

- [ ] **Step 4: Revalidar coherencia biológica**

Comprobar sobre `o3/features.parquet` causal: % migración global razonable (orden ~15%), mínimo jun-jul, picos abr y sep-oct, μ[step] migración ≫ estacionario. Si chirría con la fenología, tratar como bug (regla de coherencia biológica).

- [ ] **Step 5: Regenerar O5 y verificar que abre**

Run: `uv run python -c "from tfg_aves.viz import build_o5; build_o5()"` (ajustar al nombre real del orquestador de O5). Comprobar que los HTML se regeneran sin error.

- [ ] **Step 6: Commit de evidencia de regeneración**

Los parquets son gitignored; commitear solo el script de re-verificación si se conserva, o borrarlo. No re-taguear todavía.

---

## Fase 7 — Artefactos de O3 y decisión documentada

### Task 7: Regenerar C1–C9/D1 y documentar la decisión causal

**Files:**
- Modify: `notebooks/03_eda_o3.py`
- Create: artefactos vía `save_artifact` (figuras/tablas en `reports/`)

- [ ] **Step 1: Adaptar el notebook al esquema causal**

En `notebooks/03_eda_o3.py`, sustituir referencias a `state_a`/`state_b` por `state_a_causal`/`state_b_causal` y a las features salientes por las entrantes. Regenerar C1–C9 y D1 con `save_artifact` (mismos slugs, captions actualizados sin jerga "O3").

- [ ] **Step 2: Nuevo artefacto: acuerdo suavizado vs filtrado**

Añadir una figura/tabla que compare, sobre los días comunes, el estado suavizado del modelo anterior (`_snapshot_pre_o3causal/o3_features.parquet`) con el filtrado nuevo, mostrando el % de acuerdo. Documentar con `save_artifact` la decisión "decodificado causal en O3" (evidencia de que es el mismo modelo realineado).

- [ ] **Step 3: Revalidar coherencia biológica con las figuras**

Inspeccionar C3 (estado vs mes/latitud) regenerado: debe seguir la fenología. Confirmar antes de aceptar.

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_eda_o3.py reports/
git commit -m "O3: regenera los artefactos con estados causales y documenta el decodificado filtrado"
```

---

## Fase 8 — Memoria, ai-log y tag

### Task 8: Cierre documental

**Files:**
- Modify: `reports/memoria/05_o3_hmm.md`
- Create: `reports/ai-log/NNNN-o3-causal.md`

- [ ] **Step 1: Notas de memoria**

Actualizar `reports/memoria/05_o3_hmm.md`: narrar el régimen causal (cinemática entrante + filtrado forward-only + split temporal) y su porqué (alimenta a O4/O5 sin fuga, una sola fuente de verdad). Conservar el relato del rework/circularidad y la ablación A vs B. Sin jerga "O1/O2/...", sin rayas (—).

- [ ] **Step 2: Entrada de ai-log**

Crear la entrada `NNNN-o3-causal.md` siguiendo `reports/ai-log/README.md` (tono: el autor decide y valida; la IA propone y ejecuta lo mecánico). Es trabajo técnico sustantivo de O3.

- [ ] **Step 3: Re-tag de hitos**

```bash
git tag v0.3.2-o3-causal
```
Re-generar/etiquetar L1/L2/L3/O5 según haga falta tras la re-verificación (coordinar con el autor el re-tag de O4/O5 si los números se movieron).

- [ ] **Step 4: Actualizar CLAUDE.md y memoria de proyecto**

Actualizar el estado de O3 en `CLAUDE.md` (HMM causal canónico, columnas `state_*_causal`, split propio) y el `MEMORY.md`/`project_o3_outcome.md`.

- [ ] **Step 5: Commit final**

```bash
git add reports/ CLAUDE.md
git commit -m "O3: notas de memoria y registro de IA del HMM causal"
```

---

## Self-review (cobertura del spec)

- C1 (reemplazo total causal): Fase 3. ✓
- C2 (A y B causales, B = L3): Fase 2 (generalización A/B) + Fase 3 (fit A y B). ✓
- C3/C4 (cinemática entrante + filtrado): Fase 2. ✓
- C5/C7 (split propio en `tfg_aves.data`, columna `split`): Fase 1 + Fase 3. ✓
- C6 (hmm dueño, O4 consume, dedup L1/L2/L3): Fase 2 + Fase 4. ✓
- C8 (LL holdout temporal): Fase 3 Step 3. ✓
- C9 (revalidación biológica): Fase 6 Step 4 + Fase 7 Step 3. ✓
- §5 (gate de re-verificación): Fase 0 + Fase 6. ✓
- §6 (tests, tag, ai-log, memoria, artefactos): Fases 5, 7, 8. ✓
