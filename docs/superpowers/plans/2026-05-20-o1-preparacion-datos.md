# O1 — Preparación de datos GPS: Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar la pipeline de O1 que transforma el CSV crudo de Movebank en una tabla diaria por individuo (una fila por `(bird_id, date_utc)` con la posición del ave a la hora de referencia UTC y huecos explícitos), lista para alimentar O2.

**Architecture:** Pipeline en cuatro módulos puros bajo `src/tfg_aves/data/` (`load`, `clean`, `daily`, `build`), orquestada desde un notebook `notebooks/01_eda_o1.py` (jupytext percent) que decide umbrales con artefactos `save_artifact()`. Funciones puras devolviendo DataFrames; sólo `build_o1` escribe a disco. Constantes de rutas en `src/tfg_aves/data/_paths.py` para evitar ciclos.

**Tech Stack:** Python 3.12, pandas, pyarrow, numpy, matplotlib, folium, jupytext, pytest, ruff. Gestor de dependencias `uv` (lock ya existente). Helper `tfg_aves.reporting.save_artifact` ya disponible.

**Spec:** `docs/superpowers/specs/2026-05-20-o1-preparacion-datos-design.md`.

---

## File structure

**Crear:**
- `src/tfg_aves/data/_paths.py` — constantes `RAW_CSV`, `INTERIM`, `PROCESSED` (separadas para evitar imports circulares).
- `src/tfg_aves/data/load.py` — `load_raw(path)`.
- `src/tfg_aves/data/clean.py` — `drop_movebank_flags`, `drop_invalid_coords_and_dupes`, `drop_speed_outliers`.
- `src/tfg_aves/data/daily.py` — `coverage_by_hour`, `pick_reference_hour`, `build_daily`, `filter_birds_by_validity`.
- `src/tfg_aves/data/build.py` — `build_o1(...)`.
- `tests/conftest.py` — fixtures sintéticos compartidos.
- `tests/test_data_load.py`.
- `tests/test_data_clean.py`.
- `tests/test_data_daily.py`.
- `tests/test_data_build.py`.
- `notebooks/01_eda_o1.py` — notebook EDA en formato jupytext percent (versionado).
- `reports/memoria/03_o1_datos.md` — notas estructuradas.
- `reports/ai-log/0006-o1-eda-y-pipeline-datos.md` — entrada de uso de IA.

**Modificar:**
- `src/tfg_aves/data/__init__.py` — re-exportar API pública.

**Generados (no versionados):**
- `data/processed/daily.parquet`.
- `data/processed/fixes_clean.parquet`.

**Generados (versionados como evidencia):**
- 9 entradas en `reports/figures/`, `reports/tables/`, `reports/captions/` y `reports/INDEX.md` (artefactos D1–D4 + C1–C5).

---

## Task 1: Esqueleto de `tfg_aves.data`

**Files:**
- Create: `src/tfg_aves/data/_paths.py`
- Create: `src/tfg_aves/data/load.py`
- Create: `src/tfg_aves/data/clean.py`
- Create: `src/tfg_aves/data/daily.py`
- Create: `src/tfg_aves/data/build.py`
- Modify: `src/tfg_aves/data/__init__.py`
- Test: `tests/test_data_smoke.py` (creado en este task)

- [ ] **Step 1.1: Crear `_paths.py` con constantes de rutas**

Crear `src/tfg_aves/data/_paths.py`:

```python
"""Rutas estables del proyecto. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
RAW_CSV: Path = ROOT / "data" / "raw" / "migration_original.csv"
INTERIM: Path = ROOT / "data" / "interim"
PROCESSED: Path = ROOT / "data" / "processed"
```

- [ ] **Step 1.2: Crear `load.py` con firma `NotImplementedError`**

Crear `src/tfg_aves/data/load.py`:

```python
"""Carga y normalización del CSV crudo de Movebank."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import RAW_CSV


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV de Movebank y devuelve un DataFrame normalizado."""
    raise NotImplementedError
```

- [ ] **Step 1.3: Crear `clean.py` con tres firmas**

Crear `src/tfg_aves/data/clean.py`:

```python
"""Filtros de outliers sobre fixes GPS de Movebank."""
from __future__ import annotations

import pandas as pd


def drop_movebank_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes marcados como inválidos por Movebank."""
    raise NotImplementedError


def drop_invalid_coords_and_dupes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta coordenadas inválidas y duplicados por (bird_id, timestamp)."""
    raise NotImplementedError


def drop_speed_outliers(
    df: pd.DataFrame, max_speed_kmh: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes que implican velocidad > max_speed_kmh respecto al previo."""
    raise NotImplementedError
```

- [ ] **Step 1.4: Crear `daily.py` con cuatro firmas**

Crear `src/tfg_aves/data/daily.py`:

```python
"""Resample a una fila por (bird_id, date_utc) con huecos explícitos."""
from __future__ import annotations

import pandas as pd


def coverage_by_hour(df: pd.DataFrame, tolerance_min: float) -> pd.DataFrame:
    """Cobertura % de (ave, día) con fix en [h±tol] para h en 0..23."""
    raise NotImplementedError


def pick_reference_hour(
    df: pd.DataFrame, tolerance_min: float
) -> tuple[int, pd.DataFrame]:
    """Devuelve la hora UTC con cobertura máxima y la tabla completa."""
    raise NotImplementedError


def build_daily(
    df: pd.DataFrame,
    reference_hour_utc: int,
    tolerance_min: float,
) -> pd.DataFrame:
    """Colapsa fixes a una fila por (bird_id, date_utc) con huecos explícitos."""
    raise NotImplementedError


def filter_birds_by_validity(
    df_daily: pd.DataFrame, min_valid_days: int
) -> pd.DataFrame:
    """Descarta individuos con < min_valid_days días válidos."""
    raise NotImplementedError
```

- [ ] **Step 1.5: Crear `build.py` con orquestador**

Crear `src/tfg_aves/data/build.py`:

```python
"""Orquestación end-to-end de la pipeline de O1."""
from __future__ import annotations

from pathlib import Path

from ._paths import PROCESSED


def build_o1(
    *,
    max_speed_kmh: float,
    reference_hour_utc: int,
    tolerance_min: float,
    min_valid_days: int,
    out_dir: Path = PROCESSED,
) -> dict[str, int]:
    """Compone load → clean → build_daily → filter y materializa parquets."""
    raise NotImplementedError
```

- [ ] **Step 1.6: Actualizar `__init__.py` con la API pública**

Reemplazar `src/tfg_aves/data/__init__.py` por:

```python
"""O1 — Preparación de datos: limpieza y resample diario del GPS."""
from __future__ import annotations

from ._paths import INTERIM, PROCESSED, RAW_CSV, ROOT
from .build import build_o1
from .clean import (
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
)
from .daily import (
    build_daily,
    coverage_by_hour,
    filter_birds_by_validity,
    pick_reference_hour,
)
from .load import load_raw

__all__ = [
    "INTERIM",
    "PROCESSED",
    "RAW_CSV",
    "ROOT",
    "build_daily",
    "build_o1",
    "coverage_by_hour",
    "drop_invalid_coords_and_dupes",
    "drop_movebank_flags",
    "drop_speed_outliers",
    "filter_birds_by_validity",
    "load_raw",
    "pick_reference_hour",
]
```

- [ ] **Step 1.7: Crear smoke test de imports**

Crear `tests/test_data_smoke.py`:

```python
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
```

- [ ] **Step 1.8: Ejecutar tests y verificar verde**

```bash
uv run pytest tests/test_data_smoke.py -v
```

Expected: 1 passed.

- [ ] **Step 1.9: Linter limpio**

```bash
uv run ruff check src tests
```

Expected: `All checks passed!`

- [ ] **Step 1.10: Commit**

```bash
git add src/tfg_aves/data/ tests/test_data_smoke.py
git commit -m "Esqueleto de tfg_aves.data: load, clean, daily, build"
```

---

## Task 2: Tests unitarios de carga y limpieza (failing)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_data_load.py`
- Create: `tests/test_data_clean.py`

- [ ] **Step 2.1: Crear `tests/conftest.py` con fixtures comunes**

```python
"""Fixtures sintéticas para los tests de tfg_aves.data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Cabecera Movebank reducida a las columnas que usamos + algunas
# 'visible' duplicadas como en el CSV real.
_MOVEBANK_HEADER = (
    '"event-id","visible","timestamp","location-long","location-lat",'
    '"manually-marked-outlier","visible","sensor-type",'
    '"individual-taxon-canonical-name","tag-local-identifier",'
    '"individual-local-identifier","study-name"'
)


def _movebank_row(
    *,
    event_id: int,
    timestamp: str,
    lon: float,
    lat: float,
    outlier: bool = False,
    visible: bool = True,
    bird_id: str = "A",
) -> str:
    outlier_cell = "true" if outlier else ""
    visible_cell = "true" if visible else "false"
    return (
        f'"{event_id}","{visible_cell}","{timestamp}",'
        f'"{lon}","{lat}","{outlier_cell}","{visible_cell}","gps",'
        f'"Larus fuscus","91732","{bird_id}","study"'
    )


@pytest.fixture
def mini_movebank_csv(tmp_path: Path) -> Path:
    """CSV pequeño con formato Movebank y casos representativos."""
    rows = [
        _movebank_row(event_id=1, timestamp="2010-04-02 12:00:00.000",
                      lon=24.5, lat=61.2),
        _movebank_row(event_id=2, timestamp="2010-04-02 13:00:00.000",
                      lon=24.6, lat=61.3),
        _movebank_row(event_id=3, timestamp="2010-04-01 12:00:00.000",
                      lon=24.4, lat=61.1),
        _movebank_row(event_id=4, timestamp="2010-04-02 14:00:00.000",
                      lon=24.7, lat=61.4, visible=False),
        _movebank_row(event_id=5, timestamp="2010-04-02 15:00:00.000",
                      lon=24.8, lat=61.5, outlier=True),
    ]
    csv = tmp_path / "mini_movebank.csv"
    csv.write_text(_MOVEBANK_HEADER + "\n" + "\n".join(rows) + "\n",
                   encoding="utf-8")
    return csv


@pytest.fixture
def df_flags() -> pd.DataFrame:
    """DataFrame cargado para tests de drop_movebank_flags."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                ],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.7, 24.8],
            "lat": [61.1, 61.2, 61.3, 61.4],
            "manually_marked_outlier": [False, True, False, False],
            "visible": [True, True, False, True],
            "sensor_type": ["gps"] * 4,
            "bird_id": ["A"] * 4,
        }
    )


@pytest.fixture
def df_coords_dupes() -> pd.DataFrame:
    """DataFrame con coords inválidas y duplicado por (bird_id, timestamp)."""
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 13:00",  # duplicado con #2
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                ],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.61, 200.0, 24.8],   # 200 es inválida
            "lat": [61.1, 61.2, 61.21, 61.4, 95.0],    # 95 es inválida
            "manually_marked_outlier": [False] * 5,
            "visible": [True] * 5,
            "sensor_type": ["gps"] * 5,
            "bird_id": ["A"] * 5,
        }
    )


@pytest.fixture
def df_speed() -> pd.DataFrame:
    """Cinco fixes consecutivos donde el #3 implica un salto imposible.

    Distancias aproximadas (haversine) entre consecutivos a 1 hora:
      1→2: ~10 km (≈10 km/h) — OK
      2→3: ~3700 km (≈3700 km/h) — OUTLIER
      3→4: ~3700 km (≈3700 km/h) — OUTLIER si #3 sigue presente
      4→5: ~10 km — OK
    Al descartar iterativamente, el algoritmo debe quitar #3 y reconocer
    que 2→4 es de nuevo razonable (~10 km en 2 h ≈ 5 km/h).
    """
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-01 12:00",
                    "2010-04-01 13:00",
                    "2010-04-01 14:00",
                    "2010-04-01 15:00",
                    "2010-04-01 16:00",
                ],
                utc=True,
            ),
            "lon": [24.50, 24.60, 90.0, 24.70, 24.80],
            "lat": [61.10, 61.15, 30.0, 61.25, 61.30],
            "manually_marked_outlier": [False] * 5,
            "visible": [True] * 5,
            "sensor_type": ["gps"] * 5,
            "bird_id": ["A"] * 5,
        }
    )
```

- [ ] **Step 2.2: Crear `tests/test_data_load.py`**

```python
"""Tests de tfg_aves.data.load."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import load_raw


def test_load_raw_normalises_schema(mini_movebank_csv):
    df = load_raw(mini_movebank_csv)

    expected_cols = {
        "event_id",
        "timestamp",
        "lon",
        "lat",
        "manually_marked_outlier",
        "visible",
        "sensor_type",
        "bird_id",
    }
    assert set(df.columns) == expected_cols, df.columns.tolist()

    # timestamp tipado UTC.
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert str(df["timestamp"].dt.tz) == "UTC"

    # Ordenado por (bird_id, timestamp).
    sorted_df = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sorted_df)

    # Sin columnas ambientales.
    assert not any("ECMWF" in c or "NCEP" in c for c in df.columns)

    # Booleanos normalizados (no strings).
    assert df["visible"].dtype == bool
    assert df["manually_marked_outlier"].dtype == bool

    # event_id #4 vino con visible=false → debe seguir presente (load no filtra)
    # pero su valor visible es False.
    row_4 = df[df["event_id"] == 4].iloc[0]
    assert row_4["visible"] is False or row_4["visible"] == False  # noqa: E712

    # event_id #5 vino con outlier=true.
    row_5 = df[df["event_id"] == 5].iloc[0]
    assert row_5["manually_marked_outlier"] is True or row_5["manually_marked_outlier"] == True  # noqa: E712
```

- [ ] **Step 2.3: Crear `tests/test_data_clean.py`**

```python
"""Tests de tfg_aves.data.clean."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import (
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
)


def test_drop_movebank_flags(df_flags):
    out, report = drop_movebank_flags(df_flags)

    # event_id 2 (outlier=True) y 3 (visible=False) descartados; 1 y 4 quedan.
    assert sorted(out["event_id"].tolist()) == [1, 4]
    assert report == {
        "discarded_movebank_outlier": 1,
        "discarded_visible_false": 1,
    }


def test_drop_invalid_coords_and_dupes(df_coords_dupes):
    out, report = drop_invalid_coords_and_dupes(df_coords_dupes)

    # event_id 4 (lon=200) y 5 (lat=95) inválidos; #3 es duplicado de #2.
    # Sobreviven 1 y 2.
    assert sorted(out["event_id"].tolist()) == [1, 2]
    assert report == {
        "discarded_invalid_coords": 2,
        "discarded_duplicates": 1,
    }


def test_drop_speed_outliers_removes_impossible_fix(df_speed):
    out, report = drop_speed_outliers(df_speed, max_speed_kmh=200.0)

    # El fix #3 está a ~3700 km de los vecinos en 1 h → debe descartarse.
    assert 3 not in out["event_id"].tolist()
    # Los inocentes 1, 2, 4, 5 sobreviven.
    assert sorted(out["event_id"].tolist()) == [1, 2, 4, 5]
    assert report["discarded_speed"] == 1


def test_drop_speed_outliers_no_outliers_returns_intact():
    df = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "timestamp": pd.to_datetime(
                ["2010-04-01 12:00", "2010-04-01 13:00", "2010-04-01 14:00"],
                utc=True,
            ),
            "lon": [24.5, 24.6, 24.7],
            "lat": [61.1, 61.15, 61.20],
            "manually_marked_outlier": [False] * 3,
            "visible": [True] * 3,
            "sensor_type": ["gps"] * 3,
            "bird_id": ["A"] * 3,
        }
    )
    out, report = drop_speed_outliers(df, max_speed_kmh=200.0)

    assert out["event_id"].tolist() == [1, 2, 3]
    assert report == {"discarded_speed": 0}
```

- [ ] **Step 2.4: Ejecutar tests y verificar que fallan**

```bash
uv run pytest tests/test_data_load.py tests/test_data_clean.py -v
```

Expected: todos los tests fallan con `NotImplementedError`.

- [ ] **Step 2.5: Commit**

```bash
git add tests/conftest.py tests/test_data_load.py tests/test_data_clean.py
git commit -m "Tests unitarios de carga y limpieza"
```

---

## Task 3: Implementar `load` y `clean`

**Files:**
- Modify: `src/tfg_aves/data/load.py`
- Modify: `src/tfg_aves/data/clean.py`

- [ ] **Step 3.1: Implementar `load_raw`**

Reemplazar el cuerpo de `load_raw` en `src/tfg_aves/data/load.py`:

```python
"""Carga y normalización del CSV crudo de Movebank."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import RAW_CSV

# Columnas de Movebank → snake_case del proyecto.
_RENAME_MAP: dict[str, str] = {
    "event-id": "event_id",
    "timestamp": "timestamp",
    "location-long": "lon",
    "location-lat": "lat",
    "manually-marked-outlier": "manually_marked_outlier",
    "visible": "visible",
    "sensor-type": "sensor_type",
    "individual-local-identifier": "bird_id",
}


def _to_bool_movebank(series: pd.Series) -> pd.Series:
    """Convierte ``'true'`` / ``'false'`` / ``''`` / NaN a ``bool``.

    Movebank codifica los flags como cadenas; ``''`` y NaN se interpretan
    como ``False`` (no marcado).
    """
    return series.fillna("").astype(str).str.lower().eq("true")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV de Movebank y devuelve un DataFrame normalizado.

    - Renombra columnas a snake_case y descarta las covariables ambientales.
    - Convierte ``timestamp`` a ``datetime64[ns, UTC]``.
    - Tipa ``visible`` y ``manually_marked_outlier`` como ``bool``.
    - Ordena por ``(bird_id, timestamp)`` y reindexa.
    - El CSV de Movebank duplica la columna ``visible``; se conserva la
      primera ocurrencia.
    """
    df = pd.read_csv(path)
    # Algunos exports de Movebank repiten 'visible'; nos quedamos con la
    # primera ocurrencia.
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[list(_RENAME_MAP.keys())].rename(columns=_RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["visible"] = _to_bool_movebank(df["visible"])
    df["manually_marked_outlier"] = _to_bool_movebank(
        df["manually_marked_outlier"]
    )
    df = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    return df
```

- [ ] **Step 3.2: Implementar las tres funciones de `clean.py`**

Reemplazar el cuerpo de `src/tfg_aves/data/clean.py`:

```python
"""Filtros de outliers sobre fixes GPS de Movebank."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0088
_MAX_SPEED_ITERATIONS = 20


def drop_movebank_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta fixes con ``visible=False`` o ``manually_marked_outlier=True``.

    Devuelve el DataFrame filtrado y un dict con el conteo de descartes
    por causa.
    """
    visible_false = (~df["visible"]).sum()
    outlier_true = df["manually_marked_outlier"].sum()
    mask = df["visible"] & (~df["manually_marked_outlier"])
    out = df[mask].reset_index(drop=True)
    report = {
        "discarded_movebank_outlier": int(outlier_true),
        "discarded_visible_false": int(visible_false),
    }
    return out, report


def drop_invalid_coords_and_dupes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta coordenadas fuera de rango y duplicados (bird_id, timestamp).

    Se conserva la primera ocurrencia en caso de duplicado.
    """
    valid_coords = (
        df["lat"].between(-90, 90, inclusive="both")
        & df["lon"].between(-180, 180, inclusive="both")
    )
    discarded_invalid = int((~valid_coords).sum())
    df_coords = df[valid_coords]

    before = len(df_coords)
    df_dedup = df_coords.drop_duplicates(
        subset=["bird_id", "timestamp"], keep="first"
    )
    discarded_dupes = before - len(df_dedup)

    out = df_dedup.reset_index(drop=True)
    return out, {
        "discarded_invalid_coords": discarded_invalid,
        "discarded_duplicates": discarded_dupes,
    }


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Distancia haversine vectorizada en km entre pares de puntos."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _compute_speed_kmh(df: pd.DataFrame) -> np.ndarray:
    """Velocidad en km/h respecto al fix anterior del mismo ``bird_id``.

    Devuelve ``NaN`` en la primera observación de cada ave y donde
    ``Δt`` sea 0 (no debe ocurrir tras el deduplicado).
    """
    df = df.sort_values(["bird_id", "timestamp"])
    bird = df["bird_id"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    ts = df["timestamp"].to_numpy()

    same_bird = np.concatenate([[False], bird[1:] == bird[:-1]])
    dist = np.full(len(df), np.nan)
    dist[1:] = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_h = np.full(len(df), np.nan)
    dt_h[1:] = (
        (ts[1:] - ts[:-1]).astype("timedelta64[s]").astype(float) / 3600.0
    )

    speed = np.full(len(df), np.nan)
    valid = same_bird & (dt_h > 0)
    speed[valid] = dist[valid] / dt_h[valid]
    return speed


def drop_speed_outliers(
    df: pd.DataFrame, max_speed_kmh: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta iterativamente fixes con velocidad > ``max_speed_kmh``.

    Un único fix outlier "envenena" su salto de entrada y de salida; en
    cada iteración descartamos sólo el fix con velocidad máxima (el más
    culpable). Las velocidades de sus vecinos se recalculan en la
    siguiente iteración y, si eran víctimas inocentes, dejan de superar
    el umbral. Se itera hasta ``_MAX_SPEED_ITERATIONS`` veces o hasta
    que ningún fix supere el umbral.
    """
    if max_speed_kmh <= 0:
        raise ValueError("max_speed_kmh debe ser estrictamente positivo")

    current = df.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)
    total_discarded = 0
    for _ in range(_MAX_SPEED_ITERATIONS):
        speed = _compute_speed_kmh(current)
        bad = speed > max_speed_kmh
        if not bad.any():
            break
        # `nanargmax` sobre la máscara: usamos -inf para los no-bad de
        # forma que el argmax recaiga siempre en un fix culpable.
        masked = np.where(bad, speed, -np.inf)
        worst = int(np.argmax(masked))
        current = (
            current.drop(current.index[worst]).reset_index(drop=True)
        )
        total_discarded += 1

    return current, {"discarded_speed": total_discarded}
```

- [ ] **Step 3.3: Ejecutar tests y verificar verde**

```bash
uv run pytest tests/test_data_load.py tests/test_data_clean.py -v
```

Expected: 5 tests pasan (1 de load + 4 de clean).

- [ ] **Step 3.4: Comprobar suite completa y linter**

```bash
uv run pytest -q
uv run ruff check src tests
```

Expected: todos los tests verdes (16 previos + 1 smoke + 5 nuevos = 22), ruff sin errores.

- [ ] **Step 3.5: Commit**

```bash
git add src/tfg_aves/data/load.py src/tfg_aves/data/clean.py
git commit -m "Implementar load + clean con tests"
```

---

## Task 4: Tests unitarios de resample diario (failing)

**Files:**
- Modify: `tests/conftest.py` (añadir fixtures)
- Create: `tests/test_data_daily.py`

- [ ] **Step 4.1: Añadir fixtures de daily a `tests/conftest.py`**

Añadir al final de `tests/conftest.py`:

```python
@pytest.fixture
def df_hourly_coverage() -> pd.DataFrame:
    """Fixes concentrados a las 12:00 UTC → cobertura máxima en h=12."""
    rows = []
    eid = 1
    for day in pd.date_range("2010-04-01", "2010-04-05", tz="UTC"):
        # Un fix a 12:00 y otro a 18:00 cada día.
        for hour in (12, 18):
            rows.append(
                {
                    "event_id": eid,
                    "timestamp": day + pd.Timedelta(hours=hour),
                    "lon": 24.5,
                    "lat": 61.2,
                    "manually_marked_outlier": False,
                    "visible": True,
                    "sensor_type": "gps",
                    "bird_id": "A",
                }
            )
            eid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def df_daily_three_fixes_one_day() -> pd.DataFrame:
    """Tres fixes el mismo día, distintos minutos respecto a 12:00 UTC."""
    return pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "timestamp": pd.to_datetime(
                [
                    "2010-04-02 11:30:00",
                    "2010-04-02 12:10:00",   # más cercano a 12:00 (Δ=10 min)
                    "2010-04-02 13:00:00",
                ],
                utc=True,
            ),
            "lon": [24.40, 24.50, 24.60],
            "lat": [61.10, 61.20, 61.30],
            "manually_marked_outlier": [False] * 3,
            "visible": [True] * 3,
            "sensor_type": ["gps"] * 3,
            "bird_id": ["A"] * 3,
        }
    )


@pytest.fixture
def df_daily_with_gap() -> pd.DataFrame:
    """Ave A con fixes el 01 y 03 pero no el 02 → hueco explícito en daily."""
    return pd.DataFrame(
        {
            "event_id": [1, 2],
            "timestamp": pd.to_datetime(
                ["2010-04-01 12:00:00", "2010-04-03 12:00:00"],
                utc=True,
            ),
            "lon": [24.5, 24.7],
            "lat": [61.2, 61.4],
            "manually_marked_outlier": [False, False],
            "visible": [True, True],
            "sensor_type": ["gps", "gps"],
            "bird_id": ["A", "A"],
        }
    )


@pytest.fixture
def df_daily_far_from_reference() -> pd.DataFrame:
    """Único fix a las 15:20 (200 min de 12:00)."""
    return pd.DataFrame(
        {
            "event_id": [1],
            "timestamp": pd.to_datetime(["2010-04-02 15:20:00"], utc=True),
            "lon": [24.5],
            "lat": [61.2],
            "manually_marked_outlier": [False],
            "visible": [True],
            "sensor_type": ["gps"],
            "bird_id": ["A"],
        }
    )


@pytest.fixture
def df_daily_two_birds_unequal() -> pd.DataFrame:
    """Ave A: 3 días válidos. Ave B: 1 día válido."""
    rows = []
    eid = 1
    for day_offset in range(3):
        ts = pd.Timestamp("2010-04-01 12:00", tz="UTC") + pd.Timedelta(
            days=day_offset
        )
        rows.append(
            {
                "event_id": eid,
                "timestamp": ts,
                "lon": 24.5,
                "lat": 61.2,
                "manually_marked_outlier": False,
                "visible": True,
                "sensor_type": "gps",
                "bird_id": "A",
            }
        )
        eid += 1
    rows.append(
        {
            "event_id": eid,
            "timestamp": pd.Timestamp("2010-04-01 12:00", tz="UTC"),
            "lon": 25.0,
            "lat": 62.0,
            "manually_marked_outlier": False,
            "visible": True,
            "sensor_type": "gps",
            "bird_id": "B",
        }
    )
    return pd.DataFrame(rows)
```

- [ ] **Step 4.2: Crear `tests/test_data_daily.py`**

```python
"""Tests de tfg_aves.data.daily."""
from __future__ import annotations

import pandas as pd

from tfg_aves.data import (
    build_daily,
    coverage_by_hour,
    filter_birds_by_validity,
    pick_reference_hour,
)


def test_coverage_by_hour_peaks_at_fixed_hour(df_hourly_coverage):
    cov = coverage_by_hour(df_hourly_coverage, tolerance_min=30)

    assert set(cov["hour"]) == set(range(24))
    cov_by_hour = dict(zip(cov["hour"], cov["coverage"]))
    # 5 días con fix a 12:00 y 5 con fix a 18:00 → cobertura máxima en
    # h=12 y h=18, mínima fuera.
    assert cov_by_hour[12] == 1.0
    assert cov_by_hour[18] == 1.0
    assert cov_by_hour[0] == 0.0


def test_pick_reference_hour_returns_argmax(df_hourly_coverage):
    hour, cov = pick_reference_hour(df_hourly_coverage, tolerance_min=30)

    # Empate entre 12 y 18: la función debe devolver la primera de ellas
    # (la de menor índice horario).
    assert hour == 12
    assert set(cov["hour"]) == set(range(24))


def test_build_daily_picks_nearest_to_reference(df_daily_three_fixes_one_day):
    daily = build_daily(
        df_daily_three_fixes_one_day,
        reference_hour_utc=12,
        tolerance_min=120,
    )

    assert list(daily.columns) == [
        "bird_id",
        "date_utc",
        "lat",
        "lon",
        "is_valid",
        "source_event_id",
        "delta_minutes",
    ]
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["bird_id"] == "A"
    assert row["is_valid"] is True or row["is_valid"] == True  # noqa: E712
    assert row["source_event_id"] == 11           # fix de 12:10 elegido
    assert row["delta_minutes"] == 10.0
    assert row["lon"] == 24.50
    assert row["lat"] == 61.20


def test_build_daily_creates_explicit_gap(df_daily_with_gap):
    daily = build_daily(
        df_daily_with_gap, reference_hour_utc=12, tolerance_min=60
    )

    # 3 filas: 01, 02 (hueco), 03.
    assert len(daily) == 3
    dates = pd.to_datetime(daily["date_utc"]).dt.date.tolist()
    assert dates == [
        pd.Timestamp("2010-04-01").date(),
        pd.Timestamp("2010-04-02").date(),
        pd.Timestamp("2010-04-03").date(),
    ]

    gap = daily[daily["date_utc"] == pd.Timestamp("2010-04-02").date()].iloc[0]
    assert bool(gap["is_valid"]) is False
    assert pd.isna(gap["lat"]) and pd.isna(gap["lon"])
    assert pd.isna(gap["source_event_id"])
    assert pd.isna(gap["delta_minutes"])


def test_build_daily_respects_tolerance(df_daily_far_from_reference):
    # Fix a 200 min de la hora de referencia: con tol=120 → inválido.
    daily_strict = build_daily(
        df_daily_far_from_reference,
        reference_hour_utc=12,
        tolerance_min=120,
    )
    assert len(daily_strict) == 1
    assert bool(daily_strict.iloc[0]["is_valid"]) is False

    # Con tol=300 → válido.
    daily_loose = build_daily(
        df_daily_far_from_reference,
        reference_hour_utc=12,
        tolerance_min=300,
    )
    assert bool(daily_loose.iloc[0]["is_valid"]) is True
    assert daily_loose.iloc[0]["delta_minutes"] == 200.0


def test_filter_birds_by_validity_drops_low_coverage(df_daily_two_birds_unequal):
    # Construimos primero la tabla diaria.
    daily = build_daily(
        df_daily_two_birds_unequal,
        reference_hour_utc=12,
        tolerance_min=60,
    )
    out = filter_birds_by_validity(daily, min_valid_days=2)

    assert set(out["bird_id"]) == {"A"}
    assert (out["bird_id"] == "B").sum() == 0
```

- [ ] **Step 4.3: Ejecutar y verificar que fallan**

```bash
uv run pytest tests/test_data_daily.py -v
```

Expected: 6 tests fallan con `NotImplementedError`.

- [ ] **Step 4.4: Commit**

```bash
git add tests/conftest.py tests/test_data_daily.py
git commit -m "Tests unitarios de resample diario"
```

---

## Task 5: Implementar `daily`

**Files:**
- Modify: `src/tfg_aves/data/daily.py`

- [ ] **Step 5.1: Implementar las cuatro funciones de `daily.py`**

Reemplazar el cuerpo de `src/tfg_aves/data/daily.py`:

```python
"""Resample a una fila por (bird_id, date_utc) con huecos explícitos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_by_hour(df: pd.DataFrame, tolerance_min: float) -> pd.DataFrame:
    """Cobertura % de (ave, día) con al menos un fix en ``[h ± tol]``.

    Si la ventana cruza medianoche se considera dentro del día propio
    del fix (no envuelve a días vecinos).
    """
    if tolerance_min < 0:
        raise ValueError("tolerance_min debe ser >= 0")

    df = df.copy()
    df["date_utc"] = df["timestamp"].dt.tz_convert("UTC").dt.date
    bird_days = df[["bird_id", "date_utc"]].drop_duplicates()
    total = len(bird_days)
    if total == 0:
        return pd.DataFrame({"hour": list(range(24)), "coverage": [0.0] * 24})

    minute_of_day = (
        df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    ).to_numpy()
    bird = df["bird_id"].to_numpy()
    day = df["date_utc"].to_numpy()

    coverage = np.zeros(24, dtype=float)
    for hour in range(24):
        target = hour * 60
        within = np.abs(minute_of_day - target) <= tolerance_min
        if not within.any():
            coverage[hour] = 0.0
            continue
        covered = pd.DataFrame(
            {"bird_id": bird[within], "date_utc": day[within]}
        ).drop_duplicates()
        coverage[hour] = len(covered) / total

    return pd.DataFrame({"hour": list(range(24)), "coverage": coverage})


def pick_reference_hour(
    df: pd.DataFrame, tolerance_min: float
) -> tuple[int, pd.DataFrame]:
    """Hora UTC con mayor cobertura (empates: hora menor)."""
    cov = coverage_by_hour(df, tolerance_min)
    best = int(cov.loc[cov["coverage"].idxmax(), "hour"])
    return best, cov


def build_daily(
    df: pd.DataFrame,
    reference_hour_utc: int,
    tolerance_min: float,
) -> pd.DataFrame:
    """Colapsa fixes a una fila por (bird_id, date_utc) con huecos explícitos.

    Para cada ``bird_id`` expande el rango ``[first_date, last_date]`` día a
    día y elige el fix más cercano (en minutos) a ``reference_hour_utc``.
    Deja ``NaN`` si no hay fix en ``[reference_hour_utc ± tolerance_min]``.
    """
    if not 0 <= reference_hour_utc <= 23:
        raise ValueError("reference_hour_utc debe estar en [0, 23]")
    if tolerance_min < 0:
        raise ValueError("tolerance_min debe ser >= 0")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "bird_id",
                "date_utc",
                "lat",
                "lon",
                "is_valid",
                "source_event_id",
                "delta_minutes",
            ]
        )

    work = df.copy()
    work["date_utc"] = work["timestamp"].dt.tz_convert("UTC").dt.date
    work["delta_minutes"] = (
        work["timestamp"]
        - pd.to_datetime(work["date_utc"]).dt.tz_localize("UTC")
        - pd.Timedelta(hours=reference_hour_utc)
    ).dt.total_seconds().div(60).abs()

    # Para cada (bird_id, date_utc) nos quedamos con el fix de menor delta.
    work = work.sort_values(
        ["bird_id", "date_utc", "delta_minutes"], kind="stable"
    )
    chosen = work.drop_duplicates(subset=["bird_id", "date_utc"], keep="first")

    # Reconstruimos el rango calendario completo por ave para introducir
    # huecos explícitos.
    frames = []
    for bird_id, group in chosen.groupby("bird_id", sort=True):
        first = group["date_utc"].min()
        last = group["date_utc"].max()
        all_days = pd.date_range(first, last, freq="D").date
        full = pd.DataFrame({"date_utc": all_days, "bird_id": bird_id})
        merged = full.merge(group, on=["bird_id", "date_utc"], how="left")
        frames.append(merged)
    daily = pd.concat(frames, ignore_index=True)

    within_tol = daily["delta_minutes"] <= tolerance_min
    daily["is_valid"] = within_tol.fillna(False)
    # Anulamos lat/lon/source/delta donde no haya fix válido.
    invalid = ~daily["is_valid"]
    for col in ("lat", "lon", "source_event_id", "delta_minutes"):
        daily.loc[invalid, col] = np.nan

    daily["source_event_id"] = daily["source_event_id"].astype("Int64")
    daily = daily[
        [
            "bird_id",
            "date_utc",
            "lat",
            "lon",
            "is_valid",
            "source_event_id",
            "delta_minutes",
        ]
    ].sort_values(["bird_id", "date_utc"]).reset_index(drop=True)
    return daily


def filter_birds_by_validity(
    df_daily: pd.DataFrame, min_valid_days: int
) -> pd.DataFrame:
    """Descarta individuos cuyo ``count(is_valid) < min_valid_days``."""
    if min_valid_days < 0:
        raise ValueError("min_valid_days debe ser >= 0")

    valid_counts = (
        df_daily.groupby("bird_id")["is_valid"].sum().astype(int)
    )
    kept = valid_counts[valid_counts >= min_valid_days].index
    out = df_daily[df_daily["bird_id"].isin(kept)].reset_index(drop=True)
    return out
```

- [ ] **Step 5.2: Ejecutar tests y verificar verde**

```bash
uv run pytest tests/test_data_daily.py -v
```

Expected: 6 tests pasan.

- [ ] **Step 5.3: Suite completa y linter**

```bash
uv run pytest -q
uv run ruff check src tests
```

Expected: 28 tests verdes (16 previos + 1 smoke + 5 clean/load + 6 daily), ruff sin errores.

- [ ] **Step 5.4: Commit**

```bash
git add src/tfg_aves/data/daily.py
git commit -m "Implementar resample diario con tests"
```

---

## Task 6: Notebook EDA — generar artefactos D1–D4 y C1–C5

**Files:**
- Create: `notebooks/01_eda_o1.py`
- (Generados por el notebook): `reports/figures/o1_*.png`, `reports/tables/o1_*.csv`, `reports/captions/o1_*.md`, `reports/INDEX.md` (actualizado).

El notebook se versiona como `.py` en formato jupytext percent (`# %%` separadores). Jupyter Lab abre automáticamente el `.ipynb` emparejado con el `.py` cuando se selecciona el fichero.

- [ ] **Step 6.1: Crear `notebooks/01_eda_o1.py` con cabecera jupytext**

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # O1 — EDA y decisiones de la pipeline de datos
#
# Cuaderno de exploración para fijar los cuatro umbrales que parametrizan
# `build_o1`: `max_speed_kmh`, `reference_hour_utc`, `tolerance_min` y
# `min_valid_days`. Cada decisión se justifica con figura/tabla vía
# `save_artifact` y queda registrada en `reports/INDEX.md`.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfg_aves.data import (
    build_o1,
    coverage_by_hour,
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
    load_raw,
    pick_reference_hour,
)
from tfg_aves.data.clean import _compute_speed_kmh
from tfg_aves.reporting import save_artifact

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200})
```

- [ ] **Step 6.2: Añadir celda C1 — visión general del dataset**

Añadir al final de `notebooks/01_eda_o1.py`:

```python
# %% [markdown]
# ## C1 — Visión general del dataset

# %%
df_raw = load_raw()
print(f"Filas iniciales: {len(df_raw):,}")
print(f"Individuos: {df_raw['bird_id'].nunique()}")
print(f"Rango temporal: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

per_bird = df_raw.groupby("bird_id").size()
overview = pd.DataFrame(
    {
        "metric": [
            "n_fixes_total",
            "n_birds",
            "fixes_per_bird_median",
            "fixes_per_bird_p10",
            "fixes_per_bird_p90",
            "first_timestamp_utc",
            "last_timestamp_utc",
        ],
        "value": [
            len(df_raw),
            df_raw["bird_id"].nunique(),
            int(per_bird.median()),
            int(per_bird.quantile(0.10)),
            int(per_bird.quantile(0.90)),
            str(df_raw["timestamp"].min()),
            str(df_raw["timestamp"].max()),
        ],
    }
)
save_artifact(
    "dataset-overview",
    objective="o1",
    num=1,
    decision="Caracterización general del dataset Movebank tras carga",
    caption_es=(
        "Métricas agregadas del dataset crudo de Movebank tras la carga "
        "y normalización: número total de fixes GPS, individuos "
        "identificados, mediana e intervalo intercuartílico extendido de "
        "fixes por individuo y rango temporal cubierto."
    ),
    table=overview,
)
```

- [ ] **Step 6.3: Añadir celda C2 — distribución de Δt nativo**

```python
# %% [markdown]
# ## C2 — Distribución de intervalos entre fixes consecutivos

# %%
df_raw_sorted = df_raw.sort_values(["bird_id", "timestamp"])
dt_min = (
    df_raw_sorted.groupby("bird_id")["timestamp"]
    .diff()
    .dt.total_seconds()
    .div(60)
)
dt_min = dt_min.dropna()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(dt_min.clip(upper=600), bins=80, color="#4682B4", edgecolor="white")
ax.set_xlabel("Δt entre fixes consecutivos (min, recortado a 600)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del intervalo nativo de muestreo Movebank")

dt_summary = pd.DataFrame(
    {
        "percentil": ["p10", "p25", "p50", "p75", "p90", "p99"],
        "delta_min": [
            float(dt_min.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
        ],
    }
)
save_artifact(
    "fix-interval-distribution",
    objective="o1",
    num=2,
    decision="Caracterización del muestreo nativo de Movebank",
    caption_es=(
        "Distribución de los intervalos temporales entre fixes consecutivos "
        "de un mismo individuo. Los percentiles asociados (p10–p99) "
        "describen la frecuencia efectiva de muestreo nativa del dataset "
        "tras la carga, antes de aplicar filtros de outliers."
    ),
    fig=fig,
    table=dt_summary,
)
plt.close(fig)
```

- [ ] **Step 6.4: Añadir celda D1 — distribución de velocidades y elección de `max_speed_kmh`**

```python
# %% [markdown]
# ## D1 — Umbral de velocidad para descartar outliers GPS

# %%
df_flag_clean, report_flags = drop_movebank_flags(df_raw)
df_coord_clean, report_coords = drop_invalid_coords_and_dupes(df_flag_clean)

speeds = _compute_speed_kmh(df_coord_clean)
speeds = speeds[~np.isnan(speeds)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(speeds, 0, 300), bins=100, color="#B22222", edgecolor="white")
ax.axvline(120, color="black", linestyle="--", label="Umbral candidato 120 km/h")
ax.set_xlabel("Velocidad entre fixes consecutivos (km/h, recortada a 300)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de velocidades — Larus fuscus")
ax.legend()

speed_summary = pd.DataFrame(
    {
        "percentil": ["p50", "p90", "p95", "p99", "p99.5", "p99.9"],
        "speed_kmh": [
            float(np.quantile(speeds, q))
            for q in (0.50, 0.90, 0.95, 0.99, 0.995, 0.999)
        ],
    }
)
# Anotamos el umbral elegido. El valor final lo decide el autor a la
# vista de la figura y la biología (Larus fuscus alcanza picos de
# ~70-80 km/h con viento favorable; margen 1.5x para errores GPS aislados).
MAX_SPEED_KMH = 120.0
save_artifact(
    "speed-distribution",
    objective="o1",
    num=3,
    decision=f"Umbral de outlier de velocidad fijado en {MAX_SPEED_KMH:.0f} km/h",
    caption_es=(
        "Distribución de velocidades entre fixes GPS consecutivos del "
        "mismo individuo tras descartar marcas de Movebank y coordenadas "
        "inválidas. El umbral elegido (línea discontinua) deja un margen "
        "amplio sobre la velocidad de crucero de Larus fuscus (~50 km/h) "
        "y por encima de su pico observado con viento favorable, evitando "
        "penalizar tracks legítimos."
    ),
    fig=fig,
    table=speed_summary,
)
plt.close(fig)
```

- [ ] **Step 6.5: Añadir celda — aplicar el filtro de velocidad**

```python
# %%
df_clean, report_speed = drop_speed_outliers(df_coord_clean, max_speed_kmh=MAX_SPEED_KMH)
print(f"Tras limpieza: {len(df_clean):,} fixes ({len(df_clean)/len(df_raw):.1%} del original)")
```

- [ ] **Step 6.6: Añadir celda D2 — cobertura horaria y elección de `reference_hour_utc`**

```python
# %% [markdown]
# ## D2 — Hora UTC de referencia para el resample diario

# %%
tolerances = [60, 120, 180]
cov_long = []
for tol in tolerances:
    cov = coverage_by_hour(df_clean, tolerance_min=tol)
    cov["tolerance_min"] = tol
    cov_long.append(cov)
cov_table = pd.concat(cov_long, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 4))
for tol, sub in cov_table.groupby("tolerance_min"):
    ax.plot(sub["hour"], sub["coverage"] * 100, marker="o", label=f"±{tol} min")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hora UTC")
ax.set_ylabel("% (ave, día) con fix en la ventana")
ax.set_title("Cobertura horaria a distintas tolerancias")
ax.legend()
ax.grid(alpha=0.3)

REFERENCE_HOUR_UTC, _ = pick_reference_hour(df_clean, tolerance_min=120)
save_artifact(
    "hourly-coverage",
    objective="o1",
    num=4,
    decision=f"Hora UTC de referencia fijada en {REFERENCE_HOUR_UTC:02d}:00 (cobertura máxima a ±120 min)",
    caption_es=(
        "Cobertura del dataset diaria por hora UTC para tres tolerancias "
        "(±60, ±120 y ±180 minutos). Cada punto indica el porcentaje de "
        "pares (ave, día calendario) con al menos un fix GPS en la "
        "ventana centrada en la hora dada. La hora UTC elegida maximiza "
        "la cobertura, asegurando series diarias completas en la mayoría "
        "de individuos."
    ),
    fig=fig,
    table=cov_table,
)
plt.close(fig)
print(f"Hora de referencia elegida: {REFERENCE_HOUR_UTC:02d}:00 UTC")
```

- [ ] **Step 6.7: Añadir celda D3 — tolerancia y trade-off**

```python
# %% [markdown]
# ## D3 — Tolerancia ± min alrededor de la hora de referencia

# %%
rows = []
for tol in [30, 60, 90, 120, 180, 240]:
    cov = coverage_by_hour(df_clean, tolerance_min=tol)
    cov_at_ref = float(cov.loc[cov["hour"] == REFERENCE_HOUR_UTC, "coverage"].iloc[0])
    # Delta medio: para cada (ave, día) con fix en ventana, calculamos su
    # |Δt| al objetivo y promediamos.
    work = df_clean.copy()
    work["date_utc"] = work["timestamp"].dt.tz_convert("UTC").dt.date
    target = pd.Timedelta(hours=REFERENCE_HOUR_UTC)
    work["delta_min"] = (
        work["timestamp"]
        - pd.to_datetime(work["date_utc"]).dt.tz_localize("UTC")
        - target
    ).dt.total_seconds().div(60).abs()
    within = work[work["delta_min"] <= tol]
    closest = within.loc[
        within.groupby(["bird_id", "date_utc"])["delta_min"].idxmin()
    ]
    rows.append(
        {
            "tolerance_min": tol,
            "coverage_pct": cov_at_ref * 100,
            "delta_mean_min": float(closest["delta_min"].mean()) if len(closest) else float("nan"),
            "delta_median_min": float(closest["delta_min"].median()) if len(closest) else float("nan"),
        }
    )
tol_table = pd.DataFrame(rows)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(tol_table["tolerance_min"], tol_table["coverage_pct"],
         marker="o", color="#1f77b4", label="Cobertura (%)")
ax1.set_xlabel("Tolerancia (min)")
ax1.set_ylabel("Cobertura (%)", color="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(tol_table["tolerance_min"], tol_table["delta_mean_min"],
         marker="s", color="#d62728", label="Δ medio (min)")
ax2.set_ylabel("Δ medio al objetivo (min)", color="#d62728")
fig.suptitle("Trade-off cobertura vs precisión temporal")

TOLERANCE_MIN = 120
save_artifact(
    "tolerance-tradeoff",
    objective="o1",
    num=5,
    decision=f"Tolerancia alrededor de la hora de referencia fijada en ±{TOLERANCE_MIN} min",
    caption_es=(
        "Trade-off entre cobertura del dataset y precisión temporal "
        "del fix elegido a distintas tolerancias alrededor de la hora "
        "UTC de referencia. La tolerancia elegida equilibra una "
        "cobertura alta con un desfase medio aceptable."
    ),
    fig=fig,
    table=tol_table,
)
plt.close(fig)
print(f"Tolerancia elegida: ±{TOLERANCE_MIN} min")
```

- [ ] **Step 6.8: Añadir celda D4 — días válidos por ave y `min_valid_days`**

```python
# %% [markdown]
# ## D4 — Mínimo de días válidos por individuo

# %%
from tfg_aves.data import build_daily

daily_unfiltered = build_daily(
    df_clean,
    reference_hour_utc=REFERENCE_HOUR_UTC,
    tolerance_min=TOLERANCE_MIN,
)
valid_per_bird = (
    daily_unfiltered.groupby("bird_id")["is_valid"].sum().astype(int)
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(valid_per_bird, bins=40, color="#2E8B57", edgecolor="white")
ax.set_xlabel("Días válidos por individuo")
ax.set_ylabel("Número de aves")
ax.set_title("Distribución de días válidos por individuo")
ax.axvline(30, color="black", linestyle="--", label="Umbral candidato 30")
ax.legend()

cuts = [10, 20, 30, 50, 100]
cuts_table = pd.DataFrame(
    {
        "min_valid_days": cuts,
        "n_birds_kept": [int((valid_per_bird >= c).sum()) for c in cuts],
        "pct_birds_kept": [
            (valid_per_bird >= c).mean() * 100 for c in cuts
        ],
    }
)
MIN_VALID_DAYS = 30
save_artifact(
    "valid-days-per-bird",
    objective="o1",
    num=6,
    decision=f"Mínimo de días válidos por individuo fijado en {MIN_VALID_DAYS}",
    caption_es=(
        "Distribución del número de días válidos por individuo en la "
        "tabla diaria, junto al recuento de aves conservadas según "
        "distintos umbrales candidatos. El umbral elegido descarta "
        "individuos con seguimiento insuficiente para alimentar la "
        "cadena de Markov día a día sin penalizar a la mayoría del "
        "dataset."
    ),
    fig=fig,
    table=cuts_table,
)
plt.close(fig)
print(f"Mínimo de días válidos: {MIN_VALID_DAYS}")
```

- [ ] **Step 6.9: Añadir celda C3 — desglose de descartes**

```python
# %% [markdown]
# ## C3 — Desglose acumulado de descartes

# %%
n_initial = len(df_raw)
discards = (
    {"n_initial": n_initial}
    | report_flags
    | report_coords
    | report_speed
    | {"n_clean": len(df_clean)}
)
discard_table = pd.DataFrame(
    [
        {
            "causa": k,
            "n": v,
            "pct_sobre_inicial": v / n_initial * 100 if isinstance(v, int) else float("nan"),
        }
        for k, v in discards.items()
    ]
)
save_artifact(
    "discard-breakdown",
    objective="o1",
    num=7,
    decision="Desglose acumulado de fixes descartados por causa",
    caption_es=(
        "Conteo de fixes descartados en cada fase de la limpieza: marcas "
        "de Movebank (`visible=false` y `manually_marked_outlier`), "
        "coordenadas fuera de rango, duplicados por (individuo, "
        "timestamp) y velocidad imposible. El porcentaje se refiere al "
        "total inicial de fixes leídos del CSV crudo."
    ),
    table=discard_table,
)
```

- [ ] **Step 6.10: Añadir celda C4 — mapa espacial estático**

```python
# %% [markdown]
# ## C4 — Visión geográfica de los fixes limpios

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sample = df_clean.sample(min(20000, len(df_clean)), random_state=42)
ax.scatter(sample["lon"], sample["lat"], s=1, alpha=0.3, color="#444444")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Distribución espacial de los fixes limpios (muestreo aleatorio)")
ax.grid(alpha=0.3)

save_artifact(
    "spatial-overview",
    objective="o1",
    num=8,
    decision="Caracterización espacial del dataset limpio",
    caption_es=(
        "Distribución geográfica de los fixes GPS supervivientes tras "
        "los filtros de O1 (muestra aleatoria de 20 000 puntos). "
        "Permite verificar el dominio espacial del dataset y la "
        "consistencia con las rutas migratorias conocidas de Larus "
        "fuscus entre Europa septentrional y África occidental."
    ),
    fig=fig,
)
plt.close(fig)
```

- [ ] **Step 6.11: Añadir celda — materialización final con `build_o1` y C5**

```python
# %% [markdown]
# ## Materialización final y C5 — longitud de rachas consecutivas

# %%
metrics = build_o1(
    max_speed_kmh=MAX_SPEED_KMH,
    reference_hour_utc=REFERENCE_HOUR_UTC,
    tolerance_min=TOLERANCE_MIN,
    min_valid_days=MIN_VALID_DAYS,
)
print(metrics)

daily_final = pd.read_parquet("data/processed/daily.parquet")

# Longitudes de rachas consecutivas de is_valid=True por ave.
def _streaks(series: pd.Series) -> list[int]:
    streaks = []
    current = 0
    for v in series:
        if v:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


streak_lens: list[int] = []
for _, group in daily_final.groupby("bird_id"):
    streak_lens.extend(_streaks(group["is_valid"].to_numpy()))

streak_table = pd.DataFrame(
    {
        "percentil": ["p10", "p25", "p50", "p75", "p90", "p95", "max"],
        "longitud_dias": [
            int(np.quantile(streak_lens, q)) if streak_lens else 0
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
        ],
    }
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(streak_lens, 0, 200), bins=60, color="#6A5ACD", edgecolor="white")
ax.set_xlabel("Longitud de la racha (días consecutivos válidos)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de rachas consecutivas de días válidos")
save_artifact(
    "streak-length-distribution",
    objective="o1",
    num=9,
    decision="Caracterización de la fragmentación de las series diarias",
    caption_es=(
        "Distribución de la longitud de las rachas de días consecutivos "
        "con fix válido en la tabla diaria final, agregada sobre todos "
        "los individuos supervivientes. Indica la cantidad de "
        "transiciones día-a-día observables sin saltar huecos y "
        "constituye una entrada relevante para el diseño de la cadena de "
        "Markov de O2."
    ),
    fig=fig,
    table=streak_table,
)
plt.close(fig)
```

- [ ] **Step 6.12: Ejecutar el notebook end-to-end**

Ejecuta el notebook como script para verificar que produce los 9 artefactos y los dos parquets:

```bash
uv run jupytext --to notebook notebooks/01_eda_o1.py
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda_o1.ipynb \
    --output notebooks/01_eda_o1.ipynb
```

Expected: ejecución sin errores. `reports/INDEX.md` con 9 nuevas filas O1, `reports/figures/o1_fig0{1..9}_*.png` y `reports/tables/o1_tab0{1..9}_*.csv` presentes.

- [ ] **Step 6.13: Inspección manual de las figuras**

```bash
ls reports/figures/o1_*.png
ls reports/tables/o1_*.csv
ls reports/captions/o1_*.md
cat reports/INDEX.md | tail -12
```

Expected: 9 figuras, 9 tablas, 9 captions, 9 filas nuevas en el INDEX.

Si alguna decisión cambia tras revisar las figuras (p.ej. `MAX_SPEED_KMH` se mueve a 100), edita el notebook, **elimina los artefactos correspondientes** y vuelve a ejecutar — `save_artifact` falla si el fichero ya existe (es deliberado, no usar `overwrite=True` aquí salvo el último materialize del Step 7).

- [ ] **Step 6.14: Commit del notebook y los artefactos**

```bash
git add notebooks/01_eda_o1.py reports/figures/o1_*.png reports/tables/o1_*.csv \
        reports/captions/o1_*.md reports/INDEX.md
git commit -m "Notebook EDA: cobertura, velocidades, días por individuo"
```

---

## Task 7: Orquestación `build_o1` y test de integración

**Files:**
- Modify: `src/tfg_aves/data/build.py`
- Create: `tests/test_data_build.py`

- [ ] **Step 7.1: Test de integración (failing)**

Crear `tests/test_data_build.py`:

```python
"""Test de integración end-to-end de tfg_aves.data.build."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tfg_aves.data import build_o1


def _write_csv(path: Path) -> None:
    header = (
        '"event-id","visible","timestamp","location-long","location-lat",'
        '"manually-marked-outlier","visible","sensor-type",'
        '"individual-taxon-canonical-name","tag-local-identifier",'
        '"individual-local-identifier","study-name"'
    )
    rows = []
    eid = 1
    for day_offset in range(5):
        for hour in (10, 12, 14):
            ts = pd.Timestamp("2010-04-01", tz="UTC") + pd.Timedelta(
                days=day_offset, hours=hour
            )
            rows.append(
                f'"{eid}","true","{ts.strftime("%Y-%m-%d %H:%M:%S.000")}",'
                f'"{24.5 + day_offset * 0.01}","{61.2 + day_offset * 0.01}",'
                f'"","true","gps","Larus fuscus","91732","A","study"'
            )
            eid += 1
    # Una segunda ave con sólo 2 días — debe descartarse con min_valid_days=3.
    for day_offset in range(2):
        ts = pd.Timestamp("2010-04-01 12:00", tz="UTC") + pd.Timedelta(
            days=day_offset
        )
        rows.append(
            f'"{eid}","true","{ts.strftime("%Y-%m-%d %H:%M:%S.000")}",'
            f'"25.0","62.0","","true","gps","Larus fuscus","91732","B","study"'
        )
        eid += 1
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_o1_writes_parquets_and_returns_metrics(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    _write_csv(raw)

    # Forzamos la ruta del CSV crudo vía monkeypatch del default.
    import tfg_aves.data.load as load_mod

    monkeypatch.setattr(load_mod, "RAW_CSV", raw)

    metrics = build_o1(
        max_speed_kmh=200.0,
        reference_hour_utc=12,
        tolerance_min=120,
        min_valid_days=3,
        out_dir=out_dir,
    )

    expected_keys = {
        "n_initial",
        "discarded_movebank_outlier",
        "discarded_visible_false",
        "discarded_invalid_coords",
        "discarded_duplicates",
        "discarded_speed",
        "n_fixes_clean",
        "n_birds_initial",
        "n_birds_kept",
        "n_daily_rows",
        "n_valid_rows",
    }
    assert set(metrics.keys()) == expected_keys

    # Ave B (2 días) descartada, ave A (5 días) conservada.
    assert metrics["n_birds_initial"] == 2
    assert metrics["n_birds_kept"] == 1

    daily_path = out_dir / "daily.parquet"
    fixes_path = out_dir / "fixes_clean.parquet"
    assert daily_path.is_file()
    assert fixes_path.is_file()

    daily = pd.read_parquet(daily_path)
    assert set(daily.columns) == {
        "bird_id",
        "date_utc",
        "lat",
        "lon",
        "is_valid",
        "source_event_id",
        "delta_minutes",
    }
    assert set(daily["bird_id"]) == {"A"}
    assert daily["is_valid"].sum() == 5
```

Ejecutar y verificar fallo:

```bash
uv run pytest tests/test_data_build.py -v
```

Expected: el test falla con `NotImplementedError`.

- [ ] **Step 7.2: Implementar `build_o1`**

Reemplazar el cuerpo de `src/tfg_aves/data/build.py`:

```python
"""Orquestación end-to-end de la pipeline de O1."""
from __future__ import annotations

from pathlib import Path

from ._paths import PROCESSED
from .clean import (
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
)
from .daily import build_daily, filter_birds_by_validity
from .load import load_raw


def build_o1(
    *,
    max_speed_kmh: float,
    reference_hour_utc: int,
    tolerance_min: float,
    min_valid_days: int,
    out_dir: Path = PROCESSED,
) -> dict[str, int]:
    """Compone load → clean → build_daily → filter y materializa parquets.

    Devuelve un dict de métricas con claves fijas (ver tests). Los
    parquets se escriben en ``out_dir``: ``daily.parquet`` (entregable
    principal) y ``fixes_clean.parquet`` (auditoría).
    """
    df_raw = load_raw()
    n_initial = len(df_raw)
    n_birds_initial = df_raw["bird_id"].nunique()

    df, report_flags = drop_movebank_flags(df_raw)
    df, report_coords = drop_invalid_coords_and_dupes(df)
    df, report_speed = drop_speed_outliers(df, max_speed_kmh=max_speed_kmh)
    n_fixes_clean = len(df)

    daily = build_daily(
        df,
        reference_hour_utc=reference_hour_utc,
        tolerance_min=tolerance_min,
    )
    daily = filter_birds_by_validity(daily, min_valid_days=min_valid_days)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily.parquet", index=False)
    df.to_parquet(out_dir / "fixes_clean.parquet", index=False)

    return {
        "n_initial": int(n_initial),
        **{k: int(v) for k, v in report_flags.items()},
        **{k: int(v) for k, v in report_coords.items()},
        **{k: int(v) for k, v in report_speed.items()},
        "n_fixes_clean": int(n_fixes_clean),
        "n_birds_initial": int(n_birds_initial),
        "n_birds_kept": int(daily["bird_id"].nunique()),
        "n_daily_rows": int(len(daily)),
        "n_valid_rows": int(daily["is_valid"].sum()),
    }
```

- [ ] **Step 7.3: Ejecutar y verificar verde**

```bash
uv run pytest tests/test_data_build.py -v
```

Expected: 1 test pasa.

- [ ] **Step 7.4: Suite completa y linter**

```bash
uv run pytest -q
uv run ruff check src tests
```

Expected: 29 tests verdes (28 + 1 integración), ruff sin errores.

- [ ] **Step 7.5: Commit**

```bash
git add src/tfg_aves/data/build.py tests/test_data_build.py
git commit -m "Orquestación build_o1 + test de integración"
```

---

## Task 8: Ejecución final + memoria + ai-log + tag

**Files:**
- Modify: `notebooks/01_eda_o1.py` (ya escrito en Task 6; aquí se re-ejecuta para sellar valores)
- Generated: `data/processed/daily.parquet`, `data/processed/fixes_clean.parquet` (gitignored)
- Create: `reports/memoria/03_o1_datos.md`
- Create: `reports/ai-log/0006-o1-eda-y-pipeline-datos.md`

- [ ] **Step 8.1: Ejecutar `build_o1` con los umbrales finales**

Si los valores de la Task 6 son los definitivos tras revisar las figuras:

```bash
uv run python -c "
from tfg_aves.data import build_o1
metrics = build_o1(
    max_speed_kmh=120.0,
    reference_hour_utc=12,
    tolerance_min=120,
    min_valid_days=30,
)
print(metrics)
"
ls -la data/processed/
```

Expected: dos ficheros parquet creados y el dict de métricas impreso. Si el autor decide otros valores tras la revisión, sustituirlos en este comando y en el notebook.

- [ ] **Step 8.2: Crear `reports/memoria/03_o1_datos.md`**

Inspeccionar primero la plantilla:

```bash
cat reports/memoria/_plantilla.md
```

Crear `reports/memoria/03_o1_datos.md` siguiendo la estructura de la plantilla. Contenido mínimo a redactar:

- **Objetivo:** preparar el dataset de Larus fuscus para alimentar la cadena de Markov día-a-día.
- **Decisiones tomadas:** listar D1–D4 con sus valores definitivos y enlazar las entradas correspondientes del `reports/INDEX.md` por su `ref` (`o1_fig03_speed-distribution`, etc.).
- **Datos descartados:** referir el desglose C3 (`o1_tab07_discard-breakdown`).
- **Caracterización:** referir C1, C2, C4, C5.
- **Salida materializada:** `data/processed/daily.parquet` con N filas y M aves; `data/processed/fixes_clean.parquet` con K filas.
- **Limitaciones conocidas y problemas encontrados:** dejar notas sobre, por ejemplo, individuos con muy poco tracking, picos de velocidad ocasionales, tolerancia que aún deja días sin fix, etc.
- **Conclusiones:** entrada lista para O2 y por qué.

Sin prosa final — son notas para nutrir la sección 3 de la memoria en LaTeX (Task de redacción final, post-O5).

- [ ] **Step 8.3: Crear `reports/ai-log/0006-o1-eda-y-pipeline-datos.md`**

Inspeccionar la convención:

```bash
cat reports/ai-log/README.md
cat reports/ai-log/0005-claude-md-contexto-proyecto.md
```

Crear `reports/ai-log/0006-o1-eda-y-pipeline-datos.md` siguiendo la estructura de las entradas existentes. Tono según política `[[project_ai_use_policy]]` y `[[feedback_ai_log_scope]]`:

- **Tareas asistidas por la IA:** brainstorm de alternativas, andamiaje del notebook EDA, código boilerplate (cálculo haversine vectorizado, filtros pandas), redacción de captions en castellano y propuesta inicial de tests.
- **Decisiones del autor (no de la IA):** alcance diario, hora de referencia por EDA en lugar de a priori, representación de huecos explícitos, umbrales definitivos D1–D4, criterios de aceptación, política de granularidad de commits.
- **Validación humana:** revisión de las nueve figuras, contraste de la distribución de velocidades con literatura ornitológica sobre Larus fuscus, verificación de la consistencia del INDEX y de los tests.

Una sola entrada para toda la fase, no por commit.

- [ ] **Step 8.4: Verificar criterios de aceptación**

```bash
uv run pytest -q
uv run ruff check src tests
ls -la data/processed/ | grep parquet
grep -c '^| O1' reports/INDEX.md   # debería ser 9
ls reports/memoria/03_o1_datos.md
ls reports/ai-log/0006-*.md
```

Expected:
- 29 tests verdes.
- Ruff sin errores.
- `daily.parquet` y `fixes_clean.parquet` presentes.
- 9 entradas O1 en `reports/INDEX.md`.
- Notas de memoria y entrada de ai-log creadas.

- [ ] **Step 8.5: Commit final + tag**

```bash
git add reports/memoria/03_o1_datos.md reports/ai-log/0006-o1-eda-y-pipeline-datos.md
git commit -m "Ejecutar build_o1 con umbrales finales y materializar parquets"
git tag -a v0.1-o1-completo -m "O1 — Preparación de datos GPS completada"
git log --oneline -10
git tag --list
```

Expected: tag `v0.1-o1-completo` creado apuntando al último commit. La cadena de commits de O1 (Tasks 1–8) está en `main` sin merges.

---

## Acceptance summary

Al final del plan:

- 8 commits en `main` que cuadran con los 8 hitos del spec, todos en castellano sin trailer Co-Authored-By.
- 29 tests pasando (16 previos + 13 nuevos: 1 smoke, 1 de `load`, 4 de `clean` — incluye un caso de regresión "sin outliers" sobre los 3 del spec —, 6 de `daily`, 1 de integración).
- `uv run ruff check src tests` sin errores.
- `data/processed/daily.parquet` y `data/processed/fixes_clean.parquet` regenerables con un único `build_o1(...)`.
- 9 entradas O1 en `reports/INDEX.md` con captions castellanos en `reports/captions/`.
- `reports/memoria/03_o1_datos.md` con notas estructuradas listas para la redacción LaTeX post-O5.
- `reports/ai-log/0006-o1-eda-y-pipeline-datos.md` con la entrada de uso de IA para la fase.
- Tag `v0.1-o1-completo` creado.
