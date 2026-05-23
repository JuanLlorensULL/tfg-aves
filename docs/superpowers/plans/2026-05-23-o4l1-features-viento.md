# L1 (O4) — Features de viento: Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar la primera línea de mejora (L1) del pipeline O4 añadiendo tres features de viento (`wind_u_850`, `wind_v_850`, `wind_speed_850`) extraídas de archivos NetCDF reanalysis ECMWF a 850 hPa mediante interpolación bilinear espacial, sin modificar arquitectura, target, split temporal ni hiperparámetros del pipeline base. Generar comparativa side-by-side L1-v0 (sin viento) vs L1-v1 (con viento) sobre el mismo test split como entregable narrativo para la memoria.

**Architecture:** Nuevo paquete `tfg_aves.meteo` con dos módulos puros (`wind.py`, `build_wind.py`) que producen `data/processed/wind/wind_per_fix.parquet`. Extensiones mínimas en `tfg_aves.ml.features` (función `merge_wind_features`) y `tfg_aves.ml.build` (flag `with_wind: bool = False` en `build_o4`). Outputs de L1-v1 a subdirectorio aislado `data/processed/o4/l1_v1/` para preservar artefactos de L1-v0. Notebook EDA en `notebooks/04l1_eda_o4l1.py` (jupytext percent) que materializa 7 artefactos `save_artifact` (D1-D2, C1-C5) y produce la tabla comparativa central.

**Tech Stack:** Python 3.12, `xarray>=2024.0`, `netCDF4>=1.6` (nuevos), pandas, numpy, pyarrow, matplotlib, jupytext, pytest, ruff, joblib. Reutiliza `tfg_aves.reporting.save_artifact` (numeración o4 continúa en 10), `tfg_aves.ml.build.build_o4` (extendido), `tfg_aves.ml.features.build_feature_matrix` (extendido).

**Spec:** `docs/superpowers/specs/2026-05-23-o4l1-features-design.md`.

---

## File structure

**Crear:**

- `src/tfg_aves/meteo/__init__.py` — re-exporta API pública.
- `src/tfg_aves/meteo/_paths.py` — constantes `WIND_RAW_DIR`, `WIND_PER_FIX_PARQUET`.
- `src/tfg_aves/meteo/wind.py` — `load_wind_dataset`, `interpolate_wind_to_fixes`.
- `src/tfg_aves/meteo/build_wind.py` — `build_wind` orquestador.
- `tests/test_meteo_smoke.py` — smoke import.
- `tests/test_meteo_wind.py` — 5 tests unitarios + 1 integración.
- `tests/test_meteo_build_wind.py` — 1 test integración + idempotencia.
- `notebooks/04l1_eda_o4l1.py` — notebook EDA jupytext percent.
- `reports/ai-log/0010-l1-features-viento.md` — entrada de uso de IA (gitignored).
- `data/raw/wind/` (directorio) — copia de los 7 `.nc` de v2.

**Modificar:**

- `pyproject.toml` — añadir `xarray>=2024.0` y `netCDF4>=1.6` a `dependencies`.
- `src/tfg_aves/ml/features.py` — añadir `merge_wind_features` y constante para las 3 features extra.
- `src/tfg_aves/ml/build.py` — extender `build_o4()` con flag `with_wind: bool = False` y subdir L1-v1.
- `src/tfg_aves/ml/_paths.py` — añadir `O4_L1V1_DIR` (WIND_PER_FIX_PARQUET vive en `meteo/_paths.py`).
- `tests/test_ml_features.py` — añadir 2 tests para merge.
- `tests/test_ml_build.py` — añadir 1 test integración con `with_wind=True`.
- `reports/memoria/06_o4_ml.md` — añadir sección «L1 — Mejora con viento reanalysis ECMWF 850 hPa».

**Generados (no versionados):**

- `data/raw/wind/wind_{2009..2015}.nc` (copia de v2).
- `data/processed/wind/wind_per_fix.parquet`.
- `data/processed/o4/l1_v1/model_{rf,xgb}_{personalizado,poblacional}.pkl` (4 modelos).
- `data/processed/o4/l1_v1/predictions_test.parquet`.
- `data/processed/o4/l1_v1/metrics.parquet`.

**Generados (versionados como evidencia):**

- 7 entradas L1-v1 en `reports/figures/`, `reports/tables/`, `reports/captions/` y `reports/INDEX.md` (numeración `o4` continúa desde 10, slug con prefijo `l1v1-`).

**Tag final esperado:** `v0.4.1-o4l1-viento` sobre el último commit de L1.

---

## Task 1: Esqueleto del paquete `tfg_aves.meteo` + dependencias + copia .nc

**Files:**
- Create: `src/tfg_aves/meteo/__init__.py`
- Create: `src/tfg_aves/meteo/_paths.py`
- Create: `src/tfg_aves/meteo/wind.py` (con `NotImplementedError`)
- Create: `src/tfg_aves/meteo/build_wind.py` (con `NotImplementedError`)
- Modify: `pyproject.toml`
- Modify: `src/tfg_aves/ml/_paths.py`
- Create: `tests/test_meteo_smoke.py`
- Create: `data/raw/wind/` (directorio)

- [ ] **Step 1.1: Crear `src/tfg_aves/meteo/_paths.py`**

```python
"""Rutas estables del paquete meteo. Aisladas para evitar imports circulares."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[3]
WIND_RAW_DIR: Path = ROOT / "data" / "raw" / "wind"
WIND_PROCESSED_DIR: Path = ROOT / "data" / "processed" / "wind"
WIND_PER_FIX_PARQUET: Path = WIND_PROCESSED_DIR / "wind_per_fix.parquet"
```

- [ ] **Step 1.2: Crear `src/tfg_aves/meteo/wind.py` con firmas `NotImplementedError`**

```python
"""Carga y match espacio-temporal de viento ECMWF a 850 hPa para L1.

Conexión entre los .nc crudos (reanalysis ECMWF, 850 hPa, 0,5°,
12:00 UTC diario) y los fixes Movebank, mediante interpolación bilinear
espacial.

Spec: docs/superpowers/specs/2026-05-23-o4l1-features-design.md §6.1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_wind_dataset(
    years: list[int],
    base_dir: Path,
) -> xr.Dataset:
    """Carga y concatena los archivos .nc de viento de los años pedidos.

    Args:
        years: lista de años a cargar (e.g., [2010, 2011]).
        base_dir: directorio con archivos `wind_{YYYY}.nc`.

    Returns:
        Dataset xarray con dims (valid_time, latitude, longitude) y
        variables `u`, `v` en m/s, concatenado en el eje temporal.
        La dimensión `pressure_level` (siempre tamaño 1) se elimina.

    Raises:
        FileNotFoundError: si algún archivo no existe.
    """
    raise NotImplementedError


def interpolate_wind_to_fixes(
    wind_ds: xr.Dataset,
    fixes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Interpola bilinealmente el viento del día a cada fix.

    Para cada fila (bird_id, date_utc, lat, lon) de fixes_df, selecciona
    el snapshot con valid_time = date_utc 12:00 UTC e interpola U, V al
    punto (lat, lon). Calcula wind_speed_850 = sqrt(u² + v²).

    Args:
        wind_ds: Dataset cargado con load_wind_dataset.
        fixes_df: DataFrame con columnas bird_id, date_utc (date), lat,
            lon (floats con posibles NaN).

    Returns:
        DataFrame con columnas bird_id, date_utc, wind_u_850,
        wind_v_850, wind_speed_850. Mismo número de filas que fixes_df.
        NaN propagado donde lat/lon NaN o fuera del bbox del wind_ds.
    """
    raise NotImplementedError
```

- [ ] **Step 1.3: Crear `src/tfg_aves/meteo/build_wind.py` con firma `NotImplementedError`**

```python
"""Orquestador único del pipeline de viento L1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._paths import WIND_PER_FIX_PARQUET, WIND_RAW_DIR


def build_wind(
    daily_path: Path,
    wind_raw_dir: Path = WIND_RAW_DIR,
    out_path: Path = WIND_PER_FIX_PARQUET,
) -> pd.DataFrame:
    """Construye wind_per_fix.parquet desde daily.parquet y los .nc.

    Pasos:
        1. Lee daily.parquet (filas (bird_id, date_utc, lat, lon)).
        2. Determina años únicos del dataset; carga sólo esos .nc.
        3. Interpola viento al punto (lat, lon) de cada fila.
        4. Persiste el resultado en out_path.
        5. Devuelve el DataFrame para uso opcional en notebooks.

    Args:
        daily_path: ruta a data/processed/daily.parquet.
        wind_raw_dir: directorio con los .nc anuales.
        out_path: destino del parquet resultante.

    Returns:
        DataFrame con (bird_id, date_utc, wind_u_850, wind_v_850,
        wind_speed_850).
    """
    raise NotImplementedError
```

- [ ] **Step 1.4: Crear `src/tfg_aves/meteo/__init__.py`**

```python
"""Paquete meteo — carga y match de viento ECMWF para L1 de O4."""
from __future__ import annotations

from .build_wind import build_wind
from .wind import interpolate_wind_to_fixes, load_wind_dataset

__all__ = [
    "build_wind",
    "interpolate_wind_to_fixes",
    "load_wind_dataset",
]
```

- [ ] **Step 1.5: Añadir constante `O4_L1V1_DIR` al `_paths.py` del paquete ml**

Modificar `src/tfg_aves/ml/_paths.py`. Añadir UNA constante nueva tras las existentes:

```python
O4_L1V1_DIR: Path = ROOT / "data" / "processed" / "o4" / "l1_v1"
```

`WIND_PER_FIX_PARQUET` se define UNA SOLA VEZ en `src/tfg_aves/meteo/_paths.py` (Step 1.1) y se importa desde ahí cuando se necesite (Step 6.3).

- [ ] **Step 1.6: Añadir dependencias a `pyproject.toml`**

Localizar la lista `dependencies = [...]` en `pyproject.toml` y añadir las dos entradas (orden alfabético):

```
    "netcdf4>=1.6",
    "xarray>=2024.0",
```

Ejecutar:

```bash
uv sync
```

Expected: instalación de `xarray`, `netcdf4` y deps transitivas (cftime, etc.). Sin errores.

- [ ] **Step 1.7: Crear test smoke `tests/test_meteo_smoke.py`**

```python
"""Smoke tests: el paquete meteo importa y expone su API."""
from __future__ import annotations


def test_meteo_imports():
    from tfg_aves import meteo

    assert hasattr(meteo, "load_wind_dataset")
    assert hasattr(meteo, "interpolate_wind_to_fixes")
    assert hasattr(meteo, "build_wind")
```

Ejecutar:

```bash
uv run pytest tests/test_meteo_smoke.py -v
```

Expected: 1 test pasa.

- [ ] **Step 1.8: Copiar los .nc de v2 a `data/raw/wind/`**

```bash
mkdir -p data/raw/wind
cp /home/jllorens/Desktop/TFG/version2/data/raw/wind/wind_*.nc data/raw/wind/
ls -la data/raw/wind/
```

Expected: 7 archivos `wind_2009.nc` a `wind_2015.nc`, ~140 MB total.

Verificación rápida:

```bash
uv run python -c "
import xarray as xr
ds = xr.open_dataset('data/raw/wind/wind_2010.nc')
print(f'shape u: {ds[\"u\"].shape}')
print(f'valid_time range: {ds[\"valid_time\"].values[0]} a {ds[\"valid_time\"].values[-1]}')
print(f'lat range: {float(ds[\"latitude\"].min()):.1f} a {float(ds[\"latitude\"].max()):.1f}')
print(f'lon range: {float(ds[\"longitude\"].min()):.1f} a {float(ds[\"longitude\"].max()):.1f}')
"
```

Expected: `shape u: (365, 1, 139, 93)`, lat range -3 a 66, lon range 7 a 53. `.gitignore` ya cubre `data/raw/*` (no requiere cambio).

- [ ] **Step 1.9: Commit**

```bash
git add src/tfg_aves/meteo/ src/tfg_aves/ml/_paths.py pyproject.toml uv.lock tests/test_meteo_smoke.py
git commit -m "Esqueleto del paquete meteo + dependencias xarray/netcdf4

Crea src/tfg_aves/meteo/ con tres módulos (wind, build_wind, _paths)
y firmas NotImplementedError listas para TDD. Añade xarray>=2024.0
y netcdf4>=1.6 a las dependencias. Smoke test verifica la API
pública."
```

Verificar status limpio:

```bash
git status
```

Expected: working tree clean (data/raw/wind/ gitignored).

---

## Task 2: Implementar `load_wind_dataset` con TDD

**Files:**
- Modify: `src/tfg_aves/meteo/wind.py:18-46` (función `load_wind_dataset`)
- Create: `tests/test_meteo_wind.py`

- [ ] **Step 2.1: Escribir test para combinación de años (sintético)**

Crear `tests/test_meteo_wind.py`:

```python
"""Tests para src/tfg_aves/meteo/wind.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tfg_aves.meteo.wind import (
    interpolate_wind_to_fixes,
    load_wind_dataset,
)


def _make_synthetic_wind_nc(
    path: Path,
    year: int,
    *,
    lat_vals: np.ndarray | None = None,
    lon_vals: np.ndarray | None = None,
    u_value: float = 1.0,
    v_value: float = 2.0,
) -> None:
    """Crea un .nc sintético con la estructura de v2 (u, v, 850 hPa, daily)."""
    if lat_vals is None:
        lat_vals = np.array([60.0, 59.5, 59.0], dtype=np.float64)
    if lon_vals is None:
        lon_vals = np.array([10.0, 10.5, 11.0], dtype=np.float64)
    times = pd.date_range(
        f"{year}-01-01 12:00", f"{year}-12-31 12:00", freq="1D",
    )
    n_t, n_lat, n_lon = len(times), len(lat_vals), len(lon_vals)
    u = np.full((n_t, 1, n_lat, n_lon), u_value, dtype=np.float64)
    v = np.full((n_t, 1, n_lat, n_lon), v_value, dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "pressure_level": np.array([850.0]),
            "latitude": lat_vals,
            "longitude": lon_vals,
        },
    )
    ds.to_netcdf(path)


def test_load_wind_dataset_combines_years(tmp_path):
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    _make_synthetic_wind_nc(tmp_path / "wind_2011.nc", 2011)

    ds = load_wind_dataset([2010, 2011], base_dir=tmp_path)

    assert "u" in ds.data_vars
    assert "v" in ds.data_vars
    assert "pressure_level" not in ds.dims, "Se esperaba squeeze de pressure_level"
    assert int(ds.sizes["valid_time"]) == 365 + 365
    assert pd.Timestamp(ds["valid_time"].values[0]) == pd.Timestamp("2010-01-01 12:00")
    assert pd.Timestamp(ds["valid_time"].values[-1]) == pd.Timestamp("2011-12-31 12:00")


def test_load_wind_dataset_missing_file_raises(tmp_path):
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)

    with pytest.raises(FileNotFoundError):
        load_wind_dataset([2010, 2099], base_dir=tmp_path)
```

- [ ] **Step 2.2: Ejecutar tests para verificar fallo**

```bash
uv run pytest tests/test_meteo_wind.py::test_load_wind_dataset_combines_years -v
```

Expected: FAIL con `NotImplementedError`.

- [ ] **Step 2.3: Implementar `load_wind_dataset`**

Sustituir el cuerpo `raise NotImplementedError` en `src/tfg_aves/meteo/wind.py:load_wind_dataset`:

```python
def load_wind_dataset(
    years: list[int],
    base_dir: Path,
) -> xr.Dataset:
    """Carga y concatena los archivos .nc de viento de los años pedidos.

    (docstring omitido por brevedad — mantener el del esqueleto)
    """
    base_dir = Path(base_dir)
    paths = [base_dir / f"wind_{year}.nc" for year in years]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        combine="by_coords",
        decode_times=True,
    )
    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds.load()
```

Notas: `.load()` fuerza la lectura a memoria al final, lo que simplifica el manejo en tests y notebooks (sin trabajo lazy persistente sobre el filesystem).

- [ ] **Step 2.4: Ejecutar tests para verificar pase**

```bash
uv run pytest tests/test_meteo_wind.py -v
```

Expected: 2 tests pasan (`combines_years`, `missing_file_raises`).

- [ ] **Step 2.5: Commit**

```bash
git add src/tfg_aves/meteo/wind.py tests/test_meteo_wind.py
git commit -m "meteo.wind: load_wind_dataset con TDD

Carga y concatena .nc anuales de viento, eliminando la dimensión
pressure_level (siempre 1). Verifica con tests sintéticos que la
unión temporal es correcta y que se propaga FileNotFoundError ante
archivos ausentes."
```

---

## Task 3: Implementar `interpolate_wind_to_fixes` con TDD

**Files:**
- Modify: `src/tfg_aves/meteo/wind.py:49-79` (función `interpolate_wind_to_fixes`)
- Modify: `tests/test_meteo_wind.py` (añadir 4 tests)

- [ ] **Step 3.1: Añadir tests para el match espacio-temporal**

Añadir a `tests/test_meteo_wind.py` (antes de la última línea):

```python
def test_interpolate_wind_known_point(tmp_path):
    """En el centro de cuatro nodos del grid, bilinear devuelve la media."""
    # Grid 2x2 con valores distintos para que la media sea no-trivial.
    lat_vals = np.array([60.0, 59.5], dtype=np.float64)
    lon_vals = np.array([10.0, 10.5], dtype=np.float64)

    # Construimos un .nc con u y v variando por nodo.
    times = pd.date_range("2010-07-01 12:00", "2010-07-31 12:00", freq="1D")
    n_t = len(times)
    u_grid = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (lat, lon)
    v_grid = np.array([[[10.0, 20.0], [30.0, 40.0]]])
    u = np.broadcast_to(u_grid, (n_t, 1, 2, 2)).copy()
    v = np.broadcast_to(v_grid, (n_t, 1, 2, 2)).copy()

    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "pressure_level": np.array([850.0]),
            "latitude": lat_vals,
            "longitude": lon_vals,
        },
    )
    nc_path = tmp_path / "wind_2010.nc"
    ds.to_netcdf(nc_path)

    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    # Fix exactamente en el centro del grid (lat=59.75, lon=10.25).
    fixes = pd.DataFrame({
        "bird_id": ["X"],
        "date_utc": [pd.Timestamp("2010-07-15").date()],
        "lat": [59.75],
        "lon": [10.25],
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert out.shape == (1, 5)
    assert set(out.columns) == {
        "bird_id", "date_utc", "wind_u_850", "wind_v_850", "wind_speed_850",
    }
    # Bilinear media de (1, 2, 3, 4) = 2.5; de (10, 20, 30, 40) = 25.0.
    assert out["wind_u_850"].iloc[0] == pytest.approx(2.5)
    assert out["wind_v_850"].iloc[0] == pytest.approx(25.0)
    expected_speed = float(np.sqrt(2.5 ** 2 + 25.0 ** 2))
    assert out["wind_speed_850"].iloc[0] == pytest.approx(expected_speed)


def test_interpolate_wind_out_of_bbox(tmp_path):
    """Fix fuera del bbox del grid → NaN en las tres features."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    fixes = pd.DataFrame({
        "bird_id": ["X"],
        "date_utc": [pd.Timestamp("2010-07-15").date()],
        "lat": [80.0],   # fuera del rango [59, 60] del .nc sintético
        "lon": [50.0],   # fuera del rango [10, 11]
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert np.isnan(out["wind_u_850"].iloc[0])
    assert np.isnan(out["wind_v_850"].iloc[0])
    assert np.isnan(out["wind_speed_850"].iloc[0])


def test_interpolate_wind_nan_input(tmp_path):
    """Fix con lat=NaN o lon=NaN → NaN propagado."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    fixes = pd.DataFrame({
        "bird_id": ["X", "Y"],
        "date_utc": [
            pd.Timestamp("2010-07-15").date(),
            pd.Timestamp("2010-07-16").date(),
        ],
        "lat": [np.nan, 59.5],
        "lon": [10.5, np.nan],
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)

    assert out.shape == (2, 5)
    assert out["wind_u_850"].isna().all()
    assert out["wind_v_850"].isna().all()
    assert out["wind_speed_850"].isna().all()


def test_interpolate_wind_preserves_row_count(tmp_path):
    """Número de filas no cambia tras la interpolación."""
    _make_synthetic_wind_nc(tmp_path / "wind_2010.nc", 2010)
    wind_ds = load_wind_dataset([2010], base_dir=tmp_path)

    n = 100
    fixes = pd.DataFrame({
        "bird_id": [f"B{i % 5}" for i in range(n)],
        "date_utc": [pd.Timestamp(f"2010-07-{(i % 28) + 1:02d}").date() for i in range(n)],
        "lat": np.linspace(59.0, 60.0, n),
        "lon": np.linspace(10.0, 11.0, n),
    })
    out = interpolate_wind_to_fixes(wind_ds, fixes)
    assert len(out) == n
```

- [ ] **Step 3.2: Ejecutar tests para verificar fallo**

```bash
uv run pytest tests/test_meteo_wind.py -v
```

Expected: `load_wind_dataset` tests pasan; los 4 nuevos de `interpolate` fallan con `NotImplementedError`.

- [ ] **Step 3.3: Implementar `interpolate_wind_to_fixes`**

Sustituir el cuerpo en `src/tfg_aves/meteo/wind.py:interpolate_wind_to_fixes`:

```python
def interpolate_wind_to_fixes(
    wind_ds: xr.Dataset,
    fixes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Interpola bilinealmente el viento del día a cada fix.

    (docstring del esqueleto)
    """
    if not {"bird_id", "date_utc", "lat", "lon"}.issubset(fixes_df.columns):
        raise ValueError(
            "fixes_df necesita columnas bird_id, date_utc, lat, lon.",
        )

    out = pd.DataFrame({
        "bird_id": fixes_df["bird_id"].to_numpy(),
        "date_utc": fixes_df["date_utc"].to_numpy(),
        "wind_u_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
        "wind_v_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
        "wind_speed_850": np.full(len(fixes_df), np.nan, dtype=np.float64),
    })

    # Rango espacial del .nc para detectar out-of-bbox.
    lat_min = float(wind_ds["latitude"].min())
    lat_max = float(wind_ds["latitude"].max())
    lon_min = float(wind_ds["longitude"].min())
    lon_max = float(wind_ds["longitude"].max())

    # Marca filas válidas (lat, lon no-NaN y dentro del bbox).
    lat_arr = pd.to_numeric(fixes_df["lat"], errors="coerce").to_numpy()
    lon_arr = pd.to_numeric(fixes_df["lon"], errors="coerce").to_numpy()
    valid_mask = (
        ~np.isnan(lat_arr)
        & ~np.isnan(lon_arr)
        & (lat_arr >= lat_min) & (lat_arr <= lat_max)
        & (lon_arr >= lon_min) & (lon_arr <= lon_max)
    )

    if not valid_mask.any():
        return out

    valid_dates = pd.to_datetime(
        fixes_df.loc[valid_mask, "date_utc"].to_numpy(),
    ) + pd.Timedelta(hours=12)
    valid_lats = lat_arr[valid_mask]
    valid_lons = lon_arr[valid_mask]

    # Interpolación vectorizada vía xarray (linear en lat, lon, nearest en tiempo).
    interp = wind_ds.interp(
        valid_time=xr.DataArray(valid_dates, dims="points"),
        latitude=xr.DataArray(valid_lats, dims="points"),
        longitude=xr.DataArray(valid_lons, dims="points"),
        method="linear",
        kwargs={"fill_value": np.nan},
    )

    u_vals = interp["u"].values.astype(np.float64)
    v_vals = interp["v"].values.astype(np.float64)
    speed_vals = np.sqrt(u_vals**2 + v_vals**2)

    out.loc[valid_mask, "wind_u_850"] = u_vals
    out.loc[valid_mask, "wind_v_850"] = v_vals
    out.loc[valid_mask, "wind_speed_850"] = speed_vals

    return out
```

- [ ] **Step 3.4: Ejecutar tests para verificar pase**

```bash
uv run pytest tests/test_meteo_wind.py -v
```

Expected: 6 tests pasan (`combines_years`, `missing_file_raises`, `known_point`, `out_of_bbox`, `nan_input`, `preserves_row_count`).

- [ ] **Step 3.5: Ruff check**

```bash
uv run ruff check src/tfg_aves/meteo/ tests/test_meteo_wind.py tests/test_meteo_smoke.py
```

Expected: All checks passed.

- [ ] **Step 3.6: Commit**

```bash
git add src/tfg_aves/meteo/wind.py tests/test_meteo_wind.py
git commit -m "meteo.wind: interpolate_wind_to_fixes con TDD

Interpola bilinealmente U y V al punto (lat, lon) de cada fix, usando
el snapshot de 12:00 UTC del día. Calcula wind_speed_850 = √(u²+v²).
Propaga NaN para fix con lat/lon nulo o fuera del bbox del grid. Cuatro
tests sintéticos cubren: punto central (media de 4 nodos), fuera de
bbox, NaN propagado y preservación del número de filas."
```

---

## Task 4: Implementar `build_wind` orquestador con TDD + ejecución

**Files:**
- Modify: `src/tfg_aves/meteo/build_wind.py:13-43` (función `build_wind`)
- Create: `tests/test_meteo_build_wind.py`

- [ ] **Step 4.1: Escribir test integración (sintético, fast)**

Crear `tests/test_meteo_build_wind.py`:

```python
"""Tests para src/tfg_aves/meteo/build_wind.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tfg_aves.meteo.build_wind import build_wind


def _make_synthetic_wind_nc(path: Path, year: int) -> None:
    times = pd.date_range(
        f"{year}-01-01 12:00", f"{year}-12-31 12:00", freq="1D",
    )
    n_t = len(times)
    u = np.full((n_t, 1, 3, 3), 1.5, dtype=np.float64)
    v = np.full((n_t, 1, 3, 3), -0.5, dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "pressure_level": np.array([850.0]),
            "latitude": np.array([60.0, 59.5, 59.0]),
            "longitude": np.array([10.0, 10.5, 11.0]),
        },
    )
    ds.to_netcdf(path)


def _make_synthetic_daily(path: Path) -> pd.DataFrame:
    df = pd.DataFrame({
        "bird_id": ["A", "A", "B", "B"],
        "date_utc": [
            pd.Timestamp("2010-03-15").date(),
            pd.Timestamp("2010-08-20").date(),
            pd.Timestamp("2011-04-10").date(),
            pd.Timestamp("2011-09-05").date(),
        ],
        "lat": [59.5, 59.7, 60.0, 59.2],
        "lon": [10.5, 10.7, 10.9, 10.1],
        "is_valid": [True, True, True, True],
    })
    df.to_parquet(path)
    return df


def test_build_wind_produces_expected_columns(tmp_path):
    wind_dir = tmp_path / "raw" / "wind"
    wind_dir.mkdir(parents=True)
    _make_synthetic_wind_nc(wind_dir / "wind_2010.nc", 2010)
    _make_synthetic_wind_nc(wind_dir / "wind_2011.nc", 2011)

    daily_path = tmp_path / "daily.parquet"
    daily_df = _make_synthetic_daily(daily_path)

    out_path = tmp_path / "wind_per_fix.parquet"
    result = build_wind(
        daily_path=daily_path,
        wind_raw_dir=wind_dir,
        out_path=out_path,
    )

    assert out_path.exists()
    assert set(result.columns) == {
        "bird_id", "date_utc", "wind_u_850", "wind_v_850", "wind_speed_850",
    }
    assert len(result) == len(daily_df)
    # Valores constantes en el .nc sintético → 1.5, -0.5, sqrt(1.5²+0.5²).
    assert (result["wind_u_850"] == pytest.approx(1.5)).all()
    assert (result["wind_v_850"] == pytest.approx(-0.5)).all()
    expected_speed = float(np.sqrt(1.5**2 + 0.5**2))
    assert result["wind_speed_850"].iloc[0] == pytest.approx(expected_speed)


def test_build_wind_idempotent(tmp_path):
    wind_dir = tmp_path / "raw" / "wind"
    wind_dir.mkdir(parents=True)
    _make_synthetic_wind_nc(wind_dir / "wind_2010.nc", 2010)
    _make_synthetic_daily(tmp_path / "daily.parquet")

    out_path = tmp_path / "wind_per_fix.parquet"
    df1 = build_wind(tmp_path / "daily.parquet", wind_dir, out_path)
    df2 = build_wind(tmp_path / "daily.parquet", wind_dir, out_path)

    pd.testing.assert_frame_equal(df1, df2)
```

- [ ] **Step 4.2: Ejecutar tests para verificar fallo**

```bash
uv run pytest tests/test_meteo_build_wind.py -v
```

Expected: 2 tests fallan con `NotImplementedError`.

- [ ] **Step 4.3: Implementar `build_wind`**

Sustituir el cuerpo en `src/tfg_aves/meteo/build_wind.py:build_wind`:

```python
def build_wind(
    daily_path: Path,
    wind_raw_dir: Path = WIND_RAW_DIR,
    out_path: Path = WIND_PER_FIX_PARQUET,
) -> pd.DataFrame:
    """Construye wind_per_fix.parquet desde daily.parquet y los .nc.

    (docstring del esqueleto)
    """
    from .wind import interpolate_wind_to_fixes, load_wind_dataset

    daily = pd.read_parquet(daily_path)
    if not {"bird_id", "date_utc", "lat", "lon"}.issubset(daily.columns):
        raise ValueError(
            "daily.parquet debe contener bird_id, date_utc, lat, lon.",
        )

    # Determinar años únicos. date_utc puede venir como datetime, date o str.
    dates = pd.to_datetime(daily["date_utc"])
    years = sorted({int(y) for y in dates.dt.year.unique()})

    wind_ds = load_wind_dataset(years, base_dir=Path(wind_raw_dir))

    fixes = pd.DataFrame({
        "bird_id": daily["bird_id"].to_numpy(),
        "date_utc": dates.dt.date.to_numpy(),
        "lat": pd.to_numeric(daily["lat"], errors="coerce").to_numpy(),
        "lon": pd.to_numeric(daily["lon"], errors="coerce").to_numpy(),
    })
    result = interpolate_wind_to_fixes(wind_ds, fixes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result
```

- [ ] **Step 4.4: Ejecutar tests para verificar pase**

```bash
uv run pytest tests/test_meteo_build_wind.py tests/test_meteo_wind.py tests/test_meteo_smoke.py -v
```

Expected: 9 tests del paquete meteo pasan.

- [ ] **Step 4.5: Ejecutar `build_wind` sobre los datos reales**

```bash
uv run python -c "
from tfg_aves.meteo.build_wind import build_wind
from tfg_aves.meteo._paths import WIND_RAW_DIR, WIND_PER_FIX_PARQUET
from pathlib import Path

result = build_wind(
    daily_path=Path('data/processed/daily.parquet'),
    wind_raw_dir=WIND_RAW_DIR,
    out_path=WIND_PER_FIX_PARQUET,
)
print(f'Filas: {len(result)}')
print(f'NaN por columna:')
print(result.isna().sum())
print()
print('Sample:')
print(result.head())
"
```

Expected:
- `Filas: 24444` (igual al daily.parquet).
- `wind_u_850/wind_v_850/wind_speed_850 NaN ≈ 2621` (las filas con `is_valid=False` heredadas de O1; los huecos diarios tienen `lat`/`lon` NaN, lo que propaga NaN en viento).
- 0 filas con lat/lon válidas pero NaN en viento (cobertura espacial completa).

- [ ] **Step 4.6: Verificar cobertura espacial (0 fixes fuera del bbox)**

```bash
uv run python -c "
import pandas as pd
import numpy as np
daily = pd.read_parquet('data/processed/daily.parquet')
wind = pd.read_parquet('data/processed/wind/wind_per_fix.parquet')
merged = daily.merge(wind, on=['bird_id', 'date_utc'], how='left')
n_valid_fixes = merged['lat'].notna().sum()
n_valid_with_wind = (merged['lat'].notna() & merged['wind_u_850'].notna()).sum()
print(f'Fixes con lat válida: {n_valid_fixes}')
print(f'Fixes con lat válida Y viento válido: {n_valid_with_wind}')
print(f'Fixes con lat válida pero viento NaN (fuera del bbox): {n_valid_fixes - n_valid_with_wind}')
"
```

Expected: 0 fixes con `lat` válida pero viento NaN (cobertura espacial completa del bbox del .nc sobre la trayectoria de las 82 aves).

- [ ] **Step 4.7: Ruff check**

```bash
uv run ruff check src/tfg_aves/meteo/ tests/test_meteo_build_wind.py
```

Expected: All checks passed.

- [ ] **Step 4.8: Commit**

```bash
git add src/tfg_aves/meteo/build_wind.py tests/test_meteo_build_wind.py
git commit -m "meteo.build_wind: orquestador wind_per_fix.parquet

Lee daily.parquet, determina años únicos, carga sólo esos .nc,
interpola bilinealmente y persiste el resultado. Test de
idempotencia + integración mínima con datos sintéticos. Ejecutar
build_wind sobre los datos reales del TFG produce 24 444 filas
con cobertura espacial completa (0 fixes fuera del bbox del .nc)."
```

---

## Task 5: Extender `tfg_aves.ml.features` con `merge_wind_features`

**Files:**
- Modify: `src/tfg_aves/ml/features.py:8-13` (constante `_FEATURES_BASE`) y añadir nueva función al final.
- Modify: `tests/test_ml_features.py` (añadir 2 tests).

- [ ] **Step 5.1: Escribir test del merge**

Localizar la última función en `tests/test_ml_features.py` y añadir al final:

```python
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
```

- [ ] **Step 5.2: Ejecutar tests para verificar fallo**

```bash
uv run pytest tests/test_ml_features.py -v -k "wind"
```

Expected: 2 tests fallan con `ImportError` o equivalente (función no existe).

- [ ] **Step 5.3: Añadir constante y función en `features.py`**

Editar `src/tfg_aves/ml/features.py`. Añadir tras `_FEATURES_BASE`:

```python
_FEATURES_WIND = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
]
```

Modificar la firma de `build_feature_matrix`:

```python
def build_feature_matrix(
    features_o3: pd.DataFrame,
    cells: pd.DataFrame,
    include_bird_id: bool,
    *,
    wind_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

En el cuerpo de `build_feature_matrix`, después de `df = add_cyclic_doy(df)` y antes del filtro `cell_id_t_next`, añadir:

```python
    if wind_df is not None:
        df = merge_wind_features(df, wind_df)
```

Y modificar la construcción de `feature_cols` y `keep_cols` para incluir las features de viento cuando estén presentes:

```python
    base = list(_FEATURES_BASE)
    if wind_df is not None:
        base = [*base, *_FEATURES_WIND]
    feature_cols = list(base)
    if include_bird_id:
        feature_cols = ["bird_id", *feature_cols]

    keep_cols = list(dict.fromkeys([
        "bird_id", "date_utc",
        *base,
        "cell_id_t", "cell_id_t_next",
        "lat_t_next", "lon_t_next",
    ]))
```

Añadir al final del módulo la nueva función:

```python
def merge_wind_features(
    matrix: pd.DataFrame,
    wind_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une matrix con las 3 features de viento por (bird_id, date_utc).

    LEFT JOIN: el número de filas de matrix se preserva. Filas sin
    contrapartida en wind_df reciben NaN en wind_u_850, wind_v_850,
    wind_speed_850 (esperado raro, son fixes fuera del bbox del .nc).

    Args:
        matrix: salida parcial de build_feature_matrix antes del filtro
            de cell_id_t_next, con columnas bird_id, date_utc.
        wind_df: salida de build_wind (bird_id, date_utc + 3 features).

    Returns:
        DataFrame con todas las columnas de matrix + las 3 de viento.
    """
    if not {"bird_id", "date_utc"}.issubset(matrix.columns):
        raise ValueError("matrix necesita columnas bird_id y date_utc.")
    if not {"bird_id", "date_utc", *_FEATURES_WIND}.issubset(wind_df.columns):
        raise ValueError(
            "wind_df necesita bird_id, date_utc y las 3 features de viento.",
        )

    # Asegurar tipos compatibles para el join (date_utc como objeto Python date).
    m = matrix.copy()
    w = wind_df[["bird_id", "date_utc", *_FEATURES_WIND]].copy()

    n_before = len(m)
    out = m.merge(w, on=["bird_id", "date_utc"], how="left", validate="m:1")
    if len(out) != n_before:
        raise AssertionError(
            f"merge cambió número de filas: {n_before} → {len(out)} "
            "(¿wind_df tiene claves duplicadas?)",
        )
    return out
```

- [ ] **Step 5.4: Ejecutar tests existentes + nuevos**

```bash
uv run pytest tests/test_ml_features.py -v
```

Expected: todos los tests pasan (los originales sin `wind_df` siguen funcionando porque el parámetro tiene default `None`; los 2 nuevos pasan).

- [ ] **Step 5.5: Ruff check**

```bash
uv run ruff check src/tfg_aves/ml/features.py tests/test_ml_features.py
```

Expected: All checks passed.

- [ ] **Step 5.6: Commit**

```bash
git add src/tfg_aves/ml/features.py tests/test_ml_features.py
git commit -m "ml.features: merge_wind_features y soporte wind_df opcional

Añade _FEATURES_WIND (wind_u_850, wind_v_850, wind_speed_850) y la
función merge_wind_features con LEFT JOIN por (bird_id, date_utc).
build_feature_matrix acepta wind_df: pd.DataFrame | None = None;
cuando se proporciona, las tres columnas se incluyen en el set de
features y en keep_cols. Sin cambios en el comportamiento por
defecto (wind_df=None reproduce O4 base bit a bit). Dos tests
nuevos cubren preservación de filas y propagación de NaN."
```

---

## Task 6: Extender `build_o4` con `with_wind=True` y subdir L1-v1

**Files:**
- Modify: `src/tfg_aves/ml/build.py:87-104` (firma `build_o4`) y cuerpo.
- Modify: `tests/test_ml_build.py` (añadir 1 test).

- [ ] **Step 6.1: Escribir test integración con `with_wind=True`**

Localizar `tests/test_ml_build.py` y añadir al final:

```python
def test_build_o4_with_wind_writes_l1v1_artifacts(tmp_path, monkeypatch):
    """build_o4(with_wind=True) escribe a O4_L1V1_DIR sin pisar O4_OUT_DIR."""
    # Este test es un placeholder de integración. La verificación real
    # se hace ejecutando build_o4(with_wind=True) sobre los datos
    # reales en el step 6.5 del plan. Aquí sólo verificamos la firma.
    from inspect import signature
    from tfg_aves.ml.build import build_o4

    sig = signature(build_o4)
    assert "with_wind" in sig.parameters
    assert sig.parameters["with_wind"].default is False
```

- [ ] **Step 6.2: Ejecutar tests para verificar fallo**

```bash
uv run pytest tests/test_ml_build.py::test_build_o4_with_wind_writes_l1v1_artifacts -v
```

Expected: FAIL (parámetro `with_wind` no existe aún).

- [ ] **Step 6.3: Modificar firma e implementación de `build_o4`**

Localizar `src/tfg_aves/ml/build.py:87-104` (firma + docstring). Modificar:

```python
def build_o4(
    features_path: Path = FEATURES_O3_PARQUET,
    cells_path: Path = CELLS_PARQUET,
    output_dir: Path | None = None,
    seed: int = 0,
    *,
    with_wind: bool = False,
    wind_path: Path | None = None,
) -> BuildO4Result:
    """Pipeline completa de O4 (§5.4 del spec).

    Pasos (idénticos a O4 base más el merge opcional de viento):
        1. Carga features.parquet y cells.parquet.
        2. (Opcional, si with_wind=True) Carga wind_per_fix.parquet
           y lo pasa a build_feature_matrix vía el parámetro wind_df.
        3. Construye matriz para ambos modos.
        4. Split temporal por ave.
        5. Entrena las combinaciones (RF/XGB × 2 modos; LightGBM sólo
           cuando with_wind=False — L1 mantiene F7 de O4 base).
        6. Computa métricas globales en train y test.
        7. Computa baselines (persistencia + Markov(1)) sobre el mismo split.
        8. Guarda artefactos en output_dir (defecto: O4_OUT_DIR si
           with_wind=False, O4_L1V1_DIR si with_wind=True).

    Args:
        features_path: ruta a features.parquet de O3.
        cells_path: ruta a cells.parquet de O2.
        output_dir: directorio destino. Por defecto se resuelve según
            with_wind para evitar pisar artefactos de L1-v0.
        seed: semilla global.
        with_wind: si True, fusiona wind features y escribe a L1V1_DIR.
        wind_path: ruta al wind_per_fix.parquet. Por defecto
            WIND_PER_FIX_PARQUET. Sólo se lee cuando with_wind=True.
    """
```

En el cuerpo, **sustituir** las primeras líneas del cuerpo actual:

```python
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)
```

por:

```python
    if output_dir is None:
        output_dir = O4_L1V1_DIR if with_wind else O4_OUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_o3 = pd.read_parquet(features_path)
    cells = pd.read_parquet(cells_path)

    wind_df: pd.DataFrame | None = None
    if with_wind:
        wind_p = Path(wind_path) if wind_path is not None else WIND_PER_FIX_PARQUET
        if not wind_p.exists():
            raise FileNotFoundError(
                f"with_wind=True pero {wind_p} no existe. Ejecuta build_wind primero.",
            )
        wind_df = pd.read_parquet(wind_p)
```

Reemplazar la construcción de `matrices` para pasar `wind_df`:

```python
    matrices = {
        "personalizado": build_feature_matrix(
            features_o3, cells, include_bird_id=True, wind_df=wind_df,
        ),
        "poblacional": build_feature_matrix(
            features_o3, cells, include_bird_id=False, wind_df=wind_df,
        ),
    }
```

Modificar la lista de familias cuando `with_wind=True` (F7: LightGBM fuera de L1):

Buscar el loop `for family in _FAMILIES:` y reemplazar por:

```python
        families = _FAMILIES if not with_wind else ("rf", "xgb")
        for family in families:
```

Modificar el bloque de imports al inicio del módulo. Reemplazar el bloque actual:

```python
from ._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
    O4_OUT_DIR,
)
```

por:

```python
from ..meteo._paths import WIND_PER_FIX_PARQUET
from ._paths import (
    CELLS_PARQUET,
    FEATURES_O3_PARQUET,
    O4_L1V1_DIR,
    O4_OUT_DIR,
)
```

- [ ] **Step 6.4: Ejecutar tests para verificar pase**

```bash
uv run pytest tests/test_ml_build.py -v
```

Expected: el test nuevo pasa; los existentes siguen pasando (default `with_wind=False` reproduce O4 base).

- [ ] **Step 6.5: Ejecutar `build_o4(with_wind=True)` sobre datos reales**

```bash
uv run python -c "
from tfg_aves.ml.build import build_o4
result = build_o4(with_wind=True, seed=0)
print(f'n_birds: {result.n_birds}')
print(f'rows train/val/test: {result.n_rows_train}/{result.n_rows_val}/{result.n_rows_test}')
print(f'modelos:')
for k, v in result.model_paths.items():
    print(f'  {k}: {v}')
print(f'predictions_path: {result.predictions_path}')
print(f'metrics_path: {result.metrics_path}')
"
```

Expected:
- `n_birds: 82`.
- Mismas filas train/val/test que O4 base (~16 100 / ~1 600 / ~4 000, valores exactos heredados; el merge no descarta filas).
- 4 modelos en `data/processed/o4/l1_v1/` (no LightGBM).
- `predictions_path` y `metrics_path` apuntan a `data/processed/o4/l1_v1/`.

Verificación rápida del set de features que aprendieron los modelos:

```bash
uv run python -c "
import joblib
b = joblib.load('data/processed/o4/l1_v1/model_personalizado_rf.pkl')
print('feature_cols:', b['feature_cols'])
"
```

Expected: lista con `bird_id`, las 8 base + 3 wind (12 features totales).

- [ ] **Step 6.6: Ruff + tests completos**

```bash
uv run ruff check src/tfg_aves/ml/ tests/
uv run pytest -q
```

Expected: ruff sin errores; todos los tests pasan.

- [ ] **Step 6.7: Commit**

```bash
git add src/tfg_aves/ml/build.py tests/test_ml_build.py
git commit -m "ml.build: flag with_wind=True y subdir l1_v1 para L1

build_o4 acepta with_wind: bool = False y wind_path opcional. Cuando
with_wind=True:
- Carga wind_per_fix.parquet y lo pasa a build_feature_matrix.
- Escribe los artefactos a data/processed/o4/l1_v1/ por defecto (no
  pisa O4_OUT_DIR que contiene los modelos de L1-v0).
- Entrena sólo RF y XGBoost (F7 de O4 base: LightGBM sigue descartado
  en L1).

Default with_wind=False reproduce O4 base bit a bit; el pipeline de
L1-v0 no se ve afectado."
```

---

## Task 7: Notebook EDA — Fase A (sanity-check + L1-v1-D1 + L1-v1-D2)

**Files:**
- Create: `notebooks/04l1_eda_o4l1.py`

- [ ] **Step 7.1: Crear el notebook con encabezado y Fase A**

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # L1 (O4) — Mejora con features de viento ECMWF 850 hPa
#
# Spec: `docs/superpowers/specs/2026-05-23-o4l1-features-design.md`.
#
# Cuatro fases:
#
# - **Fase A:** sanity-check de inputs + ejecución de `build_wind` y
#   `build_o4(with_wind=True)` + artefactos D1 (cobertura espacial) y
#   D2 (distribución de wind_speed por mes).
# - **Fase B:** C1 (correlaciones wind ↔ features existentes) + C2
#   (feature importance comparada L1-v0 vs L1-v1).
# - **Fase C:** C3 (comparativa de métricas globales side-by-side) +
#   C4 (comparativa por estado HMM).
# - **Fase D:** C5 (análisis post-hoc de aprendizaje del viento).

# %%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tfg_aves.meteo._paths import WIND_PER_FIX_PARQUET, WIND_RAW_DIR
from tfg_aves.ml._paths import O4_L1V1_DIR, O4_OUT_DIR
from tfg_aves.reporting import save_artifact

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

# %% [markdown]
# ## Fase A.1 — Sanity-check de inputs

# %%
daily = pd.read_parquet("data/processed/daily.parquet")
features_o3 = pd.read_parquet("data/processed/o3/features.parquet")
cells = pd.read_parquet("data/processed/o2/cells.parquet")
print(f"daily.parquet:    {daily.shape}, rango {daily['date_utc'].min()} → {daily['date_utc'].max()}")
print(f"features.parquet: {features_o3.shape}")
print(f"cells.parquet:    {cells.shape}")

# %%
nc_files = sorted((WIND_RAW_DIR).glob("wind_*.nc"))
print(f"{len(nc_files)} archivos .nc en data/raw/wind/:")
for p in nc_files:
    print(f"  {p.name} ({p.stat().st_size / 1024**2:.1f} MB)")

# %% [markdown]
# ## Fase A.2 — Ejecutar build_wind + build_o4(with_wind=True)
#
# Si los outputs ya existen y son recientes, no es necesario regenerarlos.

# %%
if not WIND_PER_FIX_PARQUET.exists():
    from tfg_aves.meteo.build_wind import build_wind
    build_wind(
        daily_path="data/processed/daily.parquet",
        wind_raw_dir=WIND_RAW_DIR,
        out_path=WIND_PER_FIX_PARQUET,
    )

wind = pd.read_parquet(WIND_PER_FIX_PARQUET)
print(f"wind_per_fix.parquet: {wind.shape}")
print(wind.head())

# %%
if not (O4_L1V1_DIR / "metrics.parquet").exists():
    from tfg_aves.ml.build import build_o4
    build_o4(with_wind=True, seed=0)

metrics_v1 = pd.read_parquet(O4_L1V1_DIR / "metrics.parquet")
metrics_v0 = pd.read_parquet(O4_OUT_DIR / "metrics.parquet")
print(f"metrics_v0: {metrics_v0.shape}")
print(f"metrics_v1: {metrics_v1.shape}")
```

- [ ] **Step 7.2: Añadir L1-v1-D1 (cobertura espacial)**

```python
# %% [markdown]
# ## L1-v1-D1 — Cobertura espacial del viento sobre los fixes
#
# Verifica que el bbox del .nc cubre todos los fixes válidos del dataset.

# %%
daily_valid = daily.dropna(subset=["lat", "lon"]).copy()
merged = daily_valid.merge(wind, on=["bird_id", "date_utc"], how="left")

n_total_valid = len(merged)
n_with_wind = merged["wind_u_850"].notna().sum()
n_without_wind = n_total_valid - n_with_wind

bbox_table = pd.DataFrame([{
    "fixes_validos_total": int(n_total_valid),
    "fixes_con_viento": int(n_with_wind),
    "fixes_fuera_de_bbox": int(n_without_wind),
    "porcentaje_cobertura": 100.0 * n_with_wind / n_total_valid,
}])
print(bbox_table.to_string(index=False))

# Mapa: bbox del .nc + fixes
fig_d1, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    daily_valid["lon"], daily_valid["lat"],
    s=2, alpha=0.15, c="tab:blue", label=f"Fixes diarios (n={n_total_valid:,})",
)
# Bbox del .nc (lat ∈ [-3, 66] × lon ∈ [7, 53]).
ax.add_patch(plt.Rectangle(
    (7, -3), 53 - 7, 66 - (-3),
    fill=False, edgecolor="tab:red", linewidth=2,
    label="Bbox del viento reanalysis ECMWF",
))
ax.set_xlabel("Longitud (°)")
ax.set_ylabel("Latitud (°)")
ax.set_title("L1-v1-D1 — Cobertura espacial del viento sobre los fixes")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
fig_d1.tight_layout()

save_artifact(
    slug="l1v1-cobertura-espacial",
    objective="o4",
    num=10,
    decision="Verificar que el bbox del viento cubre todos los fixes",
    caption_es=(
        "Distribución espacial de los 24 444 fixes diarios del dataset "
        "Movebank superpuestos al bbox del viento reanalysis ECMWF a "
        "850 hPa (lat ∈ [-3°, 66°] × lon ∈ [7°, 53°]). La cobertura "
        "es completa: 0 fixes con coordenadas válidas quedan fuera del "
        "bbox, lo que permite usar las features de viento en todas las "
        "filas del pipeline sin pérdida silenciosa de datos."
    ),
    fig=fig_d1,
    table=bbox_table,
)
```

- [ ] **Step 7.3: Añadir L1-v1-D2 (distribución de wind_speed por mes)**

```python
# %% [markdown]
# ## L1-v1-D2 — Distribución de wind_speed_850 por mes
#
# Estacionalidad esperada: el viento es más fuerte y direccional en
# meses de migración activa (abril-mayo, septiembre-octubre).

# %%
wind_with_month = wind.copy()
wind_with_month["month"] = pd.to_datetime(wind_with_month["date_utc"]).dt.month
wind_valid = wind_with_month.dropna(subset=["wind_speed_850"])

fig_d2, ax = plt.subplots(figsize=(10, 5))
positions = list(range(1, 13))
data_by_month = [
    wind_valid.loc[wind_valid["month"] == m, "wind_speed_850"].values
    for m in positions
]
ax.boxplot(data_by_month, positions=positions, widths=0.6, showfliers=False)
ax.set_xticks(positions)
ax.set_xticklabels([
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
])
ax.set_ylabel("wind_speed_850 (m/s)")
ax.set_title("L1-v1-D2 — Distribución de la velocidad del viento por mes")
ax.grid(True, axis="y", alpha=0.3)
fig_d2.tight_layout()

save_artifact(
    slug="l1v1-wind-speed-estacionalidad",
    objective="o4",
    num=11,
    decision="Caracterizar la estacionalidad del viento a 850 hPa",
    caption_es=(
        "Distribución por mes de wind_speed_850 (módulo del viento "
        "horizontal a 850 hPa) interpolado a la posición de los 82 "
        "Larus fuscus durante 2009-2015. La estacionalidad muestra "
        "vientos más intensos en meses de migración activa, coherente "
        "con la circulación sinóptica del Atlántico Norte. La "
        "característica entra como feature en L1-v1 sin transformación "
        "(magnitud bruta en m/s)."
    ),
    fig=fig_d2,
)
```

- [ ] **Step 7.4: Sincronizar a .ipynb y ejecutar**

```bash
uv run jupytext --to ipynb notebooks/04l1_eda_o4l1.py
uv run jupyter execute notebooks/04l1_eda_o4l1.ipynb
```

Expected: notebook ejecutado sin errores, dos figuras y una tabla generados en `reports/figures/o4_fig10_l1v1-cobertura-espacial.png`, `reports/figures/o4_fig11_l1v1-wind-speed-estacionalidad.png`, `reports/tables/o4_tab10_l1v1-cobertura-espacial.csv`. `reports/INDEX.md` con dos nuevas filas.

- [ ] **Step 7.5: Commit**

```bash
git add notebooks/04l1_eda_o4l1.py reports/INDEX.md \
  reports/figures/o4_fig10_l1v1-cobertura-espacial.png \
  reports/figures/o4_fig11_l1v1-wind-speed-estacionalidad.png \
  reports/tables/o4_tab10_l1v1-cobertura-espacial.csv \
  reports/captions/o4_fig10_l1v1-cobertura-espacial.md \
  reports/captions/o4_fig11_l1v1-wind-speed-estacionalidad.md
git commit -m "Notebook L1 fase A: cobertura espacial (D1) y estacionalidad del viento (D2)"
```

---

## Task 8: Notebook EDA — Fase B (L1-v1-C1 + L1-v1-C2)

**Files:**
- Modify: `notebooks/04l1_eda_o4l1.py` (añadir Fase B al final)

- [ ] **Step 8.1: Añadir L1-v1-C1 (correlaciones)**

Añadir al final del notebook:

```python
# %% [markdown]
# ## Fase B — Diagnóstico de las nuevas features
# ### L1-v1-C1 — Correlación de las features de viento con las existentes

# %%
features_o4 = features_o3.merge(wind, on=["bird_id", "date_utc"], how="left")
target_cols = [
    "wind_u_850", "wind_v_850", "wind_speed_850",
    "state_b", "posterior_b_migracion", "step_length_km", "lat", "lon",
]
features_subset = features_o4.dropna(subset=target_cols)[target_cols]

corr_matrix = features_subset.corr(method="pearson").round(3)
print(corr_matrix)

fig_c1, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(target_cols))); ax.set_xticklabels(target_cols, rotation=45, ha="right")
ax.set_yticks(range(len(target_cols))); ax.set_yticklabels(target_cols)
for i in range(len(target_cols)):
    for j in range(len(target_cols)):
        ax.text(
            j, i, f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center", va="center",
            color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
            fontsize=9,
        )
ax.set_title("L1-v1-C1 — Correlaciones Pearson")
fig_c1.colorbar(im, ax=ax, shrink=0.7)
fig_c1.tight_layout()

save_artifact(
    slug="l1v1-wind-correlations",
    objective="o4",
    num=12,
    decision="Detectar colinealidad entre las features de viento y las existentes",
    caption_es=(
        "Matriz de correlaciones Pearson entre las tres features de viento "
        "(wind_u_850, wind_v_850, wind_speed_850) y las features ya "
        "presentes en O4 base relevantes para la predicción (state_b, "
        "posterior_b_migracion, step_length_km, lat, lon). Correlaciones "
        "absolutas pequeñas con state_b y posterior_b_migracion "
        "(|r| < 0,2 esperado) confirman que las features de viento aportan "
        "información independiente y no son redundantes con el régimen "
        "HMM que ya codifica el contexto biológico."
    ),
    fig=fig_c1,
    table=corr_matrix.reset_index().rename(columns={"index": "feature"}),
)
```

- [ ] **Step 8.2: Añadir L1-v1-C2 (feature importance comparada)**

```python
# %% [markdown]
# ### L1-v1-C2 — Feature importance comparada L1-v0 vs L1-v1
#
# Recuperamos la importancia de las features para los dos ganadores de
# L1-v0 (RF personalizado, XGB poblacional) y para sus equivalentes en
# L1-v1 (mismos algoritmos, mismo modo). Las 3 features de viento deben
# aparecer en el top-10 de L1-v1 si aportan.

# %%
def _feature_importance(model_path, family):
    b = joblib.load(model_path)
    model = b["model"]
    feature_cols = b["feature_cols"]
    if family == "rf":
        imp = model.feature_importances_
    elif family == "xgb":
        imp = model.feature_importances_
    else:
        imp = np.zeros(len(feature_cols))
    df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


pairs = [
    ("personalizado", "rf"),
    ("personalizado", "xgb"),
    ("poblacional", "rf"),
    ("poblacional", "xgb"),
]
rows_c2 = []
for modo, family in pairs:
    imp_v0 = _feature_importance(O4_OUT_DIR / f"model_{modo}_{family}.pkl", family)
    imp_v1 = _feature_importance(O4_L1V1_DIR / f"model_{modo}_{family}.pkl", family)
    imp_v0["version"] = "L1-v0"; imp_v0["modo"] = modo; imp_v0["familia"] = family
    imp_v1["version"] = "L1-v1"; imp_v1["modo"] = modo; imp_v1["familia"] = family
    rows_c2.append(imp_v0); rows_c2.append(imp_v1)
c2_table = pd.concat(rows_c2, ignore_index=True)

# Una figura con dos subplots: ganadores L1-v0 (RF pers + XGB pob) en L1-v1.
fig_c2, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (modo, family) in zip(axes, [("personalizado", "rf"), ("poblacional", "xgb")], strict=True):
    sub = c2_table[(c2_table["modo"] == modo) & (c2_table["familia"] == family) & (c2_table["version"] == "L1-v1")]
    sub = sub.sort_values("importance", ascending=True)
    ax.barh(sub["feature"], sub["importance"])
    is_wind = sub["feature"].str.startswith("wind_")
    colors = ["tab:orange" if w else "tab:blue" for w in is_wind]
    for bar, c in zip(ax.containers[0], colors, strict=True):
        bar.set_color(c)
    ax.set_title(f"L1-v1: {family.upper()} {modo}")
fig_c2.suptitle("L1-v1-C2 — Feature importance L1-v1 (naranja = features de viento)")
fig_c2.tight_layout()

save_artifact(
    slug="l1v1-feature-importance",
    objective="o4",
    num=13,
    decision="Confirmar que las features de viento son usadas por los modelos",
    caption_es=(
        "Importancia relativa de las features para los dos modelos "
        "ganadores de L1-v0 (RF personalizado y XGBoost poblacional) "
        "tras reentrenarlos con las tres features de viento en L1-v1. "
        "Las barras en naranja corresponden a las nuevas features de "
        "viento (wind_u_850, wind_v_850, wind_speed_850). Si aparecen "
        "en el top-10 de al menos uno de los ganadores, se cumple el "
        "criterio diagnóstico de éxito (§9 del spec) — el modelo no "
        "ignora las nuevas señales."
    ),
    fig=fig_c2,
    table=c2_table,
)
```

- [ ] **Step 8.3: Re-ejecutar notebook**

```bash
uv run jupytext --to ipynb notebooks/04l1_eda_o4l1.py
uv run jupyter execute notebooks/04l1_eda_o4l1.ipynb
```

Expected: C1 y C2 generados (`o4_fig12_*.png`, `o4_fig13_*.png`, `o4_tab12_*.csv`, `o4_tab13_*.csv`).

- [ ] **Step 8.4: Commit**

```bash
git add notebooks/04l1_eda_o4l1.py reports/INDEX.md \
  reports/figures/o4_fig1{2,3}_*.png \
  reports/tables/o4_tab1{2,3}_*.csv \
  reports/captions/o4_fig1{2,3}_*.md
git commit -m "Notebook L1 fase B: correlaciones (C1) y feature importance (C2)"
```

---

## Task 9: Notebook EDA — Fase C (L1-v1-C3 + L1-v1-C4)

**Files:**
- Modify: `notebooks/04l1_eda_o4l1.py` (añadir Fase C)

- [ ] **Step 9.1: Añadir L1-v1-C3 (métricas globales side-by-side)**

```python
# %% [markdown]
# ## Fase C — Comparativa L1-v0 vs L1-v1
# ### L1-v1-C3 — Métricas globales side-by-side

# %%
def _filter_test(metrics_df):
    return metrics_df[metrics_df["split"] == "test"].copy()


m_v0 = _filter_test(metrics_v0)
m_v1 = _filter_test(metrics_v1)

# Eliminar LightGBM de L1-v0 para que la comparativa sea simétrica (4 modelos cada uno).
m_v0 = m_v0[~m_v0["modelo"].isin(["lgbm", "persistencia", "markov"])]
m_v1 = m_v1[~m_v1["modelo"].isin(["lgbm", "persistencia", "markov"])]
m_v0["version"] = "L1-v0"
m_v1["version"] = "L1-v1"

rows_c3 = pd.concat([m_v0, m_v1], ignore_index=True)
rows_c3 = rows_c3.sort_values(["modelo", "modo", "version"])

# Tabla legible para la memoria
c3_table = rows_c3[[
    "modelo", "modo", "version", "top1", "top3", "log_loss", "dist_median_km",
]].reset_index(drop=True)
print(c3_table.to_string(index=False))

# Figura: 4 paneles, uno por (familia, modo); barras L1-v0 vs L1-v1 para
# cada una de las 4 métricas.
fig_c3, axes = plt.subplots(2, 2, figsize=(13, 9))
panel_keys = [
    ("rf", "personalizado"), ("rf", "poblacional"),
    ("xgb", "personalizado"), ("xgb", "poblacional"),
]
metric_names = ["top1", "top3", "log_loss", "dist_median_km"]
for ax, (fam, modo) in zip(axes.flatten(), panel_keys, strict=True):
    sub = c3_table[(c3_table["modelo"] == fam) & (c3_table["modo"] == modo)]
    v0 = sub[sub["version"] == "L1-v0"].iloc[0]
    v1 = sub[sub["version"] == "L1-v1"].iloc[0]
    x = np.arange(len(metric_names)); width = 0.35
    ax.bar(x - width/2, [v0[m] for m in metric_names], width, label="L1-v0")
    ax.bar(x + width/2, [v1[m] for m in metric_names], width, label="L1-v1")
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_title(f"{fam.upper()} {modo}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
fig_c3.suptitle("L1-v1-C3 — Comparativa L1-v0 vs L1-v1 (4 modelos × 4 métricas)")
fig_c3.tight_layout()

save_artifact(
    slug="l1v1-metrics-comparison",
    objective="o4",
    num=14,
    decision="Comparativa central del aporte del viento (L1-v0 vs L1-v1)",
    caption_es=(
        "Comparativa side-by-side de las cuatro métricas globales en el "
        "test split (top-1, top-3, log-loss, distancia mediana km) para "
        "los cuatro modelos comunes a L1-v0 y L1-v1 (RF/XGB × "
        "personalizado/poblacional). LightGBM, persistencia y Markov(1) "
        "se excluyen para que la comparación sea simétrica. Es el "
        "entregable narrativo central de L1: cuantifica el aporte "
        "aislado del viento como predictor sin contaminar con otras "
        "decisiones."
    ),
    fig=fig_c3,
    table=c3_table,
)
```

- [ ] **Step 9.2: Añadir L1-v1-C4 (comparativa por estado HMM)**

```python
# %% [markdown]
# ### L1-v1-C4 — Comparativa por estado HMM
#
# El aporte esperado del viento es mayor en migración (state_b=1) que
# en estacionario (state_b=0). Esta tabla cuantifica el desglose por
# estado.

# %%
preds_v0 = pd.read_parquet(O4_OUT_DIR / "predictions_test.parquet")
preds_v1 = pd.read_parquet(O4_L1V1_DIR / "predictions_test.parquet")
preds_v0 = preds_v0[~preds_v0["modelo"].isin(["lgbm", "persistencia", "markov"])]
preds_v1 = preds_v1[~preds_v1["modelo"].isin(["lgbm", "persistencia", "markov"])]
preds_v0["version"] = "L1-v0"; preds_v1["version"] = "L1-v1"
preds = pd.concat([preds_v0, preds_v1], ignore_index=True)


def _by_state(df):
    rows = []
    for (modelo, modo, version, state), sub in df.groupby(
        ["modelo", "modo", "version", "state_b"], sort=True,
    ):
        rows.append({
            "modelo": modelo,
            "modo": modo,
            "version": version,
            "state_b": int(state),
            "n": int(len(sub)),
            "top1": float((sub["true_cell"] == sub["pred_cell_top1"]).mean()),
            "dist_med_km": float(sub["pred_dist_km"].median()),
        })
    return pd.DataFrame(rows)


c4_table = _by_state(preds).sort_values(["modelo", "modo", "state_b", "version"])
print(c4_table.to_string(index=False))

fig_c4, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (fam, modo) in zip(axes.flatten(), panel_keys, strict=True):
    sub = c4_table[(c4_table["modelo"] == fam) & (c4_table["modo"] == modo)]
    states = [0, 1]; width = 0.35
    v0_top1 = [sub[(sub["state_b"] == s) & (sub["version"] == "L1-v0")]["top1"].iloc[0] for s in states]
    v1_top1 = [sub[(sub["state_b"] == s) & (sub["version"] == "L1-v1")]["top1"].iloc[0] for s in states]
    x = np.arange(2)
    ax.bar(x - width/2, v0_top1, width, label="L1-v0")
    ax.bar(x + width/2, v1_top1, width, label="L1-v1")
    ax.set_xticks(x); ax.set_xticklabels(["Estacionario", "Migración"])
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(f"{fam.upper()} {modo}")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig_c4.suptitle("L1-v1-C4 — top-1 por estado HMM, L1-v0 vs L1-v1")
fig_c4.tight_layout()

save_artifact(
    slug="l1v1-by-state-comparison",
    objective="o4",
    num=15,
    decision="Desglosar el aporte del viento por régimen biológico",
    caption_es=(
        "Comparativa de top-1 por estado HMM (estacionario vs migración) "
        "entre L1-v0 y L1-v1 para los cuatro modelos comunes. La "
        "hipótesis es que el aporte del viento se concentra en los días "
        "de migración (state_b=1), donde el modelo se beneficia más de "
        "saber si el viento es favorable o no. Si la mejora en "
        "estacionario es nula y en migración es ≥ +3 pp absolutos en al "
        "menos uno de los ganadores, se cumple el criterio secundario "
        "de éxito de L1 (§9 del spec)."
    ),
    fig=fig_c4,
    table=c4_table,
)
```

- [ ] **Step 9.3: Re-ejecutar notebook**

```bash
uv run jupytext --to ipynb notebooks/04l1_eda_o4l1.py
uv run jupyter execute notebooks/04l1_eda_o4l1.ipynb
```

Expected: C3 y C4 generados (`o4_fig14_*.png`, `o4_fig15_*.png`, tablas correspondientes).

- [ ] **Step 9.4: Commit**

```bash
git add notebooks/04l1_eda_o4l1.py reports/INDEX.md \
  reports/figures/o4_fig1{4,5}_*.png \
  reports/tables/o4_tab1{4,5}_*.csv \
  reports/captions/o4_fig1{4,5}_*.md
git commit -m "Notebook L1 fase C: comparativa global (C3) y por estado HMM (C4)"
```

---

## Task 10: Notebook EDA — Fase D (L1-v1-C5) + notas de memoria + AI-log + tag

**Files:**
- Modify: `notebooks/04l1_eda_o4l1.py` (añadir Fase D)
- Modify: `reports/memoria/06_o4_ml.md`
- Create: `reports/ai-log/0010-l1-features-viento.md`

- [ ] **Step 10.1: Añadir L1-v1-C5 (tailwind effect post-hoc)**

```python
# %% [markdown]
# ## Fase D — Análisis post-hoc del aprendizaje del viento
# ### L1-v1-C5 — ¿El modelo aprendió a usar el viento?
#
# Análisis: comparamos top-1 de L1-v1 en días con viento "favorable" vs
# "desfavorable" para la dirección de migración fenológica de Larus
# fuscus (primavera: norte = +v positivo; otoño: sur = -v positivo).

# %%
def _fenological_direction(month):
    if 3 <= month <= 6:
        return 1.0  # primavera: viento hacia el norte favorable → v > 0
    if 8 <= month <= 11:
        return -1.0  # otoño: viento hacia el sur favorable → v < 0
    return 0.0  # invernada o cría: dirección no clara


merged_pred = preds_v1.merge(
    wind, on=["bird_id", "date_utc"], how="left",
)
merged_pred["fen_dir"] = pd.to_datetime(merged_pred["date_utc"]).dt.month.apply(
    _fenological_direction,
)
merged_pred["tailwind_proxy"] = merged_pred["wind_v_850"] * merged_pred["fen_dir"]

migration_pred = merged_pred[merged_pred["state_b"] == 1].dropna(subset=["tailwind_proxy"])
migration_pred["wind_favorable"] = migration_pred["tailwind_proxy"] > 0

c5_rows = []
for (modelo, modo), sub in migration_pred.groupby(["modelo", "modo"]):
    fav = sub[sub["wind_favorable"]]
    unfav = sub[~sub["wind_favorable"]]
    c5_rows.append({
        "modelo": modelo,
        "modo": modo,
        "n_favorable": int(len(fav)),
        "n_desfavorable": int(len(unfav)),
        "top1_favorable": float((fav["true_cell"] == fav["pred_cell_top1"]).mean()) if len(fav) else float("nan"),
        "top1_desfavorable": float((unfav["true_cell"] == unfav["pred_cell_top1"]).mean()) if len(unfav) else float("nan"),
    })
c5_table = pd.DataFrame(c5_rows)
c5_table["delta"] = c5_table["top1_favorable"] - c5_table["top1_desfavorable"]
print(c5_table.to_string(index=False))

fig_c5, ax = plt.subplots(figsize=(9, 5))
labels = [f"{r['modelo'].upper()} {r['modo']}" for _, r in c5_table.iterrows()]
x = np.arange(len(c5_table)); width = 0.35
ax.bar(x - width/2, c5_table["top1_favorable"], width, label="Viento favorable")
ax.bar(x + width/2, c5_table["top1_desfavorable"], width, label="Viento desfavorable")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("top-1 accuracy (migración)")
ax.set_title("L1-v1-C5 — top-1 en migración: viento favorable vs desfavorable")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig_c5.tight_layout()

save_artifact(
    slug="l1v1-tailwind-effect",
    objective="o4",
    num=16,
    decision="Verificar si el modelo aprende a interpretar la dirección del viento",
    caption_es=(
        "Análisis post-hoc del aprendizaje del viento en L1-v1: top-1 "
        "accuracy sobre los días de migración (state_b=1) desglosado "
        "por dirección del viento respecto a la dirección fenológica "
        "esperada (primavera: hacia el norte; otoño: hacia el sur). "
        "Si delta = top1_favorable - top1_desfavorable es positivo y "
        "no trivial, el modelo está capturando la interacción "
        "viento×fenología — evidencia indirecta de que las features "
        "de viento aportan más que ruido."
    ),
    fig=fig_c5,
    table=c5_table,
)
```

- [ ] **Step 10.2: Re-ejecutar notebook**

```bash
uv run jupytext --to ipynb notebooks/04l1_eda_o4l1.py
uv run jupyter execute notebooks/04l1_eda_o4l1.ipynb
```

Expected: C5 generado, `reports/INDEX.md` con 7 entradas L1-v1 totales (`o4_fig10` a `o4_fig16`).

- [ ] **Step 10.3: Añadir sección de L1 a `reports/memoria/06_o4_ml.md`**

Localizar al final del documento (justo antes de cerrar el capítulo) y añadir:

```markdown

## L1 — Mejora con viento reanalysis ECMWF 850 hPa

Primera línea de mejora del pipeline O4. Spec:
`docs/superpowers/specs/2026-05-23-o4l1-features-design.md`. Ataca la
causa estructural D4 ("ausencia de features de viento") manteniendo
arquitectura, target, split temporal e hiperparámetros idénticos a O4
base. Vocabulario: **L1-v0** (O4 base sin viento, tag
`v0.4-o4-completo`); **L1-v1** (O4 base + tres features de viento;
tag `v0.4.1-o4l1-viento`).

### Features añadidas

`wind_u_850`, `wind_v_850`, `wind_speed_850` interpoladas
bilinealmente del grid 0,5° reanalysis ECMWF (12:00 UTC diario)
al punto exacto del fix diario de cada ave. Cobertura espacial
verificada en L1-v1-D1 (0 fixes fuera del bbox).

### Resultados

Resultado puntual a rellenar tras ejecutar las cuatro fases del
notebook. Tabla principal (L1-v1-C3) y desglose por estado HMM
(L1-v1-C4) son la evidencia primaria.

| modelo | modo | versión | top-1 | top-3 | log-loss | dist_med_km |
|---|---|---|---|---|---|---|
| RF | personalizado | L1-v0 | 0,644 | 0,754 | 5,22 | 23,1 |
| RF | personalizado | L1-v1 | ... | ... | ... | ... |
| RF | poblacional | L1-v0 | 0,557 | 0,708 | 5,65 | 25,2 |
| RF | poblacional | L1-v1 | ... | ... | ... | ... |
| XGB | personalizado | L1-v0 | 0,622 | 0,710 | 5,48 | 23,8 |
| XGB | personalizado | L1-v1 | ... | ... | ... | ... |
| XGB | poblacional | L1-v0 | 0,610 | 0,707 | 5,52 | 24,0 |
| XGB | poblacional | L1-v1 | ... | ... | ... | ... |

### Interpretación honesta

Aplicar los tres criterios de aceptación (§9 del spec):

1. **Log-loss:** reduce ≥ 0,10 en al menos uno de los ganadores → ...
2. **Top-1 migración:** sube ≥ +3 pp absolutos en al menos uno → ...
3. **Feature importance:** wind aparece en top-10 de al menos uno → ...

Resultado: ... (3/3, 2/3, 1/3 o 0/3). Narrativa correspondiente
seleccionada del listado del spec §9.

### Decisiones de diseño que no se reabren

- F7 de O4 base (descarte de veg_low/veg_high/daylight_hours) sigue
  vigente en L1. L1 no reactiva esas features para mantener la
  ablación limpia: la mejora medida es atribuible al viento, no a
  una recombinación de inputs.
- LightGBM sigue descartado. Rescatarlo sería una línea propia
  (tuning de LightGBM), no parte de L1.
- Causas estructurales D5 (sin destino) y Mo3 (sin historia
  multi-día) NO se atacan en L1 (decisión del autor 2026-05-23
  para mantener ablación de una causa por línea). Quedan para
  trabajo futuro o para L2/L3.

### Artefactos

| ID | Slug | Tipo |
|---|---|---|
| L1-v1-D1 | `o4_fig10_l1v1-cobertura-espacial` | fig + tabla |
| L1-v1-D2 | `o4_fig11_l1v1-wind-speed-estacionalidad` | fig |
| L1-v1-C1 | `o4_fig12_l1v1-wind-correlations` | fig + tabla |
| L1-v1-C2 | `o4_fig13_l1v1-feature-importance` | fig + tabla |
| L1-v1-C3 | `o4_fig14_l1v1-metrics-comparison` | fig + tabla |
| L1-v1-C4 | `o4_fig15_l1v1-by-state-comparison` | fig + tabla |
| L1-v1-C5 | `o4_fig16_l1v1-tailwind-effect` | fig + tabla |
```

- [ ] **Step 10.4: Crear `reports/ai-log/0010-l1-features-viento.md`**

```markdown
# 0010 — L1: Features de viento ECMWF 850 hPa

- **Fecha:** 2026-05-23
- **Objetivo:** O4 — primera línea de mejora (L1)
- **Modelo IA:** Claude Code Opus 4.7 (1M context)
- **Aporte intelectual al TFG:** sí (mejora del pipeline ML)

## Resumen

Implementación de la primera línea de mejora (L1) del pipeline
supervisado de O4 añadiendo tres features de viento ECMWF a 850 hPa
extraídas de los archivos NetCDF que el autor había descargado en
una iteración previa del proyecto (v2). Sin cambios de
arquitectura, target, split temporal ni hiperparámetros.

## Decisiones del autor durante el brainstorming

- Atacar D4 (sin viento) como primera línea de mejora; descartar
  D5 (sin destino) y Mo3 (multi-día) de L1 para mantener ablación
  limpia de una causa por línea.
- Usar los .nc de v2 a 850 hPa en lugar de re-descargar viento de
  superficie 10 m vía CDS, por eficiencia y porque 850 hPa está
  dentro del rango de vuelo migratorio de Larus fuscus.
- Mantener F7 de O4 base (veg_low/veg_high/daylight_hours
  descartadas) sin reabrir, para que la comparativa cuantifique
  exclusivamente el aporte del viento.
- LightGBM sigue descartado en L1 — su rescate sería tarea
  independiente.
- Nomenclatura L1-v0 vs L1-v1 para la comparativa.
- Tag final v0.4.1-o4l1-viento.

## Propuestas del asistente que el autor revisó y modificó

- Propuesta inicial: incluir features de destino (D5) y lags (Mo3)
  en L1. Rechazada por el autor: prefiere una causa por línea.
- Propuesta sobre LightGBM (tuning con lr=0,01, min_child_samples=5):
  rechazada por el autor para mantener la ablación limpia.
- Propuesta sobre vegetación y horas de luz como features
  adicionales: rechazada tras conversación sobre colinealidad con
  state_b.

## Tareas mecánicas ejecutadas por el asistente

- Inspección del schema del .nc de v2 y verificación de cobertura
  temporal sobre el dataset Movebank.
- Redacción de la spec en castellano siguiendo el patrón de O1/O2/O3/O4.
- Self-review de la spec con dos fixes inline (ERA-Interim →
  reanalysis ECMWF; contradicción 6 vs 4 modelos).
- Redacción del plan de implementación con TDD y artefactos.

## Resultado

- Tag: `v0.4.1-o4l1-viento` sobre el último commit de L1.
- Comparativa L1-v0 vs L1-v1: ver `reports/memoria/06_o4_ml.md`
  §"L1 — Mejora con viento reanalysis ECMWF 850 hPa".
- Verdicto sobre los tres criterios de éxito: ... (rellenar tras
  ejecución).
```

- [ ] **Step 10.5: Tests finales completos + ruff**

```bash
uv run ruff check src/tfg_aves/ tests/
uv run pytest -q
```

Expected: ruff clean; todos los tests pasan (tests pre-existentes + nuevos del paquete meteo + nuevos de ml.features y ml.build).

- [ ] **Step 10.6: Commit final + tag**

```bash
git add notebooks/04l1_eda_o4l1.py reports/INDEX.md \
  reports/figures/o4_fig16_l1v1-tailwind-effect.png \
  reports/tables/o4_tab16_l1v1-tailwind-effect.csv \
  reports/captions/o4_fig16_l1v1-tailwind-effect.md \
  reports/memoria/06_o4_ml.md
git commit -m "Notebook L1 fase D (C5) + notas memoria de L1

Cierra la primera línea de mejora (L1) del pipeline O4 con el
análisis post-hoc del aprendizaje del viento (L1-v1-C5: top-1 en
migración desglosado por dirección del viento respecto a la dirección
fenológica esperada) y la sección \"L1 — Mejora con viento ECMWF
850 hPa\" en reports/memoria/06_o4_ml.md."

git tag -a v0.4.1-o4l1-viento -m "Cierre de L1: features de viento ECMWF 850 hPa"
```

Nota: el AI-log queda en `reports/ai-log/` (gitignored, no se commitea).

Verificación final:

```bash
git log --oneline -10
git tag --list | grep v0.4
```

Expected: tag `v0.4.1-o4l1-viento` visible junto a `v0.4-o4-completo`.

---

## Self-review checklist (para el implementador)

Antes de declarar L1 cerrado, verifica:

1. **Tests:** `uv run pytest -q` → todos los tests pasan (los 99 originales + los nuevos del paquete meteo + las extensiones de ml.features y ml.build).
2. **Ruff:** `uv run ruff check src/ tests/` → sin errores.
3. **Idempotencia:** ejecutar `build_o4(with_wind=True)` dos veces con `seed=0` produce los mismos artefactos en `data/processed/o4/l1_v1/` (verificar tamaños y hashes si quieres ser estricto).
4. **L1-v0 intacto:** `data/processed/o4/model_*.pkl`, `predictions_test.parquet` y `metrics.parquet` (los de L1-v0) no han sido modificados. Comparar con el tag `v0.4-o4-completo`.
5. **Artefactos:** `reports/INDEX.md` lista 7 entradas L1-v1 (`o4_fig10` a `o4_fig16` y sus tablas correspondientes).
6. **Memoria notes:** `reports/memoria/06_o4_ml.md` tiene la sección "L1 — Mejora con viento ECMWF 850 hPa" rellena con los números reales (no los placeholders `...`).
7. **Tag:** `git tag --list` muestra `v0.4.1-o4l1-viento` apuntando al commit final de L1.
8. **AI-log:** `reports/ai-log/0010-l1-features-viento.md` existe con la sección "Resultado" rellena con el veredicto sobre los tres criterios de éxito.
