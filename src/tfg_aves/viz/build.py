"""Orquestador de O5: genera mapas folium y tablas de evidencia.

Único módulo de viz que escribe a disco. Carga las predicciones de L3 lgbm
poblacional y los modelos, produce los HTML en reports/figures/ y las tablas
vía save_artifact.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from tfg_aves.reporting import save_artifact
from tfg_aves.viz import _paths as P
from tfg_aves.viz.chain import chain_trajectory
from tfg_aves.viz.error import (
    coverage_by_cell,
    error_by_cell,
    metrics_by_month,
    metrics_by_regime,
)
from tfg_aves.viz.maps import (
    calibration_choropleth,
    error_choropleth,
    failure_vectors_map,
    multistep_demo_map,
    prediction_map,
)


def load_lgbm_predictions(path: Path = P.PREDICTIONS_L3V2_PARQUET) -> pd.DataFrame:
    """Carga y filtra las predicciones a familia lgbm, modo poblacional."""
    df = pd.read_parquet(path)
    return df[(df["familia"] == "lgbm") & (df["modo"] == "poblacional")].reset_index(drop=True)


def build_curated_table(daily: pd.DataFrame, preds: pd.DataFrame,
                        features_o3: pd.DataFrame) -> pd.DataFrame:
    """Tabla de las 4 aves curadas: días válidos, días test, % migración."""
    rows = []
    for bird in P.CURATED_BIRDS:
        dv = int(daily[(daily["bird_id"] == bird) & daily["is_valid"]].shape[0])
        dt = int(preds[preds["bird_id"] == bird].shape[0])
        fb = features_o3[features_o3["bird_id"] == bird]
        pct = float((fb["state_b"] == 1).mean() * 100) if len(fb) else float("nan")
        rows.append({"bird_id": bird, "dias_validos": dv, "dias_test": dt,
                     "pct_migracion": round(pct, 1)})
    return pd.DataFrame(rows)


def build_o5_tables(preds: pd.DataFrame, *, daily: pd.DataFrame | None = None,
                    features_o3: pd.DataFrame | None = None,
                    project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Genera y persiste las tablas de evidencia de O5 vía save_artifact."""
    if daily is None:
        daily = pd.read_parquet(P.DAILY_PARQUET)
    if features_o3 is None:
        features_o3 = pd.read_parquet(P.FEATURES_O3_PARQUET)

    curated = build_curated_table(daily, preds, features_o3)
    regime = metrics_by_regime(preds)
    month = metrics_by_month(preds)

    save_artifact("aves-curadas", objective="o5", num=1,
                  decision="4 aves curadas de la demo (las de más histórico)",
                  caption_es=("Aves seleccionadas para la demo de predicción de O5: las "
                              "cuatro con más histórico. Días válidos en daily.parquet, días "
                              "del conjunto de test de L3 y porcentaje de migración (estado B "
                              "del HMM). El criterio de máximo histórico da variedad de régimen "
                              "(91752A casi residente vs el resto ~10-13 %)."),
                  table=curated, overwrite=True, project_root=project_root)
    save_artifact("error-por-regimen", objective="o5", num=2,
                  decision="Métricas de L3 por régimen HMM (estacionario vs migración)",
                  caption_es=("Métricas de L3 LightGBM poblacional por régimen biológico. top-1, "
                              "distancia mediana nativa p50→real, cobertura marginal por eje "
                              "(nominal 0,80) y cobertura conjunta del rectángulo (menor que la "
                              "marginal por construcción). Confirma la caída en migración."),
                  table=regime, overwrite=True, project_root=project_root)
    save_artifact("error-por-mes", objective="o5", num=3,
                  decision="Métricas de L3 por mes calendario (cruce con fenología)",
                  caption_es=("Métricas mensuales de L3 LightGBM poblacional. Cruza el error "
                              "con la fenología de Larus fuscus (picos abr-may y sep-oct)."),
                  table=month, overwrite=True, project_root=project_root)
    return {"aves_curadas": curated, "por_regimen": regime, "por_mes": month}


def build_error_maps(preds: pd.DataFrame, cells: pd.DataFrame, *,
                     out_dir: Path, daily: pd.DataFrame | None = None) -> dict[str, Path]:
    """Genera y guarda las coropletas de error y calibración + vectores.

    Si ``daily`` es None se omite el mapa de vectores de fallo (necesita la
    posición real t+1). ``build_o5`` siempre lo pasa.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    err = error_by_cell(preds, cells)
    p = out_dir / "o5_fig20_error-por-celda.html"
    error_choropleth(err).save(str(p))
    paths["error"] = p

    cov = coverage_by_cell(preds, cells)
    p = out_dir / "o5_fig21_calibracion-por-celda.html"
    calibration_choropleth(cov).save(str(p))
    paths["calibracion"] = p

    if daily is not None:
        p = out_dir / "o5_fig22_vectores-fallo-migracion.html"
        failure_vectors_map(preds, daily).save(str(p))
        paths["vectores"] = p
    return paths


def build_prediction_maps(preds: pd.DataFrame, daily: pd.DataFrame, *,
                          out_dir: Path) -> dict[str, Path]:
    """Un HTML de vista A por cada ave curada."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for i, bird in enumerate(P.CURATED_BIRDS, start=1):
        m = prediction_map(preds, daily, bird_id=bird)
        p = out_dir / f"o5_fig{i:02d}_prediccion-{bird}.html"
        m.save(str(p))
        paths[bird] = p
    return paths


def _load_lgbm_axes():
    """Carga los predictores de eje lgbm poblacional (dict joblib → axis)."""
    return joblib.load(P.MODEL_LGBM_DLAT)["model"], joblib.load(P.MODEL_LGBM_DLON)["model"]


def build_multistep_demo(preds: pd.DataFrame, daily: pd.DataFrame, *,
                         out_dir: Path, bird_id: str = "91916A", k: int = 7,
                         axes=None) -> dict[str, Path]:
    """Demo B: encadena k pasos para un ave desde un día de migración real.

    Refinamiento del spec §5.2: las features del HMM se CONGELAN (el estado del
    día de arranque; posterior heurístico 0,8/0,1 según régimen) porque la
    emisión del HMM exige covariables ambientales no disponibles en posiciones
    sintéticas futuras. Demo exploratoria con banda ILUSTRATIVA no calibrada.
    ``axes`` permite inyectar predictores (test); en producción se cargan de disco.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_lat, axis_lon = axes if axes is not None else _load_lgbm_axes()
    sub = preds[preds["bird_id"] == bird_id].sort_values("date_utc")
    if sub.empty:
        return {}
    mig = sub[sub["state_b_causal"] == 1]
    start_row = mig.iloc[0] if not mig.empty else sub.iloc[0]
    start_date = pd.Timestamp(start_row["date_utc"])

    day = daily[(daily["bird_id"] == bird_id) & daily["is_valid"]].copy()
    day["date_utc"] = pd.to_datetime(day["date_utc"])
    day = day.sort_values("date_utc")
    # Seed inclusivo: incluye el propio día de arranque (t) más el anterior
    # (t-1). chain_trajectory necesita la posición real de t para generar t+1.
    prev = day[day["date_utc"] <= start_date].tail(2)
    if len(prev) < 2:
        return {}
    seed_history = list(zip(prev["lat"], prev["lon"], strict=True))
    frozen_state = int(start_row["state_b_causal"])
    frozen_post = 0.8 if frozen_state == 1 else 0.1

    chain = chain_trajectory(axis_lat, axis_lon, seed_history,
                             start_date=start_date, frozen_state_b=frozen_state,
                             frozen_posterior_mig=frozen_post, k=k)
    real = day[day["date_utc"] >= start_date].head(k + 1)[["lat", "lon"]]
    start = (float(prev["lat"].iloc[-1]), float(prev["lon"].iloc[-1]))
    p = out_dir / f"o5_fig30_demo-multipaso-{bird_id}.html"
    multistep_demo_map(chain, real, start=start).save(str(p))
    return {bird_id: p}


def build_o5(out_dir: Path | None = None) -> dict:
    """Pipeline completo de O5: tablas + predicción + error + demo multi-paso.

    Lee los artefactos reales de L3/O3/O2/O1 y escribe a reports/figures/.
    """
    out_dir = Path(out_dir) if out_dir is not None else P.FIGURES_DIR
    preds = load_lgbm_predictions()
    daily = pd.read_parquet(P.DAILY_PARQUET)
    cells = pd.read_parquet(P.CELLS_PARQUET)
    features_o3 = pd.read_parquet(P.FEATURES_O3_PARQUET)

    tables = build_o5_tables(preds, daily=daily, features_o3=features_o3)
    pred_maps = build_prediction_maps(preds, daily, out_dir=out_dir)
    error_maps = build_error_maps(preds, cells, out_dir=out_dir, daily=daily)
    demo_maps = build_multistep_demo(preds, daily, out_dir=out_dir)
    return {"tables": tables, "prediction_maps": pred_maps,
            "error_maps": error_maps, "demo_maps": demo_maps}
