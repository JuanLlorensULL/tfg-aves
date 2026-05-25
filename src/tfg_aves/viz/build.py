"""Orquestador de O5: genera mapas folium y tablas de evidencia.

Único módulo de viz que escribe a disco. Carga las predicciones de L3 lgbm
poblacional y los modelos, produce los HTML en reports/figures/ y las tablas
vía save_artifact.
"""
from __future__ import annotations

import json
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

# Plantilla de la app Leaflet a medida (HTML/CSS/JS estático con hueco de datos).
_APP_TEMPLATE: Path = Path(__file__).parent / "templates" / "prediccion_app.html"


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
        pct = float((fb["state_b_causal"] == 1).mean() * 100) if len(fb) else float("nan")
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


def build_prediction_index(pred_map_paths: dict[str, Path], *,
                           out_dir: Path) -> Path:
    """Página única con un selector de ave (arriba-derecha) sobre un iframe.

    Envuelve los HTML folium por ave —que se conservan intactos, cada uno con
    su slider/leyenda— en un ``iframe`` cuyo ``src`` cambia el ``<select>``.
    Es HTML+JS estático, sin backend (coherente con "solo folium HTML"). Los
    ``src`` son relativos: el índice debe vivir en el mismo directorio que los
    mapas por ave.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    birds = list(pred_map_paths)
    options = "\n".join(
        f'        <option value="{Path(pred_map_paths[b]).name}">{b}</option>'
        for b in birds
    )
    first = Path(pred_map_paths[birds[0]]).name if birds else ""
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>O5 — Predicción del destino por ave</title>
<style>
  html, body {{ margin: 0; height: 100%; font-family: sans-serif; }}
  #bar {{ position: fixed; top: 0; left: 0; right: 0; height: 42px;
         background: #2b3a4a; color: #fff; display: flex; align-items: center;
         justify-content: space-between; padding: 0 14px; box-sizing: border-box;
         z-index: 10000; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }}
  #bar b {{ font-size: 14px; }}
  #bar label {{ font-size: 13px; margin-right: 6px; }}
  select {{ font-size: 14px; padding: 3px 6px; }}
  iframe {{ position: fixed; top: 42px; left: 0; right: 0; bottom: 0;
           width: 100%; height: calc(100% - 42px); border: 0; }}
</style>
</head>
<body>
  <div id="bar">
    <b>O5 — Predicción del destino a 1 día</b>
    <span>
      <label for="sel">Ave:</label>
      <select id="sel" onchange="document.getElementById('map').src = this.value;">
{options}
      </select>
    </span>
  </div>
  <iframe id="map" title="Mapa de predicción" src="{first}"></iframe>
</body>
</html>"""
    p = out_dir / "o5_prediccion_index.html"
    p.write_text(html, encoding="utf-8")
    return p


def prediction_app_data(preds: pd.DataFrame, daily: pd.DataFrame, *,
                        markov_points: dict | None = None) -> dict:
    """Datos por ave/día para la app Leaflet a medida (función pura).

    Por cada ave del test con predicciones (las curadas, de más histórico,
    primero; el resto detrás), una lista de días con: fecha, año,
    origen ``o`` (posición del día t, recuperado como ``pred - p50``; centro de
    la banda y de la línea origen→p50), punto ``p`` (p50), rectángulo de banda
    ``band`` [[sur,oeste],[norte,este]], posición real ``r`` del día siguiente
    (o ``None`` si hay hueco), punto de Markov ``m`` (centroide de la celda
    0,5° que predice el baseline Markov(1), o ``None``) y error ``e`` (km,
    distancia haversine p50→real, tomada de ``dist_native_km``).
    ``markov_points`` es un dict ``(bird_id, "YYYY-MM-DD") -> [lat, lon]``.
    Coordenadas a 5 decimales. No toca disco.
    """
    mk = markov_points or {}
    has_dist = "dist_native_km" in preds.columns
    d_valid = daily[daily["is_valid"]].copy()
    d_valid["date_utc"] = pd.to_datetime(d_valid["date_utc"])
    # Detalle por día para TODAS las aves del test, ORDENADAS por histórico
    # (días válidos en daily) descendente: las de más histórico, arriba.
    present = set(preds["bird_id"].unique())
    hist = d_valid[d_valid["bird_id"].isin(present)].groupby("bird_id").size()
    order = sorted(present, key=lambda b: (-int(hist.get(b, 0)), str(b)))
    curated = [b for b in P.CURATED_BIRDS if b in present]
    birds_out: list[dict] = []
    for bird in order:
        sub = preds[preds["bird_id"] == bird].sort_values("date_utc")
        if sub.empty:
            continue
        dsub = d_valid[d_valid["bird_id"] == bird]
        real_by_date = {
            pd.Timestamp(dt): (round(float(la), 5), round(float(lo), 5))
            for dt, la, lo in zip(dsub["date_utc"], dsub["lat"], dsub["lon"], strict=True)
        }
        days_out: list[dict] = []
        for _, r in sub.iterrows():
            dt = pd.Timestamp(r["date_utc"])
            dstr = dt.strftime("%Y-%m-%d")
            lat_t = float(r["pred_lat"]) - float(r["dlat_p50"])
            lon_t = float(r["pred_lon"]) - float(r["dlon_p50"])
            south, north = lat_t + float(r["dlat_p10"]), lat_t + float(r["dlat_p90"])
            west, east = lon_t + float(r["dlon_p10"]), lon_t + float(r["dlon_p90"])
            nxt = real_by_date.get(dt + pd.Timedelta(days=1))
            days_out.append({
                "date": dstr,
                "year": int(dt.year),
                "o": [round(lat_t, 5), round(lon_t, 5)],
                "p": [round(float(r["pred_lat"]), 5), round(float(r["pred_lon"]), 5)],
                "band": [[round(south, 5), round(west, 5)],
                         [round(north, 5), round(east, 5)]],
                "r": list(nxt) if nxt is not None else None,
                "m": mk.get((bird, dstr)),
                "e": round(float(r["dist_native_km"]), 1) if has_dist else None,
                "s": int(r["state_b_causal"]),  # 0 estacionario, 1 migración (O3)
            })
        # Recorrido histórico COMPLETO (train+val+test): todos los días
        # válidos del ave, no solo los de test. Contexto opcional en la app.
        dh = dsub.sort_values("date_utc")
        hist = [
            [round(float(la), 5), round(float(lo), 5), int(pd.Timestamp(dt).year)]
            for dt, la, lo in zip(dh["date_utc"], dh["lat"], dh["lon"], strict=True)
        ]
        birds_out.append({"id": str(bird), "days": days_out, "hist": hist})

    # La vista "Todas" y el overlay de estado se derivan de birds_out
    # (hist = recorrido completo con año por punto; days = estado por día);
    # no hace falta un array aparte.
    return {"birds": birds_out, "curated": [str(b) for b in curated]}


def markov_points_for_test(features_o3: pd.DataFrame, cells: pd.DataFrame,
                           *, seed: int = 0) -> dict:
    """Punto de Markov(1) por ``(bird_id, fecha)`` del test poblacional de O4.

    Reusa el MISMO baseline que O4 (``compute_markov_baseline``: Markov(1)
    mensual reentrenado sobre el train temporal poblacional) para ser coherente
    con la memoria, y mapea la celda predicha a su centroide. Devuelve
    ``(bird_id, "YYYY-MM-DD") -> [lat, lon]``.
    """
    from tfg_aves.ml.build_l3 import _prepare_poblacional_split
    from tfg_aves.ml.evaluate import compute_markov_baseline

    train, _val, test = _prepare_poblacional_split(features_o3, cells)
    mk = compute_markov_baseline(train, test, cells=cells)
    centroid = cells.set_index("cell_id")[["lat_c", "lon_c"]]
    out: dict = {}
    for r in mk.itertuples():
        if r.pred_cell_top1 in centroid.index:
            la, lo = centroid.loc[r.pred_cell_top1]
            key = (r.bird_id, pd.Timestamp(r.date_utc).strftime("%Y-%m-%d"))
            out[key] = [round(float(la), 5), round(float(lo), 5)]
    return out


def build_prediction_app(preds: pd.DataFrame, daily: pd.DataFrame, *,
                         out_dir: Path, features_o3: pd.DataFrame | None = None,
                         cells: pd.DataFrame | None = None, seed: int = 0) -> Path:
    """App Leaflet a medida (estática) con barra lateral, selector de ave,
    ventana de días acotable por ambos lados, play/pausa y velocidad.

    Inyecta los datos (``prediction_app_data``) en la plantilla
    ``templates/prediccion_app.html``. Sin backend: todo va embebido. Se añade
    a los mapas folium por ave (no los sustituye). Si se pasan ``features_o3`` y
    ``cells`` se calcula el punto de Markov por día (baseline de O4); si no, la
    app se genera sin ese punto.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markov_points = (
        markov_points_for_test(features_o3, cells, seed=seed)
        if features_o3 is not None and cells is not None else None
    )
    data = prediction_app_data(preds, daily, markov_points=markov_points)
    template = _APP_TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("__PRED_DATA__", json.dumps(data, ensure_ascii=False))
    out = out_dir / "o5_prediccion_app.html"
    out.write_text(html, encoding="utf-8")
    return out


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
    pred_index = build_prediction_index(pred_maps, out_dir=out_dir)
    pred_app = build_prediction_app(preds, daily, out_dir=out_dir,
                                    features_o3=features_o3, cells=cells)
    error_maps = build_error_maps(preds, cells, out_dir=out_dir, daily=daily)
    demo_maps = build_multistep_demo(preds, daily, out_dir=out_dir)
    return {"tables": tables, "prediction_maps": pred_maps,
            "prediction_index": pred_index, "prediction_app": pred_app,
            "error_maps": error_maps, "demo_maps": demo_maps}
