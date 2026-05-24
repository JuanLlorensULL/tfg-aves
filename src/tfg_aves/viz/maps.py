"""Constructores de mapas folium para O5.

Reciben DataFrames ya agregados (de viz.error / viz.chain) y devuelven
objetos ``folium.Map``. No calculan métricas ni escriben a disco.
"""
from __future__ import annotations

import branca.colormap as cm
import folium
import pandas as pd
from folium.plugins import TimestampedGeoJson


def _map_centered(lat_c: pd.Series, lon_c: pd.Series, zoom: int = 4) -> folium.Map:
    """Mapa folium centrado en el centroide de los datos."""
    return folium.Map(
        location=[float(lat_c.mean()), float(lon_c.mean())],
        zoom_start=zoom, tiles="cartodbpositron",
    )


def _cell_bounds(lat_c: float, lon_c: float, cell_deg: float) -> list[list[float]]:
    """Esquinas [[sur,oeste],[norte,este]] de la celda dado su centroide."""
    half = cell_deg / 2.0
    return [[lat_c - half, lon_c - half], [lat_c + half, lon_c + half]]


def error_choropleth(error_df: pd.DataFrame, *, cell_deg: float = 0.5) -> folium.Map:
    """Coropleta de la distancia mediana p50→real por celda (verde→rojo)."""
    m = _map_centered(error_df["lat_c"], error_df["lon_c"])
    vmax = float(error_df["median_error_km"].quantile(0.95)) or 1.0
    scale = cm.LinearColormap(["#2e8b57", "#f4c430", "#dd3333"], vmin=0.0, vmax=vmax)
    scale.caption = "Distancia mediana p50→real (km)"
    for _, r in error_df.iterrows():
        folium.Rectangle(
            bounds=_cell_bounds(r["lat_c"], r["lon_c"], cell_deg),
            color=None, fill=True, fill_color=scale(min(r["median_error_km"], vmax)),
            fill_opacity=0.7,
            popup=f"{r['cell_id']}: {r['median_error_km']:.0f} km (n={int(r['n'])})",
        ).add_to(m)
    scale.add_to(m)
    return m


def calibration_choropleth(cov_df: pd.DataFrame, *, cell_deg: float = 0.5) -> folium.Map:
    """Coropleta de la cobertura marginal por celda (centrada en 0,80)."""
    m = _map_centered(cov_df["lat_c"], cov_df["lon_c"])
    scale = cm.LinearColormap(["#c9a0c9", "#3b6ea5", "#c9a0c9"], vmin=0.5, vmax=1.0)
    scale.caption = "Cobertura marginal [p10,p90] (nominal 0,80)"
    for _, r in cov_df.iterrows():
        folium.Rectangle(
            bounds=_cell_bounds(r["lat_c"], r["lon_c"], cell_deg),
            color=None, fill=True, fill_color=scale(min(max(r["coverage_marginal"], 0.5), 1.0)),
            fill_opacity=0.7,
            popup=f"{r['cell_id']}: {r['coverage_marginal']:.0%} (n={int(r['n'])})",
        ).add_to(m)
    scale.add_to(m)
    return m


def _band_ring(lat_t: float, lon_t: float, row) -> list[list[float]]:
    """Anillo GeoJSON (lon,lat) del rectángulo [p10,p90] en torno al origen."""
    s, n = lat_t + row["dlat_p10"], lat_t + row["dlat_p90"]
    w, e = lon_t + row["dlon_p10"], lon_t + row["dlon_p90"]
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def prediction_map(
    preds_bird: pd.DataFrame, daily: pd.DataFrame, *, bird_id: str,
) -> folium.Map:
    """Vista A (un día) para un ave, con slider temporal.

    Capa estática: trayectoria real de contexto. Capa animada
    (``TimestampedGeoJson``, un día por frame): por cada día de test, la banda
    rectangular [p10,p90], la línea origen→p50, el punto p50, la persistencia
    (=origen) y el real t+1. El origen del día t se recupera como
    ``pred - dlat/dlon_p50``. La capa Markov(1) (apagada por defecto, V5) se
    omite en esta versión por simplicidad: persistencia es la baseline
    didáctica relevante.
    """
    sub = preds_bird[preds_bird["bird_id"] == bird_id].sort_values("date_utc")
    day = daily[(daily["bird_id"] == bird_id) & daily["is_valid"]].sort_values("date_utc")
    m = _map_centered(day["lat"], day["lon"], zoom=5)

    folium.PolyLine(
        list(zip(day["lat"], day["lon"], strict=True)),
        color="#e08a3c", weight=2, opacity=0.5,
        tooltip=f"Trayectoria real {bird_id}",
    ).add_to(m)

    real_by_date = {
        pd.Timestamp(d): (float(la), float(lo))
        for d, la, lo in zip(day["date_utc"], day["lat"], day["lon"], strict=True)
    }
    features: list[dict] = []
    for _, r in sub.iterrows():
        d = pd.Timestamp(r["date_utc"])
        t_iso = d.isoformat()
        lat_t = r["pred_lat"] - r["dlat_p50"]
        lon_t = r["pred_lon"] - r["dlon_p50"]
        features.append({"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_band_ring(lat_t, lon_t, r)]},
            "properties": {"times": [t_iso], "style": {"color": "#3b6ea5", "weight": 1,
                "fillColor": "#3b6ea5", "fillOpacity": 0.12}}})
        features.append({"type": "Feature",
            "geometry": {"type": "LineString",
                "coordinates": [[lon_t, lat_t], [r["pred_lon"], r["pred_lat"]]]},
            "properties": {"times": [t_iso], "style": {"color": "#3b6ea5", "weight": 2}}})
        features.append({"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["pred_lon"], r["pred_lat"]]},
            "properties": {"times": [t_iso], "icon": "circle",
                "iconstyle": {"fillColor": "#3b6ea5", "fillOpacity": 1.0,
                              "stroke": False, "radius": 5}}})
        features.append({"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon_t, lat_t]},
            "properties": {"times": [t_iso], "icon": "circle",
                "iconstyle": {"color": "#2e8b57", "fillOpacity": 0.0, "radius": 7}}})
        nxt = real_by_date.get(d + pd.Timedelta(days=1))
        if nxt is not None:
            features.append({"type": "Feature",
                "geometry": {"type": "Point", "coordinates": [nxt[1], nxt[0]]},
                "properties": {"times": [t_iso], "icon": "circle",
                    "iconstyle": {"fillColor": "#e08a3c", "fillOpacity": 1.0,
                                  "stroke": False, "radius": 5}}})

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="P1D", duration="P1D", add_last_point=False,
        auto_play=False, loop=False, transition_time=200,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def failure_vectors_map(
    preds: pd.DataFrame, daily: pd.DataFrame, *, top_n: int = 30,
) -> folium.Map:
    """Flechas p50→real de los mayores errores en días de migración.

    Filtra ``state_b_causal == 1`` (migración), ordena por ``dist_native_km``
    descendente y dibuja las ``top_n`` peores. La verdad t+1 se toma de
    ``daily`` (posición real del día siguiente, mismo bird_id).
    """
    mig = preds[preds["state_b_causal"] == 1].nlargest(top_n, "dist_native_km")
    if mig.empty:
        # Sin días de migración no hay centroide; mapa base vacío.
        return folium.Map(location=[40.0, -3.0], zoom_start=4, tiles="cartodbpositron")
    m = _map_centered(mig["pred_lat"], mig["pred_lon"], zoom=4)
    day = daily[daily["is_valid"]].copy()
    day["date_utc"] = pd.to_datetime(day["date_utc"])
    for _, r in mig.iterrows():
        # Verdad t+1 = día calendario siguiente exacto (gap-aware, como la
        # vista A); si hay hueco no se dibuja el vector.
        target = pd.Timestamp(r["date_utc"]) + pd.Timedelta(days=1)
        nxt = day[(day["bird_id"] == r["bird_id"]) & (day["date_utc"] == target)]
        if nxt.empty:
            continue
        real_lat, real_lon = float(nxt["lat"].iloc[0]), float(nxt["lon"].iloc[0])
        folium.PolyLine([[r["pred_lat"], r["pred_lon"]], [real_lat, real_lon]],
                        color="#d33", weight=2, opacity=0.8,
                        popup=f"{r['dist_native_km']:.0f} km").add_to(m)
        folium.CircleMarker([r["pred_lat"], r["pred_lon"]], radius=3,
                            color="#3b6ea5", fill=True).add_to(m)
        folium.CircleMarker([real_lat, real_lon], radius=3,
                            color="#e08a3c", fill=True).add_to(m)
    return m


def multistep_demo_map(
    chain: pd.DataFrame, real: pd.DataFrame, *, start: tuple[float, float],
) -> folium.Map:
    """Demo B: track p50 encadenado + cono ilustrativo + track real.

    Rótulo explícito de que el cono es ILUSTRATIVO (no calibrado).
    """
    m = _map_centered(chain["lat"], chain["lon"], zoom=5)
    pts = [[start[0], start[1]], *chain[["lat", "lon"]].values.tolist()]
    folium.PolyLine(pts, color="#3b6ea5", weight=3, tooltip="p50 encadenado").add_to(m)
    for _, r in chain.iterrows():
        folium.Rectangle(
            bounds=[[r["lat"] - r["cone_halfwidth_lat"], r["lon"] - r["cone_halfwidth_lon"]],
                    [r["lat"] + r["cone_halfwidth_lat"], r["lon"] + r["cone_halfwidth_lon"]]],
            color="#3b6ea5", weight=1, dash_array="4,3", fill=True,
            fill_color="#3b6ea5", fill_opacity=0.10).add_to(m)
    folium.PolyLine(real[["lat", "lon"]].values.tolist(), color="#e08a3c",
                    weight=2, dash_array="2,2", tooltip="real").add_to(m)
    folium.map.Marker(
        [float(chain["lat"].iloc[0]), float(chain["lon"].iloc[0])],
        icon=folium.DivIcon(html='<div style="font-size:11px;color:#777">'
                                 'demo exploratoria · cono ilustrativo no calibrado</div>'),
    ).add_to(m)
    return m
