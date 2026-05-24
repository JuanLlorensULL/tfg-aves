"""Agregados puros de error y calibración para O5.

Todas las funciones reciben el DataFrame de predicciones de L3 (lgbm
poblacional) y devuelven DataFrames listos para el render folium o para
``save_artifact``. No tocan disco ni folium.
"""
from __future__ import annotations

import pandas as pd

from tfg_aves.ml.quantile import point_to_cell


def recover_origin(preds: pd.DataFrame) -> pd.DataFrame:
    """Recupera la posición de origen (día t) invirtiendo el p50.

    ``pred_lat = lat_t + dlat_p50`` ⟹ ``lat_t = pred_lat - dlat_p50``
    (ídem lon). Devuelve una copia con columnas ``lat_t``, ``lon_t``.
    """
    out = preds.copy()
    out["lat_t"] = out["pred_lat"].to_numpy() - out["dlat_p50"].to_numpy()
    out["lon_t"] = out["pred_lon"].to_numpy() - out["dlon_p50"].to_numpy()
    return out


def _attach_origin_cell(preds: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Añade ``origin_cell`` (celda contenedora de la posición de origen)."""
    out = recover_origin(preds)
    mapped = point_to_cell(out["lat_t"].to_numpy(), out["lon_t"].to_numpy(), cells)
    out["origin_cell"] = mapped["cell_id"].to_numpy()
    return out


def error_by_cell(preds: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Distancia mediana p50→real (``dist_native_km``) por celda de origen.

    Devuelve DataFrame [cell_id, lat_c, lon_c, n, median_error_km], solo
    celdas activas con al menos una observación de origen en ellas.
    """
    df = _attach_origin_cell(preds, cells)
    agg = (
        df.groupby("origin_cell")
        .agg(
            n=("dist_native_km", "size"),
            median_error_km=("dist_native_km", "median"),
        )
        .reset_index()
        .rename(columns={"origin_cell": "cell_id"})
    )
    out = agg.merge(cells[["cell_id", "lat_c", "lon_c"]], on="cell_id", how="inner")
    return out.sort_values("cell_id").reset_index(drop=True)


def coverage_by_cell(preds: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Cobertura empírica MARGINAL por celda de origen.

    Por fila, cobertura marginal = media de ``in_interval_lat`` e
    ``in_interval_lon`` (los dos intervalos [p10,p90] por eje, nominal ≈0,80).
    Se promedia por celda. Devuelve [cell_id, lat_c, lon_c, n,
    coverage_marginal].
    """
    df = _attach_origin_cell(preds, cells)
    df["_cov_marginal"] = 0.5 * (
        df["in_interval_lat"].astype(float) + df["in_interval_lon"].astype(float)
    )
    agg = (
        df.groupby("origin_cell")
        .agg(
            n=("_cov_marginal", "size"),
            coverage_marginal=("_cov_marginal", "mean"),
        )
        .reset_index()
        .rename(columns={"origin_cell": "cell_id"})
    )
    out = agg.merge(cells[["cell_id", "lat_c", "lon_c"]], on="cell_id", how="inner")
    return out.sort_values("cell_id").reset_index(drop=True)


_REGIME_LABEL = {0: "estacionario", 1: "migración"}


def _metric_block(df: pd.DataFrame) -> dict:
    """Métricas comunes sobre un bloque de predicciones."""
    top1 = float((df["pred_cell_top1"] == df["true_cell"]).mean())
    cov_marginal = float(
        0.5
        * (
            df["in_interval_lat"].astype(float) + df["in_interval_lon"].astype(float)
        ).mean()
    )
    cov_joint = float(
        (
            df["in_interval_lat"].astype(bool) & df["in_interval_lon"].astype(bool)
        ).mean()
    )
    return {
        "n": int(len(df)),
        "top1": top1,
        "dist_median_km": float(df["dist_native_km"].median()),
        "coverage_marginal": cov_marginal,
        "coverage_joint": cov_joint,
    }


def metrics_by_regime(preds: pd.DataFrame) -> pd.DataFrame:
    """Métricas por régimen HMM causal (estacionario vs migración).

    Columnas: regimen, n, top1, dist_median_km, coverage_marginal,
    coverage_joint. La cobertura conjunta (real dentro del rectángulo) se
    incluye junto a la marginal para documentar que la primera es menor que
    el 80 % nominal (matiz del spec §6.2).
    """
    rows = []
    for state, sub in preds.groupby("state_b_causal"):
        block = _metric_block(sub)
        block["regimen"] = _REGIME_LABEL.get(int(state), str(state))
        rows.append(block)
    cols = ["regimen", "n", "top1", "dist_median_km", "coverage_marginal", "coverage_joint"]
    return pd.DataFrame(rows)[cols].sort_values("regimen").reset_index(drop=True)


def metrics_by_month(preds: pd.DataFrame) -> pd.DataFrame:
    """Mismas métricas por mes calendario (cruce con la fenología).

    Columnas: mes, n, top1, dist_median_km, coverage_marginal, coverage_joint.
    """
    df = preds.copy()
    df["mes"] = pd.to_datetime(df["date_utc"]).dt.month
    rows = []
    for mes, sub in df.groupby("mes"):
        block = _metric_block(sub)
        block["mes"] = int(mes)
        rows.append(block)
    cols = ["mes", "n", "top1", "dist_median_km", "coverage_marginal", "coverage_joint"]
    return pd.DataFrame(rows)[cols].sort_values("mes").reset_index(drop=True)
