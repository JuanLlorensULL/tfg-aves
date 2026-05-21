"""Construcción de pares de transición y matrices de counts mensuales."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_transitions(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Convierte un DataFrame diario discretizado en pares válidos (t, t+1).

    Una transición se emite si y sólo si:
    - misma ave (``bird_id``),
    - ``is_valid`` en ambos días,
    - fechas consecutivas (``date_{t+1} - date_t == 1 día``).

    El ``month_int`` (1-12) es el del **origen** ``t``.
    """
    df = df_daily.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    # Normalizar date_utc a datetime64 para una resta robusta independientemente
    # de si la columna viene como datetime.date (parquet) o datetime64.
    df["_date_dt"] = pd.to_datetime(df["date_utc"])

    same_bird = df["bird_id"].shift(-1) == df["bird_id"]
    both_valid = df["is_valid"] & df["is_valid"].shift(-1).fillna(False).astype(bool)
    delta = df["_date_dt"].shift(-1) - df["_date_dt"]
    consecutive = delta == pd.Timedelta(days=1)

    mask = same_bird & both_valid & consecutive

    pairs = pd.DataFrame(
        {
            "bird_id": df["bird_id"][mask].values,
            "date_t": df["date_utc"][mask].values,
            "month_int": df["_date_dt"][mask].dt.month.astype(int).values,
            "cell_from": df["cell_id"][mask].values,
            "cell_to": df["cell_id"].shift(-1)[mask].values,
            "lat_from": df["lat"][mask].values,
            "lon_from": df["lon"][mask].values,
            "lat_to": df["lat"].shift(-1)[mask].values,
            "lon_to": df["lon"].shift(-1)[mask].values,
        }
    )
    return pairs.reset_index(drop=True)


def build_counts(df_transitions: pd.DataFrame, cells: list[str]) -> np.ndarray:
    """Tensor de counts ``(12, n_cells, n_cells)`` por mes y par de celdas."""
    n = len(cells)
    counts = np.zeros((12, n, n), dtype=np.int32)
    if df_transitions.empty:
        return counts

    idx_by_cell = {c: i for i, c in enumerate(cells)}
    months = df_transitions["month_int"].to_numpy() - 1
    from_idx = np.array(
        [idx_by_cell.get(c, -1) for c in df_transitions["cell_from"].to_numpy()]
    )
    to_idx = np.array(
        [idx_by_cell.get(c, -1) for c in df_transitions["cell_to"].to_numpy()]
    )

    valid = (from_idx >= 0) & (to_idx >= 0)
    np.add.at(counts, (months[valid], from_idx[valid], to_idx[valid]), 1)
    return counts
