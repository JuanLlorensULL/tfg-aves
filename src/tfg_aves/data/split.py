"""Split temporal por ave, compartido por O3 (HMM) y O4 (clasificadores).

El split se calcula por ``bird_id`` ordenando por fecha: primeros
``train_frac`` -> bloque (train+val); últimos -> test; dentro del bloque, el
último ``val_frac_of_train`` -> val. Vive en ``tfg_aves.data`` (upstream de
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
    """Devuelve una copia de ``df`` con una columna ``split`` en {train,val,test}.

    Asigna la etiqueta a cada fila según el mismo criterio que
    ``split_temporal_per_bird``. Útil para persistir el split en parquet y que
    los consumidores aguas abajo particionen sin recalcularlo.
    """
    out = df.sort_values(["bird_id", "date_utc"]).reset_index(drop=True).copy()
    out["split"] = "train"
    for _bird, sub in out.groupby("bird_id", sort=False):
        n = len(sub)
        n_train_val = int(round(n * train_frac))
        n_val = int(round(n_train_val * val_frac_of_train))
        idx = sub.index
        out.loc[idx[: n_train_val - n_val], "split"] = "train"
        out.loc[idx[n_train_val - n_val : n_train_val], "split"] = "val"
        out.loc[idx[n_train_val:], "split"] = "test"
    return out
