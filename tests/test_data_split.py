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
    min_test = test[test["bird_id"] == "A"]["date_utc"].min()
    max_train = train[train["bird_id"] == "A"]["date_utc"].max()
    assert min_test > max_train


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
