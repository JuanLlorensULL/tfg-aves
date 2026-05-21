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
