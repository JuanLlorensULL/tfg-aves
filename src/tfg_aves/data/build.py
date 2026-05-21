"""Orquestación end-to-end de la pipeline de O1."""
from __future__ import annotations

from pathlib import Path

from . import load as _load_mod
from ._paths import PROCESSED
from .clean import (
    drop_invalid_coords_and_dupes,
    drop_movebank_flags,
    drop_speed_outliers,
)
from .daily import build_daily, filter_birds_by_validity


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
    df_raw = _load_mod.load_raw(path=_load_mod.RAW_CSV)
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
