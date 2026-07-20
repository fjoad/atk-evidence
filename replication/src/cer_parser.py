"""Parser utilities for the official Irish CER half-hour consumption files.

The restricted archives are not redistributed here. This module is ready to
consume them after authorized download and checksum verification.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


CER_EPOCH = date(2009, 1, 1)


def decode_day_time_code(codes: pd.Series) -> pd.DataFrame:
    """Decode the manifest's DDDSS integer into day, slot, and timestamp."""

    numeric = pd.to_numeric(codes, errors="raise").astype("int64")
    day_number = numeric // 100
    half_hour_slot = numeric % 100
    if not day_number.between(1, 999).all():
        raise ValueError("CER day number must be between 1 and 999")
    if not half_hour_slot.between(1, 48).all():
        raise ValueError("CER half-hour slot must be between 1 and 48")
    days = pd.to_datetime(CER_EPOCH) + pd.to_timedelta(day_number - 1, unit="D")
    timestamps = days + pd.to_timedelta((half_hour_slot - 1) * 30, unit="min")
    return pd.DataFrame(
        {
            "day_number": day_number.to_numpy(),
            "half_hour_slot": half_hour_slot.to_numpy(),
            "date": days.to_numpy(),
            "timestamp": timestamps.to_numpy(),
        }
    )


def read_cer_text(path_or_buffer: object, *, nrows: int | None = None) -> pd.DataFrame:
    """Read one official three-column text member and validate its fields."""

    frame = pd.read_csv(
        path_or_buffer,
        sep=r"\s+",
        header=None,
        names=["meter_id", "day_time_code", "kwh"],
        nrows=nrows,
    )
    if frame.shape[1] != 3:
        raise ValueError("expected exactly three CER columns")
    frame["meter_id"] = pd.to_numeric(frame["meter_id"], errors="raise").astype("int64")
    frame["kwh"] = pd.to_numeric(frame["kwh"], errors="raise").astype("float64")
    if not np.isfinite(frame["kwh"]).all() or (frame["kwh"] < 0).any():
        raise ValueError("CER kWh values must be finite and non-negative")
    decoded = decode_day_time_code(frame["day_time_code"])
    return pd.concat([frame.reset_index(drop=True), decoded], axis=1)


def daily_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot validated long-form readings into one 48-value row per meter/day."""

    required = {"meter_id", "day_number", "half_hour_slot", "kwh"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    duplicates = frame.duplicated(["meter_id", "day_number", "half_hour_slot"])
    if duplicates.any():
        raise ValueError("duplicate meter/day/half-hour readings found")
    profiles = frame.pivot(
        index=["meter_id", "day_number"],
        columns="half_hour_slot",
        values="kwh",
    ).reindex(columns=range(1, 49))
    profiles.columns = [f"hh_{slot:02d}" for slot in range(1, 49)]
    return profiles.reset_index()

