"""Parser utilities for the authorized Irish CER half-hour data.

The restricted archives are not redistributed.  The functions here consume
checksum-verified local copies in bounded chunks, select residential meters
from the official allocation table (``Code == 1``), and retain only meter-days
with exactly the ordinary slots 1--48.

The official CER FAQ documents daylight-saving rows with slots 49 and 50.
They are valid source rows and therefore accepted by the parser, but a 50-slot
day is excluded from the frozen primary 48-slot daily-profile branch.  The
``timestamp`` emitted for slots 49/50 is an ordinal half-hour timestamp that
rolls beyond midnight; ``date`` and ``is_dst_extra_slot`` preserve the source
day semantics and prevent it from being mistaken for civil time.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


CER_EPOCH = date(2009, 1, 1)
ORDINARY_HALF_HOUR_SLOTS = 48
MAX_DST_HALF_HOUR_SLOT = 50
ISET_DAY_BRANCHES = {
    "complete_1_48",
    "trim_extra",
    "aggregate_duplicate_slots",
    "interpolate_grid",
}


def _integral_numeric(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    numeric_array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"CER {field} values must be finite")
    if not np.equal(numeric_array, np.floor(numeric_array)).all():
        raise ValueError(f"CER {field} values must be integers")
    return numeric.astype("int64")


def decode_day_time_code(codes: pd.Series) -> pd.DataFrame:
    """Decode ``DDDSS`` codes, accepting documented DST slots 49 and 50.

    Slot 1 is the first half hour of the encoded day.  Slots 49 and 50 occur on
    25-hour daylight-saving days; callers should use ``is_dst_extra_slot`` and
    the encoded ``date`` rather than interpreting their ordinal timestamps as
    ordinary next-day readings.
    """

    numeric = _integral_numeric(codes, field="day/time code")
    day_number = numeric // 100
    half_hour_slot = numeric % 100
    if not day_number.between(1, 999).all():
        raise ValueError("CER day number must be between 1 and 999")
    if not half_hour_slot.between(1, MAX_DST_HALF_HOUR_SLOT).all():
        raise ValueError("CER half-hour slot must be between 1 and 50")
    days = pd.to_datetime(CER_EPOCH) + pd.to_timedelta(day_number - 1, unit="D")
    ordinal_timestamps = days + pd.to_timedelta(
        (half_hour_slot - 1) * 30, unit="min"
    )
    return pd.DataFrame(
        {
            "day_number": day_number.to_numpy(),
            "half_hour_slot": half_hour_slot.to_numpy(),
            "date": days.to_numpy(),
            "timestamp": ordinal_timestamps.to_numpy(),
            "is_dst_extra_slot": (half_hour_slot > ORDINARY_HALF_HOUR_SLOTS).to_numpy(),
        }
    )


def _validate_reading_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] != 3:
        raise ValueError("expected exactly three CER columns")
    validated = frame.copy()
    validated.columns = ["meter_id", "day_time_code", "kwh"]
    validated["meter_id"] = _integral_numeric(
        validated["meter_id"], field="meter ID"
    )
    validated["kwh"] = pd.to_numeric(validated["kwh"], errors="raise").astype(
        "float64"
    )
    if not np.isfinite(validated["kwh"]).all() or (validated["kwh"] < 0).any():
        raise ValueError("CER kWh values must be finite and non-negative")
    decoded = decode_day_time_code(validated["day_time_code"])
    return pd.concat([validated.reset_index(drop=True), decoded], axis=1)


def read_cer_text(path_or_buffer: object, *, nrows: int | None = None) -> pd.DataFrame:
    """Read and validate one official whitespace-delimited text member."""

    frame = pd.read_csv(
        path_or_buffer,
        sep=r"\s+",
        header=None,
        nrows=nrows,
    )
    return _validate_reading_frame(frame)


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _allocation_source_buffer(path_or_buffer: object) -> object:
    """Decode official tab exports while tolerating a UTF-8 BOM or cp1252."""

    if isinstance(path_or_buffer, (str, Path)):
        payload = Path(path_or_buffer).read_bytes()
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("cp1252")
        return io.StringIO(text)
    if isinstance(path_or_buffer, bytes):
        try:
            text = path_or_buffer.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = path_or_buffer.decode("cp1252")
        return io.StringIO(text)
    return path_or_buffer


def _find_allocation_column(
    frame: pd.DataFrame,
    aliases: set[str],
    *,
    description: str,
) -> object:
    matches = [column for column in frame.columns if _normalise_header(column) in aliases]
    if len(matches) != 1:
        raise ValueError(
            f"allocation table must contain exactly one {description} column; "
            f"found {matches}"
        )
    return matches[0]


def read_allocation_table(path_or_buffer: object) -> pd.DataFrame:
    """Parse the official allocation TSV into ``meter_id, allocation_code``.

    Header matching ignores case, punctuation, spaces, and a possible byte
    order mark.  Exact duplicate rows are collapsed; conflicting assignments
    for one meter are rejected rather than silently selecting one.
    """

    source = _allocation_source_buffer(path_or_buffer)
    frame = pd.read_csv(source, sep=None, engine="python", skip_blank_lines=True)
    if frame.empty:
        raise ValueError("allocation table is empty")
    meter_column = _find_allocation_column(
        frame,
        {"id", "meterid", "meteridentifier"},
        description="meter ID",
    )
    code_column = _find_allocation_column(
        frame,
        {"code", "allocationcode", "participantcode"},
        description="allocation code",
    )
    allocation = pd.DataFrame(
        {
            "meter_id": _integral_numeric(frame[meter_column], field="allocation meter ID"),
            "allocation_code": _integral_numeric(
                frame[code_column], field="allocation code"
            ),
        }
    )
    conflicting = allocation.groupby("meter_id")["allocation_code"].nunique() > 1
    if conflicting.any():
        meters = conflicting[conflicting].index.astype(int).tolist()
        raise ValueError(f"conflicting allocation codes for meter IDs: {meters}")
    return (
        allocation.drop_duplicates()
        .sort_values("meter_id", kind="stable")
        .reset_index(drop=True)
    )


def residential_meter_ids(
    allocation_or_source: pd.DataFrame | object,
    *,
    residential_code: int = 1,
) -> np.ndarray:
    """Return sorted official residential meter IDs (allocation Code 1)."""

    if isinstance(allocation_or_source, pd.DataFrame) and {
        "meter_id",
        "allocation_code",
    }.issubset(allocation_or_source.columns):
        allocation = allocation_or_source
    else:
        allocation = read_allocation_table(allocation_or_source)
    meter_ids = allocation.loc[
        allocation["allocation_code"] == residential_code, "meter_id"
    ].to_numpy(dtype=np.int64)
    return np.unique(meter_ids)


def daily_profiles(
    frame: pd.DataFrame,
    *,
    complete_only: bool | None = None,
    policy: str = "complete_1_48",
) -> pd.DataFrame:
    """Pivot readings into 48-value meter-day profiles.

    The frozen primary branch uses ``policy="complete_1_48"``: a group is
    retained only when it contains exactly one of every slot 1--48 and no
    slots 49/50. ``trim_extra`` admits a 50-slot day after dropping 49/50,
    ``aggregate_duplicate_slots`` first means duplicate meter/day/slot rows,
    and ``interpolate_grid`` fills a 48-slot grid from at least two ordinary
    observations. Extraction counts are attached at
    ``result.attrs["cer_profile_stats"]``. The historical ``complete_only``
    boolean remains a compatibility alias for strict versus interpolation.
    """

    if complete_only is not None:
        policy = "complete_1_48" if complete_only else "interpolate_grid"
    if policy not in ISET_DAY_BRANCHES:
        raise ValueError(
            f"unknown ISET day branch {policy!r}; "
            f"expected one of {sorted(ISET_DAY_BRANCHES)}"
        )
    required = {"meter_id", "day_number", "half_hour_slot", "kwh"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        empty = pd.DataFrame(
            columns=["meter_id", "day_number"]
            + [f"hh_{slot:02d}" for slot in range(1, 49)]
        )
        empty.attrs["cer_profile_stats"] = {
            "meter_days_total": 0,
            "complete_48_slot_days": 0,
            "excluded_days": 0,
            "dst_extra_slot_days": 0,
        }
        return empty
    keys = ["meter_id", "day_number"]
    slot_keys = [*keys, "half_hour_slot"]
    duplicates = frame.duplicated(slot_keys, keep=False)
    if duplicates.any() and policy not in {
        "aggregate_duplicate_slots",
        "interpolate_grid",
    }:
        raise ValueError("duplicate meter/day/half-hour readings found")
    if not frame["half_hour_slot"].between(1, MAX_DST_HALF_HOUR_SLOT).all():
        raise ValueError("CER half-hour slot must be between 1 and 50")

    working = frame.loc[:, [*slot_keys, "kwh"]].copy()
    if duplicates.any():
        working = (
            working.groupby(slot_keys, sort=False, as_index=False)["kwh"]
            .mean()
        )
    grouped_slots = working.groupby(keys, sort=False)["half_hour_slot"]
    group_size = grouped_slots.size()
    group_min = grouped_slots.min()
    group_max = grouped_slots.max()
    complete_mask = (
        (group_size == ORDINARY_HALF_HOUR_SLOTS)
        & (group_min == 1)
        & (group_max == ORDINARY_HALF_HOUR_SLOTS)
    )
    dst_extra = grouped_slots.max() > ORDINARY_HALF_HOUR_SLOTS
    stats = {
        "meter_days_total": int(group_size.size),
        "complete_48_slot_days": int(complete_mask.sum()),
        "excluded_days": int((~complete_mask).sum()),
        "dst_extra_slot_days": int(dst_extra.sum()),
    }

    ordinary = working.loc[
        working["half_hour_slot"] <= ORDINARY_HALF_HOUR_SLOTS
    ].copy()
    ordinary_counts = ordinary.groupby(keys, sort=False)["half_hour_slot"].nunique()
    if policy == "complete_1_48":
        selected_keys = complete_mask[complete_mask].index
    elif policy in {"trim_extra", "aggregate_duplicate_slots"}:
        selected_keys = ordinary_counts[ordinary_counts == 48].index
    else:
        selected_keys = ordinary_counts[ordinary_counts >= 2].index
    selected = ordinary.set_index(keys)
    selected = selected.loc[selected.index.isin(selected_keys)].reset_index()
    profiles = selected.pivot(
        index=keys,
        columns="half_hour_slot",
        values="kwh",
    ).reindex(columns=range(1, ORDINARY_HALF_HOUR_SLOTS + 1))
    profiles.columns = [
        f"hh_{slot:02d}" for slot in range(1, ORDINARY_HALF_HOUR_SLOTS + 1)
    ]
    profiles = profiles.reset_index().sort_values(keys, kind="stable").reset_index(drop=True)
    if policy == "interpolate_grid":
        value_columns = [f"hh_{slot:02d}" for slot in range(1, 49)]
        profiles[value_columns] = profiles[value_columns].interpolate(
            axis=1,
            limit_direction="both",
        )
    if profiles.filter(like="hh_").isna().any().any():
        raise AssertionError(
            f"ISET day branch {policy!r} produced unresolved missing values"
        )
    profiles.attrs["cer_profile_stats"] = stats
    profiles.attrs["iset_day_branch"] = policy
    return profiles


def _archive_paths(
    archives: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(archives, (str, Path)):
        return [Path(archives)]
    return [Path(path) for path in archives]


def iter_authorized_cer_zip_chunks(
    archives: str | Path | Iterable[str | Path],
    *,
    residential_ids: Iterable[int] | np.ndarray | None = None,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Yield validated reading chunks directly from authorized ZIP members.

    No archive is extracted and no restricted data are written by this
    function.  Each official archive must contain exactly one non-hidden text
    member.  When ``residential_ids`` is supplied, non-residential readings are
    filtered before yielding.  Source provenance is available in each chunk's
    ``attrs`` as ``source_archive``, ``source_member``, and ``source_chunk``.

    Meter-days can cross chunk boundaries, so callers must concatenate or use a
    boundary-aware aggregation stage before calling :func:`daily_profiles`.
    """

    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    selected_ids: set[int] | None
    if residential_ids is None:
        selected_ids = None
    else:
        selected_ids = {int(value) for value in residential_ids}
    source_chunk = 0
    for archive_path in _archive_paths(archives):
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        try:
            archive = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid CER ZIP archive: {archive_path}") from exc
        with archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and member.filename.lower().endswith(".txt")
                and not Path(member.filename).name.startswith(".")
                and "__MACOSX" not in Path(member.filename).parts
            ]
            if len(members) != 1:
                raise ValueError(
                    f"expected one CER text member in {archive_path}, found "
                    f"{[member.filename for member in members]}"
                )
            member = members[0]
            if member.flag_bits & 0x1:
                raise ValueError(f"encrypted CER ZIP member is unsupported: {member.filename}")
            with archive.open(member, "r") as handle:
                reader = pd.read_csv(
                    handle,
                    sep=r"\s+",
                    header=None,
                    chunksize=chunksize,
                )
                for raw_chunk in reader:
                    if selected_ids is not None:
                        if raw_chunk.shape[1] != 3:
                            raise ValueError("expected exactly three CER columns")
                        # The paper selects residential meters before building
                        # profiles.  Parse only the ID column globally, apply
                        # that declared selection, and then validate the
                        # selected readings.  This prevents malformed unused
                        # SME/other time codes from becoming a silent global
                        # cleaning rule or blocking residential preparation.
                        meter_ids = _integral_numeric(
                            raw_chunk.iloc[:, 0], field="meter ID"
                        )
                        raw_chunk = raw_chunk.loc[
                            meter_ids.isin(selected_ids)
                        ].reset_index(drop=True)
                        if raw_chunk.empty:
                            continue
                    chunk = _validate_reading_frame(raw_chunk)
                    if chunk.empty:
                        continue
                    chunk.attrs.update(
                        {
                            "source_archive": str(archive_path),
                            "source_member": member.filename,
                            "source_chunk": source_chunk,
                        }
                    )
                    source_chunk += 1
                    yield chunk
