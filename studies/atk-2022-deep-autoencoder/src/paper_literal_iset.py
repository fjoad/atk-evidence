"""Paper-literal CER/ISET preparation for Tables III--V.

The production entry point accepts the six checksum-gated CER consumption
archives and the official allocation table.  A prepared 48-slot profile frame
is also accepted so deterministic fixture tests can exercise the complete
pipeline without possessing or redistributing the restricted source data.

This module deliberately retains the order printed by Takiddin et al. (2022):
feature-wise standardization is fitted jointly to benign and malicious rows
before splitting, ADASYN is applied inside the anomaly test set, and ADASYN is
applied before the supervised split.  These choices leak held-out information;
they are implemented only for the frozen exploratory paper-literal track.

Ambiguity A06 is resolved by deriving the six malicious profiles only from
meters assigned to the held-out one-third.  Consequently anomaly training and
validation contain only benign profiles, while every malicious test profile
comes from a meter unseen by those partitions.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

from attacks import generate_attack
from cer_parser import (
    ISET_DAY_BRANCHES,
    daily_profiles,
    iter_authorized_cer_zip_chunks,
    read_allocation_table,
    residential_meter_ids,
)


SCHEMA_VERSION = 1
DEFAULT_DATA_SEED = 20260721
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_ADASYN_NEIGHBORS = 5
DEFAULT_TABLE_V_SAMPLES = 3000
ATTACK_POPULATION_BRANCHES = {"all_customer_m", "heldout_b2_m"}
SCALING_BRANCHES = {
    "joint_featurewise",
    "per_class_featurewise",
    "per_profile",
    "train_benign_only",
}
ANOMALY_ADASYN_BRANCHES = {"test_set_as_printed", "none"}
SUPERVISED_ADASYN_BRANCHES = {
    "before_row_split",
    "customer_split_then_train_only",
}
ATTACK1_SCOPE_BRANCHES = {
    "per_profile",
    "per_customer_matrix",
    "per_generated_dataset",
}
ATTACK2_GRANULARITY_BRANCHES = {"per_half_hour", "per_hour_pair"}
ATTACK3_INTERVAL_BRANCHES = {
    "valid_fit_addition",
    "printed_start_truncate",
    "printed_start_wrap",
}
ATTACK_HOUR_MAPPING_BRANCHES = {"two_slots_per_hour", "direct_48_index"}
METER_POPULATION_BRANCHES = {"all_4225", "seeded_3000"}
SPLIT_UNIT_BRANCHES = {"customer_disjoint", "row_random"}
ATTACK_REGENERATION_BRANCHES = {
    "fixed_per_data_seed",
    "regenerate_per_model_seed",
    "regenerate_per_experiment",
}
HALF_HOUR_COLUMNS = tuple(f"hh_{slot:02d}" for slot in range(1, 49))
ALLOCATION_FILENAME = "SME and Residential allocations.tab"
OFFICIAL_ALLOCATION_BRANCH = "official-tab-v1"
SCIENCEDB_ALLOCATION_BRANCH = "sciencedb-csv-semantic-equivalence-v1"
SCIENCEDB_ALLOCATION_FILENAME = "SME_and_Residential_allocations.csv"
SCIENCEDB_ALLOCATION_MD5 = "89263f89253cf56b857079986ae73096"
SCIENCEDB_ALLOCATION_SHA256 = (
    "96298be047f34ba91fe281c899b440d2b28747b4f102af6f239dbbd93dd354d4"
)
SCIENCEDB_ALLOCATION_ID_CODE_SHA256 = (
    "c4e37d8bb679674f82e3365206c34e864b3050ca385a60f98a820e98e1beb696"
)
OFFICIAL_CER_MD5: dict[str, str] = {
    "File1.txt.zip": "00203f66f3f5e5201b20ed160b787684",
    "File2.txt.zip": "5e3af1474d3c8976e2e1e0f8c1969507",
    "File3.txt.zip": "b537785f8b37cb3e89103600d39da8ff",
    "File4.txt.zip": "53ec9e70c1610b74ae72417cc010a0c3",
    "File5.txt.zip": "6f8c7c9dfba3bbfbff0e5f1703e122fc",
    "File6.txt.zip": "c0a435d0359974f23ce434b5e838e251",
    ALLOCATION_FILENAME: "124c10711ab1e7c52cb7317c8f69e42e",
}
ARCHIVE_FILENAMES = tuple(f"File{index}.txt.zip" for index in range(1, 7))


def allocation_filename(branch: str) -> str:
    """Return the only admitted allocation filename for one explicit branch."""

    if branch == OFFICIAL_ALLOCATION_BRANCH:
        return ALLOCATION_FILENAME
    if branch == SCIENCEDB_ALLOCATION_BRANCH:
        return SCIENCEDB_ALLOCATION_FILENAME
    raise ValueError(
        f"unknown CER allocation branch {branch!r}; choose "
        f"{OFFICIAL_ALLOCATION_BRANCH!r} or {SCIENCEDB_ALLOCATION_BRANCH!r}"
    )


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _id_digest(sample_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids.astype(str):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _allocation_id_code_digest(allocation: pd.DataFrame) -> str:
    canonical = (
        allocation[["meter_id", "allocation_code"]]
        .sort_values("meter_id", kind="stable")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def verify_authorized_iset_files(
    archive_paths: Iterable[str | Path],
    allocation_path: str | Path,
    *,
    allocation_branch: str = OFFICIAL_ALLOCATION_BRANCH,
    expected_md5: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify the complete seven-file CER input gate before reading any data.

    ``expected_md5`` defaults to the official ISSDA manifest.  Supplying a
    fixture manifest is useful for tests, and is recorded in the returned
    provenance; production callers should never override the default.
    """

    admitted_allocation_filename = allocation_filename(allocation_branch)
    if expected_md5 is None:
        checksums = {
            filename: OFFICIAL_CER_MD5[filename]
            for filename in ARCHIVE_FILENAMES
        }
        checksums[admitted_allocation_filename] = (
            OFFICIAL_CER_MD5[ALLOCATION_FILENAME]
            if allocation_branch == OFFICIAL_ALLOCATION_BRANCH
            else SCIENCEDB_ALLOCATION_MD5
        )
    else:
        # Test fixtures may replace the bytes but not the seven-file shape.
        checksums = dict(expected_md5)
    expected_filenames = {*ARCHIVE_FILENAMES, admitted_allocation_filename}
    if set(checksums) != expected_filenames:
        missing = sorted(expected_filenames.difference(checksums))
        extra = sorted(set(checksums).difference(expected_filenames))
        raise ValueError(
            "CER checksum manifest must name exactly the six archives and "
            f"allocation table; missing={missing}, extra={extra}"
        )

    paths = [Path(path) for path in archive_paths]
    if len(paths) != 6:
        raise ValueError(f"expected exactly six CER archives, found {len(paths)}")
    by_name: dict[str, Path] = {}
    for path in paths:
        if path.name in by_name:
            raise ValueError(f"duplicate CER archive filename: {path.name}")
        by_name[path.name] = path
    if set(by_name) != set(ARCHIVE_FILENAMES):
        missing = sorted(set(ARCHIVE_FILENAMES).difference(by_name))
        extra = sorted(set(by_name).difference(ARCHIVE_FILENAMES))
        raise ValueError(
            f"CER archive set is incomplete or misnamed; missing={missing}, extra={extra}"
        )

    allocation = Path(allocation_path)
    if allocation.name != admitted_allocation_filename:
        raise ValueError(
            f"allocation table must be named {admitted_allocation_filename!r}, "
            f"found {allocation.name!r}"
        )
    all_paths = {**by_name, admitted_allocation_filename: allocation}
    records: dict[str, dict[str, Any]] = {}
    for filename in (*ARCHIVE_FILENAMES, admitted_allocation_filename):
        path = all_paths[filename]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_digest(path, "md5")
        expected = checksums[filename].lower()
        if actual.lower() != expected:
            raise ValueError(
                f"CER checksum mismatch for {filename}: {actual} != {expected}"
            )
        records[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "algorithm": "md5",
            "expected": expected,
            "actual": actual.lower(),
            "status": "verified",
        }
    if allocation_branch == SCIENCEDB_ALLOCATION_BRANCH and expected_md5 is None:
        actual_sha256 = _file_digest(allocation, "sha256")
        if actual_sha256 != SCIENCEDB_ALLOCATION_SHA256:
            raise ValueError(
                "CER ScienceDB allocation SHA-256 mismatch: "
                f"{actual_sha256} != {SCIENCEDB_ALLOCATION_SHA256}"
            )
        parsed = read_allocation_table(allocation)
        semantic_digest = _allocation_id_code_digest(parsed)
        if semantic_digest != SCIENCEDB_ALLOCATION_ID_CODE_SHA256:
            raise ValueError(
                "CER ScienceDB allocation ID/Code semantic digest mismatch: "
                f"{semantic_digest} != {SCIENCEDB_ALLOCATION_ID_CODE_SHA256}"
            )
        counts = parsed["allocation_code"].value_counts().to_dict()
        expected_counts = {1: 4225, 2: 485, 3: 1735}
        if parsed.shape[0] != 6445 or counts != expected_counts:
            raise ValueError(
                "CER ScienceDB allocation semantic cardinality mismatch: "
                f"rows={parsed.shape[0]}, counts={counts}"
            )
        records[admitted_allocation_filename].update(
            {
                "sha256": actual_sha256,
                "allocation_branch": allocation_branch,
                "semantic_id_code_sha256": semantic_digest,
                "rows": int(parsed.shape[0]),
                "allocation_counts": {str(key): int(value) for key, value in counts.items()},
                "identity": "semantic_equivalence_not_official_binary",
            }
        )
    return records


@dataclass(frozen=True)
class IsetPartition:
    """A model partition with row-level attack and source provenance."""

    values: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    source_profile_ids: np.ndarray
    meter_ids: np.ndarray
    day_numbers: np.ndarray
    attack_ids: np.ndarray
    source_refs: np.ndarray
    is_synthetic: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != 48:
            raise ValueError("ISET partition values must have shape (rows, 48)")
        rows = self.values.shape[0]
        for name, array in (
            ("labels", self.labels),
            ("sample_ids", self.sample_ids),
            ("source_profile_ids", self.source_profile_ids),
            ("meter_ids", self.meter_ids),
            ("day_numbers", self.day_numbers),
            ("attack_ids", self.attack_ids),
            ("source_refs", self.source_refs),
            ("is_synthetic", self.is_synthetic),
        ):
            if array.ndim != 1 or array.shape[0] != rows:
                raise ValueError(f"partition {name} must contain one value per row")
        if not np.isfinite(self.values).all():
            raise ValueError("ISET partition values must be finite")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("ISET partition labels must be binary")
        original = ~self.is_synthetic
        benign = original & (self.labels == 0)
        malicious = original & (self.labels == 1)
        if not (self.attack_ids[benign] == 0).all():
            raise ValueError("original benign rows must have attack ID zero")
        if not np.isin(self.attack_ids[malicious], np.arange(1, 7)).all():
            raise ValueError("original malicious rows must have attack IDs 1 through 6")
        if not (self.attack_ids[self.is_synthetic] == -1).all():
            raise ValueError("ADASYN rows must have attack ID -1")


def _empty_partition() -> IsetPartition:
    return IsetPartition(
        values=np.empty((0, 48), dtype=np.float32),
        labels=np.empty(0, dtype=np.int8),
        sample_ids=np.empty(0, dtype=str),
        source_profile_ids=np.empty(0, dtype=str),
        meter_ids=np.empty(0, dtype=str),
        day_numbers=np.empty(0, dtype=np.int64),
        attack_ids=np.empty(0, dtype=np.int8),
        source_refs=np.empty(0, dtype=str),
        is_synthetic=np.empty(0, dtype=bool),
    )


def _take(partition: IsetPartition, indices: np.ndarray) -> IsetPartition:
    index = np.asarray(indices, dtype=np.int64)
    return IsetPartition(
        values=np.ascontiguousarray(partition.values[index], dtype=np.float32),
        labels=np.ascontiguousarray(partition.labels[index], dtype=np.int8),
        sample_ids=np.ascontiguousarray(partition.sample_ids[index].astype(str)),
        source_profile_ids=np.ascontiguousarray(
            partition.source_profile_ids[index].astype(str)
        ),
        meter_ids=np.ascontiguousarray(partition.meter_ids[index].astype(str)),
        day_numbers=np.ascontiguousarray(partition.day_numbers[index], dtype=np.int64),
        attack_ids=np.ascontiguousarray(partition.attack_ids[index], dtype=np.int8),
        source_refs=np.ascontiguousarray(partition.source_refs[index].astype(str)),
        is_synthetic=np.ascontiguousarray(partition.is_synthetic[index], dtype=bool),
    )


def _concatenate(partitions: Iterable[IsetPartition]) -> IsetPartition:
    items = list(partitions)
    if not items:
        return _empty_partition()
    return IsetPartition(
        values=np.ascontiguousarray(
            np.concatenate([item.values for item in items]), dtype=np.float32
        ),
        labels=np.ascontiguousarray(
            np.concatenate([item.labels for item in items]), dtype=np.int8
        ),
        sample_ids=np.ascontiguousarray(
            np.concatenate([item.sample_ids for item in items]).astype(str)
        ),
        source_profile_ids=np.ascontiguousarray(
            np.concatenate([item.source_profile_ids for item in items]).astype(str)
        ),
        meter_ids=np.ascontiguousarray(
            np.concatenate([item.meter_ids for item in items]).astype(str)
        ),
        day_numbers=np.ascontiguousarray(
            np.concatenate([item.day_numbers for item in items]), dtype=np.int64
        ),
        attack_ids=np.ascontiguousarray(
            np.concatenate([item.attack_ids for item in items]), dtype=np.int8
        ),
        source_refs=np.ascontiguousarray(
            np.concatenate([item.source_refs for item in items]).astype(str)
        ),
        is_synthetic=np.ascontiguousarray(
            np.concatenate([item.is_synthetic for item in items]), dtype=bool
        ),
    )


@dataclass(frozen=True)
class IsetPaperLiteralData:
    """All CER/ISET partitions needed by Tables III, IV, and V."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    anomaly_train: IsetPartition
    anomaly_validation: IsetPartition
    anomaly_test: IsetPartition
    supervised_train: IsetPartition
    supervised_test: IsetPartition
    table_iv_order: np.ndarray
    table_v_benign: IsetPartition
    table_v_attacks: tuple[IsetPartition, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.scaler_mean.shape != (48,) or self.scaler_scale.shape != (48,):
            raise ValueError("ISET scaler arrays must each contain 48 values")
        if not np.isfinite(self.scaler_mean).all():
            raise ValueError("ISET scaler mean must be finite")
        if not np.isfinite(self.scaler_scale).all() or (
            self.scaler_scale <= 0
        ).any():
            raise ValueError("ISET scaler scale must be finite and positive")
        if self.table_iv_order.shape != (self.anomaly_train.values.shape[0],):
            raise ValueError("Table IV order must permute every anomaly training row")
        if not np.array_equal(
            np.sort(self.table_iv_order), np.arange(self.table_iv_order.size)
        ):
            raise ValueError("Table IV order is not a permutation")
        if len(self.table_v_attacks) != 6:
            raise ValueError("Table V requires six attack partitions")
        benign_sources = self.table_v_benign.source_profile_ids
        for attack_id, attack in enumerate(self.table_v_attacks, start=1):
            if attack.values.shape[0] != benign_sources.shape[0]:
                raise ValueError("Table V benign and attack partitions must be balanced")
            if not (attack.attack_ids == attack_id).all():
                raise ValueError("Table V attack partition has a mismatched attack ID")
            if not np.array_equal(attack.source_profile_ids, benign_sources):
                raise ValueError("Table V attacks must derive from the reused benign rows")

    def table_iv_subset(self, size: str) -> IsetPartition:
        """Return deterministic nested 1/2, 3/4, or full Table IV training rows."""

        fractions = {"half": 0.5, "three_quarter": 0.75, "full": 1.0}
        if size not in fractions:
            raise ValueError("Table IV size must be half, three_quarter, or full")
        count = int(np.floor(self.table_iv_order.size * fractions[size]))
        if size == "full":
            count = self.table_iv_order.size
        return _take(self.anomaly_train, self.table_iv_order[:count])

    def table_v_pair(self, attack_id: int) -> tuple[IsetPartition, IsetPartition]:
        """Return the one fixed benign partition and one attack-specific partition."""

        if attack_id not in range(1, 7):
            raise ValueError("Table V attack ID must be 1 through 6")
        return self.table_v_benign, self.table_v_attacks[attack_id - 1]


def _profiles_from_authorized_archives(
    archive_paths: list[Path],
    residential_ids: np.ndarray,
    *,
    chunksize: int,
    shard_count: int,
    scratch_dir: str | Path | None,
    iset_day: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bound raw-memory use by sharding residential rows on disk by meter ID."""

    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    scratch_base: Path | None = None
    if scratch_dir is not None:
        scratch_base = Path(scratch_dir)
        scratch_base.mkdir(parents=True, exist_ok=True)

    profiles_by_shard: list[pd.DataFrame] = []
    aggregate_stats = {
        "meter_days_total": 0,
        "complete_48_slot_days": 0,
        "excluded_days": 0,
        "dst_extra_slot_days": 0,
    }
    ordered_archives = sorted(
        archive_paths, key=lambda path: ARCHIVE_FILENAMES.index(path.name)
    )
    archive_index = {str(path): index for index, path in enumerate(ordered_archives)}
    with tempfile.TemporaryDirectory(prefix="atk-iset-shards-", dir=scratch_base) as temporary:
        temporary_path = Path(temporary)
        shard_has_header = np.zeros(shard_count, dtype=bool)
        for chunk in iter_authorized_cer_zip_chunks(
            ordered_archives,
            residential_ids=residential_ids,
            chunksize=chunksize,
        ):
            compact = chunk[
                ["meter_id", "day_number", "half_hour_slot", "kwh"]
            ].copy()
            compact["source_file_index"] = archive_index[chunk.attrs["source_archive"]]
            shards = np.mod(compact["meter_id"].to_numpy(dtype=np.int64), shard_count)
            for shard in np.unique(shards):
                shard_number = int(shard)
                target = temporary_path / f"shard-{shard_number:03d}.csv"
                compact.loc[shards == shard_number].to_csv(
                    target,
                    mode="a",
                    header=not shard_has_header[shard_number],
                    index=False,
                )
                shard_has_header[shard_number] = True

        for shard_number in np.flatnonzero(shard_has_header):
            source = pd.read_csv(temporary_path / f"shard-{shard_number:03d}.csv")
            profile_frame = daily_profiles(source, policy=iset_day)
            for key, value in profile_frame.attrs["cer_profile_stats"].items():
                aggregate_stats[key] += int(value)
            if profile_frame.empty:
                continue
            source_bounds = (
                source.groupby(["meter_id", "day_number"], sort=False)[
                    "source_file_index"
                ]
                .agg(["min", "max"])
                .reset_index()
            )
            profile_frame = profile_frame.merge(
                source_bounds, on=["meter_id", "day_number"], how="left", validate="one_to_one"
            )
            source_names: list[str] = []
            for first, last in zip(source_bounds["min"], source_bounds["max"], strict=True):
                first_name = ordered_archives[int(first)].name
                last_name = ordered_archives[int(last)].name
                source_names.append(
                    first_name if first_name == last_name else f"{first_name}|{last_name}"
                )
            # Merge preserves profile order, whereas source_bounds order is not
            # guaranteed by pandas across versions.  Map by the profile key.
            source_bounds["source_ref"] = source_names
            profile_frame = profile_frame.drop(columns=["min", "max"]).merge(
                source_bounds[["meter_id", "day_number", "source_ref"]],
                on=["meter_id", "day_number"],
                how="left",
                validate="one_to_one",
            )
            profiles_by_shard.append(profile_frame)

    if not profiles_by_shard:
        raise ValueError("no complete residential 48-slot CER profiles were found")
    profiles = (
        pd.concat(profiles_by_shard, ignore_index=True)
        .sort_values(["meter_id", "day_number"], kind="stable")
        .reset_index(drop=True)
    )
    return profiles, aggregate_stats


def _validate_profiles(
    profiles: pd.DataFrame,
    allocation_source: object | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"meter_id", "day_number", *HALF_HOUR_COLUMNS}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"prepared ISET profiles are missing columns: {sorted(missing)}")
    frame = profiles.copy()
    for column in ("meter_id", "day_number"):
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"prepared ISET {column} values must be finite integers")
        frame[column] = numeric.astype(np.int64)
    if not frame["day_number"].between(1, 999).all():
        raise ValueError("prepared ISET day_number values must be between 1 and 999")
    if frame.duplicated(["meter_id", "day_number"]).any():
        raise ValueError("prepared ISET profiles contain duplicate meter-days")
    values = frame.loc[:, HALF_HOUR_COLUMNS].apply(pd.to_numeric, errors="raise")
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("prepared ISET profile values must be finite and non-negative")
    frame.loc[:, HALF_HOUR_COLUMNS] = numeric

    selection: dict[str, Any]
    if allocation_source is not None:
        residential = residential_meter_ids(allocation_source, residential_code=1)
        before = frame.shape[0]
        frame = frame.loc[frame["meter_id"].isin(residential)].copy()
        selection = {
            "method": "official_allocation_code_1",
            "profiles_before_filter": int(before),
            "profiles_after_filter": int(frame.shape[0]),
            "residential_meter_ids_in_allocation": int(residential.size),
        }
    else:
        selection = {
            "method": "prepared_profiles_asserted_residential_code_1",
            "profiles_before_filter": int(frame.shape[0]),
            "profiles_after_filter": int(frame.shape[0]),
        }
    if frame.empty:
        raise ValueError("no residential ISET profiles remain")
    if "source_ref" not in frame.columns:
        frame["source_ref"] = "<prepared-profile-dataframe>"
    if frame["source_ref"].isna().any():
        raise ValueError("ISET source_ref may not be missing")
    frame["source_ref"] = frame["source_ref"].astype(str)
    frame = frame.sort_values(["meter_id", "day_number"], kind="stable").reset_index(
        drop=True
    )
    return frame, selection


def _split_meters(
    meter_ids: np.ndarray,
    *,
    train_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(meter_ids.astype(str))
    if unique.size < 3:
        raise ValueError("ISET preparation requires at least three residential meters")
    train, test = train_test_split(
        unique,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
    )
    return np.sort(train.astype(str)), np.sort(test.astype(str))


def _base_partition(
    values: np.ndarray,
    profile_ids: np.ndarray,
    meter_ids: np.ndarray,
    day_numbers: np.ndarray,
    source_refs: np.ndarray,
) -> IsetPartition:
    rows = values.shape[0]
    return IsetPartition(
        values=np.ascontiguousarray(values, dtype=np.float32),
        labels=np.zeros(rows, dtype=np.int8),
        sample_ids=np.ascontiguousarray(profile_ids.astype(str)),
        source_profile_ids=np.ascontiguousarray(profile_ids.astype(str)),
        meter_ids=np.ascontiguousarray(meter_ids.astype(str)),
        day_numbers=np.ascontiguousarray(day_numbers, dtype=np.int64),
        attack_ids=np.zeros(rows, dtype=np.int8),
        source_refs=np.ascontiguousarray(source_refs.astype(str)),
        is_synthetic=np.zeros(rows, dtype=bool),
    )


def _generate_attacks(
    benign: IsetPartition,
    *,
    seed: int,
    partition_name: str,
    attack1_scope: str = "per_profile",
    attack2_granularity: str = "per_half_hour",
    attack3_interval: str = "valid_fit_addition",
    attack_hour_mapping: str = "two_slots_per_hour",
) -> IsetPartition:
    """Generate six attack-major blocks with independent partition streams."""

    if benign.values.shape[0] == 0:
        raise ValueError("cannot generate ISET attacks from an empty partition")
    if attack1_scope not in ATTACK1_SCOPE_BRANCHES:
        raise ValueError("unsupported attack1_scope")
    if attack2_granularity not in ATTACK2_GRANULARITY_BRANCHES:
        raise ValueError("unsupported attack2_granularity")
    if attack3_interval not in ATTACK3_INTERVAL_BRANCHES:
        raise ValueError("unsupported attack3_interval")
    if attack_hour_mapping not in ATTACK_HOUR_MAPPING_BRANCHES:
        raise ValueError("unsupported attack_hour_mapping")
    name_words = np.frombuffer(
        hashlib.sha256(partition_name.encode("utf-8")).digest()[:16], dtype=np.uint32
    )
    root = np.random.SeedSequence([seed, *name_words.astype(int).tolist()])
    generators = [np.random.default_rng(child) for child in root.spawn(6)]
    rows = benign.values.shape[0]
    attacked = np.empty((rows * 6, 48), dtype=np.float32)
    for attack_id, generator in enumerate(generators, start=1):
        offset = (attack_id - 1) * rows
        dataset_factor = (
            float(generator.uniform(0.1, 0.8))
            if attack_id == 1 and attack1_scope == "per_generated_dataset"
            else None
        )
        customer_factors = (
            {
                meter_id: float(generator.uniform(0.1, 0.8))
                for meter_id in sorted(np.unique(benign.meter_ids.astype(str)))
            }
            if attack_id == 1 and attack1_scope == "per_customer_matrix"
            else {}
        )
        for row_index, profile in enumerate(benign.values):
            attack1_factor = dataset_factor
            if attack_id == 1 and attack1_scope == "per_customer_matrix":
                attack1_factor = customer_factors[str(benign.meter_ids[row_index])]
            attacked[offset + row_index] = generate_attack(
                profile,
                attack_id,
                generator,
                samples_per_hour=2,
                attack1_factor=attack1_factor,
                attack2_granularity=attack2_granularity,
                attack3_interval=attack3_interval,
                attack_hour_mapping=attack_hour_mapping,
            )
    attack_ids = np.repeat(np.arange(1, 7, dtype=np.int8), rows)
    source_profile_ids = np.tile(benign.source_profile_ids, 6)
    sample_ids = np.asarray(
        [
            f"{source_profile_id}:attack_{attack_id}"
            for attack_id in range(1, 7)
            for source_profile_id in benign.source_profile_ids
        ],
        dtype=str,
    )
    return IsetPartition(
        values=attacked,
        labels=np.ones(rows * 6, dtype=np.int8),
        sample_ids=sample_ids,
        source_profile_ids=np.ascontiguousarray(source_profile_ids.astype(str)),
        meter_ids=np.ascontiguousarray(np.tile(benign.meter_ids, 6).astype(str)),
        day_numbers=np.ascontiguousarray(np.tile(benign.day_numbers, 6), dtype=np.int64),
        attack_ids=attack_ids,
        source_refs=np.ascontiguousarray(np.tile(benign.source_refs, 6).astype(str)),
        is_synthetic=np.zeros(rows * 6, dtype=bool),
    )


def _select_meter_population(
    profiles: pd.DataFrame,
    *,
    branch: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Apply one declared residential-meter population interpretation."""

    if branch not in METER_POPULATION_BRANCHES:
        raise ValueError(
            f"unknown meter_population branch {branch!r}; "
            f"expected one of {sorted(METER_POPULATION_BRANCHES)}"
        )
    eligible_meters = np.sort(profiles["meter_id"].astype(str).unique())
    if branch == "seeded_3000" and eligible_meters.size > 3_000:
        selected_meters = np.sort(
            np.random.default_rng(seed + 3).choice(
                eligible_meters,
                size=3_000,
                replace=False,
            )
        )
        selected = profiles.loc[
            profiles["meter_id"].astype(str).isin(selected_meters)
        ].reset_index(drop=True)
    else:
        selected_meters = eligible_meters
        selected = profiles.reset_index(drop=True)
    return selected, eligible_meters, selected_meters


def _resolve_attack_seed(
    *,
    data_seed: int,
    attack_seed: int | None,
    attack_regeneration: str,
    model_seed: int | None,
    experiment_index: int | None,
) -> tuple[int, dict[str, Any]]:
    """Resolve every frozen attack-regeneration reading before generation."""

    if attack_regeneration not in ATTACK_REGENERATION_BRANCHES:
        raise ValueError(
            f"unknown attack_regeneration branch {attack_regeneration!r}; "
            f"expected one of {sorted(ATTACK_REGENERATION_BRANCHES)}"
        )
    if attack_seed is not None:
        if attack_regeneration != "fixed_per_data_seed":
            raise ValueError(
                "an explicit attack_seed may only be used with "
                "fixed_per_data_seed"
            )
        return int(attack_seed), {
            "source": "explicit_attack_seed",
            "data_seed": int(data_seed),
        }
    if attack_regeneration == "fixed_per_data_seed":
        return int(data_seed), {
            "source": "data_seed",
            "data_seed": int(data_seed),
        }
    if model_seed is None:
        raise ValueError(
            f"{attack_regeneration} requires an explicit model_seed"
        )
    if attack_regeneration == "regenerate_per_model_seed":
        return int(model_seed), {
            "source": "model_seed",
            "data_seed": int(data_seed),
            "model_seed": int(model_seed),
        }
    if experiment_index is None or experiment_index < 0:
        raise ValueError(
            "regenerate_per_experiment requires a non-negative experiment_index"
        )
    resolved = int(
        np.random.SeedSequence(
            [int(data_seed), int(model_seed), int(experiment_index), 0xA77AC]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    return resolved, {
        "source": "derived_per_experiment_seedsequence",
        "data_seed": int(data_seed),
        "model_seed": int(model_seed),
        "experiment_index": int(experiment_index),
    }


def _joint_feature_scaler(
    benign: np.ndarray,
    malicious: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute exact ddof=0 B+M moments without materializing their concat."""

    count = benign.shape[0] + malicious.shape[0]
    sums = benign.sum(axis=0, dtype=np.float64) + malicious.sum(
        axis=0, dtype=np.float64
    )
    sum_squares = np.square(benign, dtype=np.float64).sum(axis=0) + np.square(
        malicious, dtype=np.float64
    ).sum(axis=0)
    mean = sums / count
    variance = np.maximum(sum_squares / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    zero_variance = scale <= np.finfo(np.float32).eps
    scale[zero_variance] = 1.0
    return mean, scale, int(zero_variance.sum())


def _feature_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    zero_variance = scale <= np.finfo(np.float32).eps
    scale[zero_variance] = 1.0
    return mean, scale, int(zero_variance.sum())


def _standardize_partition(
    partition: IsetPartition,
    mean: np.ndarray,
    scale: np.ndarray,
) -> IsetPartition:
    return IsetPartition(
        values=np.ascontiguousarray((partition.values - mean) / scale, dtype=np.float32),
        labels=partition.labels.copy(),
        sample_ids=partition.sample_ids.copy(),
        source_profile_ids=partition.source_profile_ids.copy(),
        meter_ids=partition.meter_ids.copy(),
        day_numbers=partition.day_numbers.copy(),
        attack_ids=partition.attack_ids.copy(),
        source_refs=partition.source_refs.copy(),
        is_synthetic=partition.is_synthetic.copy(),
    )


def _per_profile_standardize(partition: IsetPartition) -> IsetPartition:
    mean = partition.values.mean(axis=1, keepdims=True, dtype=np.float64)
    scale = partition.values.std(axis=1, keepdims=True, dtype=np.float64)
    scale[scale <= np.finfo(np.float32).eps] = 1.0
    return IsetPartition(
        values=np.ascontiguousarray(
            (partition.values - mean) / scale, dtype=np.float32
        ),
        labels=partition.labels.copy(),
        sample_ids=partition.sample_ids.copy(),
        source_profile_ids=partition.source_profile_ids.copy(),
        meter_ids=partition.meter_ids.copy(),
        day_numbers=partition.day_numbers.copy(),
        attack_ids=partition.attack_ids.copy(),
        source_refs=partition.source_refs.copy(),
        is_synthetic=partition.is_synthetic.copy(),
    )


def _standardize_iset_by_branch(
    benign: IsetPartition,
    malicious: IsetPartition,
    train_benign: IsetPartition,
    *,
    branch: str,
) -> tuple[
    IsetPartition,
    IsetPartition,
    np.ndarray,
    np.ndarray,
    int,
    dict[str, Any],
]:
    """Apply a paper interpretation or corrected scaler with explicit provenance."""

    if branch not in SCALING_BRANCHES:
        raise ValueError(
            f"unknown ISET scaling branch {branch!r}; "
            f"expected one of {sorted(SCALING_BRANCHES)}"
        )
    if branch == "joint_featurewise":
        mean, scale, zero_count = _joint_feature_scaler(
            benign.values, malicious.values
        )
        return (
            _standardize_partition(benign, mean, scale),
            _standardize_partition(malicious, mean, scale),
            mean,
            scale,
            zero_count,
            {"fit_population": "benign_plus_selected_malicious_population"},
        )
    if branch == "train_benign_only":
        mean, scale, zero_count = _feature_scaler(train_benign.values)
        return (
            _standardize_partition(benign, mean, scale),
            _standardize_partition(malicious, mean, scale),
            mean,
            scale,
            zero_count,
            {"fit_population": "anomaly_train_benign_only"},
        )
    if branch == "per_profile":
        identity_mean = np.zeros(48, dtype=np.float64)
        identity_scale = np.ones(48, dtype=np.float64)
        return (
            _per_profile_standardize(benign),
            _per_profile_standardize(malicious),
            identity_mean,
            identity_scale,
            0,
            {
                "fit_population": "each_profile_independently",
                "reference_scaler": "identity_marker_only",
            },
        )

    benign_mean, benign_scale, benign_zero = _feature_scaler(benign.values)
    malicious_mean, malicious_scale, malicious_zero = _feature_scaler(
        malicious.values
    )
    return (
        _standardize_partition(benign, benign_mean, benign_scale),
        _standardize_partition(malicious, malicious_mean, malicious_scale),
        benign_mean,
        benign_scale,
        benign_zero + malicious_zero,
        {
            "fit_population": "benign_and_malicious_classes_independently",
            "reference_scaler": "benign_class",
            "malicious_mean_sha256": _array_digest(malicious_mean),
            "malicious_scale_sha256": _array_digest(malicious_scale),
        },
    )


def _original_partition(
    partition: IsetPartition,
) -> tuple[IsetPartition, dict[str, Any]]:
    counts = np.bincount(partition.labels, minlength=2)
    return partition, {
        "applied": False,
        "reason": "disabled_by_branch",
        "counts_before": counts.astype(int).tolist(),
        "counts_after": counts.astype(int).tolist(),
        "generated": 0,
    }


def _adasyn_resample(
    partition: IsetPartition,
    *,
    seed: int,
    n_neighbors: int,
    synthetic_prefix: str,
) -> tuple[IsetPartition, dict[str, Any]]:
    counts_before = np.bincount(partition.labels, minlength=2)
    if (counts_before == 0).any():
        raise ValueError("ADASYN requires benign and malicious ISET samples")
    if counts_before[0] == counts_before[1]:
        return partition, {
            "applied": False,
            "reason": "classes_already_balanced",
            "counts_before": counts_before.astype(int).tolist(),
            "counts_after": counts_before.astype(int).tolist(),
            "generated": 0,
        }
    minority_count = int(np.min(counts_before))
    if minority_count <= n_neighbors:
        raise ValueError(
            "ADASYN n_neighbors must be smaller than the ISET minority class count "
            f"({n_neighbors} >= {minority_count})"
        )
    sampler = ADASYN(random_state=seed, n_neighbors=n_neighbors)
    values, labels = sampler.fit_resample(partition.values, partition.labels)
    generated = labels.shape[0] - partition.labels.shape[0]
    generated_ids = np.asarray(
        [f"{synthetic_prefix}_{index:09d}" for index in range(generated)], dtype=str
    )
    synthetic = IsetPartition(
        values=np.ascontiguousarray(values[partition.labels.shape[0] :], dtype=np.float32),
        labels=np.ascontiguousarray(labels[partition.labels.shape[0] :], dtype=np.int8),
        sample_ids=generated_ids,
        source_profile_ids=np.full(generated, "", dtype=str),
        meter_ids=np.full(generated, "", dtype=str),
        day_numbers=np.full(generated, -1, dtype=np.int64),
        attack_ids=np.full(generated, -1, dtype=np.int8),
        source_refs=np.full(generated, "ADASYN", dtype=str),
        is_synthetic=np.ones(generated, dtype=bool),
    )
    combined = _concatenate([partition, synthetic])
    counts_after = np.bincount(combined.labels, minlength=2)
    return combined, {
        "applied": True,
        "random_state": seed,
        "n_neighbors": n_neighbors,
        "counts_before": counts_before.astype(int).tolist(),
        "counts_after": counts_after.astype(int).tolist(),
        "generated": int(generated),
    }


def prepare_iset_paper_literal(
    prepared_profiles: pd.DataFrame | None = None,
    *,
    archive_paths: Iterable[str | Path] | None = None,
    allocation_source: str | Path | pd.DataFrame | object | None = None,
    allocation_branch: str = OFFICIAL_ALLOCATION_BRANCH,
    expected_md5: Mapping[str, str] | None = None,
    data_seed: int = DEFAULT_DATA_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    adasyn_neighbors: int = DEFAULT_ADASYN_NEIGHBORS,
    table_v_samples: int = DEFAULT_TABLE_V_SAMPLES,
    attack_population: str = "heldout_b2_m",
    scaling: str = "joint_featurewise",
    anomaly_adasyn: str = "test_set_as_printed",
    supervised_adasyn: str = "before_row_split",
    attack_seed: int | None = None,
    attack1_scope: str = "per_profile",
    attack2_granularity: str = "per_half_hour",
    attack3_interval: str = "valid_fit_addition",
    attack_hour_mapping: str = "two_slots_per_hour",
    meter_population: str = "all_4225",
    iset_day: str = "complete_1_48",
    split_unit: str = "customer_disjoint",
    attack_regeneration: str = "fixed_per_data_seed",
    model_seed: int | None = None,
    experiment_index: int | None = None,
    chunksize: int = 250_000,
    shard_count: int = 64,
    scratch_dir: str | Path | None = None,
) -> IsetPaperLiteralData:
    """Prepare exact-data or fixture CER profiles for Tables III--V.

    Exactly one input route is permitted.  The archive route requires all six
    consumption ZIPs and a filesystem allocation table, verifies every MD5,
    selects official residential allocation ``Code == 1``, and applies the
    declared 48-slot day policy. The in-memory route assumes the caller has
    already applied that policy unless an allocation source is also supplied.
    """

    if (prepared_profiles is None) == (archive_paths is None):
        raise ValueError(
            "provide exactly one of prepared_profiles or archive_paths"
        )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if adasyn_neighbors < 1:
        raise ValueError("adasyn_neighbors must be at least one")
    if table_v_samples < 1:
        raise ValueError("table_v_samples must be positive")
    if iset_day not in ISET_DAY_BRANCHES:
        raise ValueError(
            f"unknown ISET day branch {iset_day!r}; "
            f"expected one of {sorted(ISET_DAY_BRANCHES)}"
        )
    if split_unit not in SPLIT_UNIT_BRANCHES:
        raise ValueError(
            f"unknown split_unit branch {split_unit!r}; "
            f"expected one of {sorted(SPLIT_UNIT_BRANCHES)}"
        )
    if attack_population not in ATTACK_POPULATION_BRANCHES:
        raise ValueError(
            f"unknown attack_population branch {attack_population!r}; "
            f"expected one of {sorted(ATTACK_POPULATION_BRANCHES)}"
        )
    if scaling not in SCALING_BRANCHES:
        raise ValueError(
            f"unknown scaling branch {scaling!r}; "
            f"expected one of {sorted(SCALING_BRANCHES)}"
        )
    if anomaly_adasyn not in ANOMALY_ADASYN_BRANCHES:
        raise ValueError(
            f"unknown anomaly_adasyn branch {anomaly_adasyn!r}; "
            f"expected one of {sorted(ANOMALY_ADASYN_BRANCHES)}"
        )
    if supervised_adasyn not in SUPERVISED_ADASYN_BRANCHES:
        raise ValueError(
            f"unknown supervised_adasyn branch {supervised_adasyn!r}; "
            f"expected one of {sorted(SUPERVISED_ADASYN_BRANCHES)}"
        )
    resolved_attack_seed, attack_seed_metadata = _resolve_attack_seed(
        data_seed=data_seed,
        attack_seed=attack_seed,
        attack_regeneration=attack_regeneration,
        model_seed=model_seed,
        experiment_index=experiment_index,
    )

    source_metadata: dict[str, Any]
    profile_stats: dict[str, int] | None = None
    if archive_paths is not None:
        if not isinstance(allocation_source, (str, Path)):
            raise ValueError(
                "the authorized archive route requires a filesystem allocation table"
            )
        paths = [Path(path) for path in archive_paths]
        verified = verify_authorized_iset_files(
            paths,
            allocation_source,
            allocation_branch=allocation_branch,
            expected_md5=expected_md5,
        )
        allocation = read_allocation_table(allocation_source)
        residential = residential_meter_ids(allocation, residential_code=1)
        if residential.size == 0:
            raise ValueError("official allocation table contains no Code=1 meters")
        raw_profiles, profile_stats = _profiles_from_authorized_archives(
            paths,
            residential,
            chunksize=chunksize,
            shard_count=shard_count,
            scratch_dir=scratch_dir,
            iset_day=iset_day,
        )
        profiles, residential_selection = _validate_profiles(raw_profiles, allocation)
        residential_selection["allocation_branch"] = allocation_branch
        source_metadata = {
            "route": "checksum_gated_authorized_archives",
            "checksum_manifest": verified,
            "official_md5_defaults": expected_md5 is None,
            "allocation_branch": allocation_branch,
        }
    else:
        assert prepared_profiles is not None
        profiles, residential_selection = _validate_profiles(
            prepared_profiles, allocation_source
        )
        source_metadata = {
            "route": "in_memory_prepared_profiles",
            "sha256": _frame_digest(prepared_profiles),
            "official_md5_defaults": False,
        }

    profiles, eligible_meters, selected_meters = _select_meter_population(
        profiles,
        branch=meter_population,
        seed=data_seed,
    )

    raw_values = profiles.loc[:, HALF_HOUR_COLUMNS].to_numpy(dtype=np.float32)
    meter_ids = profiles["meter_id"].astype(str).to_numpy()
    day_numbers = profiles["day_number"].to_numpy(dtype=np.int64)
    source_refs = profiles["source_ref"].astype(str).to_numpy()
    profile_ids = np.asarray(
        [
            f"CER:{meter_id}:day_{day_number}"
            for meter_id, day_number in zip(meter_ids, day_numbers, strict=True)
        ],
        dtype=str,
    )
    all_benign_unscaled = _base_partition(
        raw_values, profile_ids, meter_ids, day_numbers, source_refs
    )

    if split_unit == "customer_disjoint":
        b1_meters, b2_meters = _split_meters(
            meter_ids, train_size=2.0 / 3.0, seed=data_seed
        )
        if b1_meters.size < 2:
            raise ValueError("ISET B1 needs at least two meters for train/validation")
        train_meters, validation_meters = train_test_split(
            b1_meters,
            test_size=validation_fraction,
            shuffle=True,
            random_state=data_seed + 1,
        )
        train_meters = np.sort(train_meters.astype(str))
        validation_meters = np.sort(validation_meters.astype(str))
        b1_indices = np.flatnonzero(np.isin(meter_ids, b1_meters))
        b2_indices = np.flatnonzero(np.isin(meter_ids, b2_meters))
        anomaly_train_source_indices = np.flatnonzero(
            np.isin(meter_ids, train_meters)
        )
        anomaly_validation_source_indices = np.flatnonzero(
            np.isin(meter_ids, validation_meters)
        )
    else:
        profile_indices = np.arange(meter_ids.size, dtype=np.int64)
        b1_indices, b2_indices = train_test_split(
            profile_indices,
            train_size=2.0 / 3.0,
            shuffle=True,
            random_state=data_seed,
        )
        anomaly_train_source_indices, anomaly_validation_source_indices = (
            train_test_split(
                np.asarray(b1_indices, dtype=np.int64),
                test_size=validation_fraction,
                shuffle=True,
                random_state=data_seed + 1,
            )
        )
        b1_indices = np.sort(np.asarray(b1_indices, dtype=np.int64))
        b2_indices = np.sort(np.asarray(b2_indices, dtype=np.int64))
        anomaly_train_source_indices = np.sort(
            np.asarray(anomaly_train_source_indices, dtype=np.int64)
        )
        anomaly_validation_source_indices = np.sort(
            np.asarray(anomaly_validation_source_indices, dtype=np.int64)
        )
        b1_meters = np.sort(np.unique(meter_ids[b1_indices]))
        b2_meters = np.sort(np.unique(meter_ids[b2_indices]))
        train_meters = np.sort(
            np.unique(meter_ids[anomaly_train_source_indices])
        )
        validation_meters = np.sort(
            np.unique(meter_ids[anomaly_validation_source_indices])
        )
    heldout_benign_unscaled = _take(all_benign_unscaled, b2_indices)
    train_benign_unscaled = _take(
        all_benign_unscaled,
        anomaly_train_source_indices,
    )
    if attack_population == "all_customer_m":
        malicious_population_unscaled = _generate_attacks(
            all_benign_unscaled,
            seed=resolved_attack_seed,
            partition_name="iset_all_customers",
            attack1_scope=attack1_scope,
            attack2_granularity=attack2_granularity,
            attack3_interval=attack3_interval,
            attack_hour_mapping=attack_hour_mapping,
        )
    else:
        malicious_population_unscaled = _generate_attacks(
            heldout_benign_unscaled,
            seed=resolved_attack_seed,
            partition_name="iset_heldout_b2",
            attack1_scope=attack1_scope,
            attack2_granularity=attack2_granularity,
            attack3_interval=attack3_interval,
            attack_hour_mapping=attack_hour_mapping,
        )

    (
        all_benign,
        malicious_population,
        scaler_mean,
        scaler_scale,
        zero_variance_count,
        scaling_metadata,
    ) = _standardize_iset_by_branch(
        all_benign_unscaled,
        malicious_population_unscaled,
        train_benign_unscaled,
        branch=scaling,
    )
    heldout_attack_indices = np.flatnonzero(
        np.isin(
            malicious_population.source_profile_ids,
            heldout_benign_unscaled.source_profile_ids,
        )
    )
    heldout_attacks = _take(malicious_population, heldout_attack_indices)

    anomaly_train = _take(all_benign, anomaly_train_source_indices)
    anomaly_validation = _take(
        all_benign, anomaly_validation_source_indices
    )
    heldout_benign = _take(all_benign, b2_indices)
    anomaly_original = _concatenate([heldout_benign, heldout_attacks])
    if anomaly_adasyn == "test_set_as_printed":
        anomaly_test, anomaly_adasyn_metadata = _adasyn_resample(
            anomaly_original,
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_ISET_ANOMALY_TEST",
        )
    else:
        anomaly_test, anomaly_adasyn_metadata = _original_partition(
            anomaly_original
        )

    supervised_original = _concatenate([all_benign, malicious_population])
    if supervised_adasyn == "before_row_split":
        supervised_balanced, supervised_adasyn_metadata = _adasyn_resample(
            supervised_original,
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_ISET_SUPERVISED",
        )
        supervised_indices = np.arange(supervised_balanced.labels.size)
        supervised_train_indices, supervised_test_indices = train_test_split(
            supervised_indices,
            train_size=2.0 / 3.0,
            shuffle=True,
            stratify=supervised_balanced.labels,
            random_state=data_seed,
        )
        supervised_train = _take(
            supervised_balanced,
            np.asarray(supervised_train_indices, dtype=np.int64),
        )
        supervised_test = _take(
            supervised_balanced,
            np.asarray(supervised_test_indices, dtype=np.int64),
        )
        supervised_after_adasyn = supervised_balanced.labels.size
        supervised_train_meters: np.ndarray | None = None
        supervised_test_meters: np.ndarray | None = None
    else:
        if attack_population == "all_customer_m":
            if split_unit == "customer_disjoint":
                supervised_train_meters = b1_meters
                supervised_test_meters = b2_meters
            else:
                supervised_train_meters, supervised_test_meters = _split_meters(
                    all_benign.meter_ids,
                    train_size=2.0 / 3.0,
                    seed=data_seed + 2,
                )
        else:
            supervised_train_meters, supervised_test_meters = _split_meters(
                heldout_benign.meter_ids,
                train_size=2.0 / 3.0,
                seed=data_seed + 2,
            )
        train_original = _concatenate(
            [
                _take(
                    all_benign,
                    np.flatnonzero(
                        np.isin(all_benign.meter_ids, supervised_train_meters)
                    ),
                ),
                _take(
                    malicious_population,
                    np.flatnonzero(
                        np.isin(
                            malicious_population.meter_ids,
                            supervised_train_meters,
                        )
                    ),
                ),
            ]
        )
        supervised_test = _concatenate(
            [
                _take(
                    all_benign,
                    np.flatnonzero(
                        np.isin(all_benign.meter_ids, supervised_test_meters)
                    ),
                ),
                _take(
                    malicious_population,
                    np.flatnonzero(
                        np.isin(
                            malicious_population.meter_ids,
                            supervised_test_meters,
                        )
                    ),
                ),
            ]
        )
        supervised_train, supervised_adasyn_metadata = _adasyn_resample(
            train_original,
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_ISET_SUPERVISED_TRAIN",
        )
        supervised_after_adasyn = (
            supervised_train.labels.size + supervised_test.labels.size
        )

    table_iv_order = np.random.default_rng(data_seed + 4).permutation(
        anomaly_train.labels.size
    )
    table_v_count = min(table_v_samples, heldout_benign.labels.size)
    table_v_indices = np.random.default_rng(data_seed + 5).permutation(
        heldout_benign.labels.size
    )[:table_v_count]
    table_v_benign = _take(heldout_benign, table_v_indices)
    table_v_attacks: list[IsetPartition] = []
    block_size = heldout_benign.labels.size
    for attack_id in range(1, 7):
        block_start = (attack_id - 1) * block_size
        table_v_attacks.append(
            _take(heldout_attacks, block_start + table_v_indices)
        )

    partition_digest = {
        "anomaly_train": _id_digest(anomaly_train.sample_ids),
        "anomaly_validation": _id_digest(anomaly_validation.sample_ids),
        "anomaly_test": _id_digest(anomaly_test.sample_ids),
        "supervised_train": _id_digest(supervised_train.sample_ids),
        "supervised_test": _id_digest(supervised_test.sample_ids),
        "table_v_benign": _id_digest(table_v_benign.sample_ids),
        **{
            f"table_v_attack_{attack_id}": _id_digest(partition.sample_ids)
            for attack_id, partition in enumerate(table_v_attacks, start=1)
        },
    }
    counts = {
        "residential_meters": int(np.unique(meter_ids).size),
        "benign_profiles": int(all_benign.labels.size),
        "b1_meters": int(b1_meters.size),
        "b2_meters": int(b2_meters.size),
        "anomaly_train_meters": int(train_meters.size),
        "anomaly_validation_meters": int(validation_meters.size),
        "anomaly_train": int(anomaly_train.labels.size),
        "anomaly_validation": int(anomaly_validation.labels.size),
        "anomaly_b2_benign": int(heldout_benign.labels.size),
        "malicious_profiles": int(heldout_attacks.labels.size),
        "malicious_population_profiles": int(malicious_population.labels.size),
        "malicious_profiles_per_attack": int(heldout_benign.labels.size),
        "anomaly_test_original": int(anomaly_original.labels.size),
        "anomaly_test_after_adasyn": int(anomaly_test.labels.size),
        "supervised_before_adasyn": int(supervised_original.labels.size),
        "supervised_after_adasyn": int(supervised_after_adasyn),
        "supervised_train": int(supervised_train.labels.size),
        "supervised_test": int(supervised_test.labels.size),
        "table_v_per_class": int(table_v_count),
        "table_iv_half": int(np.floor(anomaly_train.labels.size * 0.5)),
        "table_iv_three_quarter": int(
            np.floor(anomaly_train.labels.size * 0.75)
        ),
        "table_iv_full": int(anomaly_train.labels.size),
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "CER/ISET",
        "track": "exploratory_paper_literal",
        "source": source_metadata,
        "residential_selection": residential_selection,
        "profile_extraction": {
            "representation": "complete_meter_day_slots_1_through_48",
            "profile_stats": profile_stats,
        },
        "config": {
            "data_seed": data_seed,
            "meter_split_train_fraction": 2.0 / 3.0,
            "validation_fraction_within_b1_meters": validation_fraction,
            "adasyn_neighbors": adasyn_neighbors,
            "table_v_requested_per_class": table_v_samples,
            "table_v_actual_per_class": table_v_count,
            "attack_population_branch": attack_population,
            "scaling_branch": scaling,
            "anomaly_adasyn_branch": anomaly_adasyn,
            "supervised_adasyn_branch": supervised_adasyn,
            "attack_seed": resolved_attack_seed,
            "attack_seed_derivation": attack_seed_metadata,
            "attack_regeneration_branch": attack_regeneration,
            "attack1_scope": attack1_scope,
            "attack2_granularity": attack2_granularity,
            "attack3_interval": attack3_interval,
            "attack_hour_mapping": attack_hour_mapping,
            "meter_population_branch": meter_population,
            "eligible_residential_meters": int(eligible_meters.size),
            "selected_residential_meters": int(selected_meters.size),
            "iset_day_branch": iset_day,
            "split_unit_branch": split_unit,
        },
        "preprocessing": {
            "malicious_source": attack_population,
            "iset_day": iset_day,
            "split_unit": split_unit,
            "attacks_per_source_profile": 6,
            "attack_rng": (
                "six_distinct_seedsequence_streams_within_selected_attack_population"
            ),
            "attack_regeneration": attack_regeneration,
            "attack_seed_derivation": attack_seed_metadata,
            "attack1_scope": attack1_scope,
            "attack2_granularity": attack2_granularity,
            "attack3_interval": attack3_interval,
            "attack_hour_mapping": attack_hour_mapping,
            "scaling": scaling,
            "scaling_details": scaling_metadata,
            "standard_deviation_ddof": 0,
            "zero_variance_features_scaled_by_one": zero_variance_count,
            "anomaly_test_adasyn": anomaly_adasyn,
            "supervised_adasyn": supervised_adasyn,
        },
        "meter_partitions": {
            "b1": b1_meters.tolist(),
            "b2": b2_meters.tolist(),
            "anomaly_train": train_meters.tolist(),
            "anomaly_validation": validation_meters.tolist(),
            "supervised_train": (
                supervised_train_meters.tolist()
                if supervised_train_meters is not None
                else None
            ),
            "supervised_test": (
                supervised_test_meters.tolist()
                if supervised_test_meters is not None
                else None
            ),
        },
        "profile_partitions": {
            "b1": _id_digest(all_benign.sample_ids[b1_indices]),
            "b2": _id_digest(all_benign.sample_ids[b2_indices]),
            "anomaly_train": _id_digest(
                all_benign.sample_ids[anomaly_train_source_indices]
            ),
            "anomaly_validation": _id_digest(
                all_benign.sample_ids[anomaly_validation_source_indices]
            ),
        },
        "counts": counts,
        "adasyn": {
            "anomaly_test": anomaly_adasyn_metadata,
            "supervised": supervised_adasyn_metadata,
            # Compatibility alias for implementation-v1 cache readers.
            "supervised_before_split": supervised_adasyn_metadata,
        },
        "partition_id_sha256": partition_digest,
        "transformation_sha256": {
            "scaler_mean": _array_digest(scaler_mean),
            "scaler_scale": _array_digest(scaler_scale),
            "table_iv_order": _array_digest(table_iv_order),
        },
        "warnings": [
            warning
            for condition, warning in (
                (
                    scaling != "train_benign_only",
                    f"Scaling branch {scaling!r} uses information beyond anomaly training.",
                ),
                (
                    anomaly_adasyn == "test_set_as_printed",
                    "ADASYN is intentionally applied within the anomaly test set.",
                ),
                (
                    supervised_adasyn == "before_row_split",
                    "ADASYN is intentionally applied before the supervised row split.",
                ),
                (
                    supervised_adasyn == "before_row_split",
                    "The supervised split cannot preserve meter disjointness after paper-ordered ADASYN.",
                ),
                (
                    True,
                    "ADASYN rows have no single meter, day, source profile, or attack ID.",
                ),
            )
            if condition
        ],
    }
    return IsetPaperLiteralData(
        scaler_mean=np.ascontiguousarray(scaler_mean, dtype=np.float64),
        scaler_scale=np.ascontiguousarray(scaler_scale, dtype=np.float64),
        anomaly_train=anomaly_train,
        anomaly_validation=anomaly_validation,
        anomaly_test=anomaly_test,
        supervised_train=supervised_train,
        supervised_test=supervised_test,
        table_iv_order=np.ascontiguousarray(table_iv_order, dtype=np.int64),
        table_v_benign=table_v_benign,
        table_v_attacks=tuple(table_v_attacks),
        metadata=metadata,
    )


_CACHE_PARTITIONS = (
    "anomaly_train",
    "anomaly_validation",
    "anomaly_test",
    "supervised_train",
    "supervised_test",
    "table_v_benign",
)
_PARTITION_ARRAYS = (
    "values",
    "labels",
    "sample_ids",
    "source_profile_ids",
    "meter_ids",
    "day_numbers",
    "attack_ids",
    "source_refs",
    "is_synthetic",
)


def _cache_partition(arrays: dict[str, np.ndarray], name: str, value: IsetPartition) -> None:
    for field in _PARTITION_ARRAYS:
        array = getattr(value, field)
        arrays[f"{name}_{field}"] = array.astype(str) if array.dtype.kind in "OU" else array


def save_prepared_iset(
    prepared: IsetPaperLiteralData,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Write a checksummed local NPZ/JSON cache intended for ``data/derived``."""

    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    manifest_path = prefix.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "scaler_mean": prepared.scaler_mean,
        "scaler_scale": prepared.scaler_scale,
        "table_iv_order": prepared.table_iv_order,
    }
    for name in _CACHE_PARTITIONS:
        _cache_partition(arrays, name, getattr(prepared, name))
    for attack_id, partition in enumerate(prepared.table_v_attacks, start=1):
        _cache_partition(arrays, f"table_v_attack_{attack_id}", partition)
    temporary = npz_path.with_suffix(npz_path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(npz_path)
    manifest = {
        "cache_schema_version": SCHEMA_VERSION,
        "npz_filename": npz_path.name,
        "npz_sha256": _file_digest(npz_path, "sha256"),
        "metadata": prepared.metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path, manifest_path


def _load_partition(arrays: Any, name: str) -> IsetPartition:
    return IsetPartition(
        **{field: arrays[f"{name}_{field}"] for field in _PARTITION_ARRAYS}
    )


def load_prepared_iset(output_prefix: str | Path) -> IsetPaperLiteralData:
    """Load and checksum-verify a cache written by :func:`save_prepared_iset`."""

    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ISET cache schema version")
    if _file_digest(npz_path, "sha256") != manifest.get("npz_sha256"):
        raise ValueError("ISET cache checksum mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        partitions = {
            name: _load_partition(arrays, name) for name in _CACHE_PARTITIONS
        }
        attacks = tuple(
            _load_partition(arrays, f"table_v_attack_{attack_id}")
            for attack_id in range(1, 7)
        )
        return IsetPaperLiteralData(
            scaler_mean=arrays["scaler_mean"],
            scaler_scale=arrays["scaler_scale"],
            anomaly_train=partitions["anomaly_train"],
            anomaly_validation=partitions["anomaly_validation"],
            anomaly_test=partitions["anomaly_test"],
            supervised_train=partitions["supervised_train"],
            supervised_test=partitions["supervised_test"],
            table_iv_order=arrays["table_iv_order"],
            table_v_benign=partitions["table_v_benign"],
            table_v_attacks=attacks,
            metadata=manifest["metadata"],
        )
