#!/usr/bin/env python3
"""Prepare Paper 1 data in the printed order from the verified sources.

The order is intentionally the paper's order:

1. parse strict 48-slot residential meter-days;
2. generate all six attacks for every customer;
3. jointly standardize benign and malicious rows before splitting;
4. split benign customers 2:1 into B1/B2;
5. train on B1;
6. test on B2 plus attacks from *all* customers; and
7. either apply the printed ADASYN step or explicitly preserve the declared
   no-test-resampling continuation when that full-scale step is unavailable.

This is not recommended methodology. It is the declared printed-method anchor.
Use ``--mode tiny`` first; ``--mode full`` processes the complete source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterator

import imblearn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from imblearn.over_sampling import ADASYN

from download_data import (
    CER_SCIENCEDB_DIR,
    SGCC_PATH,
    SGCC_SHA256,
    SCIENCEDB_ALLOCATION,
    digest,
    verify_cer_directory,
)


REPO = Path(__file__).resolve().parents[3]
RAW = CER_SCIENCEDB_DIR
DEFAULT_ROOT = REPO / "data/derived/atk-2022-deep-autoencoder/reproduction"
ARCHIVES = tuple(f"File{i}.txt.zip" for i in range(1, 7))
ALLOCATION = SCIENCEDB_ALLOCATION[0]
FEATURES = 48
DATA_SEED = 11
CHUNK_ROWS = 1_000_000
ATTACK_CHUNK = 50_000
SGCC_REPRESENTATIONS = ("last_48", "first_48", "binned_mean_48")
SGCC_METHOD_SUFFIX = {
    "last_48": "LAST48",
    "first_48": "FIRST48",
    "binned_mean_48": "BINNED-MEAN48",
}


def represent_sgcc(values: np.ndarray, representation: str) -> np.ndarray:
    """Apply one frozen architecture-preserving 1,034-to-48 completion."""

    if representation == "last_48":
        represented = values[:, -FEATURES:]
    elif representation == "first_48":
        represented = values[:, :FEATURES]
    elif representation == "binned_mean_48":
        represented = np.column_stack(
            [values[:, indices].mean(axis=1) for indices in np.array_split(np.arange(values.shape[1]), FEATURES)]
        )
    else:
        raise ValueError(f"unsupported SGCC representation: {representation}")
    return np.ascontiguousarray(represented, dtype=np.float32)


def prepare_sgcc(
    output: Path,
    *,
    seed: int,
    mode: str,
    adasyn_neighbors: int,
    representation: str = "last_48",
) -> dict[str, object]:
    """Execute one frozen 48-wide SGCC continuation for Table II."""

    started = time.perf_counter()
    if digest(SGCC_PATH, "sha256") != SGCC_SHA256:
        raise ValueError("verified SGCC source checksum mismatch")
    frame = pd.read_csv(SGCC_PATH, low_memory=False)
    required = {"CONS_NO", "FLAG"}
    if not required.issubset(frame.columns):
        raise ValueError("SGCC source is missing CONS_NO or FLAG")
    date_columns = [column for column in frame.columns if column not in required]
    if len(date_columns) != 1_034:
        raise ValueError(f"expected 1,034 SGCC dates, found {len(date_columns)}")
    dates = pd.to_datetime(date_columns, format="%Y/%m/%d", errors="raise")
    order = np.argsort(dates.to_numpy(), kind="stable")
    date_columns = [date_columns[index] for index in order]
    dates = dates.to_numpy(dtype="datetime64[D]")[order]

    labels = pd.to_numeric(frame["FLAG"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("SGCC FLAG must contain only zero and one")
    customer_ids = frame["CONS_NO"].astype(str).to_numpy()
    if pd.Series(customer_ids).duplicated().any():
        raise ValueError("SGCC customer identifiers must be unique")
    values = frame[date_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )
    if np.isinf(values).any():
        raise ValueError("SGCC readings contain infinity")
    fully_missing = np.all(np.isnan(values), axis=1)
    values, labels, customer_ids = (
        values[~fully_missing], labels[~fully_missing], customer_ids[~fully_missing]
    )
    if mode == "tiny":
        selected = np.concatenate(
            [np.flatnonzero(labels == 0)[:300], np.flatnonzero(labels == 1)[:100]]
        )
        values, labels, customer_ids = (
            values[selected], labels[selected], customer_ids[selected]
        )

    benign_index = np.flatnonzero(labels == 0)
    malicious_index = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    shuffled_benign = rng.permutation(benign_index)
    b1 = shuffled_benign[: (2 * shuffled_benign.size) // 3]
    b2 = shuffled_benign[(2 * shuffled_benign.size) // 3 :]

    # The paper is silent on missing SGCC cells. This is the frozen minimal
    # completion: interpolate only bounded gaps, then use benign-B1 medians.
    completed = pd.DataFrame(values).interpolate(
        axis=1, limit_area="inside"
    ).to_numpy(dtype=np.float32, copy=True)
    fallback = np.nanmedian(completed[b1], axis=0)
    global_fallback = float(np.nanmedian(completed[b1]))
    if not np.isfinite(global_fallback):
        raise ValueError("SGCC benign B1 has no finite readings")
    fallback = np.where(np.isfinite(fallback), fallback, global_fallback)
    missing = ~np.isfinite(completed)
    completed[missing] = np.broadcast_to(fallback, completed.shape)[missing]

    # Literal 1,034 -> 48 is undefined. Every registered continuation preserves
    # the printed width while declaring that its features are days or day bins,
    # not the half-hourly daily readings described for ISET.
    raw = represent_sgcc(completed, representation)
    mean = raw.mean(axis=0, dtype=np.float64)
    scale = raw.std(axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float32).eps] = 1.0
    standardized = ((raw - mean) / scale).astype(np.float32)

    x_train = standardized[b1]
    original_index = np.concatenate([b2, malicious_index])
    original_x = standardized[original_index]
    original_y = labels[original_index]
    anomaly_sampler = ADASYN(random_state=seed, n_neighbors=adasyn_neighbors)
    x_test, y_test = anomaly_sampler.fit_resample(original_x, original_y)
    anomaly_generated = int(y_test.size - original_y.size)

    supervised_sampler = ADASYN(random_state=seed, n_neighbors=adasyn_neighbors)
    supervised_x, supervised_y = supervised_sampler.fit_resample(
        standardized, labels
    )
    supervised_generated = int(supervised_y.size - labels.size)

    np.save(output / "x_train.npy", x_train)
    np.save(output / "table_iv_order.npy", rng.permutation(x_train.shape[0]))
    np.save(output / "train_meter_ids.npy", customer_ids[b1])
    np.save(output / "train_day_numbers.npy", np.zeros(b1.size, dtype=np.int16))
    np.save(output / "test_original_x.npy", original_x)
    np.save(output / "test_original_y.npy", original_y)
    np.save(output / "test_original_source_row.npy", original_index)
    np.save(output / "test_original_attack_id.npy", np.zeros(original_y.size, dtype=np.int8))
    np.save(output / "x_test.npy", np.asarray(x_test, dtype=np.float32))
    np.save(output / "y_test.npy", np.asarray(y_test, dtype=np.int8))
    np.save(
        output / "test_source_row.npy",
        np.concatenate([original_index, np.full(anomaly_generated, -1, dtype=np.int64)]),
    )
    np.save(output / "test_attack_id.npy", np.zeros(y_test.size, dtype=np.int8))
    np.save(
        output / "test_is_synthetic.npy",
        np.concatenate(
            [np.zeros(original_y.size, dtype=bool), np.ones(anomaly_generated, dtype=bool)]
        ),
    )
    np.save(output / "supervised_x.npy", np.asarray(supervised_x, dtype=np.float32))
    np.save(output / "supervised_y.npy", np.asarray(supervised_y, dtype=np.int8))
    np.save(output / "scaler_mean.npy", mean)
    np.save(output / "scaler_scale.npy", scale)

    files = sorted(output.glob("*.npy"))
    metadata: dict[str, object] = {
        "status": "complete",
        "schema": 1,
        "method": f"I-SGCC-{SGCC_METHOD_SUFFIX[representation]}",
        "dataset": "SGCC",
        "configuration": {
            "mode": mode,
            "seed": seed,
            "representation": representation,
            "missing": "interpolate_edge_median",
            "test_adasyn": "printed",
            "supervised_adasyn": "printed_before_split",
            "adasyn_neighbors": adasyn_neighbors,
        },
        "paper_order": [
            "joint_B_plus_M_featurewise_standardization",
            "benign_2_to_1_B1_B2",
            "anomaly_XTR_equals_B1",
            "ADASYN_inside_anomaly_test",
            "supervised_ADASYN_before_2_to_1_split",
        ],
        "source_nodes": {
            "sgcc_input": {
                "paper_claim": "SGCC is evaluated by architectures with 48 inputs",
                "literal_status": "non_executable_no_1034_to_48_rule",
                "assumption": representation,
                "semantic_warning": "48 days are not the printed 48 half-hour daily readings",
            },
            "sgcc_missing": {
                "paper_claim": "no missing-value operation stated",
                "literal_status": "non_executable_with_missing_source_cells",
                "assumption": "interior linear interpolation plus benign-B1 feature medians",
            },
        },
        "counts": {
            "source_rows": int(frame.shape[0]),
            "dropped_fully_missing": int(fully_missing.sum()),
            "retained": int(labels.size),
            "benign": int(benign_index.size),
            "malicious": int(malicious_index.size),
            "B1": int(b1.size),
            "B2": int(b2.size),
            "anomaly_test_original": int(original_y.size),
            "anomaly_test_after_adasyn": int(y_test.size),
            "supervised_after_adasyn": int(supervised_y.size),
            "anomaly_synthetic": anomaly_generated,
            "supervised_synthetic": supervised_generated,
        },
        "source": {
            "path": str(SGCC_PATH),
            "sha256": SGCC_SHA256,
            "features": 1_034,
            "first_date": str(dates[0]),
            "last_date": str(dates[-1]),
        },
        "timing_seconds": {"preparation": time.perf_counter() - started},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    save_json(output / "metadata.json", metadata)
    return metadata


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verified_source() -> dict[str, object]:
    """Recheck the exact archives and frozen allocation branch before parsing."""

    record = verify_cer_directory(
        RAW,
        allocation=SCIENCEDB_ALLOCATION,
        branch="sciencedb-csv-semantic-equivalence-v1",
    )
    if not record["ready"]:
        failed = [
            name
            for name, item in record["files"].items()
            if item["status"] != "verified"
        ]
        raise ValueError(f"ISET/CER source verification failed: {failed}")
    return record


def allocation() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(
        RAW / ALLOCATION,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )
    columns = {
        "".join(char for char in str(column).lower() if char.isalnum()): column
        for column in frame.columns
    }
    id_column = columns["id"]
    code_column = columns["code"]
    parsed = pd.DataFrame(
        {
            "meter_id": pd.to_numeric(frame[id_column], errors="raise").astype("int32"),
            "code": pd.to_numeric(frame[code_column], errors="raise").astype("int8"),
        }
    ).drop_duplicates()
    if parsed.groupby("meter_id")["code"].nunique().max() != 1:
        raise ValueError("allocation file assigns conflicting codes to one meter")
    residential = np.sort(
        parsed.loc[parsed["code"] == 1, "meter_id"].unique().astype(np.int32)
    )
    if parsed.shape[0] != 6_445 or residential.size != 4_225:
        raise ValueError(
            f"unexpected allocation cardinality rows={parsed.shape[0]} "
            f"residential={residential.size}"
        )
    return parsed, residential


def zip_chunks(paths: list[Path]) -> Iterator[pd.DataFrame]:
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            members = [
                item for item in archive.infolist()
                if not item.is_dir() and not Path(item.filename).name.startswith(".")
            ]
            if len(members) != 1:
                raise ValueError(f"{path} must contain exactly one text member")
            with archive.open(members[0]) as source:
                yield from pd.read_csv(
                    source,
                    sep=r"\s+",
                    header=None,
                    names=("meter_id", "day_time", "kwh"),
                    chunksize=CHUNK_ROWS,
                    dtype={"meter_id": "int32", "day_time": "int32", "kwh": "float32"},
                )


def strict_profiles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        return (
            np.empty((0, FEATURES), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int16),
        )
    compact = frame.copy()
    compact["day"] = compact["day_time"] // 100
    compact["slot"] = compact["day_time"] % 100
    if not np.isfinite(compact["kwh"]).all() or (compact["kwh"] < 0).any():
        raise ValueError("CER readings must be finite and nonnegative")
    grouped = compact.groupby(["meter_id", "day"], sort=False)["slot"]
    summary = grouped.agg(["size", "nunique", "min", "max"]).reset_index()
    valid = summary.loc[
        (summary["size"] == FEATURES)
        & (summary["nunique"] == FEATURES)
        & (summary["min"] == 1)
        & (summary["max"] == FEATURES),
        ["meter_id", "day"],
    ]
    selected = compact.merge(valid, on=["meter_id", "day"], how="inner")
    if selected.empty:
        return (
            np.empty((0, FEATURES), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int16),
        )
    pivot = selected.pivot(
        index=["meter_id", "day"], columns="slot", values="kwh"
    ).reindex(columns=range(1, FEATURES + 1))
    pivot = pivot.sort_index()
    values = pivot.to_numpy(dtype=np.float32)
    if values.shape[1] != FEATURES or not np.isfinite(values).all():
        raise AssertionError("strict daily pivot did not produce finite 48-slot rows")
    meters = pivot.index.get_level_values("meter_id").to_numpy(dtype=np.int32)
    days = pivot.index.get_level_values("day").to_numpy(dtype=np.int16)
    return values, meters, days


def extract_tiny(residential: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = set(map(int, residential))
    for chunk in zip_chunks([RAW / ARCHIVES[0]]):
        filtered = chunk.loc[chunk["meter_id"].isin(eligible)]
        values, meters, days = strict_profiles(filtered)
        if values.size == 0:
            continue
        counts = pd.Series(meters).value_counts()
        chosen = np.sort(counts[counts >= 20].index.to_numpy(dtype=np.int32))[:12]
        if chosen.size == 12:
            keep: list[int] = []
            for meter in chosen:
                positions = np.flatnonzero(meters == meter)[:20]
                keep.extend(positions.tolist())
            index = np.asarray(keep, dtype=np.int64)
            order = np.lexsort((days[index], meters[index]))
            index = index[order]
            return values[index], meters[index], days[index]
    raise ValueError("could not find the deterministic tiny 12-meter fixture")


def extract_full(
    residential: np.ndarray,
    scratch: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    schema = pa.schema(
        [
            ("meter_id", pa.int32()),
            ("day", pa.int16()),
            ("slot", pa.int8()),
            ("kwh", pa.float32()),
        ]
    )
    selected = set(map(int, residential))
    shard_count = 32
    writers: dict[int, pq.ParquetWriter] = {}
    source_rows = residential_rows = 0
    try:
        for chunk in zip_chunks([RAW / name for name in ARCHIVES]):
            source_rows += len(chunk)
            chunk = chunk.loc[chunk["meter_id"].isin(selected)].copy()
            residential_rows += len(chunk)
            if chunk.empty:
                continue
            chunk["day"] = (chunk["day_time"] // 100).astype("int16")
            chunk["slot"] = (chunk["day_time"] % 100).astype("int8")
            shard_ids = np.mod(chunk["meter_id"].to_numpy(dtype=np.int64), shard_count)
            for shard in np.unique(shard_ids):
                number = int(shard)
                part = chunk.loc[
                    shard_ids == number, ["meter_id", "day", "slot", "kwh"]
                ]
                table = pa.Table.from_pandas(part, schema=schema, preserve_index=False)
                if number not in writers:
                    writers[number] = pq.ParquetWriter(
                        scratch / f"rows-{number:02d}.parquet",
                        schema,
                        compression="zstd",
                    )
                writers[number].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()

    value_blocks: list[np.ndarray] = []
    meter_blocks: list[np.ndarray] = []
    day_blocks: list[np.ndarray] = []
    for shard in sorted(writers):
        rows = pd.read_parquet(scratch / f"rows-{shard:02d}.parquet")
        rows = rows.rename(columns={"day": "day_time"})
        rows["day_time"] = (
            rows["day_time"].astype("int32") * 100 + rows["slot"].astype("int32")
        )
        values, meters, days = strict_profiles(
            rows[["meter_id", "day_time", "kwh"]]
        )
        value_blocks.append(values)
        meter_blocks.append(meters)
        day_blocks.append(days)
    values = np.concatenate(value_blocks)
    meters = np.concatenate(meter_blocks)
    days = np.concatenate(day_blocks)
    order = np.lexsort((days, meters))
    return (
        values[order],
        meters[order],
        days[order],
        {
            "source_rows": source_rows,
            "residential_rows": residential_rows,
            "strict_profiles": int(values.shape[0]),
            "residential_meters_with_profiles": int(np.unique(meters).size),
        },
    )


def load_or_extract_profiles(
    output: Path,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    values_path = output / "benign_raw.npy"
    meters_path = output / "meter_ids.npy"
    days_path = output / "day_numbers.npy"
    record_path = output / "profiles.json"
    if all(path.is_file() for path in (values_path, meters_path, days_path, record_path)):
        record = json.loads(record_path.read_text())
        if record.get("mode") != mode:
            raise ValueError(
                f"cached profiles were extracted in mode {record.get('mode')}, "
                f"not requested mode {mode}"
            )
        return (
            np.load(values_path, mmap_mode="r"),
            np.load(meters_path, mmap_mode="r"),
            np.load(days_path, mmap_mode="r"),
            record,
        )

    _, residential = allocation()
    started = time.perf_counter()
    if mode == "tiny":
        values, meters, days = extract_tiny(residential)
        extraction = {"source": "first verified archive deterministic fixture"}
    else:
        scratch_parent = output / "scratch"
        scratch_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="cer-parquet-", dir=scratch_parent
        ) as temporary:
            values, meters, days, extraction = extract_full(
                residential, Path(temporary)
            )
        scratch_parent.rmdir()
    np.save(values_path, np.asarray(values, dtype=np.float32))
    np.save(meters_path, np.asarray(meters, dtype=np.int32))
    np.save(days_path, np.asarray(days, dtype=np.int16))
    record: dict[str, object] = {
        "mode": mode,
        "rows": int(values.shape[0]),
        "features": int(values.shape[1]),
        "meters": int(np.unique(meters).size),
        "elapsed_seconds": time.perf_counter() - started,
        "extraction": extraction,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (values_path, meters_path, days_path)
        },
    }
    save_json(record_path, record)
    return (
        np.load(values_path, mmap_mode="r"),
        np.load(meters_path, mmap_mode="r"),
        np.load(days_path, mmap_mode="r"),
        record,
    )


def attack_blocks(
    benign: np.ndarray,
    meters: np.ndarray,
    *,
    seed: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    unique_meters = np.unique(meters)
    alpha_rng = np.random.default_rng(seed + 101)
    alphas = alpha_rng.uniform(0.1, 0.8, size=unique_meters.size).astype(np.float32)
    positions = np.arange(FEATURES)[None, :]
    for attack_id in range(1, 7):
        rng = np.random.default_rng(seed + 100 + attack_id)
        for start in range(0, benign.shape[0], ATTACK_CHUNK):
            stop = min(start + ATTACK_CHUNK, benign.shape[0])
            source = np.asarray(benign[start:stop], dtype=np.float32)
            if attack_id == 1:
                meter_positions = np.searchsorted(unique_meters, meters[start:stop])
                attacked = source * alphas[meter_positions, None]
            elif attack_id == 2:
                attacked = source * rng.uniform(
                    0.1, 0.8, size=source.shape
                ).astype(np.float32)
            elif attack_id == 3:
                initial_hour = rng.integers(0, 20, size=source.shape[0])
                length_hour = rng.integers(4, 25, size=source.shape[0])
                final_hour = np.minimum(initial_hour + length_hour, 24)
                bypass = (
                    (positions >= 2 * initial_hour[:, None])
                    & (positions < 2 * final_hour[:, None])
                )
                attacked = source.copy()
                attacked[bypass] = 0
            elif attack_id == 4:
                attacked = np.repeat(
                    source.mean(axis=1, keepdims=True), FEATURES, axis=1
                )
            elif attack_id == 5:
                attacked = source.mean(axis=1, keepdims=True) * rng.uniform(
                    0.1, 0.8, size=source.shape
                ).astype(np.float32)
            else:
                attacked = source[:, ::-1].copy()
            yield attack_id, start, np.asarray(attacked, dtype=np.float32)


def joint_scaler(
    benign: np.ndarray,
    meters: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = benign.shape[0]
    sums = np.asarray(benign, dtype=np.float64).sum(axis=0)
    squares = np.square(np.asarray(benign, dtype=np.float64)).sum(axis=0)
    for _, _, attacked in attack_blocks(benign, meters, seed=seed):
        numeric = attacked.astype(np.float64)
        sums += numeric.sum(axis=0)
        squares += np.square(numeric).sum(axis=0)
        total += attacked.shape[0]
    mean = sums / total
    variance = np.maximum(squares / total - np.square(mean), 0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    return mean, scale


def standardized_array(
    source: np.ndarray,
    target: Path,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.memmap:
    result = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=source.shape
    )
    for start in range(0, source.shape[0], ATTACK_CHUNK):
        stop = min(start + ATTACK_CHUNK, source.shape[0])
        result[start:stop] = (
            (np.asarray(source[start:stop], dtype=np.float64) - mean) / scale
        ).astype(np.float32)
    result.flush()
    return result


def prepare_p0(
    output: Path,
    benign_raw: np.ndarray,
    meters: np.ndarray,
    days: np.ndarray,
    profiles_record: dict[str, object],
    *,
    seed: int,
    adasyn_neighbors: int,
    test_adasyn: str,
    force_expensive_adasyn: bool,
    source_record: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    mean, scale = joint_scaler(benign_raw, meters, seed)
    np.save(output / "scaler_mean.npy", mean)
    np.save(output / "scaler_scale.npy", scale)
    benign = standardized_array(
        benign_raw, output / "benign.npy", mean, scale
    )

    attack_paths: list[Path] = []
    attack_maps: dict[int, np.memmap] = {}
    for attack_id in range(1, 7):
        path = output / f"attack_{attack_id}.npy"
        attack_paths.append(path)
        attack_maps[attack_id] = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=benign_raw.shape
        )
    for attack_id, start, attacked in attack_blocks(benign_raw, meters, seed=seed):
        stop = start + attacked.shape[0]
        attack_maps[attack_id][start:stop] = (
            (attacked.astype(np.float64) - mean) / scale
        ).astype(np.float32)
    for array in attack_maps.values():
        array.flush()

    unique_meters = np.unique(meters)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_meters)
    train_meters = np.sort(shuffled[: int(np.floor(2 * shuffled.size / 3))])
    train_mask = np.isin(meters, train_meters)
    train_index = np.flatnonzero(train_mask)
    b2_index = np.flatnonzero(~train_mask)
    np.save(output / "train_index.npy", train_index)
    np.save(output / "b2_index.npy", b2_index)
    np.save(output / "x_train.npy", np.asarray(benign[train_index], dtype=np.float32))
    np.save(output / "train_meter_ids.npy", np.asarray(meters[train_index]))
    np.save(output / "train_day_numbers.npy", np.asarray(days[train_index]))
    np.save(output / "table_iv_order.npy", rng.permutation(train_index.size))

    original_rows = b2_index.size + 6 * benign.shape[0]
    original_path = output / "test_original_x.npy"
    original_x = np.lib.format.open_memmap(
        original_path, mode="w+", dtype="float32", shape=(original_rows, FEATURES)
    )
    original_y = np.empty(original_rows, dtype=np.int8)
    original_source = np.empty(original_rows, dtype=np.int64)
    original_attack = np.empty(original_rows, dtype=np.int8)
    cursor = 0
    next_cursor = b2_index.size
    original_x[cursor:next_cursor] = benign[b2_index]
    original_y[cursor:next_cursor] = 0
    original_source[cursor:next_cursor] = b2_index
    original_attack[cursor:next_cursor] = 0
    cursor = next_cursor
    all_sources = np.arange(benign.shape[0], dtype=np.int64)
    for attack_id, path in enumerate(attack_paths, start=1):
        attacked = np.load(path, mmap_mode="r")
        next_cursor = cursor + attacked.shape[0]
        original_x[cursor:next_cursor] = attacked
        original_y[cursor:next_cursor] = 1
        original_source[cursor:next_cursor] = all_sources
        original_attack[cursor:next_cursor] = attack_id
        cursor = next_cursor
    original_x.flush()
    np.save(output / "test_original_y.npy", original_y)
    np.save(output / "test_original_source_row.npy", original_source)
    np.save(output / "test_original_attack_id.npy", original_attack)

    adasyn_seconds: float | None = None
    generated: int | None = None
    test_y: np.ndarray | None = None
    adasyn_distance_queries = int(
        np.count_nonzero(original_y == 0) * original_y.size
    )
    if test_adasyn == "printed":
        if adasyn_distance_queries > 100_000_000_000 and not force_expensive_adasyn:
            raise RuntimeError(
                "printed full-test ADASYN requires about "
                f"{adasyn_distance_queries:,} first-pass query/reference pairs "
                "under the selected exact library implementation; rerun with "
                "--force-expensive-adasyn only for a deliberately budgeted attempt"
            )
        adasyn_started = time.perf_counter()
        sampler = ADASYN(random_state=seed, n_neighbors=adasyn_neighbors)
        test_x, test_y = sampler.fit_resample(original_x, original_y)
        adasyn_seconds = time.perf_counter() - adasyn_started
        generated = int(test_y.size - original_y.size)
        test_source = np.concatenate(
            [original_source, np.full(generated, -1, dtype=np.int64)]
        )
        test_attack = np.concatenate(
            [original_attack, np.full(generated, -1, dtype=np.int8)]
        )
        synthetic = np.concatenate(
            [np.zeros(original_y.size, dtype=bool), np.ones(generated, dtype=bool)]
        )
        np.save(output / "x_test.npy", np.asarray(test_x, dtype=np.float32))
        np.save(output / "y_test.npy", np.asarray(test_y, dtype=np.int8))
        np.save(output / "test_source_row.npy", test_source)
        np.save(output / "test_attack_id.npy", test_attack)
        np.save(output / "test_is_synthetic.npy", synthetic)

    files = [
        output / "benign_raw.npy",
        output / "meter_ids.npy",
        output / "day_numbers.npy",
        output / "scaler_mean.npy",
        output / "scaler_scale.npy",
        output / "benign.npy",
        *attack_paths,
        output / "x_train.npy",
        output / "test_original_x.npy",
        output / "test_original_y.npy",
    ]
    if test_adasyn == "printed":
        files.extend([output / "x_test.npy", output / "y_test.npy"])
    metadata: dict[str, object] = {
        "status": "complete",
        "schema": 1,
        "method": (
            "P0-ISET-FCSAE"
            if test_adasyn == "printed"
            else "I-ADASYN-NONE-ISET-FCSAE"
        ),
        "configuration": {
            "mode": profiles_record["mode"],
            "seed": seed,
            "test_adasyn": test_adasyn,
            "adasyn_neighbors": adasyn_neighbors,
        },
        "paper_order": [
            "strict_48_slot_profiles",
            "six_attacks_all_customers",
            "joint_B_plus_M_featurewise_standardization",
            "customer_disjoint_2_to_1_B1_B2",
            "XTR_equals_B1",
            "XTST_original_equals_B2_plus_all_customer_M",
            (
                "ADASYN_inside_test"
                if test_adasyn == "printed"
                else "declared_I-ADASYN-NONE_continuation"
            ),
        ],
        "choices": {
            "population": "all_4225_residential",
            "day": "strict_exact_slots_1_to_48",
            "attack_1": "one_alpha_per_customer_matrix",
            "attack_2_5": "independent_beta_per_half_hour",
            "attack_3": "addition_clip_two_slots_per_hour",
            "scaling": "joint_B_plus_all_M_featurewise_before_split",
            "split": "customer_disjoint_seeded",
            "malicious_test_population": "all_customers",
            "anomaly_adasyn": (
                "test_set_as_printed"
                if test_adasyn == "printed"
                else "omitted_in_explicit_I-ADASYN-NONE_branch"
            ),
            "adasyn_neighbors": adasyn_neighbors,
            "seed": seed,
        },
        "counts": {
            "benign_profiles": int(benign.shape[0]),
            "meters": int(unique_meters.size),
            "B1_profiles": int(train_index.size),
            "B2_profiles": int(b2_index.size),
            "malicious_profiles": int(6 * benign.shape[0]),
            "test_original_profiles": int(original_y.size),
            "test_original_benign": int(np.count_nonzero(original_y == 0)),
            "test_original_malicious": int(np.count_nonzero(original_y == 1)),
            "test_after_adasyn": int(test_y.size) if test_y is not None else None,
            "test_synthetic_profiles": generated,
            "test_benign": (
                int(np.count_nonzero(test_y == 0)) if test_y is not None else None
            ),
            "test_malicious": (
                int(np.count_nonzero(test_y == 1)) if test_y is not None else None
            ),
            "table_iv_full_scalar_readings": int(train_index.size * FEATURES),
        },
        "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "timing_seconds": {
            "profile_extraction": profiles_record["elapsed_seconds"],
            "adasyn": adasyn_seconds,
            "preparation_after_profile_load": time.perf_counter() - started,
        },
        "source": source_record,
        "source_nodes": {
            "attack_3_endpoint": {
                "paper_claim": "t_f = t_i - t_l for a positive theft duration",
                "literal_status": "non_executable_end_precedes_start",
                "assumption": "t_f = t_i + t_l, clipped at hour 24",
                "derived_operation": (
                    "draw integer start 0..19 and duration 4..24 inclusive; "
                    "zero the corresponding half-open two-slots-per-hour interval"
                ),
            },
            "test_adasyn": {
                "paper_claim": "apply ADASYN to B2 plus all-customer M",
                "literal_status": (
                    "executed"
                    if test_adasyn == "printed"
                    else "not_executed_in_this_interpretation"
                ),
                "assumption": (
                    "imbalanced-learn default exact neighbors"
                    if test_adasyn == "printed"
                    else "evaluate the preserved original B2+M rows without resampling"
                ),
                "derived_operation": test_adasyn,
                "first_pass_query_reference_pairs": adasyn_distance_queries,
            },
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
        "profiles": profiles_record,
    }
    save_json(output / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("iset", "sgcc"), default="iset")
    parser.add_argument(
        "--sgcc-representation",
        choices=SGCC_REPRESENTATIONS,
        default="last_48",
    )
    parser.add_argument("--mode", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    parser.add_argument("--adasyn-neighbors", type=int, default=5)
    parser.add_argument(
        "--test-adasyn",
        choices=("printed", "none"),
        default="printed",
        help=(
            "printed executes the paper's test-set ADASYN; none records and "
            "executes the separate I-ADASYN-NONE continuation"
        ),
    )
    parser.add_argument(
        "--force-expensive-adasyn",
        action="store_true",
        help="permit the known multi-trillion-pair full default-ADASYN call",
    )
    args = parser.parse_args()
    if args.adasyn_neighbors < 1:
        parser.error("--adasyn-neighbors must be positive")
    output = args.output or DEFAULT_ROOT / (
        f"sgcc-{args.sgcc_representation.replace('_', '')}-{args.mode}"
        if args.dataset == "sgcc"
        else f"p0-{args.mode}-{args.test_adasyn}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    requested = {
        "mode": args.mode,
        "seed": args.seed,
        "adasyn_neighbors": args.adasyn_neighbors,
    }
    if args.dataset == "sgcc":
        requested.update(
            {
                "representation": args.sgcc_representation,
                "missing": "interpolate_edge_median",
                "test_adasyn": "printed",
                "supervised_adasyn": "printed_before_split",
            }
        )
        source_record: dict[str, object] | None = None
    else:
        requested["test_adasyn"] = args.test_adasyn
        source_record = verified_source()
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") == "complete":
            if metadata.get("configuration") != requested:
                raise ValueError(
                    "existing prepared cache has a different configuration: "
                    f"{metadata.get('configuration')} != {requested}"
                )
            if source_record is not None and metadata.get("source") != source_record:
                raise ValueError("prepared cache source provenance no longer matches")
            print(json.dumps(metadata["counts"], indent=2))
            print(f"already complete: {output}")
            return 0
    try:
        if args.dataset == "sgcc":
            metadata = prepare_sgcc(
                output,
                seed=args.seed,
                mode=args.mode,
                adasyn_neighbors=args.adasyn_neighbors,
                representation=args.sgcc_representation,
            )
        else:
            benign, meters, days, profiles_record = load_or_extract_profiles(
                output, args.mode
            )
            assert source_record is not None
            metadata = prepare_p0(
                output,
                benign,
                meters,
                days,
                profiles_record,
                seed=args.seed,
                adasyn_neighbors=args.adasyn_neighbors,
                test_adasyn=args.test_adasyn,
                force_expensive_adasyn=args.force_expensive_adasyn,
                source_record=source_record,
            )
    except Exception as exc:
        save_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "dataset": args.dataset,
                "mode": args.mode,
                "configuration": requested,
            },
        )
        raise
    print(json.dumps(metadata["counts"], indent=2))
    print(f"prepared: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
