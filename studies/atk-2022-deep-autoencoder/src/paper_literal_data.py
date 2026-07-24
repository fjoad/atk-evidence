"""Paper-literal SGCC preparation for the exploratory Tables I--V audit.

This module intentionally preserves the data order described by Takiddin et
al. (2022), including joint feature scaling before the final split and ADASYN
inside the anomaly-detector test set.  Those choices leak information and are
not recommended methodology; they are implemented here because this is the
paper-literal exploratory track.

The primary SGCC interpretation is one full, chronologically ordered 1,034-day
row per customer.  Fully missing customers are dropped.  Interior gaps are
linearly interpolated within a customer and unresolved edge gaps are filled
from feature medians computed on the benign B1 subset.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split


SCHEMA_VERSION = 1
DEFAULT_DATA_SEED = 20260721
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_ADASYN_NEIGHBORS = 5
DEFAULT_EXPECTED_FEATURES = 1034
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
SGCC_REPRESENTATION_BRANCHES = {
    "full_1034",
    "windows_48_nonoverlap",
    "windows_48_rolling",
    "first_48",
    "last_48",
    "binned_mean_48",
}
SGCC_MISSING_BRANCHES = {
    "drop_incomplete",
    "zero_fill",
    "interpolate_edge_median",
    "customer_mean",
}
SPLIT_UNIT_BRANCHES = {"customer_disjoint", "row_random"}


@dataclass(frozen=True)
class DataPartition:
    """One prepared partition plus sample provenance.

    ``sample_ids`` contains source customer IDs for original rows and stable
    ``ADASYN_*`` identifiers for synthetic rows.  ``is_synthetic`` makes that
    distinction explicit; synthetic IDs must not be interpreted as customers.
    """

    values: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    is_synthetic: np.ndarray

    def __post_init__(self) -> None:
        row_count = self.values.shape[0]
        if self.values.ndim != 2:
            raise ValueError("partition values must be a two-dimensional array")
        for name, array in (
            ("labels", self.labels),
            ("sample_ids", self.sample_ids),
            ("is_synthetic", self.is_synthetic),
        ):
            if array.ndim != 1 or array.shape[0] != row_count:
                raise ValueError(f"partition {name} must have one entry per row")
        if not np.isfinite(self.values).all():
            raise ValueError("partition values must be finite")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("partition labels must be binary")


@dataclass(frozen=True)
class SgccPaperLiteralData:
    """All deterministic SGCC arrays needed by the exploratory runners."""

    dates: np.ndarray
    imputation_fallback: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    anomaly_train: DataPartition
    anomaly_validation: DataPartition
    anomaly_test: DataPartition
    supervised_train: DataPartition
    supervised_test: DataPartition
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        feature_count = self.dates.shape[0]
        if self.dates.ndim != 1:
            raise ValueError("dates must be one-dimensional")
        for name, array in (
            ("imputation_fallback", self.imputation_fallback),
            ("scaler_mean", self.scaler_mean),
            ("scaler_scale", self.scaler_scale),
        ):
            if array.shape != (feature_count,):
                raise ValueError(f"{name} must have one entry per feature")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
        if (self.scaler_scale <= 0).any():
            raise ValueError("scaler_scale must be strictly positive")
        for partition in (
            self.anomaly_train,
            self.anomaly_validation,
            self.anomaly_test,
            self.supervised_train,
            self.supervised_test,
        ):
            if partition.values.shape[1] != feature_count:
                raise ValueError("all partitions must use the same features")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    """Stable fixture/in-memory provenance hash.

    Production calls use the byte hash of the source CSV.  This canonical CSV
    representation exists so tiny tests and deliberate in-memory diagnostics
    retain deterministic provenance too.
    """

    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _id_digest(sample_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids.astype(str):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_sgcc(
    source: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        return frame, "<in-memory-dataframe>", _sha256_frame(frame)
    path = Path(source)
    frame = pd.read_csv(path, low_memory=False)
    return frame, str(path), _sha256_path(path)


def _parse_and_order_dates(
    frame: pd.DataFrame,
    *,
    expected_feature_count: int | None,
) -> tuple[list[str], np.ndarray]:
    required = {"CONS_NO", "FLAG"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SGCC source is missing required columns: {sorted(missing)}")
    date_columns = [column for column in frame.columns if column not in required]
    if expected_feature_count is not None and len(date_columns) != expected_feature_count:
        raise ValueError(
            f"expected {expected_feature_count} SGCC daily features, found {len(date_columns)}"
        )
    if not date_columns:
        raise ValueError("SGCC source has no daily feature columns")
    try:
        parsed_dates = pd.to_datetime(date_columns, format="%Y/%m/%d", errors="raise")
    except ValueError as exc:
        raise ValueError("all SGCC feature columns must be YYYY/M/D dates") from exc
    if parsed_dates.duplicated().any():
        raise ValueError("SGCC source contains duplicate date columns")
    order = np.argsort(parsed_dates.to_numpy(), kind="stable")
    ordered_columns = [date_columns[index] for index in order]
    ordered_dates = parsed_dates.to_numpy(dtype="datetime64[D]")[order]
    return ordered_columns, ordered_dates


def _interpolate_inside_rows(values: np.ndarray) -> np.ndarray:
    """Linearly fill only gaps bounded by observations in each row."""

    interpolated = np.asarray(values, dtype=np.float32).copy()
    positions = np.arange(interpolated.shape[1])
    for row in interpolated:
        observed = np.flatnonzero(np.isfinite(row))
        if observed.size < 2:
            continue
        first, last = int(observed[0]), int(observed[-1])
        interior = positions[first : last + 1]
        missing = ~np.isfinite(row[first : last + 1])
        if missing.any():
            row[interior[missing]] = np.interp(
                interior[missing], observed, row[observed]
            ).astype(np.float32)
    return interpolated


def _feature_median_fallback(
    interpolated: np.ndarray,
    benign_train_indices: np.ndarray,
) -> np.ndarray:
    benign_train = interpolated[benign_train_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fallback = np.nanmedian(benign_train, axis=0)
        global_fallback = float(np.nanmedian(benign_train))
    if not math.isfinite(global_fallback):
        raise ValueError("benign B1 contains no finite readings for imputation")
    fallback = np.where(np.isfinite(fallback), fallback, global_fallback)
    return fallback.astype(np.float32)


def _fill_with_fallback(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    completed = values.copy()
    missing = ~np.isfinite(completed)
    if missing.any():
        completed[missing] = np.broadcast_to(fallback, completed.shape)[missing]
    if not np.isfinite(completed).all():
        raise ValueError("SGCC imputation left non-finite values")
    return completed


def _represent_sgcc(
    values: np.ndarray,
    labels: np.ndarray,
    customer_ids: np.ndarray,
    ordered_dates: np.ndarray,
    imputation_fallback: np.ndarray,
    *,
    branch: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Turn one source-customer row into one declared model sample unit."""

    if branch not in SGCC_REPRESENTATION_BRANCHES:
        raise ValueError(
            f"unknown SGCC representation branch {branch!r}; "
            f"expected one of {sorted(SGCC_REPRESENTATION_BRANCHES)}"
        )
    days = values.shape[1]
    if branch != "full_1034" and days < 48:
        raise ValueError(
            f"SGCC representation {branch!r} requires at least 48 days, found {days}"
        )
    if branch == "full_1034":
        return (
            np.ascontiguousarray(values, dtype=np.float32),
            labels.copy(),
            customer_ids.copy(),
            customer_ids.copy(),
            ordered_dates.copy(),
            imputation_fallback.copy(),
            {"samples_per_customer": 1, "feature_semantics": "calendar_days"},
        )
    if branch in {"first_48", "last_48"}:
        selected = slice(0, 48) if branch == "first_48" else slice(days - 48, days)
        suffix = "first_48" if branch == "first_48" else "last_48"
        return (
            np.ascontiguousarray(values[:, selected], dtype=np.float32),
            labels.copy(),
            np.asarray([f"{item}::{suffix}" for item in customer_ids], dtype=str),
            customer_ids.copy(),
            ordered_dates[selected].copy(),
            imputation_fallback[selected].copy(),
            {"samples_per_customer": 1, "feature_semantics": "calendar_days"},
        )
    if branch == "binned_mean_48":
        bins = np.array_split(np.arange(days, dtype=np.int64), 48)
        represented = np.column_stack(
            [values[:, indices].mean(axis=1) for indices in bins]
        ).astype(np.float32)
        fallback = np.asarray(
            [imputation_fallback[indices].mean() for indices in bins],
            dtype=np.float32,
        )
        return (
            np.ascontiguousarray(represented),
            labels.copy(),
            np.asarray([f"{item}::binned_mean_48" for item in customer_ids], dtype=str),
            customer_ids.copy(),
            np.asarray([ordered_dates[indices[-1]] for indices in bins]),
            fallback,
            {
                "samples_per_customer": 1,
                "feature_semantics": "48_contiguous_calendar_bin_means",
                "bin_sizes": [int(indices.size) for indices in bins],
            },
        )

    step = 48 if branch == "windows_48_nonoverlap" else 1
    starts = np.arange(0, days - 48 + 1, step, dtype=np.int64)
    window_view = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=48, axis=1
    )
    represented = np.ascontiguousarray(
        window_view[:, starts, :].reshape(-1, 48),
        dtype=np.float32,
    )
    repeated_labels = np.repeat(labels, starts.size)
    repeated_customers = np.repeat(customer_ids, starts.size).astype(str)
    sample_ids = np.asarray(
        [
            f"{customer_id}::days_{start:04d}_{start + 47:04d}"
            for customer_id in customer_ids
            for start in starts
        ],
        dtype=str,
    )
    return (
        represented,
        repeated_labels,
        sample_ids,
        repeated_customers,
        np.arange(48, dtype=np.int64),
        np.zeros(48, dtype=np.float32),
        {
            "samples_per_customer": int(starts.size),
            "feature_semantics": "relative_day_within_48_day_window",
            "window_step_days": int(step),
            "discarded_tail_days": (
                int(days - (int(starts[-1]) + 48)) if step == 48 else 0
            ),
        },
    )


def _rows_for_source_indices(
    represented_source_ids: np.ndarray,
    source_customer_ids: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    """Expand an ordered source-customer selection to its represented rows."""

    if represented_source_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(
        np.concatenate(
            [
                np.ones(1, dtype=bool),
                represented_source_ids[1:] != represented_source_ids[:-1],
            ]
        )
    )
    stops = np.concatenate([starts[1:], [represented_source_ids.size]])
    if np.unique(represented_source_ids[starts]).size != starts.size:
        raise ValueError("represented SGCC rows must be contiguous by source customer")
    lookup = {
        str(represented_source_ids[start]): (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    }
    blocks = [
        np.arange(*lookup[str(source_customer_ids[index])], dtype=np.int64)
        for index in np.asarray(source_indices, dtype=np.int64)
    ]
    return (
        np.concatenate(blocks).astype(np.int64, copy=False)
        if blocks
        else np.empty(0, dtype=np.int64)
    )


def _joint_feature_standardize(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    means = np.mean(values, axis=0, dtype=np.float64)
    scales = np.std(values, axis=0, dtype=np.float64)
    zero_variance = scales <= np.finfo(np.float32).eps
    scales[zero_variance] = 1.0
    standardized = ((values - means) / scales).astype(np.float32)
    return (
        standardized,
        means.astype(np.float64),
        scales.astype(np.float64),
        int(np.sum(zero_variance)),
    )


def _apply_feature_scaler(
    values: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    fitted = values[fit_indices]
    means = np.mean(fitted, axis=0, dtype=np.float64)
    scales = np.std(fitted, axis=0, dtype=np.float64)
    zero_variance = scales <= np.finfo(np.float32).eps
    scales[zero_variance] = 1.0
    standardized = ((values - means) / scales).astype(np.float32)
    return (
        standardized,
        means.astype(np.float64),
        scales.astype(np.float64),
        int(np.sum(zero_variance)),
    )


def _standardize_by_branch(
    values: np.ndarray,
    labels: np.ndarray,
    benign_train: np.ndarray,
    *,
    branch: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    """Apply one frozen scaling interpretation without changing split identity."""

    if branch not in SCALING_BRANCHES:
        raise ValueError(
            f"unknown SGCC scaling branch {branch!r}; "
            f"expected one of {sorted(SCALING_BRANCHES)}"
        )
    if branch == "joint_featurewise":
        standardized, mean, scale, zero_count = _joint_feature_standardize(values)
        return standardized, mean, scale, zero_count, {
            "fit_population": "all_retained_benign_and_malicious_rows",
        }
    if branch == "train_benign_only":
        standardized, mean, scale, zero_count = _apply_feature_scaler(
            values, benign_train
        )
        return standardized, mean, scale, zero_count, {
            "fit_population": "anomaly_train_benign_only",
        }
    if branch == "per_profile":
        means = np.mean(values, axis=1, keepdims=True, dtype=np.float64)
        scales = np.std(values, axis=1, keepdims=True, dtype=np.float64)
        zero_variance = scales <= np.finfo(np.float32).eps
        scales[zero_variance] = 1.0
        standardized = ((values - means) / scales).astype(np.float32)
        # The cache schema stores one feature-wise reference scaler. Per-profile
        # moments are already embodied in the cached arrays and are identified
        # by the branch and transformation digest.
        reference_mean = np.zeros(values.shape[1], dtype=np.float64)
        reference_scale = np.ones(values.shape[1], dtype=np.float64)
        return (
            standardized,
            reference_mean,
            reference_scale,
            int(np.sum(zero_variance)),
            {
                "fit_population": "each_row_independently",
                "reference_scaler": "identity_marker_only",
            },
        )

    standardized = np.empty_like(values, dtype=np.float32)
    class_metadata: dict[str, Any] = {}
    reference_mean: np.ndarray | None = None
    reference_scale: np.ndarray | None = None
    zero_count = 0
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        class_values, mean, scale, class_zero_count = _apply_feature_scaler(
            values, indices
        )
        standardized[indices] = class_values[indices]
        zero_count += class_zero_count
        class_metadata[f"class_{label}_mean_sha256"] = _array_digest(mean)
        class_metadata[f"class_{label}_scale_sha256"] = _array_digest(scale)
        if label == 0:
            reference_mean = mean
            reference_scale = scale
    assert reference_mean is not None and reference_scale is not None
    return standardized, reference_mean, reference_scale, zero_count, {
        "fit_population": "each_label_class_independently",
        "reference_scaler": "benign_class",
        **class_metadata,
    }


def _original_partition(
    values: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[DataPartition, dict[str, Any]]:
    counts = np.bincount(labels, minlength=2)
    return (
        DataPartition(
            values=np.ascontiguousarray(values, dtype=np.float32),
            labels=np.ascontiguousarray(labels, dtype=np.int8),
            sample_ids=np.ascontiguousarray(sample_ids.astype(str)),
            is_synthetic=np.zeros(labels.shape[0], dtype=bool),
        ),
        {
            "applied": False,
            "reason": "disabled_by_branch",
            "counts_before": counts.astype(int).tolist(),
            "counts_after": counts.astype(int).tolist(),
            "generated": 0,
        },
    )


def _partition(
    values: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    indices: np.ndarray,
) -> DataPartition:
    return DataPartition(
        values=np.ascontiguousarray(values[indices], dtype=np.float32),
        labels=np.ascontiguousarray(labels[indices], dtype=np.int8),
        sample_ids=np.ascontiguousarray(sample_ids[indices].astype(str)),
        is_synthetic=np.zeros(indices.shape[0], dtype=bool),
    )


def _adasyn_resample(
    values: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    synthetic_prefix: str,
) -> tuple[DataPartition, dict[str, Any]]:
    counts_before = np.bincount(labels, minlength=2)
    if (counts_before == 0).any():
        raise ValueError("ADASYN requires both benign and malicious samples")
    if counts_before[0] == counts_before[1]:
        partition = DataPartition(
            values=np.ascontiguousarray(values, dtype=np.float32),
            labels=np.ascontiguousarray(labels, dtype=np.int8),
            sample_ids=np.ascontiguousarray(sample_ids.astype(str)),
            is_synthetic=np.zeros(labels.shape[0], dtype=bool),
        )
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
            "ADASYN n_neighbors must be smaller than the minority class count "
            f"({n_neighbors} >= {minority_count})"
        )
    sampler = ADASYN(random_state=seed, n_neighbors=n_neighbors)
    resampled_values, resampled_labels = sampler.fit_resample(values, labels)
    generated = int(resampled_labels.shape[0] - labels.shape[0])
    generated_ids = np.asarray(
        [f"{synthetic_prefix}_{index:09d}" for index in range(generated)],
        dtype=str,
    )
    resampled_ids = np.concatenate([sample_ids.astype(str), generated_ids])
    synthetic = np.concatenate(
        [np.zeros(labels.shape[0], dtype=bool), np.ones(generated, dtype=bool)]
    )
    partition = DataPartition(
        values=np.ascontiguousarray(resampled_values, dtype=np.float32),
        labels=np.ascontiguousarray(resampled_labels, dtype=np.int8),
        sample_ids=np.ascontiguousarray(resampled_ids),
        is_synthetic=synthetic,
    )
    counts_after = np.bincount(partition.labels, minlength=2)
    return partition, {
        "applied": True,
        "random_state": seed,
        "n_neighbors": n_neighbors,
        "counts_before": counts_before.astype(int).tolist(),
        "counts_after": counts_after.astype(int).tolist(),
        "generated": generated,
    }


def _subset_partition(partition: DataPartition, indices: np.ndarray) -> DataPartition:
    return DataPartition(
        values=np.ascontiguousarray(partition.values[indices]),
        labels=np.ascontiguousarray(partition.labels[indices]),
        sample_ids=np.ascontiguousarray(partition.sample_ids[indices]),
        is_synthetic=np.ascontiguousarray(partition.is_synthetic[indices]),
    )


def prepare_sgcc_paper_literal(
    source: str | Path | pd.DataFrame,
    *,
    data_seed: int = DEFAULT_DATA_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    adasyn_neighbors: int = DEFAULT_ADASYN_NEIGHBORS,
    expected_feature_count: int | None = DEFAULT_EXPECTED_FEATURES,
    scaling: str = "joint_featurewise",
    anomaly_adasyn: str = "test_set_as_printed",
    supervised_adasyn: str = "before_row_split",
    representation: str = "full_1034",
    missing: str = "interpolate_edge_median",
    split_unit: str = "customer_disjoint",
) -> SgccPaperLiteralData:
    """Prepare SGCC exactly according to the frozen exploratory data contract.

    The anomaly path first splits benign rows 2:1.  Fifteen percent of B1 is
    exposed separately for the frozen early-stopping rule, while the union of
    anomaly train and validation is exactly B1.  B2 is concatenated with every
    malicious row and ADASYN is applied *inside that test set*.

    The supervised path concatenates every benign and malicious row, applies
    ADASYN, and only then makes a stratified 2:1 split.  This is intentionally
    leakage-prone but follows the frozen paper-literal order.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if adasyn_neighbors < 1:
        raise ValueError("adasyn_neighbors must be at least one")
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
    if representation not in SGCC_REPRESENTATION_BRANCHES:
        raise ValueError(
            f"unknown SGCC representation branch {representation!r}; "
            f"expected one of {sorted(SGCC_REPRESENTATION_BRANCHES)}"
        )
    if missing not in SGCC_MISSING_BRANCHES:
        raise ValueError(
            f"unknown SGCC missing-data branch {missing!r}; "
            f"expected one of {sorted(SGCC_MISSING_BRANCHES)}"
        )
    if split_unit not in SPLIT_UNIT_BRANCHES:
        raise ValueError(
            f"unknown split_unit branch {split_unit!r}; "
            f"expected one of {sorted(SPLIT_UNIT_BRANCHES)}"
        )

    frame, source_name, source_sha256 = _read_sgcc(source)
    ordered_columns, ordered_dates = _parse_and_order_dates(
        frame, expected_feature_count=expected_feature_count
    )
    if frame["CONS_NO"].isna().any():
        raise ValueError("SGCC customer IDs may not be missing")
    customer_ids = frame["CONS_NO"].astype(str).to_numpy()
    if pd.Series(customer_ids).duplicated().any():
        raise ValueError("SGCC customer IDs must be unique")
    numeric_labels = pd.to_numeric(frame["FLAG"], errors="raise")
    if not np.isin(numeric_labels, [0, 1]).all():
        raise ValueError("SGCC FLAG must contain only 0 and 1")
    labels = numeric_labels.to_numpy(dtype=np.int8)
    values = frame[ordered_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )

    if np.isinf(values).any():
        raise ValueError("SGCC source contains infinite consumption values")
    fully_missing = np.all(np.isnan(values), axis=1)
    fully_missing_ids = customer_ids[fully_missing].astype(str).tolist()
    values = values[~fully_missing]
    labels = labels[~fully_missing]
    customer_ids = customer_ids[~fully_missing]
    incomplete_ids: list[str] = []
    if missing == "drop_incomplete":
        incomplete = np.any(np.isnan(values), axis=1)
        incomplete_ids = customer_ids[incomplete].astype(str).tolist()
        values = values[~incomplete]
        labels = labels[~incomplete]
        customer_ids = customer_ids[~incomplete]

    source_benign_indices = np.flatnonzero(labels == 0)
    source_malicious_indices = np.flatnonzero(labels == 1)
    if source_benign_indices.size < 3 or source_malicious_indices.size < 2:
        raise ValueError("SGCC preparation needs at least three benign and two malicious rows")
    source_benign_b1, source_benign_b2 = train_test_split(
        source_benign_indices,
        train_size=2.0 / 3.0,
        shuffle=True,
        random_state=data_seed,
    )
    source_benign_b1 = np.asarray(source_benign_b1, dtype=np.int64)
    source_benign_b2 = np.asarray(source_benign_b2, dtype=np.int64)

    if missing == "interpolate_edge_median":
        interpolated = _interpolate_inside_rows(values)
        source_fallback = _feature_median_fallback(
            interpolated, source_benign_b1
        )
        completed_source = _fill_with_fallback(interpolated, source_fallback)
        missing_metadata = {
            "interpolation": "within_customer_linear_interior_only",
            "unresolved_missing": (
                "benign_b1_feature_median_then_global_b1_median"
            ),
        }
    elif missing == "zero_fill":
        source_fallback = np.zeros(values.shape[1], dtype=np.float32)
        completed_source = np.nan_to_num(values, nan=0.0).astype(np.float32)
        missing_metadata = {"missing_fill": "constant_zero"}
    elif missing == "customer_mean":
        row_means = np.nanmean(values, axis=1, dtype=np.float64)
        completed_source = values.copy()
        missing_mask = np.isnan(completed_source)
        completed_source[missing_mask] = np.broadcast_to(
            row_means[:, None], completed_source.shape
        )[missing_mask]
        source_fallback = np.zeros(values.shape[1], dtype=np.float32)
        missing_metadata = {
            "missing_fill": "each_customer_observed_mean",
            "reference_fallback": "identity_marker_only",
        }
    else:
        source_fallback = np.zeros(values.shape[1], dtype=np.float32)
        completed_source = values.astype(np.float32, copy=True)
        missing_metadata = {
            "missing_fill": "none",
            "incomplete_customers": "dropped",
            "reference_fallback": "identity_marker_only",
        }
    if not np.isfinite(completed_source).all():
        raise ValueError("SGCC missing-data branch left non-finite values")

    (
        represented_values,
        represented_labels,
        sample_ids,
        represented_source_ids,
        feature_dates,
        fallback,
        representation_metadata,
    ) = _represent_sgcc(
        completed_source,
        labels,
        customer_ids,
        ordered_dates,
        source_fallback,
        branch=representation,
    )
    benign_indices = np.flatnonzero(represented_labels == 0)
    malicious_indices = np.flatnonzero(represented_labels == 1)

    if split_unit == "customer_disjoint":
        source_anomaly_train, source_anomaly_validation = train_test_split(
            source_benign_b1,
            test_size=validation_fraction,
            shuffle=True,
            random_state=data_seed + 1,
        )
        anomaly_train_indices = _rows_for_source_indices(
            represented_source_ids,
            customer_ids,
            np.asarray(source_anomaly_train, dtype=np.int64),
        )
        anomaly_validation_indices = _rows_for_source_indices(
            represented_source_ids,
            customer_ids,
            np.asarray(source_anomaly_validation, dtype=np.int64),
        )
        benign_b1 = _rows_for_source_indices(
            represented_source_ids, customer_ids, source_benign_b1
        )
        benign_b2 = _rows_for_source_indices(
            represented_source_ids, customer_ids, source_benign_b2
        )
    else:
        benign_b1, benign_b2 = train_test_split(
            benign_indices,
            train_size=2.0 / 3.0,
            shuffle=True,
            random_state=data_seed,
        )
        anomaly_train_indices, anomaly_validation_indices = train_test_split(
            benign_b1,
            test_size=validation_fraction,
            shuffle=True,
            random_state=data_seed + 1,
        )
        benign_b1 = np.asarray(benign_b1, dtype=np.int64)
        benign_b2 = np.asarray(benign_b2, dtype=np.int64)
        anomaly_train_indices = np.asarray(anomaly_train_indices, dtype=np.int64)
        anomaly_validation_indices = np.asarray(
            anomaly_validation_indices, dtype=np.int64
        )

    (
        standardized,
        scaler_mean,
        scaler_scale,
        zero_variance_count,
        scaling_metadata,
    ) = _standardize_by_branch(
        represented_values,
        represented_labels,
        anomaly_train_indices,
        branch=scaling,
    )

    anomaly_train = _partition(
        standardized, represented_labels, sample_ids, anomaly_train_indices
    )
    anomaly_validation = _partition(
        standardized, represented_labels, sample_ids, anomaly_validation_indices
    )

    anomaly_original_indices = np.concatenate([benign_b2, malicious_indices])
    anomaly_original_values = standardized[anomaly_original_indices]
    anomaly_original_labels = represented_labels[anomaly_original_indices]
    anomaly_original_ids = sample_ids[anomaly_original_indices]
    if anomaly_adasyn == "test_set_as_printed":
        anomaly_test, anomaly_adasyn_metadata = _adasyn_resample(
            anomaly_original_values,
            anomaly_original_labels,
            anomaly_original_ids,
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_ANOMALY_TEST",
        )
    else:
        anomaly_test, anomaly_adasyn_metadata = _original_partition(
            anomaly_original_values,
            anomaly_original_labels,
            anomaly_original_ids,
        )

    supervised_original_indices = np.concatenate([benign_indices, malicious_indices])
    if supervised_adasyn == "before_row_split":
        supervised_balanced, supervised_adasyn_metadata = _adasyn_resample(
            standardized[supervised_original_indices],
            represented_labels[supervised_original_indices],
            sample_ids[supervised_original_indices],
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_SUPERVISED",
        )
        supervised_indices = np.arange(supervised_balanced.labels.shape[0])
        supervised_train_indices, supervised_test_indices = train_test_split(
            supervised_indices,
            train_size=2.0 / 3.0,
            shuffle=True,
            stratify=supervised_balanced.labels,
            random_state=data_seed,
        )
        supervised_train = _subset_partition(
            supervised_balanced,
            np.asarray(supervised_train_indices, dtype=np.int64),
        )
        supervised_test = _subset_partition(
            supervised_balanced,
            np.asarray(supervised_test_indices, dtype=np.int64),
        )
        supervised_after_adasyn = supervised_balanced.labels.size
    else:
        source_indices = np.arange(labels.size, dtype=np.int64)
        source_train, source_test = train_test_split(
            source_indices,
            train_size=2.0 / 3.0,
            shuffle=True,
            stratify=labels,
            random_state=data_seed,
        )
        train_original = _rows_for_source_indices(
            represented_source_ids,
            customer_ids,
            np.asarray(source_train, dtype=np.int64),
        )
        test_original = _rows_for_source_indices(
            represented_source_ids,
            customer_ids,
            np.asarray(source_test, dtype=np.int64),
        )
        supervised_train, supervised_adasyn_metadata = _adasyn_resample(
            standardized[train_original],
            represented_labels[train_original],
            sample_ids[train_original],
            seed=data_seed,
            n_neighbors=adasyn_neighbors,
            synthetic_prefix="ADASYN_SUPERVISED_TRAIN",
        )
        supervised_test = _partition(
            standardized, represented_labels, sample_ids, test_original
        )
        supervised_after_adasyn = (
            supervised_train.labels.size + supervised_test.labels.size
        )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SGCC",
        "track": "exploratory_paper_literal",
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "rows_before_drop": int(frame.shape[0]),
            "features": int(len(ordered_columns)),
            "represented_features": int(feature_dates.size),
        },
        "config": {
            "data_seed": data_seed,
            "anomaly_b_split_train_fraction": 2.0 / 3.0,
            "validation_fraction_within_b1": validation_fraction,
            "supervised_split_train_fraction": 2.0 / 3.0,
            "supervised_split_stratified": True,
            "adasyn_neighbors": adasyn_neighbors,
            "scaling_branch": scaling,
            "anomaly_adasyn_branch": anomaly_adasyn,
            "supervised_adasyn_branch": supervised_adasyn,
            "representation_branch": representation,
            "missing_branch": missing,
            "split_unit_branch": split_unit,
        },
        "preprocessing": {
            "representation": representation,
            "representation_details": representation_metadata,
            "chronologically_sorted": True,
            "fully_missing_rows": "dropped",
            "missing": missing,
            "missing_details": missing_metadata,
            "scaling": scaling,
            "scaling_details": scaling_metadata,
            "standard_deviation_ddof": 0,
            "zero_variance_features_scaled_by_one": zero_variance_count,
        },
        "counts": {
            "dropped_fully_missing": len(fully_missing_ids),
            "dropped_incomplete": len(incomplete_ids),
            "dropped_customer_ids": [*fully_missing_ids, *incomplete_ids],
            "retained_customers": int(labels.shape[0]),
            "retained": int(represented_labels.shape[0]),
            "benign": int(benign_indices.size),
            "malicious": int(malicious_indices.size),
            "anomaly_b1_total": int(benign_b1.size),
            "anomaly_b1_customers": int(source_benign_b1.size),
            "anomaly_train": int(anomaly_train.labels.size),
            "anomaly_validation": int(anomaly_validation.labels.size),
            "anomaly_b2": int(benign_b2.size),
            "anomaly_test_original": int(anomaly_original_indices.size),
            "anomaly_test_after_adasyn": int(anomaly_test.labels.size),
            "supervised_before_adasyn": int(supervised_original_indices.size),
            "supervised_after_adasyn": int(supervised_after_adasyn),
            "supervised_train": int(supervised_train.labels.size),
            "supervised_test": int(supervised_test.labels.size),
        },
        "adasyn": {
            "anomaly_test": anomaly_adasyn_metadata,
            "supervised": supervised_adasyn_metadata,
            # Compatibility alias for implementation-v1 cache readers.
            "supervised_before_split": supervised_adasyn_metadata,
        },
        "partition_id_sha256": {
            "anomaly_train": _id_digest(anomaly_train.sample_ids),
            "anomaly_validation": _id_digest(anomaly_validation.sample_ids),
            "anomaly_test": _id_digest(anomaly_test.sample_ids),
            "supervised_train": _id_digest(supervised_train.sample_ids),
            "supervised_test": _id_digest(supervised_test.sample_ids),
        },
        "transformation_sha256": {
            "dates": _array_digest(feature_dates),
            "imputation_fallback": _array_digest(fallback),
            "scaler_mean": _array_digest(scaler_mean),
            "scaler_scale": _array_digest(scaler_scale),
        },
        "warnings": [
            warning
            for condition, warning in (
                (
                    scaling != "train_benign_only",
                    f"Scaling branch {scaling!r} uses information beyond benign B1.",
                ),
                (
                    anomaly_adasyn == "test_set_as_printed",
                    "ADASYN is intentionally applied within the anomaly test set.",
                ),
                (
                    supervised_adasyn == "before_row_split",
                    "ADASYN is intentionally applied before the supervised split.",
                ),
                (True, "Synthetic sample IDs are not customer IDs."),
            )
            if condition
        ],
    }
    return SgccPaperLiteralData(
        dates=feature_dates,
        imputation_fallback=fallback,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        anomaly_train=anomaly_train,
        anomaly_validation=anomaly_validation,
        anomaly_test=anomaly_test,
        supervised_train=supervised_train,
        supervised_test=supervised_test,
        metadata=metadata,
    )


_PARTITION_NAMES = (
    "anomaly_train",
    "anomaly_validation",
    "anomaly_test",
    "supervised_train",
    "supervised_test",
)


def save_prepared_sgcc(
    prepared: SgccPaperLiteralData,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Save a reproducible local cache (intended for ``data/derived``).

    The returned NPZ contains only numeric/string arrays (no pickled objects).
    Its SHA-256 and the complete source/config metadata are stored beside it in
    JSON.  Raw or derived data caches must remain outside Git.
    """

    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    manifest_path = prefix.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "dates": prepared.dates,
        "imputation_fallback": prepared.imputation_fallback,
        "scaler_mean": prepared.scaler_mean,
        "scaler_scale": prepared.scaler_scale,
    }
    for name in _PARTITION_NAMES:
        partition = getattr(prepared, name)
        arrays[f"{name}_values"] = partition.values
        arrays[f"{name}_labels"] = partition.labels
        arrays[f"{name}_sample_ids"] = partition.sample_ids.astype(str)
        arrays[f"{name}_is_synthetic"] = partition.is_synthetic
    temporary = npz_path.with_suffix(npz_path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(npz_path)
    manifest = {
        "cache_schema_version": SCHEMA_VERSION,
        "npz_filename": npz_path.name,
        "npz_sha256": _sha256_path(npz_path),
        "metadata": prepared.metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path, manifest_path


def load_prepared_sgcc(output_prefix: str | Path) -> SgccPaperLiteralData:
    """Load and checksum-verify a cache written by :func:`save_prepared_sgcc`."""

    prefix = Path(output_prefix)
    npz_path = prefix.with_suffix(".npz")
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported SGCC cache schema version")
    actual_sha256 = _sha256_path(npz_path)
    if actual_sha256 != manifest.get("npz_sha256"):
        raise ValueError("SGCC cache checksum mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        partitions = {
            name: DataPartition(
                values=arrays[f"{name}_values"],
                labels=arrays[f"{name}_labels"],
                sample_ids=arrays[f"{name}_sample_ids"],
                is_synthetic=arrays[f"{name}_is_synthetic"],
            )
            for name in _PARTITION_NAMES
        }
        return SgccPaperLiteralData(
            dates=arrays["dates"],
            imputation_fallback=arrays["imputation_fallback"],
            scaler_mean=arrays["scaler_mean"],
            scaler_scale=arrays["scaler_scale"],
            anomaly_train=partitions["anomaly_train"],
            anomaly_validation=partitions["anomaly_validation"],
            anomaly_test=partitions["anomaly_test"],
            supervised_train=partitions["supervised_train"],
            supervised_test=partitions["supervised_test"],
            metadata=manifest["metadata"],
        )
