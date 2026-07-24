"""Resumable exploratory runner primitives for Paper 1.

The command-line interface in this module remains the frozen SGCC/Table-II
runner.  Its execution and immutable-attempt primitives are also reused by the
small exact-ISET adapter for Tables III--V.

Every invocation writes a new immutable attempt directory.  A successful
attempt is skipped on resume only when its run fingerprint matches and every
artifact checksum still verifies.  ``--force`` appends another attempt; it
never overwrites prior evidence.  Failures are persisted and are not treated
as completed runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Keras chooses its backend at import time.  Models are imported lazily so
# classical-only runs do not pay the import/startup cost.
os.environ.setdefault("KERAS_BACKEND", "torch")

from paper_literal_benchmarks import (  # noqa: E402
    BenchmarkResult,
    fit_arima_completion,
    fit_arima_110,
    fit_gaussian_nb,
    fit_multiclass_svm,
    fit_one_class_svm,
)
from paper_literal_data import (  # noqa: E402
    SGCC_MISSING_BRANCHES,
    SGCC_REPRESENTATION_BRANCHES,
    SPLIT_UNIT_BRANCHES,
    SgccPaperLiteralData,
    prepare_sgcc_paper_literal,
)
from paper_literal_metrics import (  # noqa: E402
    ThresholdSelection,
    evaluate_binary_scores,
    select_threshold,
    threshold_predictions,
)


SCHEMA_VERSION = 1
EXPECTED_SGCC_SHA256 = (
    "99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7"
)
EXPECTED_SGCC_FEATURES = 1034

ANOMALY_NEURAL_MODELS = (
    "fc_sae",
    "lstm_sae",
    "fc_vae",
    "lstm_vae",
    "lstm_aea",
)
SUPERVISED_NEURAL_MODELS = (
    "supervised_feed_forward",
    "supervised_lstm",
)
CLASSICAL_MODELS = (
    "naive_bayes",
    "arima",
    "one_class_svm",
    "multiclass_svm",
)
TABLE_II_MODELS = (
    *ANOMALY_NEURAL_MODELS,
    "naive_bayes",
    "arima",
    "one_class_svm",
    *SUPERVISED_NEURAL_MODELS,
    "multiclass_svm",
)
VALIDATION_LABEL_BRANCHES = {
    "b1_generated_attacks",
    "b2_validation_carveout",
    "printed_threshold_no_derivation",
}
THRESHOLD_SCOPES = {"iset_transferred", "dataset_specific"}
VALIDATION_POLICIES = {
    "none_fixed_epochs",
    "holdout_no_refit",
    "crossval_refit_b1",
    "holdout_refit_b1",
}

_MODEL_ALIASES = {
    "fc-sae": "fc_sae",
    "lstm-sae": "lstm_sae",
    "fc-vae": "fc_vae",
    "lstm-vae": "lstm_vae",
    "lstm-aea": "lstm_aea",
    "nb": "naive_bayes",
    "naive-bayes": "naive_bayes",
    "single-class-svm": "one_class_svm",
    "one-class-svm": "one_class_svm",
    "feed-forward": "supervised_feed_forward",
    "feed_forward": "supervised_feed_forward",
    "lstm": "supervised_lstm",
    "multi-class-svm": "multiclass_svm",
}


class UnsupportedExperimentError(ValueError):
    """Raised when a request would cross the exact CER/ISET data gate."""


@dataclass(frozen=True)
class Contract:
    """Validated frozen TOML contract plus byte-level provenance."""

    path: Path
    raw: Mapping[str, Any]
    sha256: str

    @property
    def run(self) -> Mapping[str, Any]:
        return self.raw["run"]

    @property
    def data(self) -> Mapping[str, Any]:
        return self.raw["data"]

    @property
    def thresholds(self) -> Mapping[str, Any]:
        return self.raw["thresholds"]


@dataclass(frozen=True)
class ExecutionResult:
    """All in-memory evidence returned by one successful model execution."""

    scores: Mapping[str, np.ndarray]
    predictions: Mapping[str, np.ndarray]
    labels: np.ndarray
    sample_ids: np.ndarray
    is_synthetic: np.ndarray
    history: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Any]]
    fit_seconds: float
    score_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    supplemental_results: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    model: str
    seed: int
    status: str
    attempt_dir: Path
    fingerprint: str
    message: str = ""


@dataclass(frozen=True)
class RunScope:
    """Dataset/table identity and logical path for one immutable run family."""

    table: int
    dataset: str
    path_parts: tuple[str, ...]
    source_code_files: tuple[str, ...] = ()
    fingerprint_extra: Mapping[str, Any] = field(default_factory=dict)


SGCC_TABLE_II_SCOPE = RunScope(2, "SGCC", ("table_2", "sgcc"))


@dataclass(frozen=True)
class ThresholdPopulation:
    """Labeled rows used only to derive an anomaly-score threshold."""

    values: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    test_partition: Any
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        rows = self.values.shape[0]
        if self.values.ndim != 2:
            raise ValueError("threshold-population values must be two-dimensional")
        for name, array in (
            ("labels", self.labels),
            ("sample_ids", self.sample_ids),
        ):
            if array.ndim != 1 or array.shape[0] != rows:
                raise ValueError(
                    f"threshold-population {name} must contain one value per row"
                )
        if not np.isfinite(self.values).all():
            raise ValueError("threshold-population values must be finite")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("threshold-population labels must be binary")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass values and non-finite floats to strict JSON."""

    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_contract(path: str | Path) -> Contract:
    """Load and validate the fields consumed by this runner."""

    import tomllib

    contract_path = Path(path).expanduser().resolve()
    payload = contract_path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))
    for section in ("run", "data", "thresholds", "table_1"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"contract is missing [{section}]")
    required_run = {
        "data_seed",
        "model_seeds",
        "max_epochs",
        "batch_size",
        "warmup_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "validation_fraction_within_train",
        "supervised_svm_max_samples",
        "one_class_svm_max_samples",
    }
    missing_run = required_run.difference(raw["run"])
    if missing_run:
        raise ValueError(f"contract [run] is missing {sorted(missing_run)}")
    required_thresholds = {
        "arima",
        "one_class_svm",
        *ANOMALY_NEURAL_MODELS,
    }
    missing_thresholds = required_thresholds.difference(raw["thresholds"])
    if missing_thresholds:
        raise ValueError(
            f"contract [thresholds] is missing {sorted(missing_thresholds)}"
        )
    missing_models = set(ANOMALY_NEURAL_MODELS + SUPERVISED_NEURAL_MODELS).difference(
        raw["table_1"]
    )
    if missing_models:
        raise ValueError(f"contract [table_1] is missing {sorted(missing_models)}")
    return Contract(contract_path, raw, _sha256_bytes(payload))


def reject_unsupported_scope(table: int, dataset: str) -> None:
    normalized_dataset = dataset.strip().lower()
    if table == 2 and normalized_dataset == "sgcc":
        return
    if table in {3, 4, 5} or normalized_dataset in {"cer", "iset"}:
        raise UnsupportedExperimentError(
            "Tables III-V require all seven checksum-verified official CER/ISET "
            "files and are intentionally unsupported by this SGCC Table-II "
            "runner; no proxy or partial table will be produced."
        )
    raise UnsupportedExperimentError(
        "This runner implements only Paper 1, Table II on verified SGCC."
    )


def canonical_model_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "-")
    normalized = _MODEL_ALIASES.get(normalized, normalized.replace("-", "_"))
    if normalized not in TABLE_II_MODELS:
        raise ValueError(
            f"unknown Table II model {name!r}; choose from {', '.join(TABLE_II_MODELS)}"
        )
    return normalized


def resolve_models(requested: Sequence[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list(TABLE_II_MODELS)
    flattened: list[str] = []
    for item in requested:
        flattened.extend(part for part in item.split(",") if part)
    models = [canonical_model_name(item) for item in flattened]
    return list(dict.fromkeys(models))


def resolve_seeds(contract: Contract, requested: Sequence[int] | None) -> list[int]:
    declared = [int(seed) for seed in contract.run["model_seeds"]]
    seeds = declared if not requested else [int(seed) for seed in requested]
    undeclared = sorted(set(seeds).difference(declared))
    if undeclared:
        raise ValueError(
            f"seeds {undeclared} are outside the frozen model_seeds {declared}"
        )
    return list(dict.fromkeys(seeds))


def verify_and_prepare_sgcc(
    source: str | Path,
    contract: Contract,
    *,
    expected_sha256: str = EXPECTED_SGCC_SHA256,
    expected_feature_count: int = EXPECTED_SGCC_FEATURES,
    scaling: str = "joint_featurewise",
    anomaly_adasyn: str = "test_set_as_printed",
    supervised_adasyn: str = "before_row_split",
    adasyn_neighbors: int | None = None,
    representation: str = "full_1034",
    missing: str = "interpolate_edge_median",
    split_unit: str = "customer_disjoint",
) -> tuple[SgccPaperLiteralData, Mapping[str, Any], float]:
    """Checksum-gate and prepare the exact SGCC source, including timing."""

    started = time.perf_counter()
    source_path = Path(source).expanduser().resolve()
    actual_sha256 = _sha256_path(source_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "SGCC checksum mismatch: expected "
            f"{expected_sha256.lower()}, got {actual_sha256.lower()}"
        )
    prepared = prepare_sgcc_paper_literal(
        source_path,
        data_seed=int(contract.run["data_seed"]),
        validation_fraction=float(contract.run["validation_fraction_within_train"]),
        adasyn_neighbors=(
            int(contract.data["adasyn_neighbors"])
            if adasyn_neighbors is None
            else int(adasyn_neighbors)
        ),
        expected_feature_count=expected_feature_count,
        scaling=scaling,
        anomaly_adasyn=anomaly_adasyn,
        supervised_adasyn=supervised_adasyn,
        representation=representation,
        missing=missing,
        split_unit=split_unit,
    )
    elapsed = time.perf_counter() - started
    verification = {
        "source_path": str(source_path),
        "expected_sha256": expected_sha256.lower(),
        "actual_sha256": actual_sha256.lower(),
        "verified": True,
        "expected_feature_count": expected_feature_count,
    }
    return prepared, verification, elapsed


def _anomaly_training_values(prepared: SgccPaperLiteralData) -> np.ndarray:
    return prepared.anomaly_train.values


def _all_b1_values(prepared: SgccPaperLiteralData) -> np.ndarray:
    return np.concatenate(
        [prepared.anomaly_train.values, prepared.anomaly_validation.values], axis=0
    )


def _take_partition(partition: Any, indices: np.ndarray) -> Any:
    """Take aligned rows from either registered partition dataclass."""

    index = np.asarray(indices, dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for descriptor in dataclasses.fields(partition):
        array = np.asarray(getattr(partition, descriptor.name))
        if array.ndim == 0 or array.shape[0] != partition.values.shape[0]:
            raise TypeError(
                f"partition field {descriptor.name!r} is not row aligned"
            )
        values[descriptor.name] = np.ascontiguousarray(array[index])
    return type(partition)(**values)


def _concatenate_partitions(partitions: Sequence[Any]) -> Any:
    """Concatenate aligned rows without discarding ISET provenance."""

    if not partitions:
        raise ValueError("at least one partition is required")
    partition_type = type(partitions[0])
    if any(type(item) is not partition_type for item in partitions):
        raise TypeError("all partitions must have the same concrete type")
    values = {
        descriptor.name: np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(getattr(partition, descriptor.name))
                    for partition in partitions
                ],
                axis=0,
            )
        )
        for descriptor in dataclasses.fields(partitions[0])
    }
    return partition_type(**values)


def _generated_validation_attacks(
    benign: Any,
    *,
    prepared: Any,
    seed: int,
) -> Any:
    """Generate the six printed attacks from benign validation rows."""

    config = dict(prepared.metadata.get("config", {}))
    if hasattr(benign, "source_profile_ids"):
        from paper_literal_iset import _generate_attacks

        return _generate_attacks(
            benign,
            seed=seed,
            partition_name="threshold_validation_b1",
            attack1_scope=str(config.get("attack1_scope", "per_profile")),
            attack2_granularity=str(
                config.get("attack2_granularity", "per_half_hour")
            ),
            attack3_interval=str(
                config.get("attack3_interval", "valid_fit_addition")
            ),
            attack_hour_mapping=str(
                config.get("attack_hour_mapping", "two_slots_per_hour")
            ),
        )

    from attacks import generate_attack

    roots = np.random.SeedSequence(seed).spawn(6)
    rows = benign.values.shape[0]
    attacked = np.empty((rows * 6, benign.values.shape[1]), dtype=np.float32)
    for attack_id, child in enumerate(roots, start=1):
        generator = np.random.default_rng(child)
        offset = (attack_id - 1) * rows
        for row_index, profile in enumerate(benign.values):
            attacked[offset + row_index] = generate_attack(
                profile,
                attack_id,
                generator,
                samples_per_hour=2 if profile.size >= 8 else 1,
            )
    return type(benign)(
        values=attacked,
        labels=np.ones(rows * 6, dtype=np.int8),
        sample_ids=np.asarray(
            [
                f"{sample_id}:validation_attack_{attack_id}"
                for attack_id in range(1, 7)
                for sample_id in benign.sample_ids.astype(str)
            ],
            dtype=str,
        ),
        is_synthetic=np.zeros(rows * 6, dtype=bool),
    )


def build_threshold_population(
    prepared: Any,
    *,
    branch: str,
    seed: int,
    validation_fraction: float,
) -> ThresholdPopulation:
    """Execute one source-grounded resolution of the validation-label gap."""

    if branch not in VALIDATION_LABEL_BRANCHES:
        raise ValueError(
            f"unknown validation-label branch {branch!r}; "
            f"expected one of {sorted(VALIDATION_LABEL_BRANCHES)}"
        )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    test = prepared.anomaly_test
    if branch == "printed_threshold_no_derivation":
        validation = prepared.anomaly_validation
        return ThresholdPopulation(
            values=np.asarray(validation.values),
            labels=np.asarray(validation.labels, dtype=np.int8),
            sample_ids=np.asarray(validation.sample_ids).astype(str),
            test_partition=test,
            metadata={
                "branch": branch,
                "derivation": "none",
                "source_partition": "anomaly_validation_benign",
                "validation_samples": int(validation.labels.size),
                "test_samples_after_carveout": int(test.labels.size),
            },
        )

    if branch == "b1_generated_attacks":
        benign = prepared.anomaly_validation
        attacks = _generated_validation_attacks(
            benign,
            prepared=prepared,
            seed=seed,
        )
        validation = _concatenate_partitions([benign, attacks])
        return ThresholdPopulation(
            values=np.asarray(validation.values),
            labels=np.asarray(validation.labels, dtype=np.int8),
            sample_ids=np.asarray(validation.sample_ids).astype(str),
            test_partition=test,
            metadata={
                "branch": branch,
                "derivation": "six_printed_attacks_from_b1_validation",
                "generation_space": "prepared_model_input_space",
                "seed": int(seed),
                "benign_samples": int(benign.labels.size),
                "malicious_samples": int(attacks.labels.size),
                "validation_samples": int(validation.labels.size),
                "test_samples_after_carveout": int(test.labels.size),
            },
        )

    from sklearn.model_selection import train_test_split

    counts = np.bincount(np.asarray(test.labels, dtype=np.int8), minlength=2)
    if (counts < 2).any():
        raise ValueError(
            "B2 validation carve-out requires at least two rows from each class"
        )
    class_count = int(np.count_nonzero(counts))
    requested = int(math.ceil(validation_fraction * test.labels.size))
    validation_size = max(class_count, requested)
    validation_size = min(validation_size, int(test.labels.size - class_count))
    test_indices, validation_indices = train_test_split(
        np.arange(test.labels.size),
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=test.labels,
    )
    validation = _take_partition(test, np.sort(validation_indices))
    reduced_test = _take_partition(test, np.sort(test_indices))
    if set(validation.sample_ids.astype(str)) & set(
        reduced_test.sample_ids.astype(str)
    ):
        raise RuntimeError("validation and final-test row identities overlap")
    return ThresholdPopulation(
        values=np.asarray(validation.values),
        labels=np.asarray(validation.labels, dtype=np.int8),
        sample_ids=np.asarray(validation.sample_ids).astype(str),
        test_partition=reduced_test,
        metadata={
            "branch": branch,
            "derivation": "stratified_b2_test_carveout",
            "random_state": int(seed),
            "requested_fraction": float(validation_fraction),
            "validation_samples": int(validation.labels.size),
            "validation_class_counts": np.bincount(
                validation.labels, minlength=2
            )
            .astype(int)
            .tolist(),
            "test_samples_before_carveout": int(test.labels.size),
            "test_samples_after_carveout": int(reduced_test.labels.size),
            "row_identity_overlap": 0,
        },
    )


def _metrics_for_scores(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    orientations: Mapping[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Mapping[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Mapping[str, Any]] = {}
    for score_name, values in scores.items():
        threshold = float(thresholds[score_name])
        positive_if = (
            str(orientations[score_name]) if orientations is not None else "higher"
        )
        predictions[score_name] = threshold_predictions(
            values, threshold, positive_if=positive_if
        )
        metrics[score_name] = evaluate_binary_scores(
            labels, values, threshold=threshold, positive_if=positive_if
        ).as_dict()
    return predictions, metrics


def execute_classical(
    model_name: str,
    prepared: Any,
    contract: Contract,
    seed: int,
    *,
    classical_options: Mapping[str, Any] | None = None,
    threshold_rule: str = "printed_constant",
    threshold_scope: str = "iset_transferred",
    validation_labels: str = "printed_threshold_no_derivation",
    transferred_thresholds: Mapping[str, float] | None = None,
) -> ExecutionResult:
    """Execute one frozen classical benchmark and its declared ambiguities."""

    del seed  # Caps use the frozen data seed, not an outcome-selectable model seed.
    options = dict(classical_options or {})
    data_seed = int(contract.run["data_seed"])
    threshold_population: ThresholdPopulation | None = None
    if model_name in {"arima", "one_class_svm"}:
        preparation_config = dict(prepared.metadata.get("config", {}))
        threshold_population = build_threshold_population(
            prepared,
            branch=validation_labels,
            seed=int(
                preparation_config.get(
                    "attack_seed",
                    contract.run["data_seed"],
                )
            ),
            validation_fraction=float(
                contract.run["validation_fraction_within_train"]
            ),
        )
        test = threshold_population.test_partition
    else:
        test = prepared.supervised_test

    if model_name == "naive_bayes":
        result = fit_gaussian_nb(
            prepared.supervised_train.values,
            prepared.supervised_train.labels,
            test.values,
        )
        score_name = "positive_class_probability"
    elif model_name == "arima":
        completion = str(options.get("arima_completion", "p1_pooled_mse"))
        result = fit_arima_completion(
            _all_b1_values(prepared),
            test.values,
            completion=completion,
            threshold=float(contract.thresholds["arima"]),
        )
        score_name = (
            "residual_mse"
            if not classical_options and completion == "p1_pooled_mse"
            else f"arima_{completion}"
        )
    elif model_name == "one_class_svm":
        svm_training = str(
            options.get("svm_training", "resource_cap_diagnostic")
        )
        result = fit_one_class_svm(
            _all_b1_values(prepared),
            test.values,
            max_samples=(
                None
                if svm_training == "full_data"
                else int(contract.run["one_class_svm_max_samples"])
            ),
            seed=data_seed,
            threshold=float(contract.thresholds["one_class_svm"]),
        )
        score_name = "negative_decision_function"
    elif model_name == "multiclass_svm":
        label_branch = str(options.get("multiclass_labels", "binary"))
        train_values = np.asarray(prepared.supervised_train.values)
        train_labels = np.asarray(prepared.supervised_train.labels)
        if label_branch == "benign_plus_six_attacks":
            if not hasattr(prepared.supervised_train, "attack_ids"):
                raise ValueError(
                    "the seven-class SVM interpretation is non-executable on "
                    "SGCC because SGCC has no six attack-type labels"
                )
            original = ~np.asarray(
                prepared.supervised_train.is_synthetic,
                dtype=bool,
            )
            train_values = train_values[original]
            train_labels = train_labels[original]
            attack_ids = np.asarray(
                prepared.supervised_train.attack_ids
            )[original]
            malicious = train_labels == 1
            if np.any(attack_ids[malicious] < 1) or np.any(
                attack_ids[malicious] > 6
            ):
                raise ValueError(
                    "seven-class SVM requires attack IDs 1 through 6 for "
                    "every malicious training row"
                )
            train_labels = np.where(malicious, attack_ids, 0).astype(np.int8)
            from imblearn.over_sampling import ADASYN

            neighbors = int(
                prepared.metadata.get("config", {}).get(
                    "adasyn_neighbors",
                    contract.data["adasyn_neighbors"],
                )
            )
            counts = np.unique(train_labels, return_counts=True)[1]
            if counts.min() <= neighbors:
                raise ValueError(
                    "seven-class ADASYN requires more rows in every class "
                    f"than n_neighbors={neighbors}"
                )
            train_values, train_labels = ADASYN(
                random_state=data_seed,
                n_neighbors=neighbors,
            ).fit_resample(train_values, train_labels)
        elif label_branch != "binary":
            raise ValueError(
                f"unsupported multiclass-label branch {label_branch!r}"
            )
        svm_training = str(
            options.get("svm_training", "resource_cap_diagnostic")
        )
        result = fit_multiclass_svm(
            train_values,
            train_labels,
            test.values,
            max_samples=(
                None
                if svm_training == "full_data"
                else int(contract.run["supervised_svm_max_samples"])
            ),
            seed=data_seed,
        )
        score_name = "non_benign_margin"
    else:
        raise ValueError(f"{model_name!r} is not a classical Table II model")

    scores = np.asarray(result.scores, dtype=np.float64)
    threshold = float(result.threshold)
    threshold_selection: Mapping[str, Any] = {}
    threshold_fit_seconds = 0.0
    threshold_score_seconds = 0.0
    if model_name in {"arima", "one_class_svm"}:
        assert threshold_population is not None
        needs_local_validation_scores = (
            threshold_rule != "printed_constant"
            and (
                threshold_scope == "dataset_specific"
                or _prepared_dataset_name(prepared) == "iset"
            )
        )
        validation_scores: Mapping[str, np.ndarray] | None = None
        if needs_local_validation_scores:
            validation_result: BenchmarkResult
            if model_name == "arima":
                validation_result = fit_arima_completion(
                    _all_b1_values(prepared),
                    threshold_population.values,
                    completion=str(
                        options.get("arima_completion", "p1_pooled_mse")
                    ),
                    threshold=float(contract.thresholds["arima"]),
                )
            else:
                validation_result = fit_one_class_svm(
                    _all_b1_values(prepared),
                    threshold_population.values,
                    max_samples=(
                        None
                        if str(
                            options.get(
                                "svm_training",
                                "resource_cap_diagnostic",
                            )
                        )
                        == "full_data"
                        else int(contract.run["one_class_svm_max_samples"])
                    ),
                    seed=data_seed,
                    threshold=float(contract.thresholds["one_class_svm"]),
                )
            validation_scores = {
                score_name: np.asarray(
                    validation_result.scores,
                    dtype=np.float64,
                )
            }
            threshold_fit_seconds = float(validation_result.fit_seconds)
            threshold_score_seconds = float(validation_result.score_seconds)
        selected, selections = _select_anomaly_thresholds(
            model_name=model_name,
            prepared=prepared,
            contract=contract,
            population=threshold_population,
            validation_scores=validation_scores,
            orientations={score_name: str(result.positive_if)},
            score_names=[score_name],
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
        threshold = float(selected[score_name])
        threshold_selection = selections
    predictions = threshold_predictions(
        scores,
        threshold,
        positive_if=result.positive_if,
    )
    metrics = evaluate_binary_scores(
        test.labels,
        scores,
        threshold=threshold,
        positive_if=result.positive_if,
    ).as_dict()
    return ExecutionResult(
        scores={score_name: scores},
        predictions={score_name: np.asarray(predictions, dtype=np.int8)},
        labels=np.asarray(test.labels, dtype=np.int8),
        sample_ids=np.asarray(test.sample_ids).astype(str),
        is_synthetic=np.asarray(test.is_synthetic, dtype=bool),
        history={"kind": "classical", "epochs": []},
        metrics={score_name: metrics},
        fit_seconds=float(result.fit_seconds + threshold_fit_seconds),
        score_seconds=float(result.score_seconds + threshold_score_seconds),
        metadata={
            "benchmark_name": result.name,
            "threshold": threshold,
            "positive_if": result.positive_if,
            "threshold_rule": (
                threshold_rule
                if model_name in {"arima", "one_class_svm"}
                else "classifier_default"
            ),
            "threshold_scope": (
                threshold_scope
                if model_name in {"arima", "one_class_svm"}
                else "classifier_default"
            ),
            "validation_labels": (
                validation_labels
                if model_name in {"arima", "one_class_svm"}
                else "supervised_labels"
            ),
            "threshold_selection": threshold_selection,
            "threshold_population": (
                dict(threshold_population.metadata)
                if threshold_population is not None
                else None
            ),
            "train_samples_available": result.train_samples_available,
            "train_samples_used": result.train_samples_used,
            "benchmark": dict(result.metadata),
            "classical_options": options,
            "cap_seed_policy": "frozen_data_seed",
        },
    )


class _EpochTimerProtocol:
    epoch_seconds: list[float]


def _keras_epoch_timer() -> _EpochTimerProtocol:
    import keras

    class EpochTimer(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.epoch_seconds: list[float] = []
            self._started = 0.0

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            del epoch, logs
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            del epoch, logs
            self.epoch_seconds.append(time.perf_counter() - self._started)

    return EpochTimer()


def _keras_callbacks(contract: Contract) -> tuple[list[Any], _EpochTimerProtocol]:
    import keras

    timer = _keras_epoch_timer()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=float(contract.run["early_stopping_min_delta"]),
        patience=int(contract.run["early_stopping_patience"]),
        restore_best_weights=True,
        start_from_epoch=int(contract.run["warmup_epochs"]),
        verbose=0,
    )
    return [timer, early_stopping], timer


def _keras_fixed_callbacks(
    contract: Contract,
) -> tuple[list[Any], _EpochTimerProtocol]:
    del contract
    timer = _keras_epoch_timer()
    return [timer], timer


def _supervised_fit_split(
    prepared: Any,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    partition = prepared.supervised_train
    indices = np.arange(partition.labels.size)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=partition.labels,
    )
    return (
        partition.values[train_indices],
        partition.labels[train_indices],
        partition.values[validation_indices],
        partition.labels[validation_indices],
    )


def _anomaly_scores(
    bundle: Any,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    values: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Score one partition under the selected anomaly-score definition."""

    selected_vae_score = (
        str(model_config["vae_score"])
        if model_name.endswith("_vae") and "vae_score" in model_config
        else None
    )
    if selected_vae_score is not None:
        from paper_literal_models import parse_vae_score_branch

        score_spec = parse_vae_score_branch(selected_vae_score)
    else:
        score_spec = None
    if score_spec is not None and score_spec.kind == "reconstruction_probability":
        probability_scores = bundle.vae_reconstruction_probability(
            values,
            monte_carlo_samples=int(score_spec.monte_carlo_samples),
            variance=str(score_spec.variance),
            batch_size=batch_size,
        )
        return (
            {
                selected_vae_score: np.asarray(
                    probability_scores["reconstruction_probability"],
                    dtype=np.float64,
                ).reshape(-1)
            },
            {selected_vae_score: score_spec.positive_if},
        )

    raw_scores = bundle.anomaly_scores(values, batch_size=batch_size)
    if score_spec is None:
        selected_raw_scores = raw_scores
    else:
        raw_name = (
            "reconstruction_mse"
            if score_spec.kind == "reconstruction_mse"
            else "mse_plus_kl_surrogate"
        )
        selected_raw_scores = {selected_vae_score: raw_scores[raw_name]}
    scores = {
        name: np.asarray(score_values, dtype=np.float64).reshape(-1)
        for name, score_values in selected_raw_scores.items()
    }
    return scores, {name: "higher" for name in scores}


def _prepared_dataset_name(prepared: Any) -> str:
    declared = str(prepared.metadata.get("dataset", "")).lower()
    if "iset" in declared or "cer" in declared:
        return "iset"
    return "sgcc"


def _transferred_threshold(
    transferred: Mapping[str, float] | None,
    *,
    score_name: str,
    model_name: str,
) -> float:
    if transferred is None:
        raise ValueError(
            "an ISET-transferred SGCC threshold requires a frozen transferred "
            "threshold artifact"
        )
    for key in (score_name, model_name, "default"):
        if key in transferred and np.isfinite(transferred[key]):
            return float(transferred[key])
    raise ValueError(
        f"transferred threshold artifact has no value for {score_name!r} "
        f"or {model_name!r}"
    )


def _select_anomaly_thresholds(
    *,
    model_name: str,
    prepared: Any,
    contract: Contract,
    population: ThresholdPopulation,
    validation_scores: Mapping[str, np.ndarray] | None,
    orientations: Mapping[str, str],
    score_names: Sequence[str],
    threshold_rule: str,
    threshold_scope: str,
    validation_labels: str,
    transferred_thresholds: Mapping[str, float] | None,
) -> tuple[dict[str, float], dict[str, Mapping[str, Any]]]:
    """Resolve threshold formula, derivation scope, and provenance."""

    if threshold_scope not in THRESHOLD_SCOPES:
        raise ValueError(
            f"unknown threshold scope {threshold_scope!r}; "
            f"expected one of {sorted(THRESHOLD_SCOPES)}"
        )
    if threshold_rule == "printed_constant":
        if threshold_scope != "iset_transferred":
            raise ValueError(
                "the paper supplies only ISET-derived constants; "
                "printed_constant is incompatible with dataset_specific"
            )
    elif validation_labels == "printed_threshold_no_derivation":
        raise ValueError(
            f"{threshold_rule} requires labeled validation scores"
        )

    dataset = _prepared_dataset_name(prepared)
    thresholds: dict[str, float] = {}
    selections: dict[str, Mapping[str, Any]] = {}
    for score_name in score_names:
        positive_if = str(orientations[score_name])
        if threshold_rule == "printed_constant":
            selection = select_threshold(
                population.labels,
                np.zeros(population.labels.size, dtype=np.float64),
                rule="printed_constant",
                positive_if=positive_if,
                supplied_threshold=float(contract.thresholds[model_name]),
            )
        elif threshold_scope == "iset_transferred" and dataset == "sgcc":
            threshold = _transferred_threshold(
                transferred_thresholds,
                score_name=score_name,
                model_name=model_name,
            )
            counts = np.bincount(population.labels, minlength=2)
            selection = ThresholdSelection(
                rule=threshold_rule,
                threshold=threshold,
                positive_if=positive_if,
                sample_count=int(population.labels.size),
                benign_count=int(counts[0]),
                malicious_count=int(counts[1]),
                finite_roc_thresholds=0,
                details={
                    "source": "frozen_iset_transfer_artifact",
                    "target_dataset": "sgcc",
                },
            )
        else:
            if validation_scores is None or score_name not in validation_scores:
                raise ValueError(
                    f"{threshold_rule} requires validation scores for {score_name}"
                )
            selection = select_threshold(
                population.labels,
                validation_scores[score_name],
                rule=threshold_rule,
                positive_if=positive_if,
            )
        thresholds[score_name] = float(selection.threshold)
        selections[score_name] = {
            **selection.as_dict(),
            "scope": threshold_scope,
            "dataset": dataset,
            "validation_labels": validation_labels,
        }
    return thresholds, selections


@dataclass(frozen=True)
class _TrainingOutcome:
    bundle: Any
    histories: Sequence[Mapping[str, Any]]
    epoch_seconds: Sequence[float]
    fit_seconds: float
    metadata: Mapping[str, Any]


def _best_epoch(history: Mapping[str, Any]) -> int:
    validation = np.asarray(history.get("val_loss", []), dtype=np.float64)
    if validation.size and np.isfinite(validation).any():
        return int(np.nanargmin(validation)) + 1
    completed = max((len(values) for values in history.values()), default=0)
    return max(1, int(completed))


def _fit_one_bundle(
    bundle: Any,
    *,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray | None,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    epochs: int,
    batch_size: int,
    callback_factory: Callable[[Contract], tuple[list[Any], Any]],
    contract: Contract,
) -> tuple[Mapping[str, Any], Sequence[float]]:
    callbacks, timer = callback_factory(contract)
    kwargs: dict[str, Any] = {
        "x": x_train,
        "y": y_train,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "callbacks": callbacks,
        "shuffle": True,
        "verbose": 0,
    }
    if x_validation is not None:
        kwargs["validation_data"] = (x_validation, y_validation)
    history_object = bundle.model.fit(**kwargs)
    return (
        dict(getattr(history_object, "history", {})),
        list(timer.epoch_seconds),
    )


def _training_targets(
    model_name: str,
    values: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    supervised_head: str | None = None,
) -> np.ndarray | None:
    if model_name in ANOMALY_NEURAL_MODELS:
        return None if model_name.endswith("_vae") else values
    if labels is None:
        raise ValueError("supervised models require labels")
    if supervised_head == "sigmoid1_binary":
        return labels.astype(np.float32).reshape(-1, 1)
    if supervised_head in {None, "softmax2_categorical"}:
        return labels.astype(np.int64)
    raise ValueError(f"unsupported supervised head {supervised_head!r}")


def _cross_validation_indices(
    *,
    model_name: str,
    labels: np.ndarray | None,
    rows: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if model_name in ANOMALY_NEURAL_MODELS:
        from sklearn.model_selection import KFold

        folds = min(5, rows)
        if folds < 2:
            raise ValueError("cross-validation requires at least two B1 rows")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        return [
            (np.asarray(train), np.asarray(validation))
            for train, validation in splitter.split(np.arange(rows))
        ]

    from sklearn.model_selection import StratifiedKFold

    if labels is None:
        raise ValueError("supervised cross-validation requires labels")
    counts = np.bincount(labels.astype(np.int8), minlength=2)
    folds = min(5, int(counts.min()))
    if folds < 2:
        raise ValueError(
            "supervised cross-validation requires two rows from each class"
        )
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    return [
        (np.asarray(train), np.asarray(validation))
        for train, validation in splitter.split(np.arange(rows), labels)
    ]


def _fit_neural_policy(
    *,
    model_name: str,
    prepared: Any,
    contract: Contract,
    seed: int,
    input_length: int,
    model_config: Mapping[str, Any],
    model_builder: Callable[..., Any],
    callback_factory: Callable[[Contract], tuple[list[Any], Any]],
    fixed_callback_factory: Callable[[Contract], tuple[list[Any], Any]],
    validation_policy: str,
) -> _TrainingOutcome:
    """Execute the selected validation/refit semantics and preserve all fits."""

    if validation_policy not in VALIDATION_POLICIES:
        raise ValueError(
            f"unknown validation policy {validation_policy!r}; "
            f"expected one of {sorted(VALIDATION_POLICIES)}"
        )
    max_epochs = int(contract.run["max_epochs"])
    batch_size = int(contract.run["batch_size"])
    supervised_head = (
        str(model_config.get("supervised_head"))
        if model_name in SUPERVISED_NEURAL_MODELS
        else None
    )

    def targets(
        values: np.ndarray,
        labels: np.ndarray | None,
    ) -> np.ndarray | None:
        return _training_targets(
            model_name,
            values,
            labels,
            supervised_head=supervised_head,
        )

    if model_name in ANOMALY_NEURAL_MODELS:
        combined = _concatenate_partitions(
            [prepared.anomaly_train, prepared.anomaly_validation]
        )
        all_values = combined.values
        all_labels: np.ndarray | None = None
        holdout_train = prepared.anomaly_train.values
        holdout_train_labels: np.ndarray | None = None
        holdout_validation = prepared.anomaly_validation.values
        holdout_validation_labels: np.ndarray | None = None
        refit_partition = "all_b1_benign"
    else:
        combined = prepared.supervised_train
        all_values = combined.values
        all_labels = combined.labels
        (
            holdout_train,
            holdout_train_labels,
            holdout_validation,
            holdout_validation_labels,
        ) = _supervised_fit_split(
            prepared,
            validation_fraction=float(
                contract.run["validation_fraction_within_train"]
            ),
            seed=seed,
        )
        refit_partition = "all_supervised_train"

    def build() -> Any:
        return model_builder(
            model_name,
            input_length,
            dict(model_config),
            seed=seed,
        )

    histories: list[Mapping[str, Any]] = []
    epoch_seconds: list[float] = []
    started = time.perf_counter()

    if validation_policy == "none_fixed_epochs":
        bundle = build()
        history, timings = _fit_one_bundle(
            bundle,
            model_name=model_name,
            x_train=all_values,
            y_train=targets(all_values, all_labels),
            x_validation=None,
            y_validation=None,
            epochs=max_epochs,
            batch_size=batch_size,
            callback_factory=fixed_callback_factory,
            contract=contract,
        )
        histories.append(history)
        epoch_seconds.extend(timings)
        metadata = {
            "selection": "none",
            "refit": False,
            "fit_partition": refit_partition,
            "fit_samples": int(all_values.shape[0]),
            "selected_epochs": max_epochs,
        }
    elif validation_policy in {"holdout_no_refit", "holdout_refit_b1"}:
        selection_bundle = build()
        selection_history, timings = _fit_one_bundle(
            selection_bundle,
            model_name=model_name,
            x_train=holdout_train,
            y_train=targets(holdout_train, holdout_train_labels),
            x_validation=holdout_validation,
            y_validation=targets(
                holdout_validation,
                holdout_validation_labels,
            ),
            epochs=max_epochs,
            batch_size=batch_size,
            callback_factory=callback_factory,
            contract=contract,
        )
        histories.append(selection_history)
        epoch_seconds.extend(timings)
        selected_epochs = _best_epoch(selection_history)
        if validation_policy == "holdout_no_refit":
            bundle = selection_bundle
            metadata = {
                "selection": "single_holdout",
                "refit": False,
                "fit_partition": "registered_train_partition",
                "fit_samples": int(holdout_train.shape[0]),
                "validation_samples": int(holdout_validation.shape[0]),
                "selected_epochs": selected_epochs,
            }
        else:
            bundle = build()
            refit_history, timings = _fit_one_bundle(
                bundle,
                model_name=model_name,
                x_train=all_values,
                y_train=targets(all_values, all_labels),
                x_validation=None,
                y_validation=None,
                epochs=selected_epochs,
                batch_size=batch_size,
                callback_factory=fixed_callback_factory,
                contract=contract,
            )
            histories.append(refit_history)
            epoch_seconds.extend(timings)
            metadata = {
                "selection": "single_holdout",
                "refit": True,
                "refit_partition": refit_partition,
                "selection_train_samples": int(holdout_train.shape[0]),
                "validation_samples": int(holdout_validation.shape[0]),
                "refit_samples": int(all_values.shape[0]),
                "selected_epochs": selected_epochs,
            }
    else:
        splits = _cross_validation_indices(
            model_name=model_name,
            labels=all_labels,
            rows=int(all_values.shape[0]),
            seed=seed,
        )
        fold_epochs: list[int] = []
        for train_indices, validation_indices in splits:
            fold_bundle = build()
            fold_train = all_values[train_indices]
            fold_validation = all_values[validation_indices]
            fold_train_labels = (
                all_labels[train_indices] if all_labels is not None else None
            )
            fold_validation_labels = (
                all_labels[validation_indices]
                if all_labels is not None
                else None
            )
            fold_history, timings = _fit_one_bundle(
                fold_bundle,
                model_name=model_name,
                x_train=fold_train,
                y_train=targets(fold_train, fold_train_labels),
                x_validation=fold_validation,
                y_validation=targets(
                    fold_validation,
                    fold_validation_labels,
                ),
                epochs=max_epochs,
                batch_size=batch_size,
                callback_factory=callback_factory,
                contract=contract,
            )
            histories.append(fold_history)
            epoch_seconds.extend(timings)
            fold_epochs.append(_best_epoch(fold_history))
        selected_epochs = max(1, int(np.median(fold_epochs)))
        bundle = build()
        refit_history, timings = _fit_one_bundle(
            bundle,
            model_name=model_name,
            x_train=all_values,
            y_train=targets(all_values, all_labels),
            x_validation=None,
            y_validation=None,
            epochs=selected_epochs,
            batch_size=batch_size,
            callback_factory=fixed_callback_factory,
            contract=contract,
        )
        histories.append(refit_history)
        epoch_seconds.extend(timings)
        metadata = {
            "selection": "five_fold_or_maximum_feasible_cross_validation",
            "folds": len(splits),
            "fold_selected_epochs": fold_epochs,
            "selected_epoch_aggregation": "integer_median",
            "refit": True,
            "refit_partition": refit_partition,
            "refit_samples": int(all_values.shape[0]),
            "selected_epochs": selected_epochs,
        }

    return _TrainingOutcome(
        bundle=bundle,
        histories=histories,
        epoch_seconds=epoch_seconds,
        fit_seconds=time.perf_counter() - started,
        metadata={
            "policy": validation_policy,
            **metadata,
        },
    )


def execute_neural(
    model_name: str,
    prepared: Any,
    contract: Contract,
    seed: int,
    *,
    model_builder: Callable[..., Any] | None = None,
    callback_factory: Callable[[Contract], tuple[list[Any], Any]] | None = None,
    fixed_callback_factory: Callable[[Contract], tuple[list[Any], Any]] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    validation_policy: str = "holdout_no_refit",
    threshold_rule: str = "printed_constant",
    threshold_scope: str = "iset_transferred",
    validation_labels: str = "printed_threshold_no_derivation",
    transferred_thresholds: Mapping[str, float] | None = None,
) -> ExecutionResult:
    """Train and score one frozen Keras Table II model."""

    if model_builder is None:
        from paper_literal_models import build_model as model_builder

    if callback_factory is None:
        callback_factory = _keras_callbacks
    if fixed_callback_factory is None:
        fixed_callback_factory = _keras_fixed_callbacks

    input_length = int(prepared.anomaly_train.values.shape[1])
    model_config = dict(contract.raw["table_1"][model_name])
    if model_overrides:
        model_config.update(model_overrides)
    if model_name in SUPERVISED_NEURAL_MODELS:
        model_config.setdefault(
            "supervised_head",
            (
                "sigmoid1_binary"
                if model_name == "supervised_lstm"
                else "softmax2_categorical"
            ),
        )
    threshold_population: ThresholdPopulation | None = None
    if model_name in ANOMALY_NEURAL_MODELS:
        preparation_config = dict(prepared.metadata.get("config", {}))
        threshold_seed = int(
            preparation_config.get(
                "attack_seed",
                contract.run["data_seed"],
            )
        )
        threshold_population = build_threshold_population(
            prepared,
            branch=validation_labels,
            seed=threshold_seed,
            validation_fraction=float(
                contract.run["validation_fraction_within_train"]
            ),
        )
        test = threshold_population.test_partition
    elif model_name in SUPERVISED_NEURAL_MODELS:
        test = prepared.supervised_test
    else:
        raise ValueError(f"{model_name!r} is not a neural Table II model")

    training = _fit_neural_policy(
        model_name=model_name,
        prepared=prepared,
        contract=contract,
        seed=seed,
        input_length=input_length,
        model_config=model_config,
        model_builder=model_builder,
        callback_factory=callback_factory,
        fixed_callback_factory=fixed_callback_factory,
        validation_policy=validation_policy,
    )
    bundle = training.bundle
    fit_seconds = training.fit_seconds

    score_started = time.perf_counter()
    if model_name in ANOMALY_NEURAL_MODELS:
        if threshold_population is None:
            raise RuntimeError("anomaly model has no threshold population")
        batch_size = int(contract.run["batch_size"])
        scores, score_orientations = _anomaly_scores(
            bundle,
            model_name=model_name,
            model_config=model_config,
            values=test.values,
            batch_size=batch_size,
        )
        needs_local_validation_scores = (
            threshold_rule != "printed_constant"
            and (
                threshold_scope == "dataset_specific"
                or _prepared_dataset_name(prepared) == "iset"
            )
        )
        validation_scores: Mapping[str, np.ndarray] | None = None
        if needs_local_validation_scores:
            validation_scores, validation_orientations = _anomaly_scores(
                bundle,
                model_name=model_name,
                model_config=model_config,
                values=threshold_population.values,
                batch_size=batch_size,
            )
            if validation_orientations != score_orientations:
                raise RuntimeError(
                    "validation and test score orientations disagree"
                )
        score_thresholds, threshold_selections = _select_anomaly_thresholds(
            model_name=model_name,
            prepared=prepared,
            contract=contract,
            population=threshold_population,
            validation_scores=validation_scores,
            orientations=score_orientations,
            score_names=list(scores),
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
        predictions, metrics = _metrics_for_scores(
            test.labels,
            scores,
            score_thresholds,
            score_orientations,
        )
    else:
        raw = np.asarray(
            bundle.model.predict(
                test.values,
                batch_size=int(contract.run["batch_size"]),
                verbose=0,
            )
        )
        supervised_head = str(model_config["supervised_head"])
        if supervised_head == "softmax2_categorical":
            if raw.ndim != 2 or raw.shape[1] != 2:
                raise RuntimeError(
                    "softmax2 classifier must return two probabilities"
                )
            probability = raw[:, 1].astype(np.float64)
            standard_predictions = np.argmax(raw, axis=1).astype(np.int8)
        elif supervised_head == "sigmoid1_binary":
            probability = raw.reshape(-1).astype(np.float64)
            standard_predictions = (probability >= 0.5).astype(np.int8)
        else:
            raise RuntimeError(
                f"unsupported supervised head {supervised_head!r}"
            )
        threshold_predictions_from_score = threshold_predictions(probability, 0.5)
        if not np.array_equal(standard_predictions, threshold_predictions_from_score):
            raise RuntimeError(
                "standard classifier decisions disagree with the registered score rule"
            )
        scores = {"positive_class_probability": probability}
        predictions = {"positive_class_probability": standard_predictions}
        metrics = {
            "positive_class_probability": evaluate_binary_scores(
                test.labels, probability, threshold=0.5
            ).as_dict()
        }
        score_thresholds = {"positive_class_probability": 0.5}
        score_orientations = {"positive_class_probability": "higher"}
        threshold_selections = {}
    score_seconds = time.perf_counter() - score_started
    distinct_orientations = set(score_orientations.values())
    metadata_positive_if: str | Mapping[str, str] = (
        next(iter(distinct_orientations))
        if len(distinct_orientations) == 1
        else score_orientations
    )

    final_history = dict(training.histories[-1]) if training.histories else {}
    history = {
        "epochs_completed": max(
            (len(values) for values in final_history.values()), default=0
        ),
        "series": final_history,
        "all_fits": [dict(item) for item in training.histories],
        "epoch_seconds": list(training.epoch_seconds),
        "training_policy": dict(training.metadata),
    }
    return ExecutionResult(
        scores=scores,
        predictions=predictions,
        labels=np.asarray(test.labels, dtype=np.int8),
        sample_ids=np.asarray(test.sample_ids).astype(str),
        is_synthetic=np.asarray(test.is_synthetic, dtype=bool),
        history=history,
        metrics=metrics,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        metadata={
            "model_config": model_config,
            "parameter_count": int(bundle.model.count_params()),
            "score_thresholds": score_thresholds,
            "positive_if": metadata_positive_if,
            "threshold_rule": (
                threshold_rule
                if model_name in ANOMALY_NEURAL_MODELS
                else "classifier_default"
            ),
            "threshold_scope": (
                threshold_scope
                if model_name in ANOMALY_NEURAL_MODELS
                else "classifier_default"
            ),
            "validation_labels": (
                validation_labels
                if model_name in ANOMALY_NEURAL_MODELS
                else "supervised_labels"
            ),
            "threshold_selection": threshold_selections,
            "threshold_population": (
                dict(threshold_population.metadata)
                if threshold_population is not None
                else None
            ),
            "early_stopping": {
                "used_for_selection": validation_policy
                != "none_fixed_epochs",
                "monitor": "val_loss",
                "min_delta": float(contract.run["early_stopping_min_delta"]),
                "patience": int(contract.run["early_stopping_patience"]),
                "start_from_epoch": int(contract.run["warmup_epochs"]),
                "restore_best_weights": True,
            },
            "training_policy": dict(training.metadata),
        },
    )


def execute_selected_model(
    model_name: str,
    prepared: Any,
    contract: Contract,
    seed: int,
    *,
    model_overrides: Mapping[str, Any] | None = None,
    validation_policy: str = "holdout_no_refit",
    threshold_rule: str = "printed_constant",
    threshold_scope: str = "iset_transferred",
    validation_labels: str = "printed_threshold_no_derivation",
    transferred_thresholds: Mapping[str, float] | None = None,
    classical_options: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    if model_name in CLASSICAL_MODELS:
        return execute_classical(
            model_name,
            prepared,
            contract,
            seed,
            classical_options=classical_options,
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
    return execute_neural(
        model_name,
        prepared,
        contract,
        seed,
        model_overrides=model_overrides,
        validation_policy=validation_policy,
        threshold_rule=threshold_rule,
        threshold_scope=threshold_scope,
        validation_labels=validation_labels,
        transferred_thresholds=transferred_thresholds,
    )


def _runner_source_hashes(
    model_name: str, extra_files: Sequence[str] = ()
) -> Mapping[str, str]:
    names = [
        "paper_literal_runner.py",
        "paper_literal_data.py",
        "paper_literal_metrics.py",
        "paper_literal_benchmarks.py",
    ]
    if model_name not in CLASSICAL_MODELS:
        names.append("paper_literal_models.py")
    names.extend(extra_files)
    root = Path(__file__).resolve().parent
    return {name: _sha256_path(root / name) for name in dict.fromkeys(names)}


def run_fingerprint(
    model_name: str,
    seed: int,
    prepared: Any,
    contract: Contract,
    verification: Mapping[str, Any],
    *,
    scope: RunScope = SGCC_TABLE_II_SCOPE,
) -> tuple[str, Mapping[str, Any]]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "atk-2022-deep-autoencoder",
        "table": scope.table,
        "dataset": scope.dataset,
        "model": model_name,
        "seed": int(seed),
        "contract_sha256": contract.sha256,
        "source_sha256": verification["actual_sha256"],
        "partition_id_sha256": prepared.metadata["partition_id_sha256"],
        "transformation_sha256": prepared.metadata["transformation_sha256"],
        "preparation_config": prepared.metadata.get("config", {}),
        "source_code_sha256": _runner_source_hashes(
            model_name, scope.source_code_files
        ),
        "model_config": (
            contract.raw["table_1"].get(model_name)
            if model_name not in CLASSICAL_MODELS
            else None
        ),
        "run_config": contract.run,
        "threshold": (
            contract.thresholds.get(model_name)
            if model_name in ANOMALY_NEURAL_MODELS + ("arima", "one_class_svm")
            else (0.0 if model_name == "multiclass_svm" else 0.5)
        ),
        **dict(scope.fingerprint_extra),
    }
    return _sha256_bytes(_canonical_json_bytes(payload)), payload


def _logical_run_dir(
    output: Path,
    model_name: str,
    seed: int,
    scope: RunScope = SGCC_TABLE_II_SCOPE,
) -> Path:
    return output.joinpath(*scope.path_parts, model_name, f"seed_{seed}")


def _verified_completed_attempt(
    logical_dir: Path, fingerprint: str
) -> Path | None:
    attempts = logical_dir / "attempts"
    if not attempts.is_dir():
        return None
    for attempt in sorted(attempts.iterdir()):
        manifest_path = attempt / "manifest.json"
        if not attempt.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete":
            continue
        if manifest.get("fingerprint") != fingerprint:
            continue
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        valid = True
        for filename, expected_hash in artifacts.items():
            artifact = attempt / filename
            if not artifact.is_file() or _sha256_path(artifact) != expected_hash:
                valid = False
                break
        if valid:
            return attempt
    return None


def _new_attempt_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _artifact_arrays(result: ExecutionResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "labels": np.asarray(result.labels, dtype=np.int8),
        "sample_ids": np.asarray(result.sample_ids).astype(str),
        "is_synthetic": np.asarray(result.is_synthetic, dtype=bool),
    }
    for name, values in result.scores.items():
        arrays[f"score__{name}"] = np.asarray(values, dtype=np.float64)
    for name, values in result.predictions.items():
        arrays[f"prediction__{name}"] = np.asarray(values, dtype=np.int8)
    row_count = arrays["labels"].size
    if any(array.ndim != 1 or array.size != row_count for array in arrays.values()):
        raise ValueError("all persisted score/prediction/provenance arrays must align")
    return arrays


def _environment_metadata() -> Mapping[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "keras_backend_requested": os.environ.get("KERAS_BACKEND"),
        "git_commit": os.environ.get("ATK_GIT_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    return metadata


def _persist_attempt(
    logical_dir: Path,
    *,
    status: str,
    fingerprint: str,
    fingerprint_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    history: Mapping[str, Any],
    result_summary: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None,
) -> Path:
    attempts_dir = logical_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = _new_attempt_id()
    final_dir = attempts_dir / attempt_id
    temporary_dir = attempts_dir / f".{attempt_id}.tmp"
    temporary_dir.mkdir()
    try:
        _atomic_write_json(temporary_dir / "metadata.json", metadata)
        _atomic_write_json(temporary_dir / "history.json", history)
        _atomic_write_json(temporary_dir / "result.json", result_summary)
        artifact_names = ["metadata.json", "history.json", "result.json"]
        if arrays is not None:
            _atomic_write_npz(temporary_dir / "arrays.npz", arrays)
            artifact_names.append("arrays.npz")
        artifacts = {
            name: _sha256_path(temporary_dir / name) for name in artifact_names
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "artifacts": artifacts,
            "attempt_id": attempt_id,
        }
        # The manifest is written last inside the transaction directory.  The
        # directory rename then publishes the complete attempt atomically.
        _atomic_write_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            for child in temporary_dir.iterdir():
                child.unlink()
            temporary_dir.rmdir()
    return final_dir


def run_one(
    *,
    output: str | Path,
    model_name: str,
    seed: int,
    prepared: SgccPaperLiteralData,
    contract: Contract,
    verification: Mapping[str, Any],
    data_prep_seconds: float,
    force: bool = False,
    executor: Callable[[str, Any, Contract, int], ExecutionResult] | None = None,
    scope: RunScope = SGCC_TABLE_II_SCOPE,
    array_builder: Callable[[ExecutionResult], Mapping[str, np.ndarray]]
    | None = None,
) -> RunOutcome:
    """Execute, atomically persist, and resume one model/seed pair."""

    if executor is None:
        executor = execute_selected_model
    if array_builder is None:
        array_builder = _artifact_arrays
    output_path = Path(output).expanduser().resolve()
    logical_dir = _logical_run_dir(output_path, model_name, seed, scope)
    fingerprint, fingerprint_payload = run_fingerprint(
        model_name, seed, prepared, contract, verification, scope=scope
    )
    if not force:
        completed = _verified_completed_attempt(logical_dir, fingerprint)
        if completed is not None:
            return RunOutcome(
                model_name,
                seed,
                "skipped_complete",
                completed,
                fingerprint,
                "matching completed attempt and artifact checksums verified",
            )

    run_started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    runtime_branch = fingerprint_payload.get("branch_runtime")
    track = (
        "corrected_control"
        if isinstance(runtime_branch, Mapping)
        and runtime_branch.get("track") == "C"
        else "exploratory_paper_literal"
    )
    base_metadata = {
        "schema_version": SCHEMA_VERSION,
        "study": "atk-2022-deep-autoencoder",
        "track": track,
        "branch_runtime": runtime_branch,
        "table": scope.table,
        "dataset": scope.dataset,
        "model": model_name,
        "seed": int(seed),
        "started_utc": started_utc,
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "data_verification": verification,
        "prepared_data_metadata": prepared.metadata,
        "environment": _environment_metadata(),
        "fingerprint": fingerprint,
    }
    try:
        result = executor(model_name, prepared, contract, seed)
        run_seconds = time.perf_counter() - run_started
        timings = {
            "data_prep_seconds": float(data_prep_seconds),
            "fit_seconds": float(result.fit_seconds),
            "score_seconds": float(result.score_seconds),
            "run_seconds": float(run_seconds),
            "end_to_end_seconds": float(data_prep_seconds + run_seconds),
            "clock": "time.perf_counter",
            "shared_data_prep": True,
        }
        completed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata = {
            **base_metadata,
            "completed_utc": completed_utc,
            "status": "complete",
            "timings": timings,
            "execution": result.metadata,
        }
        summary = {
            "status": "complete",
            "model": model_name,
            "seed": int(seed),
            "metrics": result.metrics,
            "timings": timings,
            "score_names": sorted(result.scores),
            "n_test": int(result.labels.size),
        }
        if result.supplemental_results:
            summary["supplemental_results"] = result.supplemental_results
        attempt_dir = _persist_attempt(
            logical_dir,
            status="complete",
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            metadata=metadata,
            history=result.history,
            result_summary=summary,
            arrays=array_builder(result),
        )
        return RunOutcome(model_name, seed, "complete", attempt_dir, fingerprint)
    except BaseException as exc:  # Persist interruptions/failures before returning status.
        run_seconds = time.perf_counter() - run_started
        failure_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        timings = {
            "data_prep_seconds": float(data_prep_seconds),
            "elapsed_until_failure_seconds": float(run_seconds),
            "end_to_end_seconds": float(data_prep_seconds + run_seconds),
            "clock": "time.perf_counter",
            "shared_data_prep": True,
        }
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        metadata = {
            **base_metadata,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": failure_status,
            "timings": timings,
            "error": error,
        }
        attempt_dir = _persist_attempt(
            logical_dir,
            status=failure_status,
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            metadata=metadata,
            history={},
            result_summary={
                "status": failure_status,
                "model": model_name,
                "seed": int(seed),
                "timings": timings,
                "error": error,
            },
            arrays=None,
        )
        return RunOutcome(
            model_name,
            seed,
            failure_status,
            attempt_dir,
            fingerprint,
            f"{type(exc).__name__}: {exc}",
        )


def _preflight_payload(
    prepared: SgccPaperLiteralData,
    verification: Mapping[str, Any],
    contract: Contract,
    models: Sequence[str],
    seeds: Sequence[int],
    data_prep_seconds: float,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "scope": {
            "study": "atk-2022-deep-autoencoder",
            "table": 2,
            "dataset": "SGCC",
            "unsupported": "Tables III-V require seven verified CER/ISET files",
        },
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "data_verification": verification,
        "data_prep_seconds": data_prep_seconds,
        "counts": prepared.metadata["counts"],
        "preparation_config": prepared.metadata.get("config", {}),
        "features": int(prepared.dates.size),
        "models": list(models),
        "seeds": list(seeds),
        "planned_runs": len(models) * len(seeds),
    }


def _write_preflight(output: Path, payload: Mapping[str, Any]) -> Path:
    directory = output.expanduser().resolve() / "preflights"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_new_attempt_id()}.json"
    _atomic_write_json(path, payload)
    return path


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    default_config = Path(__file__).resolve().parents[1] / "config" / "exploratory_reproduction.toml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data", type=Path, required=True, help="verified SGCC data.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=int, default=2)
    parser.add_argument(
        "--dataset", default="sgcc", choices=("sgcc", "cer", "iset")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="model names, comma-separated names, or 'all'",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--expected-sgcc-sha256",
        default=EXPECTED_SGCC_SHA256,
        help="frozen source checksum (override remains recorded in every artifact)",
    )
    parser.add_argument(
        "--representation",
        choices=sorted(SGCC_REPRESENTATION_BRANCHES),
        default="full_1034",
    )
    parser.add_argument(
        "--missing",
        choices=sorted(SGCC_MISSING_BRANCHES),
        default="interpolate_edge_median",
    )
    parser.add_argument(
        "--split-unit",
        choices=sorted(SPLIT_UNIT_BRANCHES),
        default="customer_disjoint",
    )
    parser.add_argument(
        "--scaling",
        choices=[
            "joint_featurewise",
            "per_class_featurewise",
            "per_profile",
            "train_benign_only",
        ],
        default="joint_featurewise",
    )
    parser.add_argument(
        "--anomaly-adasyn",
        choices=["test_set_as_printed", "none"],
        default="test_set_as_printed",
    )
    parser.add_argument(
        "--supervised-adasyn",
        choices=["before_row_split", "customer_split_then_train_only"],
        default="before_row_split",
    )
    parser.add_argument("--adasyn-neighbors", type=int)
    parser.add_argument(
        "--validation-policy",
        choices=sorted(VALIDATION_POLICIES),
        default="holdout_no_refit",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=[
            "printed_constant",
            "roc_central_threshold_median",
            "threshold_iqr_midpoint",
            "threshold_iqr_median",
            "validation_youden_j",
        ],
        default="printed_constant",
    )
    parser.add_argument(
        "--threshold-scope",
        choices=sorted(THRESHOLD_SCOPES),
        default="iset_transferred",
    )
    parser.add_argument(
        "--validation-labels",
        choices=sorted(VALIDATION_LABEL_BRANCHES),
        default="printed_threshold_no_derivation",
    )
    parser.add_argument("--transferred-thresholds", type=Path)
    parser.add_argument("--branch-id")
    parser.add_argument(
        "--branch-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "config"
        / "branch_lattice.toml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper 1 exploratory SGCC Table-II runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="verify and prepare SGCC without fitting models"
    )
    _add_common_arguments(preflight)
    run = subparsers.add_parser(
        "run", help="execute resumable immutable model/seed attempts"
    )
    _add_common_arguments(run)
    run.add_argument(
        "--force",
        action="store_true",
        help="append a new attempt even when an immutable completed attempt exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reject_unsupported_scope(args.table, args.dataset)
        runtime_branch: Mapping[str, Any] | None = None
        if args.branch_id:
            from branch_runtime import assert_branch_scope, load_runtime_branch

            runtime_branch = load_runtime_branch(
                args.branch_id,
                manifest=args.branch_manifest,
            )
            assert_branch_scope(
                runtime_branch,
                dataset="sgcc",
                table=args.table,
            )
            args.models = [str(runtime_branch["model"])]
            for key, value in runtime_branch["preparation"].items():
                setattr(args, key, value)
            for key, value in runtime_branch["execution"].items():
                if hasattr(args, key):
                    setattr(args, key, value)
        contract = load_contract(args.config)
        models = resolve_models(args.models)
        seeds = resolve_seeds(contract, args.seeds)
        prepared, verification, data_prep_seconds = verify_and_prepare_sgcc(
            args.data,
            contract,
            expected_sha256=args.expected_sgcc_sha256,
            scaling=args.scaling,
            anomaly_adasyn=args.anomaly_adasyn,
            supervised_adasyn=args.supervised_adasyn,
            adasyn_neighbors=args.adasyn_neighbors,
            representation=args.representation,
            missing=args.missing,
            split_unit=args.split_unit,
        )
    except UnsupportedExperimentError as exc:
        print(json.dumps({"status": "unsupported", "message": str(exc)}))
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "preflight_failed", "type": type(exc).__name__, "message": str(exc)}
            )
        )
        return 2

    preflight = _preflight_payload(
        prepared, verification, contract, models, seeds, data_prep_seconds
    )
    if runtime_branch is not None:
        preflight = {
            **dict(preflight),
            "branch_runtime": runtime_branch,
        }
    preflight_path = _write_preflight(args.output, preflight)
    print(json.dumps({**preflight, "artifact": str(preflight_path)}, sort_keys=True))
    if args.command == "preflight":
        return 0

    outcomes: list[RunOutcome] = []
    transferred_thresholds: Mapping[str, float] | None = None
    if args.transferred_thresholds is not None:
        raw_transferred = json.loads(
            args.transferred_thresholds.read_text(encoding="utf-8")
        )
        if not isinstance(raw_transferred, dict):
            raise ValueError("transferred-threshold artifact must be a JSON object")
        transferred_thresholds = {
            str(key): float(value) for key, value in raw_transferred.items()
        }
    if runtime_branch is not None:
        scope = RunScope(
            2,
            "SGCC",
            ("branches", str(runtime_branch["branch_id"]), "table_2", "sgcc"),
            fingerprint_extra={
                "branch_runtime": runtime_branch,
                "transferred_thresholds": transferred_thresholds,
            },
        )
        model_overrides = dict(runtime_branch["model_overrides"])
        classical_options = {
            key: value
            for key, value in runtime_branch["execution"].items()
            if key in {"arima_completion", "svm_training", "multiclass_labels"}
        }
    else:
        manual_execution = {
            "validation_policy": args.validation_policy,
            "threshold_rule": args.threshold_rule,
            "threshold_scope": args.threshold_scope,
            "validation_labels": args.validation_labels,
            "transferred_thresholds": transferred_thresholds,
        }
        historical_execution = {
            "validation_policy": "holdout_no_refit",
            "threshold_rule": "printed_constant",
            "threshold_scope": "iset_transferred",
            "validation_labels": "printed_threshold_no_derivation",
            "transferred_thresholds": None,
        }
        scope = (
            SGCC_TABLE_II_SCOPE
            if manual_execution == historical_execution
            else RunScope(
                2,
                "SGCC",
                ("table_2", "sgcc"),
                fingerprint_extra={"manual_execution": manual_execution},
            )
        )
        model_overrides = {}
        classical_options = {}

    def executor(
        model_name: str,
        prepared_data: Any,
        frozen_contract: Contract,
        model_seed: int,
    ) -> ExecutionResult:
        return execute_selected_model(
            model_name,
            prepared_data,
            frozen_contract,
            model_seed,
            model_overrides=model_overrides,
            validation_policy=args.validation_policy,
            threshold_rule=args.threshold_rule,
            threshold_scope=args.threshold_scope,
            validation_labels=args.validation_labels,
            transferred_thresholds=transferred_thresholds,
            classical_options=classical_options,
        )

    for model_name in models:
        for seed in seeds:
            outcome = run_one(
                output=args.output,
                model_name=model_name,
                seed=seed,
                prepared=prepared,
                contract=contract,
                verification=verification,
                data_prep_seconds=data_prep_seconds,
                force=args.force,
                executor=executor,
                scope=scope,
            )
            outcomes.append(outcome)
            print(json.dumps(_jsonable(outcome), sort_keys=True))
            if model_name not in CLASSICAL_MODELS:
                try:
                    import keras

                    keras.backend.clear_session()
                except Exception:
                    pass
            if outcome.status == "interrupted":
                break
        if outcomes and outcomes[-1].status == "interrupted":
            break
    failures = [
        outcome for outcome in outcomes if outcome.status in {"failed", "interrupted"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
