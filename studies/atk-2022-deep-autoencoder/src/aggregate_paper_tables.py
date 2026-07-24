"""Deterministically aggregate immutable Paper 1 runner attempts.

The SGCC runner stores one append-only attempt directory per model and seed.
This module treats those directories as untrusted inputs: it verifies the
manifest fingerprint, every declared artifact digest, the frozen contract,
the logical model/seed path, and the recorded source-code digests before an
attempt can contribute to an aggregate.  Older successes and every failure
remain visible in the individual-run ledger; the earliest matching valid
success is selected for a model/seed.  Forced reruns therefore cannot replace
an already observed primary outcome.

Ordinary single-process and the cluster four-V100 DDP attempts are separate
execution branches.  The current neural rows accept only the fingerprinted DDP
branch, including its runner source and linked Slurm/Git provenance, so an
earlier local attempt cannot silently displace a the cluster result.

Tables III--V cannot be computed from the SGCC runner.  Their generated CSVs
therefore preserve every published cell and put ``BLOCKED_EXACT_DATA`` in
every corresponding reproduction cell, together with the complete seven-file
CER/ISET gate.  No unavailable value is synthesized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from paper_literal_metrics import evaluate_binary_scores, threshold_predictions


SCHEMA_VERSION = 1
STUDY_ID = "atk-2022-deep-autoencoder"
BLOCKED = "BLOCKED_EXACT_DATA"
EXPECTED_SGCC_SHA256 = (
    "99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7"
)

PAPER_METRICS = ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC")
RESULT_METRICS = {
    "DR": "dr",
    "FA": "fa",
    "SP": "sp",
    "PR": "precision",
    "ACC": "balanced_accuracy",
    "F1": "f1",
    "AUC": "auc",
}
COUNT_FIELDS = ("tp", "fp", "tn", "fn")

PAPER_TO_INTERNAL = {
    "FC-SAE": "fc_sae",
    "LSTM-SAE": "lstm_sae",
    "FC-VAE": "fc_vae",
    "LSTM-VAE": "lstm_vae",
    "LSTM-AEA": "lstm_aea",
    "Naive Bayes": "naive_bayes",
    "ARIMA": "arima",
    "Single-class SVM": "one_class_svm",
    "Feed forward": "supervised_feed_forward",
    "LSTM": "supervised_lstm",
    "Multi-class SVM": "multiclass_svm",
}
INTERNAL_TO_PAPER = {value: key for key, value in PAPER_TO_INTERNAL.items()}

PRIMARY_SCORE_BRANCH = {
    "fc_sae": "reconstruction_mse",
    "lstm_sae": "reconstruction_mse",
    "fc_vae": "reconstruction_mse",
    "lstm_vae": "reconstruction_mse",
    "lstm_aea": "reconstruction_mse",
    "naive_bayes": "positive_class_probability",
    "arima": "residual_mse",
    "one_class_svm": "negative_decision_function",
    "supervised_feed_forward": "positive_class_probability",
    "supervised_lstm": "positive_class_probability",
    "multiclass_svm": "non_benign_margin",
}
VAE_DIAGNOSTIC_BRANCH = "mse_plus_kl_surrogate"

ORDINARY_EXECUTION_BRANCH = "ordinary_single_process"
PANTHER_DDP_EXECUTION_BRANCH = "panther_four_v100_ddp"
PANTHER_DDP_IMPLEMENTATION = "keras-torch-ddp-v1"
PANTHER_DDP_WORLD_SIZE = 4
PANTHER_DDP_SOURCE = "paper_literal_ddp.py"
NEURAL_MODELS = {
    "fc_sae",
    "lstm_sae",
    "fc_vae",
    "lstm_vae",
    "lstm_aea",
    "supervised_feed_forward",
    "supervised_lstm",
}

CER_GATE = (
    ("File1.txt.zip", "00203f66f3f5e5201b20ed160b787684"),
    ("File2.txt.zip", "5e3af1474d3c8976e2e1e0f8c1969507"),
    ("File3.txt.zip", "b537785f8b37cb3e89103600d39da8ff"),
    ("File4.txt.zip", "53ec9e70c1610b74ae72417cc010a0c3"),
    ("File5.txt.zip", "6f8c7c9dfba3bbfbff0e5f1703e122fc"),
    ("File6.txt.zip", "c0a435d0359974f23ce434b5e838e251"),
    (
        "SME and Residential allocations.tab",
        "124c10711ab1e7c52cb7317c8f69e42e",
    ),
)

_ATTEMPT_ID = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{12}$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_scalar(row.get(field)) for field in fieldnames})
    return stream.getvalue().encode("utf-8")


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        # Stable and lossless enough for result summaries without noisy tails.
        return format(value, ".12g")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        sample_sd = 0.0
    else:
        sample_sd = math.sqrt(
            sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
        )
    return {
        "n": len(finite),
        "mean": mean,
        "sample_sd": sample_sd,
        "min": min(finite),
        "max": max(finite),
    }


@dataclass
class AttemptRecord:
    model: str
    seed: int | None
    attempt_id: str
    relative_path: str
    manifest_sha256: str | None = None
    attempt_status: str = "unknown"
    verification_status: str = "INVALID"
    verification_detail: str = ""
    fingerprint: str = ""
    matching: bool = False
    selected: bool = False
    result: Mapping[str, Any] | None = None
    history: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    recorded_source_code_sha256: Mapping[str, str] = field(default_factory=dict)
    execution_branch: str = "unknown"
    execution_source_code_sha256: Mapping[str, str] = field(default_factory=dict)
    slurm_job_id: str | None = None
    git_commit: str | None = None

    @property
    def paper_model(self) -> str:
        return INTERNAL_TO_PAPER.get(self.model, self.model)


def _logical_attempt_dirs(runner_root: Path) -> list[tuple[str, int | None, Path]]:
    base = runner_root / "table_2" / "sgcc"
    discovered: list[tuple[str, int | None, Path]] = []
    if not base.is_dir():
        return discovered
    for model_dir in sorted((item for item in base.iterdir() if item.is_dir()), key=lambda p: p.name):
        for seed_dir in sorted((item for item in model_dir.iterdir() if item.is_dir()), key=lambda p: p.name):
            seed = None
            if seed_dir.name.startswith("seed_"):
                try:
                    seed = int(seed_dir.name.removeprefix("seed_"))
                except ValueError:
                    pass
            attempts_dir = seed_dir / "attempts"
            if not attempts_dir.is_dir():
                continue
            for attempt_dir in sorted(
                (item for item in attempts_dir.iterdir() if item.is_dir()),
                key=lambda p: p.name,
            ):
                discovered.append((model_dir.name, seed, attempt_dir))
    return discovered


def _verify_artifacts(
    attempt_dir: Path, artifacts: Any
) -> tuple[bool, str, dict[str, str]]:
    if not isinstance(artifacts, dict) or not artifacts:
        return False, "manifest artifacts must be a non-empty object", {}
    actual: dict[str, str] = {}
    for filename in sorted(artifacts):
        expected = artifacts[filename]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).is_absolute()
        ):
            return False, f"unsafe artifact name {filename!r}", actual
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            return False, f"invalid SHA-256 for artifact {filename!r}", actual
        artifact = attempt_dir / filename
        if not artifact.is_file():
            return False, f"missing artifact {filename!r}", actual
        digest = _sha256_path(artifact)
        actual[filename] = digest
        if digest != expected:
            return False, f"artifact hash mismatch for {filename!r}", actual
    return True, "all declared artifact hashes verified", actual


def _metric_values_match(recorded: Any, recalculated: Any) -> bool:
    if isinstance(recalculated, str):
        return recorded == recalculated
    if isinstance(recalculated, int):
        return _integer(recorded) == recalculated
    recorded_float = _finite_float(recorded)
    recalculated_float = _finite_float(recalculated)
    if recorded_float is None or recalculated_float is None:
        return recorded_float is None and recalculated_float is None
    return math.isclose(
        recorded_float, recalculated_float, rel_tol=1e-12, abs_tol=1e-12
    )


def _verify_complete_arrays(
    attempt_dir: Path, result: Mapping[str, Any]
) -> tuple[bool, str]:
    """Recalculate every metric from the persisted raw score arrays."""

    score_names = result.get("score_names")
    metrics = result.get("metrics")
    if (
        not isinstance(score_names, list)
        or not score_names
        or not all(isinstance(name, str) and name for name in score_names)
        or len(set(score_names)) != len(score_names)
        or not isinstance(metrics, dict)
        or set(score_names) != set(metrics)
    ):
        return False, "result score_names and metrics branches do not agree"
    expected_arrays = {"labels", "sample_ids", "is_synthetic"}
    expected_arrays.update(f"score__{name}" for name in score_names)
    expected_arrays.update(f"prediction__{name}" for name in score_names)
    try:
        with np.load(attempt_dir / "arrays.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != expected_arrays:
                return False, "arrays.npz members do not match result score_names"
            labels = np.asarray(arrays["labels"])
            sample_ids = np.asarray(arrays["sample_ids"])
            synthetic = np.asarray(arrays["is_synthetic"])
            if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, [0, 1]).all():
                return False, "arrays.npz labels must be a non-empty binary vector"
            row_count = labels.size
            if (
                sample_ids.ndim != 1
                or synthetic.ndim != 1
                or sample_ids.size != row_count
                or synthetic.size != row_count
            ):
                return False, "arrays.npz provenance arrays do not align with labels"
            if len(set(sample_ids.astype(str).tolist())) != row_count:
                return False, "arrays.npz sample_ids are not unique"
            if _integer(result.get("n_test")) != row_count:
                return False, "result n_test disagrees with arrays.npz"

            for branch in score_names:
                recorded = metrics.get(branch)
                if not isinstance(recorded, dict):
                    return False, f"result metrics branch {branch!r} is not an object"
                scores = np.asarray(arrays[f"score__{branch}"], dtype=np.float64)
                predictions = np.asarray(arrays[f"prediction__{branch}"])
                if (
                    scores.ndim != 1
                    or predictions.ndim != 1
                    or scores.size != row_count
                    or predictions.size != row_count
                    or not np.isfinite(scores).all()
                    or not np.isin(predictions, [0, 1]).all()
                ):
                    return False, f"arrays for score branch {branch!r} are invalid"
                threshold = _finite_float(recorded.get("threshold"))
                orientation = recorded.get("positive_if")
                if threshold is None or orientation not in {"higher", "lower"}:
                    return False, f"metrics branch {branch!r} has no valid score rule"
                expected_predictions = threshold_predictions(
                    scores, threshold, positive_if=orientation
                )
                if not np.array_equal(predictions.astype(np.int8), expected_predictions):
                    return False, f"stored predictions disagree for branch {branch!r}"
                recalculated = evaluate_binary_scores(
                    labels,
                    scores,
                    threshold=threshold,
                    positive_if=orientation,
                ).as_dict()
                for field_name, expected in recalculated.items():
                    if field_name not in recorded or not _metric_values_match(
                        recorded[field_name], expected
                    ):
                        return (
                            False,
                            f"recorded metric {branch}.{field_name} disagrees with arrays.npz",
                        )
    except (OSError, ValueError, KeyError) as exc:
        return False, f"arrays.npz is unreadable: {type(exc).__name__}: {exc}"
    return True, "raw arrays and all recorded metrics independently verified"


def _metadata_reasons(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    model: str,
    seed: int | None,
    status: str,
    contract_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    expected_fields = {"model": model, "seed": seed, "status": status}
    for field_name, expected in expected_fields.items():
        if metadata.get(field_name) != expected:
            reasons.append(f"metadata {field_name} disagrees with logical attempt")
    if metadata.get("fingerprint") != _sha256_bytes(_canonical_json_bytes(payload)):
        reasons.append("metadata fingerprint disagrees with manifest")
    contract = metadata.get("contract")
    if not isinstance(contract, dict) or contract.get("sha256") != contract_sha256:
        reasons.append("metadata contract SHA-256 disagrees with frozen contract")
    verification = metadata.get("data_verification")
    if not isinstance(verification, dict) or verification.get("verified") is not True:
        reasons.append("metadata does not record successful SGCC verification")
    else:
        actual = verification.get("actual_sha256")
        expected = verification.get("expected_sha256")
        if actual != EXPECTED_SGCC_SHA256 or expected != EXPECTED_SGCC_SHA256:
            reasons.append("metadata SGCC SHA-256 is not the frozen exact source")
        if actual != payload.get("source_sha256"):
            reasons.append("metadata and fingerprint SGCC SHA-256 disagree")
    prepared = metadata.get("prepared_data_metadata")
    if not isinstance(prepared, dict):
        reasons.append("metadata omits prepared_data_metadata")
    else:
        for field_name in ("partition_id_sha256", "transformation_sha256"):
            if prepared.get(field_name) != payload.get(field_name):
                reasons.append(f"prepared metadata {field_name} disagrees with fingerprint")
    return reasons


def _execution_branch(payload: Mapping[str, Any]) -> str:
    distributed = payload.get("distributed_execution")
    if distributed is None:
        return ORDINARY_EXECUTION_BRANCH
    if (
        isinstance(distributed, dict)
        and distributed.get("implementation") == PANTHER_DDP_IMPLEMENTATION
    ):
        return PANTHER_DDP_EXECUTION_BRANCH
    return "unsupported_distributed_execution"


def _expected_execution_branch(model: str) -> str:
    return (
        PANTHER_DDP_EXECUTION_BRANCH
        if model in NEURAL_MODELS
        else ORDINARY_EXECUTION_BRANCH
    )


def _execution_integrity_reasons(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    execution_branch: str,
    status: str,
) -> list[str]:
    """Verify internally linked provenance for the declared execution branch."""

    if execution_branch != PANTHER_DDP_EXECUTION_BRANCH:
        return []

    reasons: list[str] = []
    distributed = payload.get("distributed_execution")
    if not isinstance(distributed, dict):
        return ["the cluster DDP fingerprint has no distributed_execution object"]

    implementation_hash = distributed.get("implementation_source_sha256")
    if not isinstance(implementation_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", implementation_hash
    ):
        reasons.append("the cluster DDP implementation source SHA-256 is invalid")

    git_commit = distributed.get("git_commit")
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        reasons.append("the cluster DDP fingerprint Git commit is invalid")

    provenance = metadata.get("execution_provenance")
    if not isinstance(provenance, dict):
        reasons.append("the cluster DDP metadata omits execution_provenance")
    else:
        slurm_job_id = provenance.get("slurm_job_id")
        if not isinstance(slurm_job_id, str) or not re.fullmatch(r"[0-9]+", slurm_job_id):
            reasons.append("the cluster DDP SLURM job ID is missing or invalid")
        if provenance.get("git_commit") != git_commit:
            reasons.append("the cluster DDP metadata and fingerprint Git commits disagree")

    world_size = _integer(distributed.get("world_size"))
    environment = metadata.get("environment")
    if status == "complete":
        if not isinstance(environment, dict) or _integer(
            environment.get("distributed_world_size")
        ) != world_size:
            reasons.append("the cluster DDP metadata and fingerprint world sizes disagree")
        execution = metadata.get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("distributed_execution") != distributed
        ):
            reasons.append(
                "the cluster DDP result metadata does not preserve its fingerprinted "
                "execution specification"
            )
    return reasons


def _panther_ddp_matching_reasons(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    source_dir: Path,
) -> list[str]:
    """Check the frozen the cluster four-V100 execution policy and current source."""

    reasons: list[str] = []
    distributed = payload.get("distributed_execution")
    if not isinstance(distributed, dict):
        return ["expected the cluster DDP distributed_execution fingerprint"]

    implementation_hash = distributed.get("implementation_source_sha256")
    implementation_source = source_dir / PANTHER_DDP_SOURCE
    if not implementation_source.is_file():
        reasons.append(f"current DDP runner source is absent: {PANTHER_DDP_SOURCE}")
    elif (
        not isinstance(implementation_hash, str)
        or _sha256_path(implementation_source) != implementation_hash
    ):
        reasons.append("recorded DDP runner source hash does not match current file")

    if _integer(distributed.get("world_size")) != PANTHER_DDP_WORLD_SIZE:
        reasons.append(
            f"the cluster DDP world size is not {PANTHER_DDP_WORLD_SIZE}"
        )
    run_config = contract.get("run")
    expected_global_batch = (
        _integer(run_config.get("batch_size")) if isinstance(run_config, dict) else None
    )
    if _integer(distributed.get("global_batch_size")) != expected_global_batch:
        reasons.append("the cluster DDP global batch disagrees with frozen contract")
    expected_inference_batch = (
        expected_global_batch // PANTHER_DDP_WORLD_SIZE
        if expected_global_batch is not None
        and expected_global_batch % PANTHER_DDP_WORLD_SIZE == 0
        else None
    )
    if _integer(distributed.get("rank_zero_inference_batch_size")) != expected_inference_batch:
        reasons.append("the cluster DDP inference batch is not the frozen local batch")

    threads = distributed.get("thread_environment")
    if not isinstance(threads, dict) or {
        "OMP_NUM_THREADS": threads.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": threads.get("MKL_NUM_THREADS"),
    } != {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}:
        reasons.append("the cluster DDP OMP/MKL thread environment is not frozen at 2")

    inventory = distributed.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != PANTHER_DDP_WORLD_SIZE:
        reasons.append("the cluster DDP GPU inventory does not contain four ranks")
    else:
        ranks: set[int] = set()
        local_ranks: set[int] = set()
        for gpu in inventory:
            if not isinstance(gpu, dict):
                reasons.append("the cluster DDP GPU inventory entry is not an object")
                break
            rank = _integer(gpu.get("rank"))
            local_rank = _integer(gpu.get("local_rank"))
            name = gpu.get("name")
            memory = _integer(gpu.get("total_memory_bytes"))
            if rank is not None:
                ranks.add(rank)
            if local_rank is not None:
                local_ranks.add(local_rank)
            if not isinstance(name, str) or "V100" not in name.upper():
                reasons.append("the cluster DDP GPU inventory is not V100")
                break
            if memory is None or not 15 * 1024**3 <= memory < 20 * 1024**3:
                reasons.append("the cluster DDP GPU inventory is not the 16-GB allocation")
                break
        expected_ranks = set(range(PANTHER_DDP_WORLD_SIZE))
        if ranks != expected_ranks or local_ranks != expected_ranks:
            reasons.append("the cluster DDP GPU rank inventory is incomplete")
    return reasons


def _matching_payload_reasons(
    payload: Mapping[str, Any],
    *,
    model: str,
    seed: int | None,
    contract_sha256: str,
    contract: Mapping[str, Any],
    declared_seeds: set[int],
    source_dir: Path,
    execution_branch: str,
) -> list[str]:
    reasons: list[str] = []
    expected_scope = {
        "study": STUDY_ID,
        "table": 2,
        "dataset": "SGCC",
        "model": model,
        "seed": seed,
        "contract_sha256": contract_sha256,
    }
    for field_name, expected in expected_scope.items():
        if payload.get(field_name) != expected:
            reasons.append(
                f"fingerprint payload {field_name}={payload.get(field_name)!r}, expected {expected!r}"
            )
    if seed is None or seed not in declared_seeds:
        reasons.append(f"seed {seed!r} is outside declared seeds {sorted(declared_seeds)}")
    if model not in PRIMARY_SCORE_BRANCH:
        reasons.append(f"unrecognized Table II model {model!r}")
    if payload.get("source_sha256") != EXPECTED_SGCC_SHA256:
        reasons.append("fingerprint source_sha256 is not the frozen exact SGCC source")
    expected_execution = _expected_execution_branch(model)
    if execution_branch != expected_execution:
        reasons.append(
            f"execution branch {execution_branch!r}, expected {expected_execution!r}"
        )
    elif execution_branch == PANTHER_DDP_EXECUTION_BRANCH:
        reasons.extend(
            _panther_ddp_matching_reasons(
                payload,
                contract=contract,
                source_dir=source_dir,
            )
        )
    if payload.get("run_config") != contract.get("run"):
        reasons.append("fingerprint run_config disagrees with frozen contract")
    table_1 = contract.get("table_1")
    expected_model_config = (
        table_1.get(model)
        if isinstance(table_1, dict) and model not in {
            "naive_bayes",
            "arima",
            "one_class_svm",
            "multiclass_svm",
        }
        else None
    )
    if payload.get("model_config") != expected_model_config:
        reasons.append("fingerprint model_config disagrees with frozen contract")
    thresholds = contract.get("thresholds")
    if model in {"arima", "one_class_svm", "fc_sae", "lstm_sae", "fc_vae", "lstm_vae", "lstm_aea"}:
        expected_threshold = (
            thresholds.get(model) if isinstance(thresholds, dict) else None
        )
    elif model == "multiclass_svm":
        expected_threshold = 0.0
    else:
        expected_threshold = 0.5
    recorded_threshold = _finite_float(payload.get("threshold"))
    expected_threshold_float = _finite_float(expected_threshold)
    if (
        recorded_threshold is None
        or expected_threshold_float is None
        or not math.isclose(
            recorded_threshold,
            expected_threshold_float,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        reasons.append("fingerprint threshold disagrees with frozen contract")

    source_hashes = payload.get("source_code_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        reasons.append("fingerprint payload has no source_code_sha256 map")
    else:
        for filename in sorted(source_hashes):
            expected_hash = source_hashes[filename]
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or Path(filename).is_absolute()
            ):
                reasons.append(f"unsafe source filename {filename!r}")
                continue
            source = source_dir / filename
            if not source.is_file():
                reasons.append(f"recorded source file is absent: {filename}")
                continue
            if not isinstance(expected_hash, str) or _sha256_path(source) != expected_hash:
                reasons.append(f"recorded source hash does not match current file: {filename}")
    return reasons


def verify_attempts(
    runner_root: str | Path,
    *,
    contract_sha256: str,
    contract: Mapping[str, Any],
    declared_seeds: Sequence[int],
    source_dir: str | Path,
) -> list[AttemptRecord]:
    """Verify and inventory every attempt directory under the runner root."""

    root = Path(runner_root).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    records: list[AttemptRecord] = []
    declared = {int(seed) for seed in declared_seeds}

    for model, seed, attempt_dir in _logical_attempt_dirs(root):
        relative = attempt_dir.relative_to(root).as_posix()
        record = AttemptRecord(model, seed, attempt_dir.name, relative)
        records.append(record)
        if not _ATTEMPT_ID.fullmatch(attempt_dir.name):
            record.verification_detail = "attempt directory name is not a runner attempt ID"
            continue
        manifest_path = attempt_dir / "manifest.json"
        if not manifest_path.is_file():
            record.verification_detail = "manifest.json is missing"
            continue
        record.manifest_sha256 = _sha256_path(manifest_path)
        try:
            manifest = _read_json(manifest_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record.verification_detail = f"manifest is unreadable: {type(exc).__name__}: {exc}"
            continue

        record.attempt_status = str(manifest.get("status", "unknown"))
        record.fingerprint = str(manifest.get("fingerprint", ""))
        if manifest.get("schema_version") != 1:
            record.verification_detail = "unsupported or missing manifest schema_version"
            continue
        if record.attempt_status not in {"complete", "failed", "interrupted"}:
            record.verification_detail = f"unsupported attempt status {record.attempt_status!r}"
            continue
        if manifest.get("attempt_id") != attempt_dir.name:
            record.verification_detail = "manifest attempt_id does not match directory"
            continue
        payload = manifest.get("fingerprint_payload")
        if not isinstance(payload, dict):
            record.verification_detail = "manifest fingerprint_payload is not an object"
            continue
        record.execution_branch = _execution_branch(payload)
        source_hashes = payload.get("source_code_sha256")
        if isinstance(source_hashes, dict):
            record.recorded_source_code_sha256 = {
                str(filename): str(digest)
                for filename, digest in sorted(source_hashes.items())
            }
        distributed = payload.get("distributed_execution")
        if isinstance(distributed, dict):
            implementation_hash = distributed.get("implementation_source_sha256")
            if isinstance(implementation_hash, str):
                record.execution_source_code_sha256 = {
                    PANTHER_DDP_SOURCE: implementation_hash
                }
        expected_fingerprint = _sha256_bytes(_canonical_json_bytes(payload))
        if record.fingerprint != expected_fingerprint:
            record.verification_detail = "manifest fingerprint does not hash fingerprint_payload"
            continue

        artifacts_ok, artifact_detail, actual_hashes = _verify_artifacts(
            attempt_dir, manifest.get("artifacts")
        )
        record.artifact_sha256 = actual_hashes
        if not artifacts_ok:
            record.verification_detail = artifact_detail
            continue
        required = {"metadata.json", "history.json", "result.json"}
        if not required.issubset(actual_hashes):
            record.verification_detail = "manifest omits one or more required JSON artifacts"
            continue
        if record.attempt_status == "complete" and "arrays.npz" not in actual_hashes:
            record.verification_detail = "complete attempt manifest omits arrays.npz"
            continue
        try:
            record.metadata = _read_json(attempt_dir / "metadata.json")
            record.history = _read_json(attempt_dir / "history.json")
            record.result = _read_json(attempt_dir / "result.json")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record.verification_detail = f"verified JSON artifact is unreadable: {type(exc).__name__}: {exc}"
            continue
        if record.result.get("status") != record.attempt_status:
            record.verification_detail = "manifest and result statuses disagree"
            continue
        if record.result.get("model") != model or _integer(record.result.get("seed")) != seed:
            record.verification_detail = "result model/seed disagrees with logical path"
            continue
        execution_provenance = record.metadata.get("execution_provenance")
        if isinstance(execution_provenance, dict):
            slurm_job_id = execution_provenance.get("slurm_job_id")
            git_commit = execution_provenance.get("git_commit")
            record.slurm_job_id = (
                str(slurm_job_id) if slurm_job_id is not None else None
            )
            record.git_commit = str(git_commit) if git_commit is not None else None

        metadata_reasons = _metadata_reasons(
            record.metadata,
            payload,
            model=model,
            seed=seed,
            status=record.attempt_status,
            contract_sha256=contract_sha256,
        )
        if metadata_reasons:
            record.verification_detail = "; ".join(metadata_reasons)
            continue
        execution_integrity_reasons = _execution_integrity_reasons(
            payload,
            record.metadata,
            execution_branch=record.execution_branch,
            status=record.attempt_status,
        )
        if execution_integrity_reasons:
            record.verification_detail = "; ".join(execution_integrity_reasons)
            continue
        if record.attempt_status == "complete":
            arrays_ok, arrays_detail = _verify_complete_arrays(attempt_dir, record.result)
            if not arrays_ok:
                record.verification_detail = arrays_detail
                continue
            primary = PRIMARY_SCORE_BRANCH.get(model)
            result_metrics = record.result.get("metrics")
            primary_metrics = (
                result_metrics.get(primary) if isinstance(result_metrics, dict) else None
            )
            if (
                not isinstance(primary_metrics, dict)
                or _finite_float(primary_metrics.get("threshold"))
                != _finite_float(payload.get("threshold"))
                or primary_metrics.get("positive_if") != "higher"
            ):
                record.verification_detail = (
                    "primary result score rule disagrees with fingerprint"
                )
                continue

        mismatch_reasons = _matching_payload_reasons(
            payload,
            model=model,
            seed=seed,
            contract_sha256=contract_sha256,
            contract=contract,
            declared_seeds=declared,
            source_dir=source,
            execution_branch=record.execution_branch,
        )
        if mismatch_reasons:
            record.verification_status = "VERIFIED_NONMATCHING"
            record.verification_detail = "; ".join(mismatch_reasons)
            continue

        record.matching = True
        record.verification_status = "VERIFIED_MATCHING"
        record.verification_detail = (
            f"{artifact_detail}; {arrays_detail}"
            if record.attempt_status == "complete"
            else artifact_detail
        )
        if record.attempt_status == "complete":
            primary = PRIMARY_SCORE_BRANCH.get(model)
            metrics = record.result.get("metrics")
            if not isinstance(metrics, dict) or not isinstance(metrics.get(primary), dict):
                record.matching = False
                record.verification_status = "INVALID"
                record.verification_detail = f"complete result omits primary score branch {primary!r}"

    # Lexical order is chronological for the runner's UTC-prefixed attempt IDs.
    # Select the first valid completion: a later forced rerun is evidence, not
    # a replacement opportunity for an already observed outcome.
    earliest: dict[tuple[str, int], AttemptRecord] = {}
    for record in records:
        if (
            record.matching
            and record.attempt_status == "complete"
            and record.seed is not None
        ):
            key = (record.model, record.seed)
            if key not in earliest or record.attempt_id < earliest[key].attempt_id:
                earliest[key] = record
    for record in earliest.values():
        record.selected = True
    return records


def _metric_row(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for paper_name, result_name in RESULT_METRICS.items():
        value = _finite_float(metrics.get(result_name)) if metrics else None
        output[paper_name] = value * 100.0 if value is not None else None
    for field_name in COUNT_FIELDS:
        output[field_name.upper()] = _integer(metrics.get(field_name)) if metrics else None
    return output


def _timing_row(record: AttemptRecord) -> dict[str, Any]:
    timings = record.result.get("timings", {}) if record.result else {}
    history = record.history or {}
    epoch_values = history.get("epoch_seconds", [])
    finite_epoch_values = (
        [value for item in epoch_values if (value := _finite_float(item)) is not None]
        if isinstance(epoch_values, list)
        else []
    )
    epochs_completed = _integer(history.get("epochs_completed"))
    if epochs_completed is None and history.get("kind") == "classical":
        epochs_completed = 0
    return {
        "fit_seconds": _finite_float(timings.get("fit_seconds")),
        "score_seconds": _finite_float(timings.get("score_seconds")),
        "epochs_completed": epochs_completed,
        "epoch_seconds_total": sum(finite_epoch_values) if finite_epoch_values else 0.0,
        "epoch_seconds_mean": (
            sum(finite_epoch_values) / len(finite_epoch_values)
            if finite_epoch_values
            else 0.0
        ),
    }


def _attempt_branches(record: AttemptRecord) -> list[tuple[str, str]]:
    if record.result is None or record.attempt_status != "complete":
        return [("", "none")]
    metrics = record.result.get("metrics")
    if not isinstance(metrics, dict):
        return [("", "none")]
    primary = PRIMARY_SCORE_BRANCH.get(record.model, "")
    branches: list[tuple[str, str]] = []
    if primary in metrics:
        branches.append((primary, "primary"))
    if record.model in {"fc_vae", "lstm_vae"} and VAE_DIAGNOSTIC_BRANCH in metrics:
        branches.append((VAE_DIAGNOSTIC_BRANCH, "vae_surrogate_diagnostic"))
    return branches or [("", "none")]


INDIVIDUAL_FIELDS = (
    "model",
    "internal_model",
    "seed",
    "execution_branch",
    "slurm_job_id",
    "git_commit",
    "execution_source_sha256",
    "attempt_id",
    "attempt_path",
    "selected",
    "attempt_status",
    "verification_status",
    "verification_detail",
    "manifest_sha256",
    "fingerprint",
    "score_branch",
    "score_role",
    "n_test",
    "TP",
    "FP",
    "TN",
    "FN",
    *PAPER_METRICS,
    "fit_seconds",
    "score_seconds",
    "epochs_completed",
    "epoch_seconds_total",
    "epoch_seconds_mean",
    "error_type",
    "error_message",
)


def individual_rows(records: Sequence[AttemptRecord]) -> list[dict[str, Any]]:
    """Return an attempt-complete ledger, including failures and old successes."""

    order = {model: index for index, model in enumerate(PAPER_TO_INTERNAL.values())}
    sorted_records = sorted(
        records,
        key=lambda row: (
            order.get(row.model, len(order)),
            row.seed if row.seed is not None else 2**31,
            row.attempt_id,
        ),
    )
    rows: list[dict[str, Any]] = []
    for record in sorted_records:
        result_metrics = record.result.get("metrics", {}) if record.result else {}
        timing = _timing_row(record)
        error = record.result.get("error", {}) if record.result else {}
        for branch, role in _attempt_branches(record):
            metrics = result_metrics.get(branch) if isinstance(result_metrics, dict) else None
            row = {
                "model": record.paper_model,
                "internal_model": record.model,
                "seed": record.seed,
                "execution_branch": record.execution_branch,
                "slurm_job_id": record.slurm_job_id,
                "git_commit": record.git_commit,
                "execution_source_sha256": record.execution_source_code_sha256.get(
                    PANTHER_DDP_SOURCE
                ),
                "attempt_id": record.attempt_id,
                "attempt_path": record.relative_path,
                "selected": record.selected,
                "attempt_status": record.attempt_status,
                "verification_status": record.verification_status,
                "verification_detail": record.verification_detail,
                "manifest_sha256": record.manifest_sha256,
                "fingerprint": record.fingerprint,
                "score_branch": branch,
                "score_role": role,
                "n_test": _integer(record.result.get("n_test")) if record.result else None,
                **_metric_row(metrics if isinstance(metrics, dict) else None),
                **timing,
                "error_type": error.get("type") if isinstance(error, dict) else None,
                "error_message": error.get("message") if isinstance(error, dict) else None,
            }
            rows.append(row)
    return rows


def _selected_branch_rows(
    records: Sequence[AttemptRecord], model: str, branch: str
) -> list[tuple[AttemptRecord, Mapping[str, Any]]]:
    rows: list[tuple[AttemptRecord, Mapping[str, Any]]] = []
    for record in records:
        if not record.selected or record.model != model or record.result is None:
            continue
        metrics = record.result.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get(branch), dict):
            rows.append((record, metrics[branch]))
    return sorted(rows, key=lambda item: int(item[0].seed or 0))


def _aggregate_table_2_row(
    *,
    reported: Mapping[str, str],
    model: str,
    branch: str,
    role: str,
    records: Sequence[AttemptRecord],
    declared_seeds: Sequence[int],
) -> dict[str, Any]:
    selected = _selected_branch_rows(records, model, branch)
    failed_attempt_count = sum(
        1
        for record in records
        if record.model == model
        and record.matching
        and record.attempt_status in {"failed", "interrupted"}
    )
    row: dict[str, Any] = {
        "model": reported["model"],
        "internal_model": model,
        "execution_branch": _expected_execution_branch(model),
        "score_branch": branch,
        "score_role": role,
        "declared_seed_count": len(declared_seeds),
        "selected_seed_count": len(selected),
        "selected_seeds": ";".join(str(record.seed) for record, _ in selected),
        "failed_attempt_count": failed_attempt_count,
        "required_close_seed_count": 2,
    }

    close_seed_count = 0
    for paper_metric, result_metric in RESULT_METRICS.items():
        values_fraction = [_finite_float(metrics.get(result_metric)) for _, metrics in selected]
        values_percent = [value * 100.0 if value is not None else None for value in values_fraction]
        stats = _summary(values_percent)
        target = _finite_float(reported.get(paper_metric))
        row[f"{paper_metric}_paper"] = target
        for statistic in ("mean", "sample_sd", "min", "max"):
            row[f"{paper_metric}_{statistic}"] = stats[statistic]
        row[f"{paper_metric}_delta"] = (
            float(stats["mean"]) - target
            if stats["mean"] is not None and target is not None
            else None
        )

    for record, metrics in selected:
        del record
        deltas: list[float] = []
        for paper_metric, result_metric in RESULT_METRICS.items():
            value = _finite_float(metrics.get(result_metric))
            target = _finite_float(reported.get(paper_metric))
            if value is None or target is None:
                deltas = []
                break
            deltas.append(abs(value * 100.0 - target))
        if len(deltas) == len(PAPER_METRICS) and all(delta <= 2.0 for delta in deltas):
            close_seed_count += 1
    row["close_seed_count"] = close_seed_count

    timing_fields = (
        "fit_seconds",
        "score_seconds",
        "epochs_completed",
        "epoch_seconds_total",
        "epoch_seconds_mean",
    )
    for timing_field in timing_fields:
        stats = _summary(_timing_row(record)[timing_field] for record, _ in selected)
        for statistic in ("mean", "sample_sd", "min", "max"):
            row[f"{timing_field}_{statistic}"] = stats[statistic]

    if role == "vae_surrogate_diagnostic":
        row["status"] = "DIAGNOSTIC_ONLY"
    elif close_seed_count >= 2:
        row["status"] = "CLOSE_MATCH"
    elif len(selected) < 2:
        row["status"] = "INSUFFICIENT_VALID_SEEDS"
    else:
        row["status"] = "NOT_CLOSE_MATCH"
    return row


def _table_2_fields() -> list[str]:
    fields = [
        "model",
        "internal_model",
        "execution_branch",
        "score_branch",
        "score_role",
        "declared_seed_count",
        "selected_seed_count",
        "selected_seeds",
        "failed_attempt_count",
        "required_close_seed_count",
        "close_seed_count",
        "status",
    ]
    for metric in PAPER_METRICS:
        fields.extend(
            [
                f"{metric}_paper",
                f"{metric}_mean",
                f"{metric}_sample_sd",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_delta",
            ]
        )
    for timing in (
        "fit_seconds",
        "score_seconds",
        "epochs_completed",
        "epoch_seconds_total",
        "epoch_seconds_mean",
    ):
        fields.extend(
            f"{timing}_{statistic}"
            for statistic in ("mean", "sample_sd", "min", "max")
        )
    return fields


def table_2_rows(
    reported_rows: Sequence[Mapping[str, str]],
    records: Sequence[AttemptRecord],
    declared_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reported in reported_rows:
        model = PAPER_TO_INTERNAL.get(reported.get("model", ""))
        if model is None:
            raise ValueError(f"unrecognized Table II paper model {reported.get('model')!r}")
        rows.append(
            _aggregate_table_2_row(
                reported=reported,
                model=model,
                branch=PRIMARY_SCORE_BRANCH[model],
                role="primary",
                records=records,
                declared_seeds=declared_seeds,
            )
        )
        if model in {"fc_vae", "lstm_vae"}:
            rows.append(
                _aggregate_table_2_row(
                    reported=reported,
                    model=model,
                    branch=VAE_DIAGNOSTIC_BRANCH,
                    role="vae_surrogate_diagnostic",
                    records=records,
                    declared_seeds=declared_seeds,
                )
            )
    return rows


def _normalized_text(value: Any) -> str:
    return str(value).strip().lower()


def _equivalent_number(left: Any, right: Any) -> bool:
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-12)
    )


def table_1_rows(
    reported_rows: Sequence[Mapping[str, str]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    configuration = contract.get("table_1")
    if not isinstance(configuration, dict):
        raise ValueError("config is missing [table_1]")
    fields = (
        "layers_total",
        "encoder_widths",
        "optimizer",
        "dropout",
        "hidden_activation",
        "output_activation",
    )
    rows: list[dict[str, Any]] = []
    for reported in reported_rows:
        model = PAPER_TO_INTERNAL.get(reported.get("model", ""))
        if model is None:
            raise ValueError(f"unrecognized Table I paper model {reported.get('model')!r}")
        reconstructed = configuration.get(model)
        if not isinstance(reconstructed, dict):
            reconstructed = {}
        row: dict[str, Any] = {
            "model": reported["model"],
            "internal_model": model,
        }
        matches: list[bool] = []
        for field_name in fields:
            paper_value: Any = reported.get(field_name)
            reconstructed_value: Any = reconstructed.get(field_name)
            if field_name == "encoder_widths" and isinstance(reconstructed_value, list):
                reconstructed_value = ";".join(str(item) for item in reconstructed_value)
            row[f"{field_name}_paper"] = paper_value
            row[f"{field_name}_reconstructed"] = reconstructed_value
            if field_name in {"layers_total", "dropout"}:
                matches.append(_equivalent_number(paper_value, reconstructed_value))
            else:
                matches.append(_normalized_text(paper_value) == _normalized_text(reconstructed_value))
        row["status"] = "MATCH_CONFIG" if all(matches) else "CONFIG_MISMATCH"
        rows.append(row)
    return rows


def _table_1_fields() -> list[str]:
    fields = ["model", "internal_model"]
    for field_name in (
        "layers_total",
        "encoder_widths",
        "optimizer",
        "dropout",
        "hidden_activation",
        "output_activation",
    ):
        fields.extend([f"{field_name}_paper", f"{field_name}_reconstructed"])
    fields.append("status")
    return fields


def _gate_text() -> str:
    return ";".join(f"{filename}|md5={digest}" for filename, digest in CER_GATE)


def blocked_table(
    table_number: int, source_fields: Sequence[str], source_rows: Sequence[Mapping[str, str]]
) -> tuple[list[str], list[dict[str, Any]]]:
    if table_number == 3:
        reproduced_fields = list(PAPER_METRICS)
    elif table_number == 4:
        reproduced_fields = ["half_train", "three_quarter_train", "full_train"]
    elif table_number == 5:
        reproduced_fields = [
            "attack_1",
            "attack_2",
            "attack_3",
            "attack_4",
            "attack_5",
            "attack_6",
            "average",
        ]
    else:
        raise ValueError("only Tables III-V have blocked exact-data renderers")
    missing = [field_name for field_name in reproduced_fields if field_name not in source_fields]
    if missing:
        raise ValueError(f"reported Table {table_number} is missing columns {missing}")
    fields = [
        *source_fields,
        *(f"reproduction_{field_name}" for field_name in reproduced_fields),
        "status",
        "missing_file_count",
        "missing_file_gate",
    ]
    rows = []
    for source_row in source_rows:
        row: dict[str, Any] = dict(source_row)
        row.update({f"reproduction_{field_name}": BLOCKED for field_name in reproduced_fields})
        row.update(
            {
                "status": BLOCKED,
                "missing_file_count": len(CER_GATE),
                "missing_file_gate": _gate_text(),
            }
        )
        rows.append(row)
    return fields, rows


def aggregate(
    *,
    runner_root: str | Path,
    reported_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    source_dir: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify inputs, select attempts, and write all Paper 1 table artifacts."""

    import tomllib

    runner = Path(runner_root).expanduser().resolve()
    reported = Path(reported_dir).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source = (
        Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else Path(__file__).resolve().parent
    )
    if not runner.is_dir():
        raise FileNotFoundError(f"immutable runner root does not exist: {runner}")
    if not reported.is_dir():
        raise FileNotFoundError(f"reported CSV directory does not exist: {reported}")
    if not source.is_dir():
        raise FileNotFoundError(f"runner source directory does not exist: {source}")
    config_bytes = config.read_bytes()
    contract = tomllib.loads(config_bytes.decode("utf-8"))
    run_config = contract.get("run")
    if not isinstance(run_config, dict) or not isinstance(run_config.get("model_seeds"), list):
        raise ValueError("config [run] must define model_seeds")
    declared_seeds = [int(seed) for seed in run_config["model_seeds"]]
    if len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("config model_seeds must be unique")
    contract_sha256 = _sha256_bytes(config_bytes)

    reported_inputs: dict[int, tuple[list[str], list[dict[str, str]]]] = {}
    reported_hashes: dict[str, str] = {}
    for table_number in range(1, 6):
        path = reported / f"table_{table_number}.csv"
        reported_inputs[table_number] = _read_csv(path)
        reported_hashes[path.name] = _sha256_path(path)

    records = verify_attempts(
        runner,
        contract_sha256=contract_sha256,
        contract=contract,
        declared_seeds=declared_seeds,
        source_dir=source,
    )

    table1 = table_1_rows(reported_inputs[1][1], contract)
    table2_individual = individual_rows(records)
    table2 = table_2_rows(reported_inputs[2][1], records, declared_seeds)
    generated: dict[str, bytes] = {
        "table_1_reconstructed.csv": _csv_bytes(_table_1_fields(), table1),
        "table_2_individual.csv": _csv_bytes(INDIVIDUAL_FIELDS, table2_individual),
        "table_2_reproduction.csv": _csv_bytes(_table_2_fields(), table2),
    }
    blocked_counts: dict[str, int] = {}
    for table_number in (3, 4, 5):
        fields, rows = blocked_table(
            table_number,
            reported_inputs[table_number][0],
            reported_inputs[table_number][1],
        )
        filename = f"table_{table_number}_reproduction.csv"
        generated[filename] = _csv_bytes(fields, rows)
        blocked_counts[str(table_number)] = len(rows)

    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in generated.items():
        _atomic_write(output / filename, payload)

    selected = [record for record in records if record.selected]
    integrity_counts: dict[str, int] = {}
    for record in records:
        integrity_counts[record.verification_status] = (
            integrity_counts.get(record.verification_status, 0) + 1
        )
    failures = [
        {
            "model": record.model,
            "seed": record.seed,
            "execution_branch": record.execution_branch,
            "slurm_job_id": record.slurm_job_id,
            "git_commit": record.git_commit,
            "attempt_id": record.attempt_id,
            "attempt_status": record.attempt_status,
            "verification_status": record.verification_status,
            "detail": record.verification_detail,
            "manifest_sha256": record.manifest_sha256,
        }
        for record in records
        if record.attempt_status in {"failed", "interrupted"}
        or record.verification_status == "INVALID"
    ]
    inventory = [
        {
            "path": record.relative_path,
            "manifest_sha256": record.manifest_sha256,
            "artifact_sha256": record.artifact_sha256,
            "fingerprint": record.fingerprint,
            "execution_branch": record.execution_branch,
            "execution_source_code_sha256": record.execution_source_code_sha256,
            "slurm_job_id": record.slurm_job_id,
            "git_commit": record.git_commit,
            "verification_status": record.verification_status,
            "attempt_status": record.attempt_status,
            "selected": record.selected,
        }
        for record in records
    ]
    referenced_source_names = sorted(
        {
            filename
            for record in records
            for filename in (
                set(record.recorded_source_code_sha256)
                | set(record.execution_source_code_sha256)
            )
            if Path(filename).name == filename
        }
    )
    current_runner_source_hashes = {
        filename: _sha256_path(source / filename)
        for filename in referenced_source_names
        if (source / filename).is_file()
    }
    result_document = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY_ID,
        "track": "exploratory_paper_literal",
        "provenance": {
            "contract_sha256": contract_sha256,
            "reported_csv_sha256": reported_hashes,
            "aggregation_code_sha256": _sha256_path(Path(__file__).resolve()),
            "current_runner_source_code_sha256": current_runner_source_hashes,
            "runner_inventory_sha256": _sha256_bytes(_canonical_json_bytes(inventory)),
            "generated_csv_sha256": {
                filename: _sha256_bytes(payload) for filename, payload in sorted(generated.items())
            },
        },
        "attempts": {
            "discovered": len(records),
            "integrity_status_counts": integrity_counts,
            "selected": [
                {
                    "model": record.model,
                    "seed": record.seed,
                    "execution_branch": record.execution_branch,
                    "slurm_job_id": record.slurm_job_id,
                    "git_commit": record.git_commit,
                    "attempt_id": record.attempt_id,
                    "fingerprint": record.fingerprint,
                    "manifest_sha256": record.manifest_sha256,
                    "recorded_source_code_sha256": record.recorded_source_code_sha256,
                    "execution_source_code_sha256": (
                        record.execution_source_code_sha256
                    ),
                }
                for record in selected
            ],
            "failures_and_invalid": failures,
        },
        "assessment": {
            "metric_unit": "percent",
            "dispersion": "sample standard deviation (ddof=1; zero for n=1)",
            "signed_delta": "reproduction mean minus paper target, percentage points",
            "close_match_rule": (
                "all seven metrics within 2 percentage points in at least two "
                "of the three declared seeds"
            ),
            "table_2_rows": [
                {
                    "model": row["model"],
                    "execution_branch": row["execution_branch"],
                    "score_branch": row["score_branch"],
                    "score_role": row["score_role"],
                    "selected_seed_count": row["selected_seed_count"],
                    "close_seed_count": row["close_seed_count"],
                    "status": row["status"],
                }
                for row in table2
            ],
        },
        "exact_data_gate": {
            "status": BLOCKED,
            "required_file_count": len(CER_GATE),
            "required_files": [
                {"filename": filename, "md5": digest} for filename, digest in CER_GATE
            ],
            "blocked_table_row_counts": blocked_counts,
        },
    }
    _write_json(output / "paper_1_results.json", result_document)
    return result_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--reported-dir", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing the runner source files named in fingerprints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = aggregate(
        runner_root=args.runner_root,
        reported_dir=args.reported_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
    )
    print(json.dumps(document["attempts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
