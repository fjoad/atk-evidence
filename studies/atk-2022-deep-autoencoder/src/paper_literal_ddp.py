"""Production four-GPU DDP runner for exploratory Paper 1 Table II.

The ordinary runner is the reference implementation for SGCC preparation,
model construction, compiled Keras losses and optimizers, scoring, metrics, and
immutable attempt artifacts.  This module implements a separately fingerprinted
distributed execution branch: one declared neural model/seed is trained by four
equal peers under PyTorch DistributedDataParallel while retaining the frozen
global batch size.

The distributed trajectory is a documented execution branch, not a claim of
bitwise identity with Keras ``Model.fit`` on one device.  Rank 0 broadcasts one
global permutation per epoch.  Every real sample appears exactly once, and a
partial global batch is divided as evenly as possible.  DDP's gradient mean is
corrected by ``world_size * local_count / global_count`` so the update is the
gradient of the global sample mean.  Empty final shards execute a zero-weight
dummy forward only to participate in collectives; their stochastic-generator
states are restored immediately afterward.

Only rank 0 scores and publishes an attempt.  Recoverable failures are also
published as immutable failed attempts.  A process-level CUDA/NCCL termination
can preclude Python cleanup; the corresponding immutable Slurm log remains the
failure record in that case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from paper_literal_data import (  # noqa: E402
    ANOMALY_ADASYN_BRANCHES,
    SCALING_BRANCHES,
    SGCC_MISSING_BRANCHES,
    SGCC_REPRESENTATION_BRANCHES,
    SPLIT_UNIT_BRANCHES,
    SUPERVISED_ADASYN_BRANCHES,
)
from paper_literal_metrics import evaluate_binary_scores, threshold_predictions  # noqa: E402
from paper_literal_runner import (  # noqa: E402
    ANOMALY_NEURAL_MODELS,
    EXPECTED_SGCC_SHA256,
    SCHEMA_VERSION,
    SGCC_TABLE_II_SCOPE,
    SUPERVISED_NEURAL_MODELS,
    THRESHOLD_SCOPES,
    VALIDATION_LABEL_BRANCHES,
    VALIDATION_POLICIES,
    Contract,
    ExecutionResult,
    RunScope,
    RunOutcome,
    ThresholdPopulation,
    _anomaly_scores,
    _artifact_arrays,
    _atomic_write_json,
    _best_epoch,
    _canonical_json_bytes,
    _concatenate_partitions,
    _cross_validation_indices,
    _environment_metadata,
    _jsonable,
    _keras_callbacks,
    _keras_fixed_callbacks,
    _logical_run_dir,
    _metrics_for_scores,
    _persist_attempt,
    _preflight_payload,
    _select_anomaly_thresholds,
    _sha256_bytes,
    _sha256_path,
    _supervised_fit_split,
    _verified_completed_attempt,
    _write_preflight,
    build_threshold_population,
    canonical_model_name,
    load_contract,
    resolve_seeds,
    verify_and_prepare_sgcc,
)
from branch_runtime import (  # noqa: E402
    DEFAULT_LATTICE,
    assert_branch_scope,
    load_runtime_branch,
)


DDP_IMPLEMENTATION = "keras-torch-ddp-v2-validation-threshold-branches"
DDP_MODELS = (*ANOMALY_NEURAL_MODELS, *SUPERVISED_NEURAL_MODELS)


@dataclass(frozen=True)
class TrainingPartitions:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test: Any
    metadata: Mapping[str, Any]


def balanced_shard_bounds(
    sample_count: int, world_size: int, rank: int
) -> tuple[int, int]:
    """Return a contiguous balanced shard, allowing a zero-sized last rank."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within the distributed world")
    base, remainder = divmod(sample_count, world_size)
    start = rank * base + min(rank, remainder)
    stop = start + base + int(rank < remainder)
    return start, stop


def shard_indices(indices: np.ndarray, world_size: int, rank: int) -> np.ndarray:
    """Return one rank's contiguous piece of a global batch permutation."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    start, stop = balanced_shard_bounds(values.size, world_size, rank)
    return values[start:stop]


def ddp_loss_scale(local_count: int, global_count: int, world_size: int) -> float:
    """Compensate for DDP's mean gradient reduction."""

    if local_count < 0 or global_count <= 0 or world_size <= 0:
        raise ValueError("counts/world_size are outside the valid loss-scale domain")
    if local_count > global_count:
        raise ValueError("local_count cannot exceed global_count")
    return float(world_size * local_count / global_count)


def weighted_mean(parts: Sequence[tuple[float, int]]) -> float:
    """Pure reference for globally sample-weighted loss aggregation."""

    count = sum(int(item_count) for _, item_count in parts)
    if count <= 0:
        raise ValueError("weighted mean requires at least one sample")
    return float(
        sum(float(mean) * int(item_count) for mean, item_count in parts) / count
    )


def inference_batch_size(global_batch: int, world_size: int) -> int:
    """Keep rank-0 inference within the per-GPU training batch envelope."""

    if global_batch <= 0 or world_size <= 0 or global_batch % world_size:
        raise ValueError("global batch must divide evenly across distributed ranks")
    return global_batch // world_size


def timing_payload(
    *,
    data_prep_seconds: float,
    run_seconds: float,
    fit_seconds: float | None = None,
    score_seconds: float | None = None,
    failed: bool = False,
) -> Mapping[str, Any]:
    """Build timings without counting the preparation phase twice."""

    payload: dict[str, Any] = {
        "data_prep_seconds": float(data_prep_seconds),
        "run_seconds": float(run_seconds),
        "end_to_end_seconds": float(data_prep_seconds + run_seconds),
        "clock": "time.perf_counter",
        "shared_data_prep": False,
        "data_prep_aggregation": "maximum across ranks",
    }
    if failed:
        payload["elapsed_until_failure_seconds"] = payload.pop("run_seconds")
    if fit_seconds is not None:
        payload["fit_seconds"] = float(fit_seconds)
    if score_seconds is not None:
        payload["score_seconds"] = float(score_seconds)
    return payload


def epoch_permutation(sample_count: int, seed: int, epoch: int) -> np.ndarray:
    """Deterministically generate the rank-0 global shuffle for one epoch."""

    if sample_count < 0 or epoch < 0:
        raise ValueError("sample_count and epoch must be non-negative")
    seed_sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF, 0x41544B31]
    )
    return np.random.default_rng(seed_sequence).permutation(sample_count).astype(
        np.int64, copy=False
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _training_partitions(
    model_name: str,
    prepared: Any,
    contract: Contract,
    seed: int,
    *,
    supervised_head: str | None = None,
) -> TrainingPartitions:
    if model_name in ANOMALY_NEURAL_MODELS:
        train_x = prepared.anomaly_train.values
        validation_x = prepared.anomaly_validation.values
        return TrainingPartitions(
            train_x=np.ascontiguousarray(train_x, dtype=np.float32),
            train_y=np.ascontiguousarray(train_x, dtype=np.float32),
            validation_x=np.ascontiguousarray(validation_x, dtype=np.float32),
            validation_y=np.ascontiguousarray(validation_x, dtype=np.float32),
            test=prepared.anomaly_test,
            metadata={
                "train_partition": "anomaly_train_benign",
                "validation_partition": "anomaly_validation_benign",
                "target_policy": (
                    "none_attached_loss"
                    if model_name.endswith("_vae")
                    else "x_equals_y"
                ),
            },
        )
    if model_name in SUPERVISED_NEURAL_MODELS:
        train_x, train_y, validation_x, validation_y = _supervised_fit_split(
            prepared,
            validation_fraction=float(contract.run["validation_fraction_within_train"]),
            seed=seed,
        )
        if supervised_head == "sigmoid1_binary":
            train_y = train_y.astype(np.float32).reshape(-1, 1)
            validation_y = validation_y.astype(np.float32).reshape(-1, 1)
        elif supervised_head in {None, "softmax2_categorical"}:
            train_y = train_y.astype(np.int64)
            validation_y = validation_y.astype(np.int64)
        else:
            raise ValueError(f"unsupported supervised head {supervised_head!r}")
        return TrainingPartitions(
            train_x=np.ascontiguousarray(train_x, dtype=np.float32),
            train_y=np.ascontiguousarray(train_y),
            validation_x=np.ascontiguousarray(validation_x, dtype=np.float32),
            validation_y=np.ascontiguousarray(validation_y),
            test=prepared.supervised_test,
            metadata={
                "train_partition": "supervised_train_subsplit",
                "validation_partition": "supervised_train_subsplit",
                "validation_random_state": int(seed),
                "validation_stratified": True,
                "target_policy": "class_labels",
            },
        )
    raise ValueError(f"{model_name!r} is not a supported neural Table-II model")


def _all_training_partitions(
    model_name: str,
    prepared: Any,
    *,
    supervised_head: str | None = None,
) -> TrainingPartitions:
    """Return the complete registered training population for final refits."""

    if model_name in ANOMALY_NEURAL_MODELS:
        combined = _concatenate_partitions(
            [prepared.anomaly_train, prepared.anomaly_validation]
        )
        values = np.ascontiguousarray(combined.values, dtype=np.float32)
        return TrainingPartitions(
            train_x=values,
            train_y=values,
            validation_x=np.empty((0, values.shape[1]), dtype=np.float32),
            validation_y=np.empty((0, values.shape[1]), dtype=np.float32),
            test=prepared.anomaly_test,
            metadata={
                "train_partition": "all_b1_benign",
                "validation_partition": "none",
                "target_policy": (
                    "none_attached_loss"
                    if model_name.endswith("_vae")
                    else "x_equals_y"
                ),
            },
        )
    if model_name in SUPERVISED_NEURAL_MODELS:
        values = np.ascontiguousarray(
            prepared.supervised_train.values,
            dtype=np.float32,
        )
        labels = np.asarray(prepared.supervised_train.labels)
        if supervised_head == "sigmoid1_binary":
            labels = labels.astype(np.float32).reshape(-1, 1)
        elif supervised_head in {None, "softmax2_categorical"}:
            labels = labels.astype(np.int64)
        else:
            raise ValueError(f"unsupported supervised head {supervised_head!r}")
        return TrainingPartitions(
            train_x=values,
            train_y=np.ascontiguousarray(labels),
            validation_x=np.empty((0, values.shape[1]), dtype=np.float32),
            validation_y=np.empty((0, *labels.shape[1:]), dtype=labels.dtype),
            test=prepared.supervised_test,
            metadata={
                "train_partition": "all_supervised_train",
                "validation_partition": "none",
                "target_policy": "class_labels",
            },
        )
    raise ValueError(f"{model_name!r} is not a supported neural Table-II model")


def _dependency_versions() -> Mapping[str, str]:
    import keras
    import imblearn
    import sklearn
    import torch

    return {
        "keras": keras.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "nccl": str(torch.cuda.nccl.version()),
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
    }


def _distributed_runtime_provenance(
    *, rank: int, local_rank: int, world_size: int
) -> Mapping[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(local_rank)
    local_gpu = {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }
    inventory = sorted(
        _all_rank_values(local_gpu, world_size), key=lambda item: item["rank"]
    )
    return {
        "software": dict(_dependency_versions()),
        "gpu_inventory": inventory,
        "git_commit": os.environ.get("ATK_GIT_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def execution_specification(
    *,
    world_size: int,
    global_batch: int,
    partitions: TrainingPartitions,
    runtime_provenance: Mapping[str, Any],
    branch_execution: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Machine-readable distributed contract included in every fingerprint."""

    return {
        "implementation": DDP_IMPLEMENTATION,
        "implementation_source_sha256": _sha256_path(Path(__file__).resolve()),
        "world_size": int(world_size),
        "global_batch_size": int(global_batch),
        "rank_zero_inference_batch_size": inference_batch_size(
            global_batch, world_size
        ),
        "optimizer": "compiled Keras optimizer; DDP-reduced gradients",
        "loss": "compiled Keras compute_loss",
        "gradient_weighting": "world_size * local_count / global_count",
        "sharding": "balanced contiguous shards of each global batch; no drop/pad",
        "shuffle": "rank-0 NumPy PCG64 permutation broadcast once per epoch",
        "validation": (
            "branch-specific fixed/holdout/cross-validation policy; every "
            "validation loss is a globally sample-weighted mean"
        ),
        "early_stopping": (
            "selection fits use the existing Keras callback driven by global "
            "val_loss on rank 0; fixed/refit stages use no early stopping"
        ),
        "best_weight_restore": (
            "rank-0 Keras restore on selection fits; trainable weights "
            "broadcast before selection or final scoring"
        ),
        "scoring": "complete original-order test partition on rank 0",
        "failure_start_record": (
            "rank-0 immutable preflight precedes training; Slurm output retains "
            "fatal CUDA/NCCL terminations"
        ),
        "stochastic_streams": (
            "deterministic SHA-256-derived rank-specific Keras SeedGenerator states; "
            "empty dummy forwards restore stochastic state"
        ),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "dependency_versions": dict(runtime_provenance["software"]),
        "gpu_inventory": list(runtime_provenance["gpu_inventory"]),
        "git_commit": runtime_provenance.get("git_commit"),
        "branch_execution": dict(branch_execution or {}),
        "cardinalities": {
            "train": int(partitions.train_x.shape[0]),
            "validation": int(partitions.validation_x.shape[0]),
            "test": int(partitions.test.values.shape[0]),
        },
        "partition_array_sha256": {
            "train_x": _array_sha256(partitions.train_x),
            "train_y": _array_sha256(partitions.train_y),
            "validation_x": _array_sha256(partitions.validation_x),
            "validation_y": _array_sha256(partitions.validation_y),
        },
    }


def distributed_fingerprint(
    *,
    base_payload: Mapping[str, Any],
    execution_spec: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    payload = {**dict(base_payload), "distributed_execution": dict(execution_spec)}
    return _sha256_bytes(_canonical_json_bytes(payload)), payload


def _parameter_sha256(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seed_generators(model: Any) -> list[Any]:
    generators: list[Any] = []
    seen: set[int] = set()
    for layer in model._flatten_layers():
        generator = getattr(layer, "seed_generator", None)
        if generator is not None and id(generator) not in seen:
            seen.add(id(generator))
            generators.append(generator)
    return generators


def _rank_stream_seed(model_name: str, seed: int, rank: int, index: int) -> int:
    payload = f"{DDP_IMPLEMENTATION}:{model_name}:{seed}:{rank}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _assign_rank_stochastic_streams(
    model: Any, model_name: str, seed: int, rank: int
) -> list[list[int]]:
    states: list[list[int]] = []
    for index, generator in enumerate(_seed_generators(model)):
        state = np.asarray(
            [_rank_stream_seed(model_name, seed, rank, index), 0], dtype=np.int32
        )
        generator.state.assign(state)
        states.append(state.astype(int).tolist())
    return states


def _snapshot_stochastic_streams(model: Any) -> list[Any]:
    return [generator.state.value.detach().clone() for generator in _seed_generators(model)]


def _restore_stochastic_streams(model: Any, snapshots: Sequence[Any]) -> None:
    generators = _seed_generators(model)
    if len(generators) != len(snapshots):
        raise RuntimeError("the model's stochastic-generator set changed during execution")
    for generator, snapshot in zip(generators, snapshots, strict=True):
        generator.state.assign(snapshot)


class _LossModuleFactory:
    @staticmethod
    def build(model: Any) -> Any:
        import torch

        class LossModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.paper_model = model

            def forward(self, inputs: Any, targets: Any) -> Any:
                del targets
                return self.paper_model(inputs, training=True)

        return LossModule()


def _all_rank_values(value: Any, world_size: int) -> list[Any]:
    import torch.distributed as distributed

    gathered: list[Any] = [None] * world_size
    distributed.all_gather_object(gathered, value)
    return gathered


def _assert_identical_trainable_state(model: Any, world_size: int, label: str) -> str:
    parameters = [weight.value for weight in model.trainable_weights]
    digest = _parameter_sha256(parameters)
    digests = _all_rank_values(digest, world_size)
    if len(set(digests)) != 1:
        raise RuntimeError(f"trainable parameters diverged {label}: {digests}")
    return digest


def _assert_identical_optimizer_state(model: Any, world_size: int) -> str:
    variables = [variable.value for variable in model.optimizer.variables]
    digest = _parameter_sha256(variables)
    digests = _all_rank_values(digest, world_size)
    if len(set(digests)) != 1:
        raise RuntimeError(f"compiled Keras optimizer states diverged: {digests}")
    return digest


def _broadcast_global_permutation(
    sample_count: int, seed: int, epoch: int, rank: int, device: Any
) -> np.ndarray:
    import torch
    import torch.distributed as distributed

    if rank == 0:
        tensor = torch.from_numpy(epoch_permutation(sample_count, seed, epoch)).to(
            device=device
        )
    else:
        tensor = torch.empty(sample_count, dtype=torch.int64, device=device)
    distributed.broadcast(tensor, src=0)
    return tensor.cpu().numpy()


def _to_device(array: np.ndarray, device: Any) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True)


def _global_loss_pair(local_loss: Any, local_count: int, device: Any) -> Any:
    import torch

    detached = local_loss.detach().to(dtype=torch.float64)
    return torch.stack(
        (
            detached * float(local_count),
            torch.tensor(float(local_count), dtype=torch.float64, device=device),
        )
    )


def _distributed_error_gate(local_code: int, device: Any, boundary: str) -> None:
    """Make an ordinary rank-local error visible before the next collective."""

    import torch
    import torch.distributed as distributed

    flag = torch.tensor(int(local_code), dtype=torch.int32, device=device)
    distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
    code = int(flag.item())
    if code:
        raise FloatingPointError(
            f"distributed {boundary} gate failed on at least one rank (code={code})"
        )


def _finite_tensor_code(tensors: Sequence[Any]) -> int:
    """Return a compact local code without a collective."""

    import torch

    if not tensors or any(tensor is None for tensor in tensors):
        return 1
    for tensor in tensors:
        if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
            return 2
    return 0


def _forward_loss_with_gate(
    *,
    parallel_model: Any,
    paper_model: Any,
    inputs: Any,
    targets: Any,
    training: bool,
    supervised_probability: bool,
    device: Any,
    boundary: str,
) -> Any:
    """Compute the compiled Keras loss and synchronize failure before backward."""

    import torch
    import torch.distributed as distributed

    local_code = 0
    local_loss: Any = None
    try:
        if training:
            outputs = parallel_model(inputs, targets)
        else:
            outputs = paper_model(inputs, training=False)
        if not bool(torch.isfinite(outputs).all().detach().cpu().item()):
            local_code = 2
        elif supervised_probability and (
            float(outputs.min().detach().cpu().item()) < 0.0
            or float(outputs.max().detach().cpu().item()) > 1.0
        ):
            local_code = 3
        else:
            local_loss = paper_model.compute_loss(
                x=inputs,
                y=targets,
                y_pred=outputs,
                training=training,
            )
            if local_loss is None:
                local_code = 4
            elif not bool(torch.isfinite(local_loss).detach().cpu().item()):
                local_code = 5
    except Exception as exc:
        local_code = 6
        print(
            json.dumps(
                {
                    "status": "rank_local_forward_failure",
                    "rank": int(distributed.get_rank()),
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    _distributed_error_gate(local_code, device, boundary)
    return local_loss


def _apply_compiled_optimizer(model: Any) -> None:
    import torch

    weights = list(model.trainable_weights)
    gradients = [weight.value.grad for weight in weights]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("a trainable Keras weight has no DDP-reduced gradient")
    with torch.no_grad():
        model.optimizer.apply(gradients, weights)


def _batch_slices(sample_count: int, global_batch: int) -> Sequence[tuple[int, int]]:
    return [
        (start, min(start + global_batch, sample_count))
        for start in range(0, sample_count, global_batch)
    ]


def _train_epoch(
    *,
    parallel_model: Any,
    paper_model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    permutation: np.ndarray,
    global_batch: int,
    rank: int,
    world_size: int,
    device: Any,
    supervised_probability: bool,
) -> float:
    import torch
    import torch.distributed as distributed

    parallel_model.train()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for global_start, global_stop in _batch_slices(train_x.shape[0], global_batch):
        global_indices = permutation[global_start:global_stop]
        local_indices = shard_indices(global_indices, world_size, rank)
        local_count = int(local_indices.size)
        snapshots: list[Any] | None = None
        if local_count:
            x_tensor = _to_device(train_x[local_indices], device)
            y_tensor = _to_device(train_y[local_indices], device)
        else:
            # DDP requires every peer to enter the same backward collective.
            # The dummy contributes exactly zero and does not consume this
            # rank's deterministic stochastic stream.
            snapshots = _snapshot_stochastic_streams(paper_model)
            x_tensor = _to_device(train_x[:1], device)
            y_tensor = _to_device(train_y[:1], device)

        parallel_model.zero_grad(set_to_none=True)
        local_loss = _forward_loss_with_gate(
            parallel_model=parallel_model,
            paper_model=paper_model,
            inputs=x_tensor,
            targets=y_tensor,
            training=True,
            supervised_probability=supervised_probability,
            device=device,
            boundary="training forward/loss",
        )
        scale = ddp_loss_scale(local_count, global_indices.size, world_size)
        scaled_loss = local_loss * scale
        scale_loss = getattr(paper_model.optimizer, "scale_loss", None)
        if callable(scale_loss):
            scaled_loss = scale_loss(scaled_loss)
        scaled_loss.backward()
        gradients = [weight.value.grad for weight in paper_model.trainable_weights]
        _distributed_error_gate(
            _finite_tensor_code(gradients), device, "post-backward gradient"
        )
        optimizer_error = 0
        try:
            _apply_compiled_optimizer(paper_model)
        except Exception as exc:
            optimizer_error = 1
            print(
                json.dumps(
                    {
                        "status": "rank_local_optimizer_failure",
                        "rank": int(distributed.get_rank()),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        _distributed_error_gate(
            optimizer_error, device, "compiled Keras optimizer application"
        )
        parameter_and_optimizer_state = [
            *(weight.value for weight in paper_model.trainable_weights),
            *(variable.value for variable in paper_model.optimizer.variables),
        ]
        _distributed_error_gate(
            _finite_tensor_code(parameter_and_optimizer_state),
            device,
            "post-optimizer parameter/state",
        )
        if snapshots is not None:
            _restore_stochastic_streams(paper_model, snapshots)
        if local_count:
            total += _global_loss_pair(local_loss, local_count, device)

    distributed.all_reduce(total, op=distributed.ReduceOp.SUM)
    if float(total[1].item()) != float(train_x.shape[0]):
        raise RuntimeError("distributed train accounting did not cover every sample")
    return float((total[0] / total[1]).item())


def _validation_epoch(
    *,
    paper_model: Any,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    global_batch: int,
    rank: int,
    world_size: int,
    device: Any,
    supervised_probability: bool,
) -> float:
    import torch
    import torch.distributed as distributed

    paper_model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for global_start, global_stop in _batch_slices(
            validation_x.shape[0], global_batch
        ):
            global_indices = np.arange(global_start, global_stop, dtype=np.int64)
            local_indices = shard_indices(global_indices, world_size, rank)
            local_count = int(local_indices.size)
            if local_count:
                x_tensor = _to_device(validation_x[local_indices], device)
                y_tensor = _to_device(validation_y[local_indices], device)
                local_loss = _forward_loss_with_gate(
                    parallel_model=None,
                    paper_model=paper_model,
                    inputs=x_tensor,
                    targets=y_tensor,
                    training=False,
                    supervised_probability=supervised_probability,
                    device=device,
                    boundary="validation forward/loss",
                )
                total += _global_loss_pair(local_loss, local_count, device)
            else:
                _distributed_error_gate(0, device, "validation forward/loss")

    distributed.all_reduce(total, op=distributed.ReduceOp.SUM)
    if float(total[1].item()) != float(validation_x.shape[0]):
        raise RuntimeError("distributed validation accounting did not cover every sample")
    return float((total[0] / total[1]).item())


def _broadcast_stop(stop: bool, device: Any) -> bool:
    import torch
    import torch.distributed as distributed

    value = torch.tensor(int(stop), dtype=torch.int32, device=device)
    distributed.broadcast(value, src=0)
    return bool(value.item())


def _broadcast_restored_weights(model: Any) -> None:
    import torch.distributed as distributed

    for weight in model.trainable_weights:
        distributed.broadcast(weight.value.data, src=0)


def _score_rank_zero(
    model_name: str,
    bundle: Any,
    test: Any,
    prepared: Any,
    contract: Contract,
    inference_batch: int,
    *,
    model_config: Mapping[str, Any],
    threshold_population: ThresholdPopulation | None,
    threshold_rule: str,
    threshold_scope: str,
    validation_labels: str,
    transferred_thresholds: Mapping[str, float] | None,
) -> tuple[
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, float],
    Mapping[str, str],
    Mapping[str, Mapping[str, Any]],
]:
    if model_name in ANOMALY_NEURAL_MODELS:
        if threshold_population is None:
            raise RuntimeError("anomaly DDP scoring has no threshold population")
        scores, orientations = _anomaly_scores(
            bundle,
            model_name=model_name,
            model_config=model_config,
            values=test.values,
            batch_size=inference_batch,
        )
        needs_local_validation_scores = (
            threshold_rule != "printed_constant"
            and threshold_scope == "dataset_specific"
        )
        validation_scores: Mapping[str, np.ndarray] | None = None
        if needs_local_validation_scores:
            validation_scores, validation_orientations = _anomaly_scores(
                bundle,
                model_name=model_name,
                model_config=model_config,
                values=threshold_population.values,
                batch_size=inference_batch,
            )
            if validation_orientations != orientations:
                raise RuntimeError(
                    "validation and test score orientations disagree"
                )
        thresholds, selections = _select_anomaly_thresholds(
            model_name=model_name,
            prepared=prepared,
            contract=contract,
            population=threshold_population,
            validation_scores=validation_scores,
            orientations=orientations,
            score_names=list(scores),
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
        predictions, metrics = _metrics_for_scores(
            test.labels,
            scores,
            thresholds,
            orientations,
        )
        return (
            scores,
            predictions,
            metrics,
            thresholds,
            orientations,
            selections,
        )

    raw = np.asarray(
        bundle.model.predict(test.values, batch_size=inference_batch, verbose=0)
    )
    supervised_head = str(model_config["supervised_head"])
    if supervised_head == "softmax2_categorical":
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise RuntimeError(
                "softmax2 classifier must return two probabilities"
            )
        probability = raw[:, 1].astype(np.float64)
        predictions = np.argmax(raw, axis=1).astype(np.int8)
    elif supervised_head == "sigmoid1_binary":
        probability = raw.reshape(-1).astype(np.float64)
        predictions = (probability >= 0.5).astype(np.int8)
    else:
        raise RuntimeError(f"unsupported supervised head {supervised_head!r}")
    thresholded = threshold_predictions(probability, 0.5)
    if not np.array_equal(predictions, thresholded):
        raise RuntimeError("classifier decisions disagree with the registered score rule")
    return (
        {"positive_class_probability": probability},
        {"positive_class_probability": predictions},
        {
            "positive_class_probability": evaluate_binary_scores(
                test.labels, probability, threshold=0.5
            ).as_dict()
        },
        {"positive_class_probability": 0.5},
        {"positive_class_probability": "higher"},
        {},
    )


@dataclass(frozen=True)
class DistributedFitOutcome:
    bundle: Any
    history: Mapping[str, Sequence[float]]
    epoch_seconds: Sequence[float]
    fit_seconds: float
    provenance: Mapping[str, Any]


def _broadcast_integer(value: int, *, rank: int, device: Any) -> int:
    import torch
    import torch.distributed as distributed

    tensor = torch.tensor(
        int(value) if rank == 0 else 0,
        dtype=torch.int64,
        device=device,
    )
    distributed.broadcast(tensor, src=0)
    return int(tensor.item())


def _distributed_fit_once(
    *,
    model_name: str,
    seed: int,
    model_config: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    epochs: int,
    early_stopping: bool,
    stage: str,
    contract: Contract,
    rank: int,
    local_rank: int,
    world_size: int,
    device: Any,
) -> DistributedFitOutcome:
    """Run one fresh distributed fit used by selection or final refit."""

    import keras
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from paper_literal_models import build_model

    keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with keras.device(f"cuda:{local_rank}"):
        bundle = build_model(
            model_name,
            int(train_x.shape[1]),
            dict(model_config),
            seed=seed,
        )
    loss_module = _LossModuleFactory.build(bundle.model).to(device)
    parallel_model = DistributedDataParallel(
        loss_module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    initial_parameter_sha256 = _assert_identical_trainable_state(
        bundle.model,
        world_size,
        f"after {stage} DDP initialization",
    )
    stochastic_states = _assign_rank_stochastic_streams(
        bundle.model,
        model_name,
        seed,
        rank,
    )
    gathered_streams = _all_rank_values(stochastic_states, world_size)
    if stochastic_states and len(
        {json.dumps(item) for item in gathered_streams}
    ) != world_size:
        raise RuntimeError("rank stochastic streams are not independent")

    callbacks: list[Any] = []
    epoch_timer: Any = None
    if rank == 0:
        factory = _keras_callbacks if early_stopping else _keras_fixed_callbacks
        callbacks, epoch_timer = factory(contract)
        for callback in callbacks:
            callback.set_model(bundle.model)
            callback.on_train_begin()

    global_batch = int(contract.run["batch_size"])
    history_series: dict[str, list[float]] = {"loss": []}
    if validation_x is not None:
        history_series["val_loss"] = []
    supervised_probability = model_name in SUPERVISED_NEURAL_MODELS
    fit_started = time.perf_counter()
    for epoch in range(int(epochs)):
        if rank == 0:
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
        permutation = _broadcast_global_permutation(
            train_x.shape[0],
            seed,
            epoch,
            rank,
            device,
        )
        train_loss = _train_epoch(
            parallel_model=parallel_model,
            paper_model=bundle.model,
            train_x=train_x,
            train_y=train_y,
            permutation=permutation,
            global_batch=global_batch,
            rank=rank,
            world_size=world_size,
            device=device,
            supervised_probability=supervised_probability,
        )
        validation_loss: float | None = None
        if validation_x is not None:
            if validation_y is None:
                raise RuntimeError("validation targets are missing")
            validation_loss = _validation_epoch(
                paper_model=bundle.model,
                validation_x=validation_x,
                validation_y=validation_y,
                global_batch=global_batch,
                rank=rank,
                world_size=world_size,
                device=device,
                supervised_probability=supervised_probability,
            )
        stop = False
        if rank == 0:
            logs = {"loss": train_loss}
            history_series["loss"].append(train_loss)
            if validation_loss is not None:
                logs["val_loss"] = validation_loss
                history_series["val_loss"].append(validation_loss)
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            stop = bool(bundle.model.stop_training)
        if _broadcast_stop(stop, device):
            break

    if rank == 0:
        for callback in callbacks:
            callback.on_train_end()
    _broadcast_restored_weights(bundle.model)
    fit_seconds = time.perf_counter() - fit_started
    final_parameter_sha256 = _assert_identical_trainable_state(
        bundle.model,
        world_size,
        f"after {stage} completion",
    )
    final_optimizer_sha256 = _assert_identical_optimizer_state(
        bundle.model,
        world_size,
    )
    return DistributedFitOutcome(
        bundle=bundle,
        history=history_series,
        epoch_seconds=(
            list(epoch_timer.epoch_seconds) if rank == 0 else []
        ),
        fit_seconds=float(fit_seconds),
        provenance={
            "stage": stage,
            "train_samples": int(train_x.shape[0]),
            "validation_samples": (
                int(validation_x.shape[0]) if validation_x is not None else 0
            ),
            "epochs_requested": int(epochs),
            "epochs_completed": int(len(history_series["loss"])),
            "early_stopping": bool(early_stopping),
            "initial_parameter_sha256": initial_parameter_sha256,
            "final_parameter_sha256": final_parameter_sha256,
            "final_optimizer_sha256": final_optimizer_sha256,
            "rank_stochastic_initial_states": gathered_streams,
        },
    )


def _execute_ddp(
    *,
    model_name: str,
    seed: int,
    prepared: Any,
    partitions: TrainingPartitions,
    contract: Contract,
    execution_spec: Mapping[str, Any],
    rank: int,
    local_rank: int,
    world_size: int,
    validation_policy: str,
    threshold_population: ThresholdPopulation | None,
    threshold_rule: str,
    threshold_scope: str,
    validation_labels: str,
    transferred_thresholds: Mapping[str, float] | None,
    model_overrides: Mapping[str, Any] | None = None,
) -> ExecutionResult | None:
    import torch
    import torch.distributed as distributed

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
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
    supervised_head = (
        str(model_config.get("supervised_head"))
        if model_name in SUPERVISED_NEURAL_MODELS
        else None
    )
    all_partitions = replace(
        _all_training_partitions(
            model_name,
            prepared,
            supervised_head=supervised_head,
        ),
        test=partitions.test,
    )
    max_epochs = int(contract.run["max_epochs"])
    fit_records: list[Mapping[str, Any]] = []
    histories: list[Mapping[str, Sequence[float]]] = []
    epoch_seconds: list[float] = []
    total_fit_seconds = 0.0

    def record_fit(outcome: DistributedFitOutcome) -> None:
        nonlocal total_fit_seconds
        fit_records.append(dict(outcome.provenance))
        histories.append(dict(outcome.history))
        epoch_seconds.extend(outcome.epoch_seconds)
        total_fit_seconds += outcome.fit_seconds

    training_metadata: dict[str, Any]
    if validation_policy == "none_fixed_epochs":
        final_fit = _distributed_fit_once(
            model_name=model_name,
            seed=seed,
            model_config=model_config,
            train_x=all_partitions.train_x,
            train_y=all_partitions.train_y,
            validation_x=None,
            validation_y=None,
            epochs=max_epochs,
            early_stopping=False,
            stage="fixed_all_training",
            contract=contract,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        record_fit(final_fit)
        training_metadata = {
            "policy": validation_policy,
            "selection": "none",
            "refit": False,
            "selected_epochs": max_epochs,
            "fit_samples": int(all_partitions.train_x.shape[0]),
        }
    elif validation_policy in {"holdout_no_refit", "holdout_refit_b1"}:
        selection_fit = _distributed_fit_once(
            model_name=model_name,
            seed=seed,
            model_config=model_config,
            train_x=partitions.train_x,
            train_y=partitions.train_y,
            validation_x=partitions.validation_x,
            validation_y=partitions.validation_y,
            epochs=max_epochs,
            early_stopping=True,
            stage="holdout_selection",
            contract=contract,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        record_fit(selection_fit)
        selected_epochs = _broadcast_integer(
            _best_epoch(selection_fit.history) if rank == 0 else 0,
            rank=rank,
            device=device,
        )
        if validation_policy == "holdout_no_refit":
            final_fit = selection_fit
            training_metadata = {
                "policy": validation_policy,
                "selection": "single_holdout",
                "refit": False,
                "selected_epochs": selected_epochs,
                "fit_samples": int(partitions.train_x.shape[0]),
                "validation_samples": int(partitions.validation_x.shape[0]),
            }
        else:
            del selection_fit
            torch.cuda.empty_cache()
            final_fit = _distributed_fit_once(
                model_name=model_name,
                seed=seed,
                model_config=model_config,
                train_x=all_partitions.train_x,
                train_y=all_partitions.train_y,
                validation_x=None,
                validation_y=None,
                epochs=selected_epochs,
                early_stopping=False,
                stage="holdout_selected_all_training_refit",
                contract=contract,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
            )
            record_fit(final_fit)
            training_metadata = {
                "policy": validation_policy,
                "selection": "single_holdout",
                "refit": True,
                "selected_epochs": selected_epochs,
                "selection_train_samples": int(partitions.train_x.shape[0]),
                "validation_samples": int(partitions.validation_x.shape[0]),
                "refit_samples": int(all_partitions.train_x.shape[0]),
            }
    elif validation_policy == "crossval_refit_b1":
        split_labels = (
            np.asarray(prepared.supervised_train.labels, dtype=np.int8)
            if model_name in SUPERVISED_NEURAL_MODELS
            else None
        )
        splits = _cross_validation_indices(
            model_name=model_name,
            labels=split_labels,
            rows=int(all_partitions.train_x.shape[0]),
            seed=seed,
        )
        fold_epochs: list[int] = []
        for fold_index, (train_indices, validation_indices) in enumerate(splits):
            fold_fit = _distributed_fit_once(
                model_name=model_name,
                seed=seed,
                model_config=model_config,
                train_x=all_partitions.train_x[train_indices],
                train_y=all_partitions.train_y[train_indices],
                validation_x=all_partitions.train_x[validation_indices],
                validation_y=all_partitions.train_y[validation_indices],
                epochs=max_epochs,
                early_stopping=True,
                stage=f"cross_validation_fold_{fold_index + 1}",
                contract=contract,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
            )
            record_fit(fold_fit)
            fold_epoch = _broadcast_integer(
                _best_epoch(fold_fit.history) if rank == 0 else 0,
                rank=rank,
                device=device,
            )
            fold_epochs.append(fold_epoch)
            del fold_fit
            torch.cuda.empty_cache()
        selected_epochs = _broadcast_integer(
            max(1, int(np.median(fold_epochs))) if rank == 0 else 0,
            rank=rank,
            device=device,
        )
        final_fit = _distributed_fit_once(
            model_name=model_name,
            seed=seed,
            model_config=model_config,
            train_x=all_partitions.train_x,
            train_y=all_partitions.train_y,
            validation_x=None,
            validation_y=None,
            epochs=selected_epochs,
            early_stopping=False,
            stage="cross_validation_selected_all_training_refit",
            contract=contract,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        record_fit(final_fit)
        training_metadata = {
            "policy": validation_policy,
            "selection": "five_fold_or_maximum_feasible_cross_validation",
            "folds": len(splits),
            "fold_selected_epochs": fold_epochs,
            "selected_epoch_aggregation": "integer_median",
            "refit": True,
            "selected_epochs": selected_epochs,
            "refit_samples": int(all_partitions.train_x.shape[0]),
        }
    else:
        raise ValueError(
            f"unsupported distributed validation policy {validation_policy!r}"
        )

    bundle = final_fit.bundle
    final_provenance = dict(final_fit.provenance)

    result: ExecutionResult | None = None
    if rank == 0:
        score_started = time.perf_counter()
        (
            scores,
            predictions,
            metrics,
            score_thresholds,
            score_orientations,
            threshold_selections,
        ) = _score_rank_zero(
            model_name,
            bundle,
            partitions.test,
            prepared,
            contract,
            int(execution_spec["rank_zero_inference_batch_size"]),
            model_config=model_config,
            threshold_population=threshold_population,
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
        score_seconds = time.perf_counter() - score_started
        distinct_orientations = set(score_orientations.values())
        positive_if: str | Mapping[str, str] = (
            next(iter(distinct_orientations))
            if len(distinct_orientations) == 1
            else score_orientations
        )
        result = ExecutionResult(
            scores=scores,
            predictions=predictions,
            labels=np.asarray(partitions.test.labels, dtype=np.int8),
            sample_ids=np.asarray(partitions.test.sample_ids).astype(str),
            is_synthetic=np.asarray(partitions.test.is_synthetic, dtype=bool),
            history={
                "epochs_completed": len(final_fit.history["loss"]),
                "series": dict(final_fit.history),
                "all_fits": histories,
                "epoch_seconds": epoch_seconds,
                "training_policy": training_metadata,
            },
            metrics=metrics,
            fit_seconds=float(total_fit_seconds),
            score_seconds=float(score_seconds),
            metadata={
                "model_config": model_config,
                "parameter_count": int(bundle.model.count_params()),
                "score_thresholds": score_thresholds,
                "positive_if": positive_if,
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
                "training_policy": training_metadata,
                "distributed_fits": fit_records,
                "early_stopping": {
                    "used_for_selection": validation_policy
                    != "none_fixed_epochs",
                    "monitor": "val_loss",
                    "min_delta": float(contract.run["early_stopping_min_delta"]),
                    "patience": int(contract.run["early_stopping_patience"]),
                    "start_from_epoch": int(contract.run["warmup_epochs"]),
                    "restore_best_weights": True,
                },
                "distributed_execution": dict(execution_spec),
                "initial_parameter_sha256": final_provenance[
                    "initial_parameter_sha256"
                ],
                "final_parameter_sha256": final_provenance[
                    "final_parameter_sha256"
                ],
                "final_optimizer_sha256": final_provenance[
                    "final_optimizer_sha256"
                ],
                "rank_stochastic_initial_states": final_provenance[
                    "rank_stochastic_initial_states"
                ],
                **dict(partitions.metadata),
            },
        )
    distributed.barrier()
    return result


def _base_fingerprint_payload(
    model_name: str,
    seed: int,
    prepared: Any,
    contract: Contract,
    verification: Mapping[str, Any],
    *,
    scope: RunScope,
) -> Mapping[str, Any]:
    # Keep the ordinary runner's fingerprint fields, then extend rather than
    # replacing them with distributed-specific identifiers.
    from paper_literal_runner import run_fingerprint

    _, payload = run_fingerprint(
        model_name,
        seed,
        prepared,
        contract,
        verification,
        scope=scope,
    )
    return payload


def _persist_success(
    *,
    logical_dir: Path,
    fingerprint: str,
    fingerprint_payload: Mapping[str, Any],
    result: ExecutionResult,
    model_name: str,
    seed: int,
    contract: Contract,
    verification: Mapping[str, Any],
    prepared: Any,
    data_prep_seconds: float,
    started_utc: str,
    run_started: float,
) -> RunOutcome:
    run_seconds = time.perf_counter() - run_started
    branch_runtime = fingerprint_payload.get("branch_runtime")
    track = (
        "corrected_control"
        if isinstance(branch_runtime, Mapping)
        and branch_runtime.get("track") == "C"
        else "exploratory_paper_literal"
    )
    timings = timing_payload(
        data_prep_seconds=data_prep_seconds,
        run_seconds=run_seconds,
        fit_seconds=result.fit_seconds,
        score_seconds=result.score_seconds,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "study": "atk-2022-deep-autoencoder",
        "track": track,
        "branch_runtime": branch_runtime,
        "table": 2,
        "dataset": "SGCC",
        "model": model_name,
        "seed": int(seed),
        "started_utc": started_utc,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "data_verification": verification,
        "prepared_data_metadata": prepared.metadata,
        "environment": {
            **dict(_environment_metadata()),
            "python_platform": platform.platform(),
            "distributed_world_size": fingerprint_payload["distributed_execution"][
                "world_size"
            ],
        },
        "fingerprint": fingerprint,
        "timings": timings,
        "execution": result.metadata,
        "execution_provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": os.environ.get("ATK_GIT_COMMIT"),
        },
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
    attempt_dir = _persist_attempt(
        logical_dir,
        status="complete",
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        metadata=metadata,
        history=result.history,
        result_summary=summary,
        arrays=_artifact_arrays(result),
    )
    return RunOutcome(model_name, seed, "complete", attempt_dir, fingerprint)


def _persist_failure(
    *,
    logical_dir: Path,
    fingerprint: str,
    fingerprint_payload: Mapping[str, Any],
    model_name: str,
    seed: int,
    contract: Contract,
    verification: Mapping[str, Any],
    prepared: Any,
    data_prep_seconds: float,
    started_utc: str,
    run_started: float,
    exc: BaseException,
) -> RunOutcome:
    elapsed = time.perf_counter() - run_started
    branch_runtime = fingerprint_payload.get("branch_runtime")
    track = (
        "corrected_control"
        if isinstance(branch_runtime, Mapping)
        and branch_runtime.get("track") == "C"
        else "exploratory_paper_literal"
    )
    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    error = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }
    timings = timing_payload(
        data_prep_seconds=data_prep_seconds,
        run_seconds=elapsed,
        failed=True,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "study": "atk-2022-deep-autoencoder",
        "track": track,
        "branch_runtime": branch_runtime,
        "table": 2,
        "dataset": "SGCC",
        "model": model_name,
        "seed": int(seed),
        "started_utc": started_utc,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "data_verification": verification,
        "prepared_data_metadata": prepared.metadata,
        "environment": _environment_metadata(),
        "fingerprint": fingerprint,
        "timings": timings,
        "error": error,
        "execution_provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": os.environ.get("ATK_GIT_COMMIT"),
        },
    }
    attempt_dir = _persist_attempt(
        logical_dir,
        status=status,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        metadata=metadata,
        history={},
        result_summary={
            "status": status,
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
        status,
        attempt_dir,
        fingerprint,
        f"{type(exc).__name__}: {exc}",
    )


def build_parser() -> argparse.ArgumentParser:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Paper 1 production four-GPU SGCC Table-II neural runner"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=study / "config" / "exploratory_reproduction.toml",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--expected-sgcc-sha256", default=EXPECTED_SGCC_SHA256)
    parser.add_argument("--branch-id")
    parser.add_argument(
        "--branch-manifest",
        type=Path,
        default=DEFAULT_LATTICE,
    )
    parser.add_argument(
        "--scaling",
        choices=sorted(SCALING_BRANCHES),
        default="joint_featurewise",
    )
    parser.add_argument(
        "--anomaly-adasyn",
        choices=sorted(ANOMALY_ADASYN_BRANCHES),
        default="test_set_as_printed",
    )
    parser.add_argument(
        "--supervised-adasyn",
        choices=sorted(SUPERVISED_ADASYN_BRANCHES),
        default="before_row_split",
    )
    parser.add_argument("--adasyn-neighbors", type=int)
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
    parser.add_argument(
        "--transferred-thresholds",
        type=Path,
        help=(
            "JSON mapping of score/model names to frozen ISET-derived "
            "thresholds, required for non-printed ISET-to-SGCC transfer"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    import torch.distributed as distributed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    distributed.init_process_group(
        backend="nccl",
        timeout=dt.timedelta(minutes=15),
        device_id=device,
    )
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    fingerprint = ""
    fingerprint_payload: Mapping[str, Any] = {}
    logical_dir: Path | None = None
    prepared: Any = None
    verification: Mapping[str, Any] = {}
    contract: Contract | None = None
    data_prep_seconds = 0.0
    run_started = 0.0
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    model_name = str(args.model)
    runtime_branch: Mapping[str, Any] | None = None
    model_overrides: Mapping[str, Any] = {}
    scope = SGCC_TABLE_II_SCOPE
    try:
        if args.branch_id:
            runtime_branch = load_runtime_branch(
                args.branch_id,
                manifest=args.branch_manifest,
            )
            assert_branch_scope(
                runtime_branch,
                dataset="sgcc",
                table=2,
            )
            args.model = str(runtime_branch["model"])
            for key, value in runtime_branch["preparation"].items():
                if hasattr(args, key):
                    setattr(args, key, value)
            for key, value in runtime_branch["execution"].items():
                if hasattr(args, key):
                    setattr(args, key, value)
            model_overrides = dict(runtime_branch["model_overrides"])
            scope = RunScope(
                2,
                "SGCC",
                (
                    "branches",
                    str(runtime_branch["branch_id"]),
                    "table_2",
                    "sgcc",
                ),
                fingerprint_extra={"branch_runtime": runtime_branch},
            )
        model_name = canonical_model_name(args.model)
        if model_name not in DDP_MODELS:
            raise ValueError("the DDP runner accepts only neural Table-II models")
        if world_size != args.expected_gpus:
            raise RuntimeError(
                f"expected {args.expected_gpus} DDP ranks, received {world_size}"
            )
        contract = load_contract(args.config)
        resolve_seeds(contract, [args.seed])
        global_batch = int(contract.run["batch_size"])
        if global_batch <= 0 or global_batch % world_size:
            raise RuntimeError(
                "the declared global batch must be positive and divide evenly "
                "across DDP ranks"
            )

        prep_started = time.perf_counter()
        prepared, verification, _ = verify_and_prepare_sgcc(
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
        resolved_model_config = dict(contract.raw["table_1"][model_name])
        resolved_model_config.update(model_overrides)
        resolved_model_config.setdefault(
            "supervised_head",
            (
                "sigmoid1_binary"
                if model_name == "supervised_lstm"
                else "softmax2_categorical"
            ),
        )
        partitions = _training_partitions(
            model_name,
            prepared,
            contract,
            args.seed,
            supervised_head=(
                str(resolved_model_config["supervised_head"])
                if model_name in SUPERVISED_NEURAL_MODELS
                else None
            ),
        )
        threshold_population: ThresholdPopulation | None = None
        transferred_thresholds: Mapping[str, float] | None = None
        if args.transferred_thresholds is not None:
            raw_thresholds = json.loads(
                args.transferred_thresholds.read_text(encoding="utf-8")
            )
            if not isinstance(raw_thresholds, dict):
                raise ValueError("transferred-threshold artifact must be a JSON object")
            transferred_thresholds = {
                str(key): float(value) for key, value in raw_thresholds.items()
            }
        if model_name in ANOMALY_NEURAL_MODELS:
            if (
                args.threshold_rule == "printed_constant"
                and args.threshold_scope != "iset_transferred"
            ):
                raise ValueError(
                    "printed_constant is incompatible with dataset_specific"
                )
            if (
                args.threshold_rule != "printed_constant"
                and args.validation_labels == "printed_threshold_no_derivation"
            ):
                raise ValueError(
                    f"{args.threshold_rule} requires labeled validation scores"
                )
            if (
                args.threshold_rule != "printed_constant"
                and args.threshold_scope == "iset_transferred"
                and transferred_thresholds is None
            ):
                raise ValueError(
                    "an ISET-transferred SGCC threshold requires "
                    "--transferred-thresholds"
                )
            preparation_config = dict(prepared.metadata.get("config", {}))
            threshold_population = build_threshold_population(
                prepared,
                branch=args.validation_labels,
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
            partitions = replace(
                partitions,
                test=threshold_population.test_partition,
            )
        branch_execution = {
            "runtime_branch": runtime_branch,
            "model_overrides": model_overrides,
            "validation_policy": args.validation_policy,
            "threshold_rule": args.threshold_rule,
            "threshold_scope": args.threshold_scope,
            "validation_labels": args.validation_labels,
            "transferred_thresholds": transferred_thresholds,
            "threshold_population": (
                {
                    "metadata": dict(threshold_population.metadata),
                    "values_sha256": _array_sha256(
                        threshold_population.values
                    ),
                    "labels_sha256": _array_sha256(
                        threshold_population.labels
                    ),
                    "sample_ids_sha256": _array_sha256(
                        threshold_population.sample_ids.astype(str)
                    ),
                }
                if threshold_population is not None
                else None
            ),
        }
        runtime_provenance = _distributed_runtime_provenance(
            rank=rank, local_rank=local_rank, world_size=world_size
        )
        execution_spec = execution_specification(
            world_size=world_size,
            global_batch=global_batch,
            partitions=partitions,
            runtime_provenance=runtime_provenance,
            branch_execution=branch_execution,
        )
        base_payload = _base_fingerprint_payload(
            model_name,
            args.seed,
            prepared,
            contract,
            verification,
            scope=scope,
        )
        fingerprint, fingerprint_payload = distributed_fingerprint(
            base_payload=base_payload, execution_spec=execution_spec
        )
        fingerprints = _all_rank_values(fingerprint, world_size)
        if len(set(fingerprints)) != 1:
            raise RuntimeError(
                "data preparation or execution fingerprints differ across ranks: "
                f"{fingerprints}"
            )
        distributed.barrier()
        prep_tensor = torch.tensor(
            time.perf_counter() - prep_started,
            dtype=torch.float64,
            device=device,
        )
        distributed.all_reduce(prep_tensor, op=distributed.ReduceOp.MAX)
        data_prep_seconds = float(prep_tensor.item())
        run_started = time.perf_counter()
        logical_dir = _logical_run_dir(
            Path(args.output).resolve(),
            model_name,
            args.seed,
            scope,
        )

        skip_path = ""
        if rank == 0 and not args.force:
            completed = _verified_completed_attempt(logical_dir, fingerprint)
            if completed is not None:
                skip_path = str(completed)
        skipped = _all_rank_values(skip_path if rank == 0 else "", world_size)[0]
        if skipped:
            if rank == 0:
                print(
                    json.dumps(
                        _jsonable(
                            RunOutcome(
                                model_name,
                                args.seed,
                                "skipped_complete",
                                Path(skipped),
                                fingerprint,
                                "matching completed attempt and checksums verified",
                            )
                        ),
                        sort_keys=True,
                    )
                )
            return 0

        if rank == 0:
            preflight = {
                **dict(
                    _preflight_payload(
                        prepared,
                        verification,
                        contract,
                        [model_name],
                        [args.seed],
                        data_prep_seconds,
                    )
                ),
                "distributed_execution": execution_spec,
                "execution_provenance": runtime_provenance,
                "fingerprint": fingerprint,
                "branch_runtime": runtime_branch,
            }
            preflight_path = _write_preflight(Path(args.output), preflight)
            print(json.dumps({**preflight, "artifact": str(preflight_path)}, sort_keys=True))

        result = _execute_ddp(
            model_name=model_name,
            seed=args.seed,
            prepared=prepared,
            partitions=partitions,
            contract=contract,
            execution_spec=execution_spec,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            validation_policy=args.validation_policy,
            threshold_population=threshold_population,
            threshold_rule=args.threshold_rule,
            threshold_scope=args.threshold_scope,
            validation_labels=args.validation_labels,
            transferred_thresholds=transferred_thresholds,
            model_overrides=model_overrides,
        )
        if rank == 0:
            if result is None:
                raise RuntimeError("rank 0 did not receive a scoring result")
            outcome = _persist_success(
                logical_dir=logical_dir,
                fingerprint=fingerprint,
                fingerprint_payload=fingerprint_payload,
                result=result,
                model_name=model_name,
                seed=args.seed,
                contract=contract,
                verification=verification,
                prepared=prepared,
                data_prep_seconds=data_prep_seconds,
                started_utc=started_utc,
                run_started=run_started,
            )
            print(json.dumps(_jsonable(outcome), sort_keys=True))
        distributed.barrier()
        return 0
    except BaseException as exc:
        if (
            rank == 0
            and fingerprint
            and logical_dir is not None
            and prepared is not None
            and contract is not None
        ):
            try:
                outcome = _persist_failure(
                    logical_dir=logical_dir,
                    fingerprint=fingerprint,
                    fingerprint_payload=fingerprint_payload,
                    model_name=model_name,
                    seed=args.seed,
                    contract=contract,
                    verification=verification,
                    prepared=prepared,
                    data_prep_seconds=data_prep_seconds,
                    started_utc=started_utc,
                    run_started=run_started,
                    exc=exc,
                )
                print(json.dumps(_jsonable(outcome), sort_keys=True))
            except Exception as persistence_exc:
                print(
                    json.dumps(
                        {
                            "status": "failure_persistence_failed",
                            "original_error": str(exc),
                            "persistence_error": str(persistence_exc),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
