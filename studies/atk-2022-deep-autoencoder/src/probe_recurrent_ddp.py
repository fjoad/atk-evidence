"""Exact-shape multi-GPU resource probe for Paper 1 recurrent models.

This is not a result runner.  It uses deterministic subsets of the verified
SGCC preparation to measure whether one frozen global batch fits, whether an
optimizer step changes weights, and how long a few full-shape steps take.  The
default path applies the model's compiled Keras optimizer to DDP-reduced
gradients; a Torch-native comparison remains available for timing diagnostics.
The probe does not claim stochastic, stopping, scoring, or persistence
equivalence.  No probe metric may be reported as Table II reproduction
evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from paper_literal_runner import (  # noqa: E402
    ANOMALY_NEURAL_MODELS,
    EXPECTED_SGCC_SHA256,
    _supervised_fit_split,
    load_contract,
    verify_and_prepare_sgcc,
)


PROBE_MODELS = ("lstm_sae", "lstm_vae", "lstm_aea", "supervised_lstm")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimizer_for_probe(model: Any, optimizer_name: str) -> Any:
    """Match the Keras default optimizer settings used by the frozen model."""

    import torch

    normalized = optimizer_name.strip().lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if normalized == "adam":
        return torch.optim.Adam(
            parameters,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=0.0,
            amsgrad=False,
        )
    if normalized == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=0.01,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
        )
    raise ValueError(f"resource probe does not support optimizer {optimizer_name!r}")


def _parameter_sha256(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _global_probability_diagnostics(outputs: Any) -> tuple[bool, float, float]:
    """Gather one compact probability diagnostic so every rank fails together."""

    import torch
    import torch.distributed as distributed

    finite = torch.isfinite(outputs)
    local = torch.stack(
        (
            torch.logical_not(finite).any().to(dtype=outputs.dtype),
            torch.where(finite, outputs, torch.inf).min(),
            torch.where(finite, outputs, -torch.inf).max(),
        )
    )
    if distributed.is_available() and distributed.is_initialized():
        gathered = [torch.empty_like(local) for _ in range(distributed.get_world_size())]
        distributed.all_gather(gathered, local)
        diagnostics = torch.stack(gathered)
    else:
        diagnostics = local.reshape(1, 3)
    return (
        bool((diagnostics[:, 0] > 0).any().detach().cpu().item()),
        float(diagnostics[:, 1].min().detach().cpu().item()),
        float(diagnostics[:, 2].max().detach().cpu().item()),
    )


def _require_distributed_finite(tensors: Sequence[Any], label: str) -> None:
    """Raise on every rank if any rank observes a non-finite tensor."""

    import torch
    import torch.distributed as distributed

    if not tensors:
        raise RuntimeError(f"cannot check an empty {label} tensor sequence")
    local_bad = any(
        not bool(torch.isfinite(tensor).all().detach().cpu().item())
        for tensor in tensors
    )
    flag = torch.tensor(
        int(local_bad),
        dtype=torch.int32,
        device=tensors[0].device,
    )
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
    if bool(flag.detach().cpu().item()):
        raise FloatingPointError(f"{label} became non-finite on at least one rank")


def _probe_loss(
    model_name: str,
    model: Any,
    inputs: Any,
    targets: Any,
    *,
    training: bool,
) -> Any:
    """Return the local mean loss corresponding to the compiled Keras model."""

    outputs = model(inputs, training=training)
    if model_name == "supervised_lstm":
        nonfinite, minimum, maximum = _global_probability_diagnostics(outputs)
        if nonfinite:
            raise FloatingPointError(
                "supervised LSTM produced a non-finite sigmoid probability on "
                "at least one rank"
            )
        if minimum < 0.0 or maximum > 1.0:
            raise FloatingPointError(
                "supervised LSTM probability left [0, 1]: "
                f"minimum={minimum}, maximum={maximum}"
            )
    loss = model.compute_loss(
        x=inputs,
        y=targets,
        y_pred=outputs,
        training=training,
    )
    if loss is None:
        raise RuntimeError(
            f"the compiled Keras {model_name} model did not produce a loss"
        )
    return loss


class ProbeLossModule:
    """Factory wrapper kept separate so importing this module stays lightweight."""

    @staticmethod
    def build(model_name: str, model: Any) -> Any:
        import torch

        class _LossModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.paper_model = model

            def forward(self, inputs: Any, targets: Any) -> Any:
                return _probe_loss(
                    model_name,
                    self.paper_model,
                    inputs,
                    targets,
                    training=self.training,
                )

        return _LossModule()


def _probe_partitions(
    model_name: str,
    prepared: Any,
    *,
    seed: int,
    validation_fraction: float,
    train_samples: int,
    validation_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if model_name in ANOMALY_NEURAL_MODELS:
        full_train_x = prepared.anomaly_train.values
        full_train_y = full_train_x
        full_validation_x = prepared.anomaly_validation.values
        full_validation_y = full_validation_x
    elif model_name == "supervised_lstm":
        (
            full_train_x,
            full_train_y,
            full_validation_x,
            full_validation_y,
        ) = _supervised_fit_split(
            prepared,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported probe model {model_name!r}")

    if train_samples > full_train_x.shape[0]:
        raise ValueError("probe train sample cap exceeds the paper-literal partition")
    if validation_samples > full_validation_x.shape[0]:
        raise ValueError("probe validation cap exceeds the paper-literal partition")
    return (
        np.ascontiguousarray(full_train_x[:train_samples], dtype=np.float32),
        np.ascontiguousarray(full_train_y[:train_samples], dtype=np.float32),
        np.ascontiguousarray(full_validation_x[:validation_samples], dtype=np.float32),
        np.ascontiguousarray(full_validation_y[:validation_samples], dtype=np.float32),
        int(full_train_x.shape[0]),
        int(full_validation_x.shape[0]),
    )


def _synchronize(torch: Any, device: Any) -> None:
    torch.cuda.synchronize(device)


def _apply_optimizer_step(
    *,
    optimizer_engine: str,
    torch_optimizer: Any,
    paper_model: Any,
) -> None:
    import torch

    if optimizer_engine == "torch":
        torch_optimizer.step()
        return
    weights = list(paper_model.trainable_weights)
    gradients = [weight.value.grad for weight in weights]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("a trainable Keras weight has no DDP-reduced gradient")
    with torch.no_grad():
        paper_model.optimizer.apply(gradients, weights)


def _run_probe(args: argparse.Namespace) -> Mapping[str, Any]:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")

    distributed.init_process_group(backend="nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    import keras

    if keras.backend.backend() != "torch":
        raise RuntimeError("the DDP probe requires KERAS_BACKEND=torch")

    if world_size != args.expected_gpus:
        raise RuntimeError(
            f"expected {args.expected_gpus} DDP ranks, received {world_size}"
        )
    if args.global_batch % world_size:
        raise ValueError("global batch must divide evenly across DDP ranks")
    if args.train_samples % args.global_batch:
        raise ValueError("probe train samples must be a multiple of global batch")
    if args.validation_samples % args.global_batch:
        raise ValueError("probe validation samples must be a multiple of global batch")

    contract = load_contract(args.config)
    if int(contract.run["batch_size"]) != args.global_batch:
        raise ValueError(
            "probe global batch must equal the frozen primary contract batch size"
        )
    prepared, verification, data_prep_seconds = verify_and_prepare_sgcc(
        args.data,
        contract,
        expected_sha256=args.expected_sgcc_sha256,
    )
    (
        train_x,
        train_y,
        validation_x,
        validation_y,
        full_train_samples,
        full_validation_samples,
    ) = _probe_partitions(
        args.model,
        prepared,
        seed=args.seed,
        validation_fraction=float(contract.run["validation_fraction_within_train"]),
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
    )
    input_length = int(train_x.shape[1])
    if input_length != 1034:
        raise RuntimeError(f"expected the full 1,034-feature input, got {input_length}")
    del prepared
    gc.collect()

    from paper_literal_models import build_model

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_config = dict(contract.raw["table_1"][args.model])
    with keras.device(f"cuda:{local_rank}"):
        bundle = build_model(args.model, input_length, model_config, seed=args.seed)
    loss_module = ProbeLossModule.build(args.model, bundle.model).to(device)
    parallel_model = DistributedDataParallel(
        loss_module,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    optimizer = (
        _optimizer_for_probe(parallel_model, model_config["optimizer"])
        if args.optimizer_engine == "torch"
        else None
    )
    trainable_parameters = [
        parameter for parameter in parallel_model.parameters() if parameter.requires_grad
    ]
    parameter_count = int(sum(parameter.numel() for parameter in trainable_parameters))
    first_parameter_before = trainable_parameters[0].detach().clone()

    local_batch = args.global_batch // world_size
    local_train_start = rank * local_batch
    local_validation_start = rank * local_batch

    def local_batch_at(
        values: np.ndarray, targets: np.ndarray, global_step: int, local_start: int
    ) -> tuple[Any, Any]:
        start = global_step * args.global_batch + local_start
        stop = start + local_batch
        x_tensor = torch.from_numpy(values[start:stop]).to(device, non_blocking=True)
        y_tensor = torch.from_numpy(targets[start:stop]).to(device, non_blocking=True)
        return x_tensor, y_tensor

    parallel_model.train()
    warmup_x, warmup_y = local_batch_at(train_x, train_y, 0, local_train_start)
    distributed.barrier()
    _synchronize(torch, device)
    warmup_started = time.perf_counter()
    parallel_model.zero_grad(set_to_none=True)
    warmup_loss = parallel_model(warmup_x, warmup_y)
    if args.model == "supervised_lstm":
        _require_distributed_finite((warmup_loss,), "warm-up loss")
    warmup_loss.backward()
    if args.model == "supervised_lstm":
        _require_distributed_finite(
            tuple(parameter.grad for parameter in trainable_parameters),
            "warm-up gradients",
        )
    _apply_optimizer_step(
        optimizer_engine=args.optimizer_engine,
        torch_optimizer=optimizer,
        paper_model=bundle.model,
    )
    if args.model == "supervised_lstm":
        _require_distributed_finite(trainable_parameters, "warm-up parameters")
        _require_distributed_finite(
            tuple(variable.value for variable in bundle.model.optimizer.variables),
            "warm-up optimizer state",
        )
    _synchronize(torch, device)
    distributed.barrier()
    warmup_seconds = time.perf_counter() - warmup_started
    weight_delta_l2 = float(
        torch.linalg.vector_norm(trainable_parameters[0].detach() - first_parameter_before)
        .cpu()
        .item()
    )
    if not math.isfinite(weight_delta_l2) or weight_delta_l2 <= 0.0:
        raise RuntimeError("warm-up optimizer step did not change the first trainable weight")

    torch.cuda.reset_peak_memory_stats(device)
    train_steps = args.train_samples // args.global_batch
    train_losses: list[float] = []
    distributed.barrier()
    _synchronize(torch, device)
    train_started = time.perf_counter()
    for step in range(train_steps):
        x_tensor, y_tensor = local_batch_at(train_x, train_y, step, local_train_start)
        parallel_model.zero_grad(set_to_none=True)
        loss = parallel_model(x_tensor, y_tensor)
        if args.model == "supervised_lstm":
            _require_distributed_finite((loss,), f"train step {step} loss")
        loss.backward()
        if args.model == "supervised_lstm":
            _require_distributed_finite(
                tuple(parameter.grad for parameter in trainable_parameters),
                f"train step {step} gradients",
            )
        _apply_optimizer_step(
            optimizer_engine=args.optimizer_engine,
            torch_optimizer=optimizer,
            paper_model=bundle.model,
        )
        if args.model == "supervised_lstm":
            _require_distributed_finite(
                trainable_parameters,
                f"train step {step} parameters",
            )
            _require_distributed_finite(
                tuple(variable.value for variable in bundle.model.optimizer.variables),
                f"train step {step} optimizer state",
            )
        train_losses.append(float(loss.detach().cpu().item()))
    _synchronize(torch, device)
    distributed.barrier()
    timed_train_seconds = time.perf_counter() - train_started

    parallel_model.eval()
    validation_steps = args.validation_samples // args.global_batch
    validation_losses: list[float] = []
    distributed.barrier()
    _synchronize(torch, device)
    validation_started = time.perf_counter()
    with torch.no_grad():
        for step in range(validation_steps):
            x_tensor, y_tensor = local_batch_at(
                validation_x, validation_y, step, local_validation_start
            )
            loss = parallel_model(x_tensor, y_tensor)
            validation_losses.append(float(loss.detach().cpu().item()))
    _synchronize(torch, device)
    distributed.barrier()
    timed_validation_seconds = time.perf_counter() - validation_started

    rank_record = {
        "rank": rank,
        "local_rank": local_rank,
        "device_name": torch.cuda.get_device_name(device),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "warmup_loss": float(warmup_loss.detach().cpu().item()),
        "weight_delta_l2": weight_delta_l2,
        "final_train_loss": train_losses[-1],
        "final_validation_loss": validation_losses[-1],
        "parameter_sha256": _parameter_sha256(trainable_parameters),
    }
    rank_records: list[Any] = [None] * world_size
    distributed.all_gather_object(rank_records, rank_record)
    parameter_hashes = {record["parameter_sha256"] for record in rank_records}
    if len(parameter_hashes) != 1:
        raise RuntimeError("DDP ranks ended with different model parameters")

    train_seconds_per_step = timed_train_seconds / train_steps
    validation_seconds_per_step = timed_validation_seconds / validation_steps
    full_train_steps = math.ceil(full_train_samples / args.global_batch)
    full_validation_steps = math.ceil(full_validation_samples / args.global_batch)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "evidentiary_scope": "resource_timing_probe_not_table_ii_result",
        "model": args.model,
        "seed": args.seed,
        "started_from_commit": os.environ.get("ATK_GIT_COMMIT"),
        "host": platform.node(),
        "backend": "keras_torch_ddp",
        "optimizer_execution": args.optimizer_engine,
        "loss_execution": "compiled_keras_compute_loss",
        "supervised_numerical_diagnostics": args.model == "supervised_lstm",
        "versions": {
            "python": sys.version.split()[0],
            "keras": keras.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "data": {
            "verification": verification,
            "preparation_seconds_per_rank": data_prep_seconds,
            "input_features": input_length,
            "probe_train_samples": args.train_samples,
            "probe_validation_samples": args.validation_samples,
            "full_train_samples": full_train_samples,
            "full_validation_samples": full_validation_samples,
        },
        "model_contract": {
            "config_path": str(contract.path),
            "config_sha256": contract.sha256,
            "table_1": model_config,
            "trainable_parameters": parameter_count,
        },
        "distribution": {
            "world_size": world_size,
            "global_batch": args.global_batch,
            "local_batch": local_batch,
            "gradient_reduction": "DDP mean of equal-sized local mean losses",
            "rank_records": rank_records,
        },
        "timing": {
            "warmup_optimizer_step_seconds": warmup_seconds,
            "timed_train_steps": train_steps,
            "timed_train_seconds": timed_train_seconds,
            "train_seconds_per_global_step": train_seconds_per_step,
            "timed_validation_steps": validation_steps,
            "timed_validation_seconds": timed_validation_seconds,
            "validation_seconds_per_global_step": validation_seconds_per_step,
            "full_train_steps_per_epoch": full_train_steps,
            "full_validation_steps_per_epoch": full_validation_steps,
            "extrapolated_full_epoch_seconds": (
                full_train_steps * train_seconds_per_step
                + full_validation_steps * validation_seconds_per_step
            ),
            "partial_final_steps_approximated_as_full_steps": True,
            "extrapolation_is_not_a_result": True,
        },
        "source_sha256": {
            name: _sha256_path(Path(__file__).resolve().parent / name)
            for name in (
                "probe_recurrent_ddp.py",
                "paper_literal_models.py",
                "paper_literal_runner.py",
                "paper_literal_data.py",
            )
        },
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    distributed.destroy_process_group()
    return payload


def build_parser() -> argparse.ArgumentParser:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROBE_MODELS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--config", type=Path, default=study / "config/exploratory_reproduction.toml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument(
        "--optimizer-engine",
        choices=("keras", "torch"),
        default="keras",
        help="use the compiled Keras optimizer by default; torch is a timing-only comparison",
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--expected-sgcc-sha256", default=EXPECTED_SGCC_SHA256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output.expanduser().resolve() / f"{started}-{args.model}-seed-{args.seed}.json"
    try:
        payload = _run_probe(args)
    except BaseException as exc:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "evidentiary_scope": "resource_timing_probe_not_table_ii_result",
            "model": args.model,
            "seed": args.seed,
            "rank": rank,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if rank == 0:
            _atomic_json(output, payload)
        print(json.dumps(payload, allow_nan=False, sort_keys=True), flush=True)
        return 1
    if rank == 0:
        _atomic_json(output, payload)
        print(json.dumps({**payload, "artifact": str(output)}, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
