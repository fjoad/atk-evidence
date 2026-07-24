"""Diagnose the first distributed FC-VAE Adam update without training onward.

This is a bounded failure-localization tool, not a Table-II runner.  It reuses
the production DDP runner's verified data preparation, first-epoch shuffle,
balanced sharding, Keras model, loss, gradient weighting, and compiled optimizer.
Each rank executes exactly the first global batch, reports gradient-square and
post-Adam finite-state diagnostics, and stops.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from paper_literal_ddp import (  # noqa: E402
    _LossModuleFactory,
    _apply_compiled_optimizer,
    _assign_rank_stochastic_streams,
    _assert_identical_trainable_state,
    _broadcast_global_permutation,
    _distributed_error_gate,
    _finite_tensor_code,
    _forward_loss_with_gate,
    _training_partitions,
    ddp_loss_scale,
    shard_indices,
)
from paper_literal_runner import (  # noqa: E402
    EXPECTED_SGCC_SHA256,
    load_contract,
    resolve_seeds,
    verify_and_prepare_sgcc,
)


def _path(variable: Any) -> str:
    return str(getattr(variable, "path", getattr(variable, "name", "<unnamed>")))


def gradient_diagnostics(
    named_gradients: Iterable[tuple[str, Any]],
) -> Mapping[str, Any]:
    """Summarize finite gradients and overflow in their float32 squares."""

    import torch

    gradients = list(named_gradients)
    if not gradients:
        raise ValueError("gradient diagnostics require at least one tensor")
    maximum_path = ""
    maximum_abs = -1.0
    square_nonfinite_count = 0
    square_nonfinite_paths: list[Mapping[str, Any]] = []
    gradient_nonfinite_count = 0
    for path, gradient in gradients:
        detached = gradient.detach()
        finite = torch.isfinite(detached)
        gradient_nonfinite_count += int((~finite).sum().cpu().item())
        finite_values = detached[finite]
        local_max = (
            float(finite_values.abs().max().cpu().item())
            if finite_values.numel()
            else float("nan")
        )
        if local_max > maximum_abs:
            maximum_abs = local_max
            maximum_path = path
        squared = detached * detached
        count = int((~torch.isfinite(squared)).sum().cpu().item())
        square_nonfinite_count += count
        if count:
            square_nonfinite_paths.append({"path": path, "elements": count})
    return {
        "gradient_tensor_count": len(gradients),
        "gradient_nonfinite_elements": gradient_nonfinite_count,
        "max_gradient_path": maximum_path,
        "max_gradient_abs": maximum_abs,
        "gradient_square_nonfinite_elements": square_nonfinite_count,
        "gradient_square_nonfinite_tensors": square_nonfinite_paths,
    }


def nonfinite_state_diagnostics(
    named_tensors: Iterable[tuple[str, Any]],
) -> Mapping[str, Any]:
    """Return names and element counts for tensors containing NaN or Inf."""

    import torch

    records: list[Mapping[str, Any]] = []
    for path, tensor in named_tensors:
        detached = tensor.detach()
        nan_count = int(torch.isnan(detached).sum().cpu().item())
        positive_inf_count = int(torch.isposinf(detached).sum().cpu().item())
        negative_inf_count = int(torch.isneginf(detached).sum().cpu().item())
        nonfinite_count = nan_count + positive_inf_count + negative_inf_count
        if nonfinite_count:
            records.append(
                {
                    "path": path,
                    "elements": nonfinite_count,
                    "nan": nan_count,
                    "positive_inf": positive_inf_count,
                    "negative_inf": negative_inf_count,
                }
            )
    return {
        "nonfinite_tensor_count": len(records),
        "nonfinite_element_count": sum(int(item["elements"]) for item in records),
        "nonfinite_tensors": records,
    }


def _index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def _rank_lines(payload: Mapping[str, Any], world_size: int) -> list[Mapping[str, Any]]:
    import torch.distributed as distributed

    gathered: list[Any] = [None] * world_size
    distributed.all_gather_object(gathered, dict(payload))
    return sorted(gathered, key=lambda item: int(item["rank"]))


def build_parser() -> argparse.ArgumentParser:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="One-batch four-GPU FC-VAE finite-state diagnostic"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--config",
        type=Path,
        default=study / "config" / "exploratory_reproduction.toml",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--expected-sgcc-sha256", default=EXPECTED_SGCC_SHA256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import keras
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel

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
    try:
        if world_size != args.expected_gpus:
            raise RuntimeError(
                f"expected {args.expected_gpus} DDP ranks, received {world_size}"
            )
        contract = load_contract(args.config)
        resolve_seeds(contract, [args.seed])
        global_batch = int(contract.run["batch_size"])
        if global_batch != 512:
            raise RuntimeError("the diagnostic requires frozen global batch 512")

        prepared, _, _ = verify_and_prepare_sgcc(
            args.data,
            contract,
            expected_sha256=args.expected_sgcc_sha256,
        )
        partitions = _training_partitions("fc_vae", prepared, contract, args.seed)

        keras.utils.set_random_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        from paper_literal_models import build_model

        with keras.device(f"cuda:{local_rank}"):
            bundle = build_model(
                "fc_vae",
                int(partitions.train_x.shape[1]),
                dict(contract.raw["table_1"]["fc_vae"]),
                seed=args.seed,
            )
        parallel_model = DistributedDataParallel(
            _LossModuleFactory.build(bundle.model).to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        _assert_identical_trainable_state(
            bundle.model, world_size, "before FC-VAE diagnostic"
        )
        _assign_rank_stochastic_streams(
            bundle.model, "fc_vae", args.seed, rank
        )

        permutation = _broadcast_global_permutation(
            partitions.train_x.shape[0], args.seed, 0, rank, device
        )
        global_indices = permutation[:global_batch]
        local_indices = shard_indices(global_indices, world_size, rank)
        if local_indices.size != global_batch // world_size:
            raise RuntimeError("the first diagnostic batch did not shard equally")
        inputs = torch.from_numpy(
            np.ascontiguousarray(partitions.train_x[local_indices])
        ).to(device, non_blocking=True)
        targets = inputs

        parallel_model.train()
        parallel_model.zero_grad(set_to_none=True)
        loss = _forward_loss_with_gate(
            parallel_model=parallel_model,
            paper_model=bundle.model,
            inputs=inputs,
            targets=targets,
            training=True,
            supervised_probability=False,
            device=device,
            boundary="diagnostic training forward/loss",
        )
        scaled_loss = loss * ddp_loss_scale(
            int(local_indices.size), int(global_indices.size), world_size
        )
        scale_loss = getattr(bundle.model.optimizer, "scale_loss", None)
        if callable(scale_loss):
            scaled_loss = scale_loss(scaled_loss)
        scaled_loss.backward()

        named_gradients = [
            (_path(weight), weight.value.grad)
            for weight in bundle.model.trainable_weights
        ]
        gradient_payload = {
            "diagnostic": "fc_vae_first_ddp_adam_step",
            "stage": "pre_adam",
            "rank": rank,
            "seed": int(args.seed),
            "global_batch": int(global_indices.size),
            "local_batch": int(local_indices.size),
            "local_index_sha256": _index_sha256(local_indices),
            "local_input_max_abs": float(inputs.abs().max().detach().cpu().item()),
            "local_loss": float(loss.detach().cpu().item()),
            **gradient_diagnostics(named_gradients),
        }
        for line in _rank_lines(gradient_payload, world_size):
            if rank == 0:
                print(json.dumps(line, sort_keys=True), flush=True)

        # Match the production runner's pre-optimizer finite gate and compiled
        # Keras optimizer call exactly.  We intentionally inspect rather than
        # reject the post-update state below.
        gradients = [gradient for _, gradient in named_gradients]
        _distributed_error_gate(
            _finite_tensor_code(gradients),
            device,
            "diagnostic post-backward gradient",
        )
        _apply_compiled_optimizer(bundle.model)

        parameter_state = nonfinite_state_diagnostics(
            (_path(weight), weight.value) for weight in bundle.model.trainable_weights
        )
        optimizer_state = nonfinite_state_diagnostics(
            (_path(variable), variable.value)
            for variable in bundle.model.optimizer.variables
        )
        state_payload = {
            "diagnostic": "fc_vae_first_ddp_adam_step",
            "stage": "post_adam",
            "rank": rank,
            "seed": int(args.seed),
            "parameters": parameter_state,
            "optimizer": optimizer_state,
        }
        for line in _rank_lines(state_payload, world_size):
            if rank == 0:
                print(json.dumps(line, sort_keys=True), flush=True)
        distributed.barrier()
        return 0
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
