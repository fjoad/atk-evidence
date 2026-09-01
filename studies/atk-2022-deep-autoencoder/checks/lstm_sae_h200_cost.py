#!/usr/bin/env python3
"""Bounded H200 timing and decision-stability pilot for CR-ISET-LSTMSAE-01."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import keras
import numpy as np
import torch

import remaining_score_recovery as recovery


pilot = recovery.pilot
CONTRACT = "lstm-sae-h200-cost-v1"
CONTRACT_PATH = recovery.STUDY / "LSTM_SAE_ANCHOR_PROMOTION.md"
SOURCE_MODEL = "lstm_sae"
SOURCE_ATTEMPT_ID = "5f53ca7217aa"
SOURCE_ATTEMPT_SHA256 = recovery.ATTEMPTS[SOURCE_MODEL]
FIT_ROWS = 32_768
SCORE_ROWS = 12_119
FULL_FIT_ROWS = 1_500_523
FULL_SCORE_ROWS = 8_884_989
EPOCHS = 2
BATCH_SIZE = 32
BATCHES = recovery.BATCHES
SAFETY_FACTOR = 1.5
FULL_HOUR_LIMIT = 72.0
AUC_DELTA_LIMIT_PERCENTAGE_POINTS = 0.001
MAX_TRANSFER_LABEL_CHANGES = 1


def runtime_errors() -> list[str]:
    errors: list[str] = []
    if not os.environ.get("SLURM_JOB_ID"):
        errors.append("H200 cost pilot must run inside Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu-H200":
        errors.append("H200 cost pilot requires SLURM_JOB_PARTITION='gpu-H200'")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "16":
        errors.append("H200 cost pilot requires SLURM_CPUS_PER_TASK=16")
    if os.environ.get("SLURM_MEM_PER_NODE") != "98304":
        errors.append("H200 cost pilot requires SLURM_MEM_PER_NODE=98304 MiB")
    if not torch.cuda.is_available():
        errors.append("H200 cost pilot requires one visible CUDA GPU")
    elif torch.cuda.device_count() != 1:
        errors.append(
            f"H200 cost pilot requires one visible GPU, observed {torch.cuda.device_count()}"
        )
    elif "H200" not in torch.cuda.get_device_name(0).upper():
        errors.append(
            f"H200 cost pilot requires an H200, observed {torch.cuda.get_device_name(0)}"
        )
    return errors


def cost_projection(
    *, slowest_epoch_seconds: float, score_seconds: float
) -> dict[str, float | bool]:
    full_epoch_seconds = slowest_epoch_seconds * FULL_FIT_ROWS / FIT_ROWS
    full_score_seconds = score_seconds * FULL_SCORE_ROWS / SCORE_ROWS
    minimum_hours = SAFETY_FACTOR * (
        10 * full_epoch_seconds + full_score_seconds
    ) / 3600.0
    maximum_hours = SAFETY_FACTOR * (
        100 * full_epoch_seconds + full_score_seconds
    ) / 3600.0
    return {
        "method": (
            "1.5_times_slowest_pilot_epoch_scaled_by_steps_plus_"
            "batch_256_score_scaled_by_rows"
        ),
        "slowest_pilot_epoch_seconds": slowest_epoch_seconds,
        "batch_256_score_seconds": score_seconds,
        "projected_full_epoch_seconds": full_epoch_seconds,
        "projected_full_score_seconds": full_score_seconds,
        "minimum_10_epoch_hours": minimum_hours,
        "worst_case_100_epoch_hours": maximum_hours,
        "full_hour_limit": FULL_HOUR_LIMIT,
        "passes_100_epoch_gate": maximum_hours <= FULL_HOUR_LIMIT,
    }


def stability_summary(
    labels: np.ndarray, by_batch: dict[int, np.ndarray]
) -> dict[str, object]:
    threshold = recovery.SPECS[SOURCE_MODEL].threshold
    direction = recovery.SPECS[SOURCE_MODEL].anomaly_direction
    views = {
        batch: recovery.primary_view(
            labels,
            scores,
            threshold=threshold,
            direction=direction,
        )
        for batch, scores in by_batch.items()
    }
    pairwise: dict[str, object] = {}
    maximum_printed_changes = 0
    maximum_auc_delta = 0.0
    maximum_cap_dr_delta = 0.0
    maximum_cap_fa_delta = 0.0
    for left, right in itertools.combinations(BATCHES, 2):
        printed_changes = int(
            np.count_nonzero(
                recovery.strict_predictions(by_batch[left], threshold, direction)
                != recovery.strict_predictions(by_batch[right], threshold, direction)
            )
        )
        auc_delta = abs(
            float(views[right]["envelope"]["AUC"])
            - float(views[left]["envelope"]["AUC"])
        )
        cap_delta: dict[str, object] = {}
        for cap in recovery.FA_CAPS:
            left_cap = views[left]["envelope"]["at_FA_cap"][str(cap)]
            right_cap = views[right]["envelope"]["at_FA_cap"][str(cap)]
            dr_delta = abs(float(right_cap["DR"]) - float(left_cap["DR"]))
            fa_delta = abs(float(right_cap["FA"]) - float(left_cap["FA"]))
            maximum_cap_dr_delta = max(maximum_cap_dr_delta, dr_delta)
            maximum_cap_fa_delta = max(maximum_cap_fa_delta, fa_delta)
            cap_delta[str(cap)] = {"DR": dr_delta, "FA": fa_delta}
        maximum_printed_changes = max(maximum_printed_changes, printed_changes)
        maximum_auc_delta = max(maximum_auc_delta, auc_delta)
        pairwise[f"{left}_vs_{right}"] = {
            "score_difference": recovery.numerical_difference(
                by_batch[left], by_batch[right]
            ),
            "printed_cutoff_label_changes": printed_changes,
            "AUC_delta_percentage_points": auc_delta,
            "FA_cap_absolute_delta_percentage_points": cap_delta,
        }

    reference = views[256]["envelope"]
    threshold_transfer: dict[str, object] = {}
    maximum_transfer_changes = 0
    for cap in recovery.FA_CAPS:
        cutoff = reference["at_FA_cap"][str(cap)]["strict_cutoff"]
        if cutoff is None:
            raise AssertionError("selected H200 reference cutoff is infinite")
        reference_predictions = recovery.strict_predictions(
            by_batch[256], cutoff, direction
        )
        applied: dict[str, object] = {}
        for batch in BATCHES:
            predictions = recovery.strict_predictions(
                by_batch[batch], cutoff, direction
            )
            changes = int(np.count_nonzero(predictions != reference_predictions))
            maximum_transfer_changes = max(maximum_transfer_changes, changes)
            applied[str(batch)] = {
                "label_changes_from_batch_256": changes,
                "metrics": pilot.confusion_metrics(
                    labels,
                    by_batch[batch],
                    threshold=cutoff,
                    direction=direction,
                ),
            }
        threshold_transfer[str(cap)] = {
            "batch_256_strict_cutoff": cutoff,
            "applied_unchanged": applied,
        }

    malicious_row_pp = 100.0 / int(np.count_nonzero(labels == 1))
    benign_row_pp = 100.0 / int(np.count_nonzero(labels == 0))
    gates = {
        "printed_cutoff_zero_label_changes": maximum_printed_changes == 0,
        "fixed_threshold_at_most_one_label_change": (
            maximum_transfer_changes <= MAX_TRANSFER_LABEL_CHANGES
        ),
        "AUC_delta_at_most_0_001_percentage_points": (
            maximum_auc_delta <= AUC_DELTA_LIMIT_PERCENTAGE_POINTS
        ),
        "FA_cap_DR_delta_at_most_one_malicious_row": (
            maximum_cap_dr_delta <= malicious_row_pp + 1e-12
        ),
        "FA_cap_FA_delta_at_most_one_benign_row": (
            maximum_cap_fa_delta <= benign_row_pp + 1e-12
        ),
    }
    return {
        "views": views,
        "pairwise": pairwise,
        "batch_256_threshold_transfer": threshold_transfer,
        "maximums": {
            "printed_cutoff_label_changes": maximum_printed_changes,
            "fixed_threshold_transfer_label_changes": maximum_transfer_changes,
            "AUC_delta_percentage_points": maximum_auc_delta,
            "FA_cap_DR_delta_percentage_points": maximum_cap_dr_delta,
            "FA_cap_FA_delta_percentage_points": maximum_cap_fa_delta,
        },
        "one_row_percentage_points": {
            "malicious": malicious_row_pp,
            "benign": benign_row_pp,
        },
        "gates": gates,
    }


def verify_training_data(
    data: Path, selection: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, str]]:
    metadata_path = data / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_sha = recovery.digest(metadata_path)
    expected_shapes = {
        "x_train.npy": (FULL_FIT_ROWS, 48),
        "x_test.npy": (FULL_SCORE_ROWS, 48),
        "y_test.npy": (FULL_SCORE_ROWS,),
        "table_iv_order.npy": (FULL_FIT_ROWS,),
        "test_attack_id.npy": (FULL_SCORE_ROWS,),
        "test_source_row.npy": (FULL_SCORE_ROWS,),
    }
    expected, _, _ = pilot.remaining_pilot_checksum_manifest(
        metadata, metadata_sha, tuple(expected_shapes)
    )
    observed = {name: recovery.digest(data / name) for name in expected_shapes}
    if observed != expected:
        raise ValueError("H200 cost-pilot input bytes drifted")
    x_train = np.load(data / "x_train.npy", mmap_mode="r")
    if x_train.shape != expected_shapes["x_train.npy"]:
        raise ValueError("H200 cost-pilot training shape drifted")
    values = np.asarray(x_train[selection["fit"]], dtype=np.float32)
    if values.shape != (FIT_ROWS, 48) or not np.isfinite(values).all():
        raise FloatingPointError("H200 cost-pilot training selection is invalid")
    return values, observed


def run(args: argparse.Namespace) -> int:
    errors = runtime_errors()
    if errors:
        raise RuntimeError("H200 cost-pilot runtime mismatch:\n- " + "\n- ".join(errors))
    source_config, selection = recovery.verify_source_attempt(
        SOURCE_MODEL, args.attempt
    )
    x_score, labels, score_input_sha = recovery.verify_and_select_data(
        args.data, selection
    )
    x_fit, all_input_sha = verify_training_data(args.data, selection)
    if tuple(args.batches) != BATCHES:
        raise ValueError(f"score batches must remain {BATCHES}")
    if recovery.digest(recovery.SOURCE_RECORD) != recovery.SOURCE_RECORD_SHA256:
        raise ValueError("committed feasibility record drifted")
    for name, expected in recovery.SOURCE_IMPLEMENTATION_SHA256.items():
        if recovery.digest(recovery.REPRODUCTION / name) != expected:
            raise ValueError(f"source implementation drifted for {name}")

    streams = pilot.remaining_pilot_seed_streams(int(source_config["seed"]))
    configuration: dict[str, object] = {
        "contract": CONTRACT,
        "eligibility": "operational_X_not_N_M_or_A",
        "scientific_method_changed": False,
        "model": SOURCE_MODEL,
        "source_attempt_id": SOURCE_ATTEMPT_ID,
        "source_commit": recovery.SOURCE_COMMIT,
        "diagnostic_commit": pilot.git_commit(),
        "seed": int(source_config["seed"]),
        "seed_streams": streams,
        "fit_rows": FIT_ROWS,
        "score_rows": SCORE_ROWS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "score_batches": list(BATCHES),
        "partition": "gpu-H200",
        "gpu_count": 1,
        "cpus": 16,
        "ram_gib": 96,
        "time_limit_hours": 2,
        "source_attempt_sha256": SOURCE_ATTEMPT_SHA256,
        "source_implementation_sha256": recovery.SOURCE_IMPLEMENTATION_SHA256,
        "source_record_sha256": recovery.SOURCE_RECORD_SHA256,
        "contract_sha256": recovery.digest(CONTRACT_PATH),
        "script_sha256": recovery.digest(Path(__file__)),
    }
    configuration["attempt_id"] = pilot.stable_id(configuration)
    output = (
        args.output
        / "hardware_cost"
        / SOURCE_MODEL
        / f"h200_{configuration['attempt_id']}"
    )
    if output.exists():
        raise RuntimeError(f"immutable H200 cost pilot already exists: {output}")
    output.mkdir(parents=True)
    pilot.save_json(output / "config.json", configuration)
    started = time.perf_counter()
    try:
        determinism = pilot.configure_remaining_pilot_determinism()
        torch.cuda.reset_peak_memory_stats()
        bundle = recovery.build_remaining_paper_model(
            SOURCE_MODEL,
            seed=streams["initialization"],
            latent_seed=streams["latent_training"],
        )
        initial_weight_digest = pilot.weight_digest(bundle.model)
        trace = pilot.FeasibilityTrace()
        keras.utils.set_random_seed(streams["shuffle"])
        fit_started = time.perf_counter()
        history = bundle.model.fit(
            x_fit,
            x_fit,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[trace],
            verbose=2,
        )
        fit_seconds = time.perf_counter() - fit_started
        bundle.model.save_weights(output / "model.weights.h5")
        fitted_weight_digest = pilot.weight_digest(bundle.model)
        if trace.budget_stopped or len(trace.epochs) != EPOCHS:
            raise TimeoutError("H200 cost pilot did not complete two epochs")
        if initial_weight_digest == fitted_weight_digest:
            raise AssertionError("H200 cost-pilot weights did not update")

        del bundle
        keras.backend.clear_session()
        bundle = recovery.build_remaining_paper_model(
            SOURCE_MODEL,
            seed=streams["initialization"],
            latent_seed=streams["latent_training"],
        )
        bundle.model.load_weights(output / "model.weights.h5")
        reloaded_weight_digest = pilot.weight_digest(bundle.model)
        if reloaded_weight_digest != fitted_weight_digest:
            raise AssertionError("H200 saved-weight reload changed fitted weights")

        by_batch: dict[int, np.ndarray] = {}
        score_seconds: dict[str, float] = {}
        score_archive: dict[str, np.ndarray] = {"labels": labels}
        for batch in BATCHES:
            scores, seconds = pilot.score_remaining_bundle(
                bundle,
                x_score,
                batch_size=batch,
                monte_carlo_samples=1,
                scoring_seed=streams["scoring"],
            )
            primary = np.asarray(scores["primary"], dtype=np.float64)
            if not np.isfinite(primary).all():
                raise FloatingPointError(f"H200 batch {batch} scores are nonfinite")
            by_batch[batch] = primary
            score_seconds[str(batch)] = seconds
            score_archive[f"batch_{batch}_primary"] = primary
        np.savez(output / "scores_by_batch.npz", **score_archive)
        stability = stability_summary(labels, by_batch)
        projection = cost_projection(
            slowest_epoch_seconds=max(
                float(epoch["seconds"]) for epoch in trace.epochs
            ),
            score_seconds=score_seconds["256"],
        )
        memory = pilot._memory_snapshot()
        memory_gate = (
            float(memory["peak_rss_fraction"]) <= 0.75
            and memory["peak_gpu_fraction"] is not None
            and memory["peak_gpu_reserved_fraction"] is not None
            and max(
                float(memory["peak_gpu_fraction"]),
                float(memory["peak_gpu_reserved_fraction"]),
            )
            <= 0.75
        )
        gates = {
            "two_epochs_complete": len(trace.epochs) == EPOCHS,
            "weights_updated": initial_weight_digest != fitted_weight_digest,
            "saved_weights_reload_exact": (
                reloaded_weight_digest == fitted_weight_digest
            ),
            **stability["gates"],
            "peak_memory_at_most_75_percent": memory_gate,
            "projected_100_epoch_total_at_most_72_hours": projection[
                "passes_100_epoch_gate"
            ],
        }
        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        pilot.save_json(output / "history.json", history_payload)
        result = {
            "status": "passed" if all(gates.values()) else "gate_failed",
            "eligibility": configuration["eligibility"],
            "configuration": configuration,
            "source": {
                "attempt": str(args.attempt),
                "source_config_sha256": recovery.digest(args.attempt / "config.json"),
                "source_selection_sha256": recovery.digest(args.attempt / "selection.npz"),
                "input_sha256": all_input_sha,
                "score_input_sha256": score_input_sha,
            },
            "runtime": {
                "cuda_device": torch.cuda.get_device_name(0),
                "keras": keras.__version__,
                "torch": torch.__version__,
                "determinism": determinism,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "model": {
                "parameters": int(bundle.model.count_params()),
                "initial_weight_digest": initial_weight_digest,
                "fitted_weight_digest": fitted_weight_digest,
                "reloaded_weight_digest": reloaded_weight_digest,
            },
            "training": {
                "fit_seconds": fit_seconds,
                "trace": trace.epochs,
                "history": history_payload,
                "updates": len(trace.batch_seconds),
            },
            "scoring": {
                "seconds": score_seconds,
                "stability": stability,
            },
            "projection": projection,
            "memory": memory,
            "gates": gates,
            "timing_seconds": {"total": time.perf_counter() - started},
            "decision": (
                "eligible_to_freeze_one_full_anchor"
                if all(gates.values())
                else "stop_without_full_anchor"
            ),
            "artifacts": {
                name: recovery.digest(output / filename)
                for name, filename in {
                    "config.json": "config.json",
                    "history.json": "history.json",
                    "model.weights.h5": "model.weights.h5",
                    "scores_by_batch.npz": "scores_by_batch.npz",
                }.items()
            },
        }
        pilot.save_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "device": result["runtime"]["cuda_device"],
                    "epoch_seconds": [row["seconds"] for row in trace.epochs],
                    "projection": projection,
                    "stability_maximums": stability["maximums"],
                    "gates": gates,
                    "attempt": str(output),
                },
                indent=2,
            )
        )
        return 0 if result["status"] == "passed" else 3
    except Exception as exc:
        pilot.save_json(
            output / "failure.json",
            {
                "status": "failed",
                "eligibility": configuration["eligibility"],
                "configuration": configuration,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
