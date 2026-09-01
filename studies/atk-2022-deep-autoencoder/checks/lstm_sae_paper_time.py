#!/usr/bin/env python3
"""Run CR-ISET-LSTMSAE-01 inside the paper's 183-minute fit budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from sklearn.metrics import roc_auc_score, roc_curve

CHECKS = Path(__file__).resolve().parent
sys.path.insert(0, str(CHECKS))
import remaining_score_recovery as recovery


pilot = recovery.pilot
STUDY = recovery.STUDY
REPRODUCTION = recovery.REPRODUCTION
CONTRACT = "lstm-sae-paper-time-v1"
CONTRACT_PATH = STUDY / "PAPER_TIME_BUDGET_CONTRACT.md"
MODEL = "lstm_sae"
SOURCE_ATTEMPT_ID = "5f53ca7217aa"
FULL_FIT_ROWS = 1_500_523
FULL_SCORE_ROWS = 8_884_989
FEATURES = 48
BATCH_SIZE = 32
MAX_EPOCHS = 100
FIT_SECONDS_LIMIT = 183 * 60
SCORE_BATCH = 256
FA_CAPS = (13.0, 13.5, 15.0, 15.5)
REPORTED = pilot.REPORTED[MODEL]
EXPECTED_SHAPES = {
    "x_train.npy": (FULL_FIT_ROWS, FEATURES),
    "x_test.npy": (FULL_SCORE_ROWS, FEATURES),
    "y_test.npy": (FULL_SCORE_ROWS,),
    "table_iv_order.npy": (FULL_FIT_ROWS,),
    "test_attack_id.npy": (FULL_SCORE_ROWS,),
    "test_source_row.npy": (FULL_SCORE_ROWS,),
}


def runtime_errors() -> list[str]:
    errors: list[str] = []
    if not os.environ.get("SLURM_JOB_ID"):
        errors.append("paper-time run must execute inside Slurm")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu-all":
        errors.append("paper-time run requires partition gpu-all")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "16":
        errors.append("paper-time run requires 16 CPUs")
    if os.environ.get("SLURM_MEM_PER_NODE") != "98304":
        errors.append("paper-time run requires 98304 MiB RAM")
    if not torch.cuda.is_available():
        errors.append("paper-time run requires one visible CUDA GPU")
    elif torch.cuda.device_count() != 1:
        errors.append(
            f"paper-time run requires one GPU, observed {torch.cuda.device_count()}"
        )
    elif "V100" not in torch.cuda.get_device_name(0).upper():
        errors.append(
            f"paper-time run requires a V100, observed {torch.cuda.get_device_name(0)}"
        )
    return errors


class PaperTimeTrace(keras.callbacks.Callback):
    """Stop at the first completed batch at or beyond the paper's fit budget."""

    def __init__(self, *, limit_seconds: float, batches_per_epoch: int) -> None:
        super().__init__()
        self.limit_seconds = float(limit_seconds)
        self.batches_per_epoch = int(batches_per_epoch)
        self.epochs: list[dict[str, object]] = []
        self.batch_seconds: list[float] = []
        self.batch_losses: list[float] = []
        self.budget_stopped = False
        self.current_epoch_batches = 0

    def on_train_begin(self, logs: dict[str, float] | None = None) -> None:
        del logs
        self.fit_started = time.perf_counter()

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, float] | None = None
    ) -> None:
        del epoch, logs
        self.epoch_started = time.perf_counter()
        self.current_epoch_batches = 0

    def on_train_batch_begin(
        self, batch: int, logs: dict[str, float] | None = None
    ) -> None:
        del batch, logs
        self.batch_started = time.perf_counter()

    def on_train_batch_end(
        self, batch: int, logs: dict[str, float] | None = None
    ) -> None:
        del batch
        elapsed = time.perf_counter() - self.batch_started
        loss = None if logs is None else logs.get("loss")
        if loss is None or not np.isfinite(float(loss)):
            raise FloatingPointError("paper-time fit produced a nonfinite objective")
        self.batch_seconds.append(float(elapsed))
        self.batch_losses.append(float(loss))
        self.current_epoch_batches += 1
        if time.perf_counter() - self.fit_started >= self.limit_seconds:
            self.budget_stopped = True
            self.model.stop_training = True

    def on_epoch_end(
        self, epoch: int, logs: dict[str, float] | None = None
    ) -> None:
        values = {
            key: float(value) for key, value in ({} if logs is None else logs).items()
        }
        if not values or not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("paper-time epoch summary is nonfinite")
        self.epochs.append(
            {
                "epoch": int(epoch + 1),
                "seconds": float(time.perf_counter() - self.epoch_started),
                "batches": int(self.current_epoch_batches),
                "complete": self.current_epoch_batches == self.batches_per_epoch,
                **values,
            }
        )

    def summary(self, fit_seconds: float) -> dict[str, object]:
        complete = [row for row in self.epochs if bool(row["complete"])]
        partial = [row for row in self.epochs if not bool(row["complete"])]
        return {
            "fit_seconds_limit": self.limit_seconds,
            "fit_seconds_observed": float(fit_seconds),
            "budget_overshoot_seconds": max(0.0, float(fit_seconds) - self.limit_seconds),
            "budget_stopped": self.budget_stopped,
            "batches_per_complete_epoch": self.batches_per_epoch,
            "updates": len(self.batch_seconds),
            "complete_epochs": len(complete),
            "partial_epochs": len(partial),
            "epochs": self.epochs,
            "batch_seconds": {
                "minimum": float(np.min(self.batch_seconds)),
                "median": float(np.median(self.batch_seconds)),
                "maximum": float(np.max(self.batch_seconds)),
            },
            "last_batch_loss": self.batch_losses[-1],
        }


def verify_full_data(
    data: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    metadata_path = data / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_sha = recovery.digest(metadata_path)
    if metadata_sha != "5f3e9d8ea038f8dddede879f73f420a679124cd24a5d2311a2e7e4838a9e869e":
        raise ValueError("paper-time prepared metadata identity drifted")
    expected, sources, manifest = pilot.remaining_pilot_checksum_manifest(
        metadata, metadata_sha, tuple(EXPECTED_SHAPES)
    )
    observed = {name: recovery.digest(data / name) for name in EXPECTED_SHAPES}
    if observed != expected:
        raise ValueError("paper-time prepared input bytes drifted")
    arrays = {
        name: np.load(data / name, mmap_mode="r") for name in EXPECTED_SHAPES
    }
    for name, shape in EXPECTED_SHAPES.items():
        if arrays[name].shape != shape:
            raise ValueError(f"paper-time prepared shape drifted for {name}")
    labels = arrays["y_test.npy"]
    if int(np.count_nonzero(labels == 0)) != 4_380_387:
        raise ValueError("paper-time benign score population drifted")
    if int(np.count_nonzero(labels == 1)) != 4_504_602:
        raise ValueError("paper-time malicious score population drifted")
    return (
        arrays["x_train.npy"],
        arrays["x_test.npy"],
        labels,
        {
            "metadata_sha256": metadata_sha,
            "input_sha256": observed,
            "checksum_sources": sources,
            "checksum_manifest": manifest,
        },
    )


def _strict_cutoff(threshold: float, direction: str) -> float | None:
    if not np.isfinite(threshold):
        return None
    return (
        float(np.nextafter(threshold, -np.inf))
        if direction == "higher"
        else float(np.nextafter(-threshold, np.inf))
    )


def score_envelope(
    labels: np.ndarray, scores: np.ndarray, *, direction: str
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    oriented = scores if direction == "higher" else -scores
    fpr, tpr, thresholds = roc_curve(labels, oriented, drop_intermediate=False)
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    tp = tpr * positives
    fp = fpr * negatives
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    specificity = 1.0 - fpr
    balanced = 0.5 * (tpr + specificity)
    f1 = np.divide(
        2.0 * tpr * precision,
        tpr + precision,
        out=np.zeros_like(tpr),
        where=(tpr + precision) > 0,
    )
    auc = float(100.0 * roc_auc_score(labels, oriented))
    matrix = 100.0 * np.column_stack(
        (tpr, fpr, specificity, precision, balanced, f1)
    )
    target = np.asarray(
        [REPORTED[name] for name in ("DR", "FA", "SP", "PR", "ACC", "F1")],
        dtype=np.float64,
    )
    metric_delta = matrix - target
    auc_delta = auc - float(REPORTED["AUC"])
    maximum_gap = np.maximum(np.max(np.abs(metric_delta), axis=1), abs(auc_delta))
    euclidean_gap = np.sqrt(np.sum(np.square(metric_delta), axis=1) + auc_delta**2)
    best_max = int(np.argmin(maximum_gap))
    best_l2 = int(np.argmin(euclidean_gap))
    best_balanced = int(np.argmax(balanced))

    def selected(index: int) -> dict[str, object]:
        cutoff = _strict_cutoff(float(thresholds[index]), direction)
        applied = (
            (float("inf") if direction == "higher" else float("-inf"))
            if cutoff is None
            else cutoff
        )
        return {
            "roc_index": index,
            "strict_cutoff": cutoff,
            "metrics": pilot.confusion_metrics(
                labels, scores, threshold=applied, direction=direction
            ),
            "maximum_absolute_gap_percentage_points": float(maximum_gap[index]),
            "euclidean_gap_percentage_points": float(euclidean_gap[index]),
        }

    caps: dict[str, object] = {}
    for cap in FA_CAPS:
        eligible = np.flatnonzero(fpr <= cap / 100.0)
        chosen_tpr = np.max(tpr[eligible])
        ties = eligible[np.flatnonzero(tpr[eligible] == chosen_tpr)]
        index = int(ties[-1])
        caps[str(cap)] = selected(index)

    curve_digest = hashlib.sha256()
    for array in (fpr, tpr, thresholds):
        curve_digest.update(np.asarray(array, dtype=np.float64).tobytes())
    return {
        "direction": direction,
        "threshold_candidates": int(len(thresholds)),
        "AUC": auc,
        "best_balanced_accuracy": selected(best_balanced),
        "smallest_maximum_metric_gap": selected(best_max),
        "smallest_euclidean_metric_gap": selected(best_l2),
        "at_FA_cap": caps,
        "reported_corner_reached": (
            float(caps["13.0"]["metrics"]["DR"]) >= float(REPORTED["DR"])
        ),
        "rounded_corner_reached": (
            float(caps["13.5"]["metrics"]["DR"])
            >= float(REPORTED["DR"]) - 0.5
        ),
        "curve_sha256": curve_digest.hexdigest(),
    }


def run(args: argparse.Namespace) -> int:
    errors = runtime_errors()
    if errors:
        raise RuntimeError("paper-time runtime mismatch:\n- " + "\n- ".join(errors))
    source_config, _ = recovery.verify_source_attempt(MODEL, args.attempt)
    if recovery.digest(recovery.SOURCE_RECORD) != recovery.SOURCE_RECORD_SHA256:
        raise ValueError("paper-time feasibility record drifted")
    for name, expected in recovery.SOURCE_IMPLEMENTATION_SHA256.items():
        if recovery.digest(REPRODUCTION / name) != expected:
            raise ValueError(f"paper-time source implementation drifted for {name}")
    x_train, x_test, labels, data_identity = verify_full_data(args.data)
    streams = pilot.remaining_pilot_seed_streams(int(source_config["seed"]))
    configuration: dict[str, object] = {
        "contract": CONTRACT,
        "eligibility": "time_bounded_N_and_A_for_one_declared_completion",
        "model": MODEL,
        "source_attempt_id": SOURCE_ATTEMPT_ID,
        "source_commit": recovery.SOURCE_COMMIT,
        "run_commit": pilot.git_commit(),
        "seed": int(source_config["seed"]),
        "seed_streams": streams,
        "fit_rows": FULL_FIT_ROWS,
        "score_rows": FULL_SCORE_ROWS,
        "features": FEATURES,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "fit_seconds_limit": FIT_SECONDS_LIMIT,
        "score_batch": SCORE_BATCH,
        "partition": "gpu-all",
        "gpu_type": "v100_16GB",
        "gpu_count": 1,
        "cpus": 16,
        "ram_gib": 96,
        "slurm_time_limit_hours": 6,
        "printed_threshold": recovery.SPECS[MODEL].threshold,
        "printed_direction": recovery.SPECS[MODEL].anomaly_direction,
        "reported_table_3": REPORTED,
        "reported_training_minutes": 183,
        "source_implementation_sha256": recovery.SOURCE_IMPLEMENTATION_SHA256,
        "source_record_sha256": recovery.SOURCE_RECORD_SHA256,
        "contract_sha256": recovery.digest(CONTRACT_PATH),
        "script_sha256": recovery.digest(Path(__file__)),
    }
    configuration["attempt_id"] = pilot.stable_id(configuration)
    output = (
        args.output
        / "paper_time"
        / MODEL
        / f"v100_{configuration['attempt_id']}"
    )
    if output.exists():
        raise RuntimeError(f"immutable paper-time attempt already exists: {output}")
    output.mkdir(parents=True)
    pilot.save_json(output / "config.json", configuration)
    started = time.perf_counter()
    try:
        determinism = pilot.configure_remaining_pilot_determinism()
        torch.cuda.reset_peak_memory_stats()
        bundle = recovery.build_remaining_paper_model(
            MODEL,
            seed=streams["initialization"],
            latent_seed=streams["latent_training"],
        )
        initial_weight_digest = pilot.weight_digest(bundle.model)
        trace = PaperTimeTrace(
            limit_seconds=FIT_SECONDS_LIMIT,
            batches_per_epoch=math.ceil(FULL_FIT_ROWS / BATCH_SIZE),
        )
        keras.utils.set_random_seed(streams["shuffle"])
        fit_started = time.perf_counter()
        history = bundle.model.fit(
            x_train,
            x_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[trace],
            verbose=2,
        )
        fit_seconds = time.perf_counter() - fit_started
        training = trace.summary(fit_seconds)
        if not trace.budget_stopped and len(trace.epochs) != MAX_EPOCHS:
            raise AssertionError(
                "paper-time fit ended before either its time boundary or maximum epochs"
            )
        training["termination"] = (
            "paper_time_boundary" if trace.budget_stopped else "maximum_epochs"
        )
        bundle.model.save_weights(output / "model.weights.h5")
        fitted_weight_digest = pilot.weight_digest(bundle.model)
        if initial_weight_digest == fitted_weight_digest:
            raise AssertionError("paper-time model weights did not update")
        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        pilot.save_json(output / "history.json", history_payload)
        pilot.save_json(output / "training_trace.json", training)

        del bundle
        keras.backend.clear_session()
        bundle = recovery.build_remaining_paper_model(
            MODEL,
            seed=streams["initialization"],
            latent_seed=streams["latent_training"],
        )
        bundle.model.load_weights(output / "model.weights.h5")
        reloaded_weight_digest = pilot.weight_digest(bundle.model)
        if reloaded_weight_digest != fitted_weight_digest:
            raise AssertionError("paper-time saved weights changed on reload")

        scores, score_seconds = pilot.score_mse(
            bundle.model,
            x_test,
            output / "scores.npy",
            batch_size=SCORE_BATCH,
            score_kind="mse",
        )
        if scores.shape != (FULL_SCORE_ROWS,) or not np.isfinite(scores).all():
            raise FloatingPointError("paper-time full scores are invalid")
        printed = pilot.confusion_metrics(
            labels,
            scores,
            threshold=recovery.SPECS[MODEL].threshold,
            direction=recovery.SPECS[MODEL].anomaly_direction,
        )
        paper_envelope = score_envelope(labels, scores, direction="higher")
        reversed_envelope = score_envelope(labels, scores, direction="lower")
        memory = pilot._memory_snapshot()
        result = {
            "status": "completed",
            "eligibility": configuration["eligibility"],
            "configuration": configuration,
            "source": data_identity,
            "runtime": {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_device_memory_bytes": int(
                    torch.cuda.get_device_properties(0).total_memory
                ),
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
            "training": training,
            "history": history_payload,
            "scoring": {
                "seconds": float(score_seconds),
                "batch_size": SCORE_BATCH,
                "score_summary": recovery.score_summary(scores),
            },
            "printed_cutoff_metrics": printed,
            "paper_direction_envelope": paper_envelope,
            "reversed_direction_envelope": reversed_envelope,
            "reported_result_reproduced_at_printed_cutoff": all(
                abs(float(printed[name]) - float(REPORTED[name])) <= 0.5
                for name in REPORTED
            ),
            "reported_corner_reached_at_any_cutoff": paper_envelope[
                "reported_corner_reached"
            ],
            "memory": memory,
            "timing_seconds": {"total": time.perf_counter() - started},
            "decision_language": (
                "reported_result_recovered_within_declared_time_and_completion"
                if paper_envelope["reported_corner_reached"]
                else "reported_result_not_recovered_within_declared_time_and_completion"
            ),
        }
        result["artifacts"] = {
            name: recovery.digest(output / filename)
            for name, filename in {
                "config.json": "config.json",
                "history.json": "history.json",
                "training_trace.json": "training_trace.json",
                "model.weights.h5": "model.weights.h5",
                "scores.npy": "scores.npy",
            }.items()
        }
        pilot.save_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "device": result["runtime"]["cuda_device"],
                    "training": training,
                    "printed_cutoff_metrics": printed,
                    "paper_direction_at_FA_13": paper_envelope["at_FA_cap"]["13.0"],
                    "decision_language": result["decision_language"],
                    "attempt": str(output),
                },
                indent=2,
            )
        )
        return 0
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
