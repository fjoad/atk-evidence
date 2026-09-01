#!/usr/bin/env python3
"""Decision-level recovery for two failed recurrent feasibility score gates."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import roc_auc_score, roc_curve


STUDY = Path(__file__).resolve().parent.parent
REPRODUCTION = STUDY / "reproduction"
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPRODUCTION))

import run_experiment as pilot  # noqa: E402
from models import SPECS  # noqa: E402
from remaining_models import build_remaining_paper_model  # noqa: E402


CONTRACT = "remaining-score-recovery-v1"
SOURCE_COMMIT = "052ac373b77786ad58829b0ffe35568e971bb92d"
SOURCE_RECORD = STUDY / "results/remaining_pilot_feasibility_20260901.json"
SOURCE_RECORD_SHA256 = (
    "ab1f62e32956e0160d5251ecf7bdbb95a4f64cde56378eb4bc2590c524ea2d79"
)
CONTRACT_PATH = STUDY / "REMAINING_SCORE_RECOVERY.md"
SOURCE_IMPLEMENTATION_SHA256 = {
    "remaining_models.py": (
        "d0d9c42f3b7c846f6f6f1b579fb9fa3b114e3b523fda84dc86754e57c78cb35d"
    ),
    "run_experiment.py": (
        "5443060ee7fefeeaf37a687371652325303230c371eaa2c8d8c423eb1865436f"
    ),
}
INPUT_SHA256 = {
    "x_test.npy": "a942e2affb3d76b5ca3b25f1f023878c2eeda1449b9dbd3d8bce037361a5ece0",
    "y_test.npy": "cec8c078b62557535a0b30ff80da2908115639a0c3c52821a44227378537c887",
    "test_attack_id.npy": (
        "ca76e2e01c830f64de5a6646ddc6d3f1650d2ba81e238f2a00af9ebe4b7dec7b"
    ),
    "test_source_row.npy": (
        "0d2f99d21e27c3b521e3fc4842ccea77673e3ebc4360ab61124134c453ab6964"
    ),
}
EXPECTED_SHAPES = {
    "x_test.npy": (8_884_989, 48),
    "y_test.npy": (8_884_989,),
    "test_attack_id.npy": (8_884_989,),
    "test_source_row.npy": (8_884_989,),
}
ATTEMPTS = {
    "lstm_sae": {
        "attempt_id": "5f53ca7217aa",
        "config_sha256": (
            "1fc10e0d940c35c40c30a0ef292b87a1c51bc43b8311813261b56cd87734ad71"
        ),
        "failure_sha256": (
            "49aea6a2b7f6ab491445846170156a12637f85d7a53d67c97e1770472d4590e6"
        ),
        "weights_sha256": (
            "b26dc72463c75e8e1af73955e7e3021b023a3c80234ea0c20b5218f05ac01b62"
        ),
        "selection_sha256": (
            "5e9a718e6012645296b725ee9e898d5e0cf9542f5d12a1960efa77a3d620e93a"
        ),
    },
    "lstm_vae": {
        "attempt_id": "1d6360ddcead",
        "config_sha256": (
            "b8c798a37cc9e7a3221d8c869c4b598ee4c501ba23969c5d9a71a8227d332db6"
        ),
        "failure_sha256": (
            "5d21e425cbaded402e7df24b40c7d5ee13b368751dd586fef6b2bd56212695d4"
        ),
        "weights_sha256": (
            "11953b8dd4b681f27ebbd2a9fb49ff29397577fce1081487b47f4759dcb5344e"
        ),
        "selection_sha256": (
            "5e9a718e6012645296b725ee9e898d5e0cf9542f5d12a1960efa77a3d620e93a"
        ),
    },
}
BATCHES = (256, 128, 64, 32)
FA_CAPS = (15.0, 15.5)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def strict_predictions(
    scores: np.ndarray, threshold: float, direction: str
) -> np.ndarray:
    if direction == "higher":
        return np.asarray(scores) > threshold
    if direction == "lower":
        return np.asarray(scores) < threshold
    raise ValueError(f"unsupported score direction {direction}")


def score_envelope(
    labels: np.ndarray, scores: np.ndarray, direction: str
) -> dict[str, object]:
    """Complete fixed-score ROC envelope with strict-cutoff representatives."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    oriented = scores if direction == "higher" else -scores
    fpr, tpr, thresholds = roc_curve(
        labels, oriented, drop_intermediate=False
    )
    balanced_accuracy = 50.0 * (1.0 + tpr - fpr)
    best = int(np.argmax(balanced_accuracy))

    def strict_cutoff(index: int) -> float | None:
        oriented_threshold = float(thresholds[index])
        if not np.isfinite(oriented_threshold):
            return None
        return (
            float(np.nextafter(oriented_threshold, -np.inf))
            if direction == "higher"
            else float(np.nextafter(-oriented_threshold, np.inf))
        )

    def metrics_for(index: int, cutoff: float | None) -> dict[str, float | int]:
        applied = (
            (float("inf") if direction == "higher" else float("-inf"))
            if cutoff is None
            else cutoff
        )
        metrics = pilot.confusion_metrics(
            labels, scores, threshold=applied, direction=direction
        )
        if abs(float(metrics["DR"]) - 100.0 * tpr[index]) > 1e-10:
            raise AssertionError("strict ROC cutoff changed detection rate")
        if abs(float(metrics["FA"]) - 100.0 * fpr[index]) > 1e-10:
            raise AssertionError("strict ROC cutoff changed false-alarm rate")
        return metrics

    capped: dict[str, object] = {}
    for cap in FA_CAPS:
        eligible = np.flatnonzero(fpr <= cap / 100.0)
        if not eligible.size:
            raise AssertionError(f"ROC curve has no point under FA cap {cap}")
        selected_tpr = np.max(tpr[eligible])
        tied = eligible[np.flatnonzero(tpr[eligible] == selected_tpr)]
        index = int(tied[-1])
        cutoff = strict_cutoff(index)
        capped[str(cap)] = {
            "DR": float(100.0 * tpr[index]),
            "FA": float(100.0 * fpr[index]),
            "strict_cutoff": cutoff,
            "roc_index": index,
            "metrics": metrics_for(index, cutoff),
        }
    best_cutoff = strict_cutoff(best)
    return {
        "threshold_candidates": int(len(thresholds)),
        "AUC": float(100.0 * roc_auc_score(labels, oriented)),
        "best_balanced_ACC": float(balanced_accuracy[best]),
        "best_roc_index": best,
        "best_strict_cutoff": best_cutoff,
        "best_metrics": metrics_for(best, best_cutoff),
        "at_FA_cap": capped,
        "curve_sha256": hashlib.sha256(
            np.column_stack((fpr, tpr)).astype(np.float64).tobytes()
        ).hexdigest(),
    }


def score_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
    }


def numerical_difference(
    left: np.ndarray, right: np.ndarray
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    absolute = np.abs(left - right)
    score_range = max(float(np.ptp(left)), float(np.ptp(right)))
    return {
        "maximum_absolute": float(np.max(absolute)),
        "mean_absolute": float(np.mean(absolute)),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "maximum_over_score_range": (
            0.0 if score_range == 0.0 else float(np.max(absolute) / score_range)
        ),
        "exactly_equal_rows": int(np.count_nonzero(left == right)),
        "rows": int(left.size),
    }


def metric_delta(
    left: dict[str, float | int], right: dict[str, float | int]
) -> dict[str, float]:
    return {
        name: float(right[name]) - float(left[name])
        for name in ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC")
    }


def primary_view(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> dict[str, object]:
    return {
        "printed_cutoff": threshold,
        "metrics": pilot.confusion_metrics(
            labels, scores, threshold=threshold, direction=direction
        ),
        "positive_predictions": int(
            np.count_nonzero(strict_predictions(scores, threshold, direction))
        ),
        "envelope": score_envelope(labels, scores, direction),
    }


def verify_source_attempt(
    model: str, attempt: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    expected = ATTEMPTS[model]
    required = {
        "config.json": expected["config_sha256"],
        "failure.json": expected["failure_sha256"],
        "model.weights.h5": expected["weights_sha256"],
        "selection.npz": expected["selection_sha256"],
    }
    observed = {name: digest(attempt / name) for name in required}
    if observed != required:
        raise ValueError(
            f"preserved {model} attempt drifted: {observed} versus {required}"
        )
    config = json.loads((attempt / "config.json").read_text())
    failure = json.loads((attempt / "failure.json").read_text())
    if config.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("preserved attempt names a different source commit")
    if config.get("model") != model:
        raise ValueError("preserved attempt model identity drifted")
    if config.get("attempt_id") != expected["attempt_id"]:
        raise ValueError("preserved attempt id drifted")
    if failure.get("status") != "failed" or failure.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("preserved failure record identity drifted")
    with np.load(attempt / "selection.npz") as archive:
        selection = {name: np.asarray(archive[name]) for name in archive.files}
    if set(selection) != {"fit", "score", "source_days"}:
        raise ValueError("preserved selection keys drifted")
    return config, selection


def verify_and_select_data(
    data: Path, selection: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    metadata_path = data / "metadata.json"
    metadata_sha = digest(metadata_path)
    if metadata_sha != "5f3e9d8ea038f8dddede879f73f420a679124cd24a5d2311a2e7e4838a9e869e":
        raise ValueError("prepared metadata identity drifted")
    observed = {name: digest(data / name) for name in EXPECTED_SHAPES}
    if observed != INPUT_SHA256:
        raise ValueError(f"score-recovery input bytes drifted: {observed}")
    arrays = {
        name: np.load(data / name, mmap_mode="r") for name in EXPECTED_SHAPES
    }
    for name, shape in EXPECTED_SHAPES.items():
        if arrays[name].shape != shape:
            raise ValueError(f"prepared shape drifted for {name}")
    pilot.verify_remaining_pilot_identities(
        selection,
        labels=arrays["y_test.npy"],
        attacks=arrays["test_attack_id.npy"],
        sources=arrays["test_source_row.npy"],
        original_group_rows=750_767,
    )
    score_rows = selection["score"]
    values = np.asarray(arrays["x_test.npy"][score_rows], dtype=np.float32)
    labels = np.asarray(arrays["y_test.npy"][score_rows], dtype=np.int8)
    if values.shape != (12_119, 48) or labels.shape != (12_119,):
        raise ValueError("selected score population drifted")
    if not np.isfinite(values).all():
        raise FloatingPointError("selected score inputs are nonfinite")
    return values, labels, observed


def run(args: argparse.Namespace) -> int:
    if args.model not in ATTEMPTS:
        raise ValueError(f"unsupported score-recovery model {args.model}")
    runtime_errors = pilot._pilot_runtime_errors()
    if runtime_errors:
        raise RuntimeError("score-recovery runtime mismatch:\n- " + "\n- ".join(runtime_errors))
    if tuple(args.batches) != BATCHES:
        raise ValueError(f"score batches must remain {BATCHES}")
    if digest(SOURCE_RECORD) != SOURCE_RECORD_SHA256:
        raise ValueError("committed feasibility record drifted")
    for name, expected in SOURCE_IMPLEMENTATION_SHA256.items():
        if digest(REPRODUCTION / name) != expected:
            raise ValueError(f"source implementation drifted for {name}")

    expected_attempt = ATTEMPTS[args.model]
    configuration: dict[str, object] = {
        "contract": CONTRACT,
        "eligibility": "operational_X_not_N_M_or_A",
        "no_training": True,
        "model": args.model,
        "source_commit": SOURCE_COMMIT,
        "source_attempt_id": expected_attempt["attempt_id"],
        "source_attempt_sha256": {
            name: expected_attempt[f"{name.split('.')[0]}_sha256"]
            for name in ("config.json", "failure.json")
        },
        "source_weights_sha256": expected_attempt["weights_sha256"],
        "source_selection_sha256": expected_attempt["selection_sha256"],
        "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
        "source_record_sha256": SOURCE_RECORD_SHA256,
        "prepared_metadata_sha256": (
            "5f3e9d8ea038f8dddede879f73f420a679124cd24a5d2311a2e7e4838a9e869e"
        ),
        "score_batches": list(BATCHES),
        "score_rows": 12_119,
        "monte_carlo_samples": 10 if args.model.endswith("_vae") else 1,
        "score_directions": [SPECS[args.model].anomaly_direction, "reversed_control"],
        "printed_threshold": SPECS[args.model].threshold,
        "diagnostic_commit": pilot.git_commit(),
        "script_sha256": digest(Path(__file__)),
        "contract_sha256": digest(CONTRACT_PATH),
    }
    configuration["attempt_id"] = pilot.stable_id(configuration)
    output = (
        args.output
        / "score_recovery"
        / args.model
        / f"source_{expected_attempt['attempt_id']}_{configuration['attempt_id']}"
    )
    if output.exists():
        raise RuntimeError(f"immutable score recovery already exists: {output}")
    output.mkdir(parents=True)
    pilot.save_json(output / "config.json", configuration)
    started = time.perf_counter()
    try:
        source_config, selection = verify_source_attempt(args.model, args.attempt)
        values, labels, input_sha256 = verify_and_select_data(args.data, selection)
        determinism = pilot.configure_remaining_pilot_determinism()
        streams = pilot.remaining_pilot_seed_streams(int(source_config["seed"]))
        torch.cuda.reset_peak_memory_stats()
        bundle = build_remaining_paper_model(
            args.model,
            seed=streams["initialization"],
            latent_seed=streams["latent_training"],
        )
        bundle.model.load_weights(args.attempt / "model.weights.h5")
        loaded_weight_digest = pilot.weight_digest(bundle.model)

        by_batch: dict[int, dict[str, np.ndarray]] = {}
        batch_details: dict[str, object] = {}
        for batch in BATCHES:
            scores, seconds = pilot.score_remaining_bundle(
                bundle,
                values,
                batch_size=batch,
                monte_carlo_samples=configuration["monte_carlo_samples"],
                scoring_seed=streams["scoring"],
            )
            if not all(np.isfinite(array).all() for array in scores.values()):
                raise FloatingPointError(f"batch {batch} produced nonfinite scores")
            by_batch[batch] = scores
            printed_direction = SPECS[args.model].anomaly_direction
            reversed_direction = (
                "lower" if printed_direction == "higher" else "higher"
            )
            batch_details[str(batch)] = {
                "seconds": seconds,
                "score_summaries": {
                    name: score_summary(array) for name, array in scores.items()
                },
                "printed": primary_view(
                    labels,
                    scores["primary"],
                    threshold=SPECS[args.model].threshold,
                    direction=printed_direction,
                ),
                "reversed_control": primary_view(
                    labels,
                    scores["primary"],
                    threshold=SPECS[args.model].threshold,
                    direction=reversed_direction,
                ),
            }

        comparisons: dict[str, object] = {}
        threshold = SPECS[args.model].threshold
        printed_direction = SPECS[args.model].anomaly_direction
        reversed_direction = "lower" if printed_direction == "higher" else "higher"
        for left_batch, right_batch in itertools.combinations(BATCHES, 2):
            left_scores = by_batch[left_batch]
            right_scores = by_batch[right_batch]
            pair: dict[str, object] = {
                "score_differences": {
                    name: numerical_difference(left_scores[name], right_scores[name])
                    for name in left_scores
                }
            }
            for label, direction in (
                ("printed", printed_direction),
                ("reversed_control", reversed_direction),
            ):
                left_view = batch_details[str(left_batch)][label]
                right_view = batch_details[str(right_batch)][label]
                pair[label] = {
                    "printed_cutoff_label_changes": int(
                        np.count_nonzero(
                            strict_predictions(left_scores["primary"], threshold, direction)
                            != strict_predictions(right_scores["primary"], threshold, direction)
                        )
                    ),
                    "metric_delta": metric_delta(
                        left_view["metrics"], right_view["metrics"]
                    ),
                    "AUC_delta": float(
                        right_view["envelope"]["AUC"]
                        - left_view["envelope"]["AUC"]
                    ),
                    "best_balanced_ACC_delta": float(
                        right_view["envelope"]["best_balanced_ACC"]
                        - left_view["envelope"]["best_balanced_ACC"]
                    ),
                    "FA_cap_delta": {
                        str(cap): {
                            "DR": float(
                                right_view["envelope"]["at_FA_cap"][str(cap)]["DR"]
                                - left_view["envelope"]["at_FA_cap"][str(cap)]["DR"]
                            ),
                            "FA": float(
                                right_view["envelope"]["at_FA_cap"][str(cap)]["FA"]
                                - left_view["envelope"]["at_FA_cap"][str(cap)]["FA"]
                            ),
                        }
                        for cap in FA_CAPS
                    },
                }
            comparisons[f"{left_batch}_vs_{right_batch}"] = pair

        reference_transfer: dict[str, object] = {}
        reference_scores = by_batch[256]["primary"]
        for label, direction in (
            ("printed", printed_direction),
            ("reversed_control", reversed_direction),
        ):
            reference_transfer[label] = {}
            reference_envelope = batch_details["256"][label]["envelope"]
            for cap in FA_CAPS:
                cutoff = reference_envelope["at_FA_cap"][str(cap)]["strict_cutoff"]
                if cutoff is None:
                    raise AssertionError("selected reference FA threshold is infinite")
                reference_predictions = strict_predictions(
                    reference_scores, cutoff, direction
                )
                applied: dict[str, object] = {}
                for batch in BATCHES:
                    scores = by_batch[batch]["primary"]
                    predictions = strict_predictions(scores, cutoff, direction)
                    applied[str(batch)] = {
                        "label_changes_from_batch_256": int(
                            np.count_nonzero(predictions != reference_predictions)
                        ),
                        "metrics": pilot.confusion_metrics(
                            labels, scores, threshold=cutoff, direction=direction
                        ),
                    }
                reference_transfer[label][str(cap)] = {
                    "batch_256_strict_cutoff": cutoff,
                    "applied_unchanged": applied,
                }

        archive = {"labels": labels}
        for batch, scores in by_batch.items():
            for name, array in scores.items():
                archive[f"batch_{batch}_{name}"] = array
        np.savez(output / "scores_by_batch.npz", **archive)
        result = {
            "status": "complete",
            "eligibility": configuration["eligibility"],
            "no_training": True,
            "configuration": configuration,
            "source": {
                "attempt": str(args.attempt),
                "weights_file_sha256": digest(args.attempt / "model.weights.h5"),
                "loaded_weight_digest": loaded_weight_digest,
                "selection_file_sha256": digest(args.attempt / "selection.npz"),
                "input_sha256": input_sha256,
            },
            "population": {
                "rows": int(labels.size),
                "malicious": int(np.count_nonzero(labels == 1)),
                "benign": int(np.count_nonzero(labels == 0)),
            },
            "runtime": {
                "determinism": determinism,
                "cuda_device": torch.cuda.get_device_name(0),
                "keras": keras.__version__,
                "torch": torch.__version__,
            },
            "batches": batch_details,
            "pairwise": comparisons,
            "batch_256_threshold_transfer": reference_transfer,
            "memory": pilot._memory_snapshot(),
            "timing_seconds": {"total": time.perf_counter() - started},
            "artifacts": {
                "config.json": digest(output / "config.json"),
                "scores_by_batch.npz": digest(output / "scores_by_batch.npz"),
            },
            "decision": "stop_for_discussion_without_changing_original_gate",
        }
        pilot.save_json(output / "result.json", result)
        print(json.dumps({
            "status": result["status"],
            "model": args.model,
            "attempt": str(output),
            "pairwise": comparisons,
        }, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "eligibility": configuration["eligibility"],
            "no_training": True,
            "configuration": configuration,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        pilot.save_json(output / "failure.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(ATTEMPTS), required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=list(BATCHES)
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
