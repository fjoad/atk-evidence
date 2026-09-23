#!/usr/bin/env python3
"""Read-only authorization/identity checks for the same-seed GRU completion."""

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
from analyze_results import audit_result, digest, verify_source

PREVIOUS_SHA256 = "fa19293a1bac847fd7fb3ab8fe37b25f3ea92f52d88ad8e53cee129bbbdd4602"
UNCHANGED_FUNCTIONS = ("fit_gru", "sequence_inputs", "normalized_softmax", "gru_kernel_norms",
                       "configure_tensorflow", "load_preparation", "weight_hash", "require_compute")


def runtime_budget(history):
    times = [r["seconds"] for r in history if r["epoch_complete"]]
    if not times or not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError("Need finite positive full-epoch timing")
    projection = 50 * max(times)
    guard = 300 * math.ceil(projection / 300)
    return {"full_epochs_measured": len(times), "slowest_epoch_seconds": max(times),
            "projected_50_epochs_from_slowest_seconds": projection,
            "projected_50_epochs_from_mean_seconds": 50 * sum(times) / len(times),
            "fit_guard_seconds": guard, "job_limit_seconds": 2 * guard + 900}


def scientific_functions_unchanged(before, after):
    functions = lambda source: {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(before), functions(after)
    for name in UNCHANGED_FUNCTIONS:
        if old.get(name) is None or old[name] != new.get(name):
            raise ValueError(f"Scientific function changed: {name}")


def verify(previous):
    directory = previous / "p00"
    if digest(directory / "result.json") != PREVIOUS_SHA256:
        raise ValueError("Prior partial result identity changed")
    r = json.loads((directory / "result.json").read_text())
    if r["status"] != "partial" or r["neural"]["epochs_completed"] != 32 or (previous / "p30").exists():
        raise ValueError("Unexpected prior attempt scope")
    for name, expected in r["output_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Prior artifact changed: {name}")
    for relative, expected in r["source_sha256"].items():
        verify_source(relative, expected, r["code_commit"])
        if relative != "reproduction/run_experiment.py" and digest(STUDY / relative) != expected:
            raise ValueError(f"Scientific file changed: {relative}")
    relative = "studies/takiddin-2021-robust-poisoning/reproduction/run_experiment.py"
    before = subprocess.run(["git", "-C", str(STUDY.parents[1]), "show", f"{r['code_commit']}:{relative}"],
                            check=True, capture_output=True, text=True).stdout
    scientific_functions_unchanged(before, (STUDY / "reproduction/run_experiment.py").read_text())
    budget = runtime_budget(json.loads((directory / "history.json").read_text()))
    if budget["fit_guard_seconds"] != 2100 or budget["job_limit_seconds"] != 5100:
        raise ValueError("Measured budget differs from the approved 35/85-minute completion")
    return {"status": "verified_same_science_revised_operational_budget", "prior_result_sha256": PREVIOUS_SHA256,
            "prior_commit": r["code_commit"], "budget": budget,
            "unchanged_scientific_functions": list(UNCHANGED_FUNCTIONS),
            "approval_sha256": digest(STUDY / "GRU_COMPLETION.md"),
            "initial_weights_sha256_required": r["neural"]["initial_weights_sha256"]}


def compare_prefix(previous, completed):
    old = json.loads((previous / "p00/result.json").read_text())
    audited = audit_result(completed)
    new = json.loads((completed / "result.json").read_text())
    if (digest(previous / "p00/result.json") != PREVIOUS_SHA256
            or old["neural"]["initial_weights_sha256"] != new["neural"]["initial_weights_sha256"]):
        raise ValueError("Same-seed initialization identity changed")
    if new["execution_budget"]["fit_guard_seconds"] != 2100 or new["neural"]["epochs_completed"] != 50:
        raise ValueError("Completion schedule/budget differs")
    old_h = json.loads((previous / "p00/history.json").read_text())[:32]
    new_h = json.loads((completed / "history.json").read_text())[:32]
    keys = ("epoch", "loss", "categorical_accuracy", "optimizer_updates", "epoch_complete")
    mismatches = [a["epoch"] for a, b in zip(old_h, new_h) if any(a[k] != b[k] for k in keys)]
    if len(new_h) != 32:
        raise ValueError("Missing repeated training prefix")
    return {"status": "compared_without_selecting_a_run", "completed_result_sha256": audited["result_sha256"],
            "initial_weights_match_partial": True, "first_32_epoch_metrics_match_exactly": not mismatches,
            "mismatched_epoch_numbers": mismatches, "timing_compared_for_equality": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--completed-p00", type=Path)
    args = parser.parse_args()
    result = compare_prefix(args.previous, args.completed_p00) if args.completed_p00 else verify(args.previous)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
