"""Capped no-training Sigmoid checks; see ../SIGMOID_SANITY.md."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from post_anchor_diagnostics import RESULT_SHA, THRESHOLD, digest
from source_assumption_check import compact_envelope, cube_bounds


REFERENCE_SHA = "3ef1cf59cae7bc7e9dcec2f2d8119b65b221f26b5d2b455b314b14546362dbd8"
CHUNK = 32768
TIME_CAP = 240


def selection(base, total, stage):
    if stage == "full":
        return None, 7 * base
    days = np.unique(np.linspace(0, base - 1, min(64, base), dtype=int))
    synthetic = np.unique(np.linspace(7 * base, total - 1, min(64, total - 7 * base), dtype=int))
    return np.concatenate([days + group * base for group in range(7)] + [synthetic]), 7 * len(days)


def verify_identities(y, attacks, sources, original):
    base = original // 7
    assert len(y) == len(attacks) == len(sources) and original == 7 * base
    assert np.array_equal(y[:original], np.repeat([0, 1, 1, 1, 1, 1, 1], base))
    assert np.array_equal(attacks[:original], np.repeat(np.arange(7), base))
    assert np.all(sources[:base] >= 0)
    assert np.array_equal(sources[:original], np.tile(sources[:base], 7))
    assert np.all(y[original:] == 0)
    assert np.all(sources[original:] == -1) and np.all(attacks[original:] == -1)


def fixed_metrics(y, error, reverse=False):
    flagged = error < THRESHOLD if reverse else error > THRESHOLD
    positive = y == 1
    tp, fp = int(np.sum(flagged & positive)), int(np.sum(flagged & ~positive))
    fn, tn = int(np.sum(~flagged & positive)), int(np.sum(~flagged & ~positive))
    dr, fa = tp / (tp + fn), fp / (fp + tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "DR": 100 * dr, "FA": 100 * fa, "SP": 100 * (1 - fa),
            "PR": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "ACC": 50 * (dr + 1 - fa),
            "F1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def control_summary(y, errors, reverse=False):
    # Equal endpoints enumerate thresholds of this fixed, label-blind score.
    envelope = compact_envelope(y, errors, errors, reverse=reverse)
    metrics = fixed_metrics(y, errors, reverse=reverse)
    metrics["AUC"] = envelope["max_AUC"]
    return {"fixed_cutoff": metrics, "all_cutoffs_diagnostic": envelope}


def run(args, record):
    started = time.perf_counter()
    assert digest(args.result) == RESULT_SHA
    assert digest(args.reference) == REFERENCE_SHA
    result = json.loads(args.result.read_text())
    reference = json.loads(args.reference.read_text())
    data = args.root / result["data"]["path"]
    assert digest(data / "metadata.json") == result["data"]["metadata_sha256"]
    metadata = json.loads((data / "metadata.json").read_text())
    expected = {key: value["sha256"] for key, value in metadata["files"].items()}
    expected.update(result["data"]["files"])
    names = ("x_test.npy", "y_test.npy", "test_attack_id.npy", "test_source_row.npy")
    record["input_sha256"] = {name: digest(data / name) for name in names}
    for name, actual in record["input_sha256"].items():
        assert actual == expected[name], name
    record["metadata_sha256"] = result["data"]["metadata_sha256"]
    record["verification_seconds"] = time.perf_counter() - started
    arrays = {name: np.load(data / name, mmap_mode="r") for name in names}
    x = arrays["x_test.npy"]
    base, total = metadata["counts"]["B2_profiles"], len(x)
    assert base == 750767 and x.shape == (8884989, 48)
    indices, original = selection(base, total, args.stage)
    select = lambda values: np.asarray(values) if indices is None else np.asarray(values[indices])
    y, attacks, sources = [select(arrays[name]) for name in names[1:]]
    verify_identities(y, attacks, sources, original)
    n = len(y)
    record["population"] = {"rows": n, "original_rows": original,
                            "attacks": int(y.sum()), "synthetic_benign": n - original}
    record["selection"] = ("all rows in saved order" if indices is None else
                           hashlib.sha256(indices.tobytes()).hexdigest())
    lower, upper, clipped, half = (np.empty(n, dtype=np.float64) for _ in range(4))
    geometry_at = time.perf_counter()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        values = np.asarray(x[start:stop] if indices is None else x[indices[start:stop]], dtype=np.float64)
        assert np.isfinite(values).all()
        lo, hi = cube_bounds(values)
        constant = np.mean((values - 0.5) ** 2, axis=1)
        assert np.all(lo <= constant + 1e-10) and np.all(constant <= hi + 1e-10)
        epsilon = 1e-5 * (1 + hi)
        lower[start:stop], upper[start:stop] = np.maximum(lo - epsilon, 0), hi + epsilon
        clipped[start:stop], half[start:stop] = lo, constant
    record["geometry_seconds"] = time.perf_counter() - geometry_at
    assert all(np.isfinite(vector).all() for vector in (lower, upper, clipped, half))
    record["views"] = {}
    for view, rows in (("full", slice(None)), ("original", slice(0, original))):
        labels, lo, hi = y[rows], lower[rows], upper[rows]
        record["views"][view] = details = {"bounds": {}, "controls": {}}
        for direction, reverse in (("printed", False), ("reversed_control", True)):
            bound = compact_envelope(labels, lo, hi, reverse=reverse)
            if reverse:
                bound["at_original_error_cutoff"] = {
                    "max_DR": float(100 * np.mean(lo[labels == 1] < THRESHOLD)),
                    "min_FA": float(100 * np.mean(hi[labels == 0] < THRESHOLD))}
            details["bounds"][direction] = bound
        for name, scores in (("clipped_input", clipped), ("constant_half", half)):
            details["controls"][name] = {
                direction: control_summary(labels, scores[rows], reverse=reverse)
                for direction, reverse in (("printed", False), ("reversed_control", True))}
        print(view, "bounds", details["bounds"], flush=True)
    if args.stage == "full":
        for direction in ("printed", "reversed_control"):
            actual = record["views"]["original"]["bounds"][direction]
            expected_bound = reference["branches"]["joint_feature_sigmoid_control"][direction]
            for key in ("max_ACC", "max_AUC"):
                assert abs(actual[key] - expected_bound[key]) < 1e-7
            for cap in ("15.0", "15.5"):
                assert abs(actual["at_FA_cap"][cap]["max_DR"] - expected_bound["at_FA_cap"][cap]["max_DR"]) < 1e-7
        record["original_reference_reproduced"] = True
    record["synthetic_benign_controls"] = {
        name: {"FA_printed": float(100 * np.mean(scores[original:] > THRESHOLD)),
               "FA_reversed": float(100 * np.mean(scores[original:] < THRESHOLD))}
        for name, scores in (("clipped_input", clipped), ("constant_half", half))}
    record["checks_passed"] = ["hashes", "identities", "finite_inputs_and_scores", "constant_containment"]
    record["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "result", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experimental scoring requires a cluster compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    record = {"status": "started", "classification": "X/A exploratory no-training Sigmoid checks",
              "stage": args.stage, "analysis_commit": args.analysis_commit,
              "job_id": os.environ["SLURM_JOB_ID"], "script_sha256": digest(script),
              "contract_sha256": digest(script.parent.parent / "SIGMOID_SANITY.md"),
              "helper_sha256": {name: digest(script.parent / name) for name in
                                ("source_assumption_check.py", "post_anchor_diagnostics.py")},
              "source_result_sha256": RESULT_SHA, "reference_sha256": REFERENCE_SHA,
              "time_cap_seconds": TIME_CAP, "threshold": THRESHOLD,
              "scope": "Unchanged prepared inputs; Sigmoid cube and two no-training controls; not a fitted-model reproduction",
              "numerical_scope": "outward-padded float64 analytic bound, not certified interval arithmetic",
              "versions": {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__}}
    started = time.perf_counter()
    def timeout_handler(signum, frame):
        raise TimeoutError("Predeclared 240-second analysis cap reached")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIME_CAP)
    try:
        run(args, record)
    except Exception:
        record["status"], record["error"] = "failed", traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        record["elapsed_seconds"] = time.perf_counter() - started
        with (args.output / "result.json").open("x") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: record[key] for key in ("status", "stage", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
