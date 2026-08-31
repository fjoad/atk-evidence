"""Bounded no-training source sensitivity; see SOURCE_ASSUMPTION_CHECK.md."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from post_anchor_diagnostics import RESULT_SHA, digest, oracle_envelope, simplex_bounds


BRANCHES = ("joint_feature_softmax", "joint_scalar_softmax",
            "separate_class_feature_softmax", "joint_feature_sigmoid_control")
REFERENCE_SHA = "0e337db5a7f424dafc46cc3aac0643b5fac77c61799b0558b2434143bf5cd372"
CHUNK = 32768


def moments(values):
    total = np.zeros(values.shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(values), CHUNK):
        block = np.asarray(values[start:start + CHUNK], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite raw inputs")
        total += block.sum(axis=0)
        squares += (block * block).sum(axis=0)
    return total / len(values), squares / len(values)


def scales_from_moments(mean, second):
    variance = np.asarray(second) - np.asarray(mean) ** 2
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Expected strictly positive finite feature variance")
    return np.sqrt(variance)


def alternative_scalers(joint_mean, joint_scale, benign_mean, benign_second):
    """Joint population has one benign and six attack rows per source day."""
    joint_second = joint_scale ** 2 + joint_mean ** 2
    scalar_mean = float(joint_mean.mean())
    scalar_scale = float(scales_from_moments(scalar_mean, joint_second.mean()))
    malicious_mean = (7 * joint_mean - benign_mean) / 6
    malicious_second = (7 * joint_second - benign_second) / 6
    return {"joint_scalar": {"mean": scalar_mean, "scale": scalar_scale},
            "benign": {"mean": benign_mean, "scale": scales_from_moments(benign_mean, benign_second)},
            "malicious": {"mean": malicious_mean, "scale": scales_from_moments(malicious_mean, malicious_second)}}


def transform(raw, y, scalers, branch):
    if branch == "joint_scalar_softmax":
        fitted = scalers["joint_scalar"]
        return (raw - fitted["mean"]) / fitted["scale"]
    if branch != "separate_class_feature_softmax":
        raise ValueError(f"Unsupported normalization branch: {branch}")
    benign, malicious = scalers["benign"], scalers["malicious"]
    return np.where(y[:, None] == 0,
                    (raw - benign["mean"]) / benign["scale"],
                    (raw - malicious["mean"]) / malicious["scale"])


def cube_bounds(x):
    x = np.asarray(x, dtype=np.float64)
    return (np.mean((x - np.clip(x, 0, 1)) ** 2, axis=1),
            np.mean(np.maximum(x ** 2, (x - 1) ** 2), axis=1))


def selected_rows(base, stage):
    days = np.unique(np.linspace(0, base - 1, min(64, base), dtype=int))
    return (np.arange(7 * base) if stage == "full" else
            np.concatenate([days + group * base for group in range(7)]))


def compact_envelope(y, lower, upper, reverse=False):
    result = oracle_envelope(y, lower, upper, reverse=reverse)
    result.pop("curve")  # No new figures: preserve exact enumeration summaries.
    return result


def analyze(args, record):
    verified_at = time.perf_counter()
    result = json.loads(args.result.read_text())
    assert digest(args.result) == RESULT_SHA
    assert digest(args.reference) == REFERENCE_SHA
    reference = json.loads(args.reference.read_text())
    data = args.root / result["data"]["path"]
    assert digest(data / "metadata.json") == result["data"]["metadata_sha256"]
    meta = json.loads((data / "metadata.json").read_text())
    hashes = {}
    for name in ("benign_raw.npy", "x_test.npy", "y_test.npy", "test_source_row.npy"):
        hashes[name] = digest(data / name)
        expected = (meta["files"][name]["sha256"] if name == "benign_raw.npy"
                    else result["data"]["files"][name])
        assert hashes[name] == expected, name
    record["input_sha256"] = hashes
    record["metadata_sha256"] = result["data"]["metadata_sha256"]
    record["verification_seconds"] = time.perf_counter() - verified_at
    x = np.load(data / "x_test.npy", mmap_mode="r")
    y = np.load(data / "y_test.npy", mmap_mode="r")
    source = np.load(data / "test_source_row.npy", mmap_mode="r")
    benign_raw = np.load(data / "benign_raw.npy", mmap_mode="r")
    base = meta["counts"]["B2_profiles"]
    assert base == 750767 and x.shape[1] == 48
    assert len(benign_raw) == meta["profiles"]["rows"]
    for group in range(7):
        assert np.all(y[group * base:(group + 1) * base] == (group > 0))
        assert np.array_equal(source[group * base:(group + 1) * base], source[:base])
    assert np.all((source[:base] >= 0) & (source[:base] < len(benign_raw)))
    joint_mean = np.asarray(meta["scaler"]["mean"], dtype=np.float64)
    joint_scale = np.asarray(meta["scaler"]["scale"], dtype=np.float64)
    fitted_at = time.perf_counter()
    benign_mean, benign_second = moments(benign_raw)
    scalers = alternative_scalers(joint_mean, joint_scale, benign_mean, benign_second)
    record["fitting_moments_seconds"] = time.perf_counter() - fitted_at
    record["scalers"] = {name: {key: np.asarray(value).tolist() for key, value in fitted.items()}
                         for name, fitted in scalers.items()}
    record["fitted_population"] = {"benign_rows": len(benign_raw),
                                   "malicious_rows": 6 * len(benign_raw)}
    check = np.unique(np.linspace(0, base - 1, 256, dtype=int))
    recovered = np.asarray(x[check], dtype=np.float64) * joint_scale + joint_mean
    actual = np.asarray(benign_raw[source[check]], dtype=np.float64)
    tolerance = 2e-5 * (1 + np.abs(actual))
    assert np.all(np.abs(recovered - actual) <= tolerance)
    record["round_trip"] = {"checked_benign_rows": len(check),
                            "max_absolute_error": float(np.max(np.abs(recovered - actual))),
                            "passed": True}
    rows = selected_rows(base, args.stage)
    labels = np.asarray(y[rows])
    record["rows"] = len(rows)
    record["source_days"] = len(rows) // 7
    record["source_row_indices_sha256"] = hashlib.sha256(rows.tobytes()).hexdigest()
    bounds = {name: (np.empty(len(rows)), np.empty(len(rows))) for name in BRANCHES}
    started = time.perf_counter()
    for start in range(0, len(rows), CHUNK):
        stop = min(start + CHUNK, len(rows))
        inputs = np.asarray(x[rows[start:stop]], dtype=np.float64)
        raw = inputs * joint_scale + joint_mean
        reference_lower = reference_upper = None
        for branch in BRANCHES:
            values = (inputs if branch.startswith("joint_feature") else
                      transform(raw, labels[start:stop], scalers, branch))
            assert np.isfinite(values).all()
            if branch.endswith("sigmoid_control"):
                lower, upper = cube_bounds(values)
                assert np.all(lower <= reference_lower + 1e-10)
                assert np.all(upper >= reference_upper - 1e-10)
                witness = np.mean((values - 0.5) ** 2, axis=1)
            else:
                lower, upper, witness, _ = simplex_bounds(values)
                if branch == BRANCHES[0]:
                    reference_lower, reference_upper = lower, upper
            assert np.all(lower <= witness + 1e-10)
            assert np.all(witness <= upper + 1e-10)
            epsilon = 1e-5 * (1 + upper)
            bounds[branch][0][start:stop] = np.maximum(lower - epsilon, 0)
            bounds[branch][1][start:stop] = upper + epsilon
    record["geometry_seconds"] = time.perf_counter() - started
    record["branches"] = {}
    for branch, (lower, upper) in bounds.items():
        printed = compact_envelope(labels, lower, upper)
        record["branches"][branch] = {
            "printed": printed,
            "reversed_control": compact_envelope(labels, lower, upper, reverse=True),
            "printed_cutoff_rounded_DR_target_excluded": printed["at_printed_threshold"]["max_DR"] < 80.5,
            "range_quantiles": {name: np.quantile(vector, [0, .25, .5, .75, 1]).tolist()
                                for name, vector in (("lower", lower), ("upper", upper))},
        }
    if args.stage == "full":
        for direction in ("printed", "reversed_control"):
            actual_bound = record["branches"][BRANCHES[0]][direction]
            expected_bound = reference["bounds"]["original"][direction]
            for metric in ("max_ACC", "max_AUC"):
                assert abs(actual_bound[metric] - expected_bound[metric]) < 1e-7
            assert abs(actual_bound["at_FA_cap"]["15.0"]["max_DR"] -
                       expected_bound["at_FA_cap"]["15.0"]["max_DR"]) < 1e-7
        record["reference_original_bound_reproduced"] = True
    record["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experimental scoring requires a cluster compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    script = Path(__file__).resolve()
    record = {"status": "started", "classification": "C/A bound diagnostics on explicit I input alternatives",
              "stage": args.stage, "analysis_commit": args.analysis_commit,
              "script_sha256": digest(script),
              "helper_sha256": digest(script.parent / "post_anchor_diagnostics.py"),
              "contract_sha256": digest(script.parent.parent / "SOURCE_ASSUMPTION_CHECK.md"),
              "source_result_sha256": RESULT_SHA, "reference_sha256": REFERENCE_SHA,
              "job_id": os.environ["SLURM_JOB_ID"],
              "scope": "Original held-out rows only; fixed-cutoff attack DR survives benign-only resampling; other original-row limits do not automatically cover ADASYN",
              "numerical_scope": "float64 extrema padded outward by 1e-5*(1+U), not certified interval arithmetic",
              "versions": {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__}}
    try:
        analyze(args, record)
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        with (args.output / "result.json").open("x") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: record[key] for key in ("status", "stage", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
