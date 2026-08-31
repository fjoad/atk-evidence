"""One capped, adaptive no-training control; see ENERGY_BAND_CONTROL.md."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from post_anchor_diagnostics import (
    ANALYSIS_SEED, RESULT_SHA, digest, energy_band_rankings,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experimental scoring requires a cluster compute node")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    result = json.loads(args.result.read_text())
    assert digest(args.result) == RESULT_SHA
    data, run = args.root / result["data"]["path"], args.result.parent
    assert digest(data / "metadata.json") == result["data"]["metadata_sha256"]
    meta = json.loads((data / "metadata.json").read_text())
    hashes = {}
    for name in ("x_test.npy", "y_test.npy", "test_source_row.npy"):
        hashes[name] = digest(data / name)
        assert hashes[name] == result["data"]["files"][name]
    arrays = {}
    for name, file in [("trained", "scores.npy"), ("energy", "zero_reconstruction_scores.npy"),
                       ("projection", "softmax_projection_floor_scores.npy")]:
        hashes[file] = digest(run / file)
        assert hashes[file] == result["artifact_sha256"][file]
        arrays[name] = np.load(run / file, mmap_mode="r")
    base = meta["counts"]["B2_profiles"]
    days = np.sort(np.random.default_rng(ANALYSIS_SEED).choice(base, 10000, replace=False))
    rows = np.concatenate([days + group * base for group in range(7)])
    scores = {name: np.asarray(a[rows]) for name, a in arrays.items()}
    x = np.asarray(np.load(data / "x_test.npy", mmap_mode="r")[rows], dtype=np.float64)
    scores["uniform"] = np.mean((x - 1 / 48) ** 2, axis=1)
    y = np.asarray(np.load(data / "y_test.npy", mmap_mode="r")[rows])
    source = np.asarray(np.load(data / "test_source_row.npy", mmap_mode="r")[rows])
    assert np.array_equal(source, np.tile(source[:10000], 7))
    assert np.array_equal(y, np.repeat([0, 1, 1, 1, 1, 1, 1], 10000))
    checks = {name: energy_band_rankings(y, scores["energy"], score)
              for name, score in scores.items() if name != "energy"}
    aucs = {"energy": checks["trained"]["pair_weighted_within_bin_AUC"]["energy"],
            **{name: values["pair_weighted_within_bin_AUC"]["trained"] for name, values in checks.items()}}
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "sample_day_indices.npy", days)
    script = Path(__file__).resolve()
    record = {"classification": "X/M exploratory control", "status": "passed",
        "source_result_sha256": RESULT_SHA, "input_sha256": hashes,
        "script_sha256": digest(script), "helper_sha256": digest(script.parent / "post_anchor_diagnostics.py"),
        "contract_sha256": digest(script.parent.parent / "ENERGY_BAND_CONTROL.md"),
        "analysis_commit": subprocess.check_output(["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ["SLURM_JOB_ID"], "seed": ANALYSIS_SEED,
        "sample_source_days": len(days), "rows": len(rows),
        "sample_day_indices_sha256": digest(args.output / "sample_day_indices.npy"),
        "pair_weighted_within_bin_AUC": aucs,
        "within_energy_checks": checks, "elapsed_seconds": time.perf_counter() - started}
    with (args.output / "control.json").open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
                     "within_energy_AUC": aucs}, indent=2))


if __name__ == "__main__":
    main()
