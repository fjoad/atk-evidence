#!/usr/bin/env python3
"""Read-only verification of saved preparation artifacts; no model scoring."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Preparation did not complete")
    checked = []
    arrays_by_name = {}
    for run in summary["runs"]:
        path = directory / run["name"]
        metadata = json.loads((path / "metadata.json").read_text())
        arrays = {}
        for name, expected in metadata["files"].items():
            actual = np.load(path / name, allow_pickle=False, mmap_mode="r")
            assert sha256(path / name) == expected["sha256"], (run["name"], name, "hash")
            assert list(actual.shape) == expected["shape"] and str(actual.dtype) == expected["dtype"]
            assert np.isfinite(actual).all(), (run["name"], name, "nonfinite")
            arrays[name.removesuffix(".npy")] = actual
        assert np.array_equal(arrays["test_true_y"], arrays["test_observed_y"])
        originals = {}
        for part in ("train", "test"):
            assert len(arrays[part + "_x"]) == metadata["training_rows" if part == "train" else "test_rows"]
            counts = np.bincount(arrays[part + "_true_y"], minlength=2).tolist()
            assert counts == metadata[part + "_true_counts"]
            rebuilt = (arrays[part + "_raw_x"].astype(np.float64) - arrays["scaler_mean"]) / arrays["scaler_scale"]
            np.testing.assert_allclose(rebuilt, arrays[part + "_x"], rtol=5e-7, atol=5e-7)
            for index in np.flatnonzero(~arrays[part + "_synthetic"]):
                uid = int(arrays[part + "_uid"][index])
                value = arrays[part + "_raw_x"][index]
                label = int(arrays[part + "_true_y"][index])
                if uid in originals:
                    np.testing.assert_array_equal(value, originals[uid][0])
                    assert label == originals[uid][1]
                originals[uid] = (value, label)
        mean = arrays["train_raw_x"].mean(axis=0, dtype=np.float64)
        np.testing.assert_allclose(mean, arrays["scaler_mean"], rtol=1e-12, atol=1e-12)
        assert np.bincount(arrays["train_observed_y"], minlength=2).tolist() == metadata["train_observed_counts"]
        for part in ("train", "test"):
            indices = np.flatnonzero(arrays[part + "_synthetic"])
            for index in indices:
                left, left_label = originals[int(arrays[part + "_parent_a"][index])]
                right, right_label = originals[int(arrays[part + "_parent_b"][index])]
                weight = arrays[part + "_mix"][index]
                assert left_label == right_label == 0 and 0 <= weight <= 1
                rebuilt = (left.astype(np.float64) + weight * (right - left)).astype(np.float32)
                np.testing.assert_array_equal(rebuilt, arrays[part + "_raw_x"][index])
        overlap = np.intersect1d(arrays["train_uid"][~arrays["train_synthetic"]],
                                 arrays["test_uid"][~arrays["test_synthetic"]])
        assert len(overlap) == metadata["shared_original_row_identities"]
        changed = np.count_nonzero(arrays["train_true_y"] != arrays["train_observed_y"])
        assert changed == metadata["poisoning"]["changed_rows"]
        arrays_by_name[run["name"]] = arrays
        checked.append({"name": run["name"], "files": len(arrays), "status": "verified",
                        "changed_rows": int(changed), "original_identity_overlap": len(overlap)})
    for name, clean in arrays_by_name.items():
        if not name.endswith("p00"):
            continue
        for candidate, altered in arrays_by_name.items():
            if candidate.rsplit("-p", 1)[0] != name.rsplit("-p", 1)[0]:
                continue
            for key in ("test_raw_x", "test_true_y", "test_uid"):
                np.testing.assert_array_equal(clean[key], altered[key])
            if "two-class" in name:
                np.testing.assert_array_equal(clean["train_raw_x"], altered["train_raw_x"])
    return {"status": "verified", "code_commit": summary["code_commit"], "job_id": summary["job_id"],
            "summary_sha256": sha256(directory / "summary.json"), "runs": checked,
            "checks": ["all saved array hashes, shapes, dtypes and finite values",
                       "true/observed labels and confusion-independent class counts",
                       "training-fitted scaler mean and every standardized input",
                       "every synthetic interpolation and benign parent",
                       "reported train/test identity overlap",
                       "fixed raw test population across poison levels"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
