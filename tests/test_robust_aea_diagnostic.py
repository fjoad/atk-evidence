"""End-to-end diagnostic fixtures made from constructed rows, never CER data."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_baseline import A, R, load
from tests.test_robust_aea_geometry import G, PATH

with patch.dict("sys.modules", {"aea_geometry": G, "analyze_results": A, "run_experiment": R}):
    D = load("robust_aea_diagnostic", PATH.with_name("aea_diagnostic.py"))


def original(source, raw, attack):
    source, raw, attack = np.asarray(source, np.int64), np.asarray(raw, np.float32), np.asarray(attack, np.int8)
    uid = 7 * source + attack
    return {"raw_x": raw, "meter": (source // 1000).astype(np.int32), "day": (source % 1000).astype(np.int32),
            "source_id": source, "attack": attack, "uid": uid, "true_y": (attack > 0).astype(np.int8),
            "observed_y": (attack > 0).astype(np.int8), "synthetic": np.zeros(len(source), bool),
            "parent_a": uid.copy(), "parent_b": np.full(len(source), -1, np.int64), "mix": np.zeros(len(source))}


def fixture(poisoned=False):
    profile = np.arange(48, dtype=np.float32) / 100 + 1
    train = original([1001, 1002, 2001], np.array([.8, 1., 1.2])[:, None] * profile, [0, 0, 0])
    test = original([2002, 3001, 1001, 1002, 2001], np.array([.9, 1.1, .4, .5, .6])[:, None] * profile, [0, 0, 1, 1, 1])
    synthetic = original([-1], np.zeros((1, 48)), [0])
    synthetic.update(meter=np.array([-1], np.int32), day=np.array([-1], np.int32),
                     uid=np.array([-1], np.int64), synthetic=np.array([True]),
                     parent_a=test["uid"][:1].copy(), parent_b=test["uid"][1:2].copy(), mix=np.array([.25]))
    synthetic["raw_x"] = (test["raw_x"][:1].astype(np.float64) + .25 *
                            (test["raw_x"][1:2] - test["raw_x"][:1])).astype(np.float32)
    test = {k: np.concatenate((test[k], synthetic[k])) for k in test}
    if poisoned:
        for key in train:
            train[key][0] = test[key][2]
    train["observed_y"][:] = 0
    mean, scale = train["raw_x"].mean(axis=0, dtype=np.float64), train["raw_x"].std(axis=0, dtype=np.float64)
    arrays = {f"{part}_{key}": value for part, row in (("train", train), ("test", test)) for key, value in row.items()}
    arrays.update(scaler_mean=mean, scaler_scale=scale)
    for part in ("train", "test"):
        arrays[f"{part}_x"] = ((arrays[f"{part}_raw_x"].astype(np.float64) - mean) / scale).astype(np.float32)
    metadata = {"mode": "novelty", "scope": "generalized", "training_rows": 3, "test_rows": 6,
                "train_true_counts": [2, 1] if poisoned else [3, 0], "test_true_counts": [3, 3],
                "train_observed_counts": [3, 0], "shared_original_row_identities": int(poisoned),
                "shared_source_days": 3, "resampling": {"generated": 1},
                "poisoning": {"changed_rows": int(poisoned), "selected_customer_count": int(poisoned)}}
    return arrays, metadata


def write_fixture(root):
    hashes = {}
    for level in ("p00", "p30"):
        arrays, m = fixture(level == "p30")
        directory = root / f"generalized-novelty-{level}"
        directory.mkdir(parents=True)
        m["files"] = {}
        for name, value in arrays.items():
            path = directory / (name + ".npy")
            np.save(path, value, allow_pickle=False)
            m["files"][path.name] = {"sha256": A.digest(path), "shape": list(value.shape), "dtype": str(value.dtype)}
        path = directory / "metadata.json"
        path.write_text(json.dumps(m))
        hashes[level] = A.digest(path)
    return hashes


class AEADiagnosticTests(unittest.TestCase):
    def test_clean_and_contaminated_provenance_and_scaler(self):
        for poison in (False, True):
            arrays, m = fixture(poison)
            report = D.validate_case(arrays, m)
            self.assertEqual(report["exact_train_test_overlaps"], int(poison))
            self.assertEqual(report["contaminated_training_rows"], int(poison))
            self.assertTrue(report["synthetic_ancestry_verified"])

    def test_invalid_scaling_labels_ancestry_or_overlap_are_rejected(self):
        for key in ("test_x", "train_observed_y", "test_raw_x"):
            arrays, m = fixture()
            arrays[key][-1] += 1
            with self.assertRaises(AssertionError):
                D.validate_case(arrays, m)
        arrays, m = fixture(True)
        m["shared_original_row_identities"] = 0
        with self.assertRaises(AssertionError):
            D.validate_case(arrays, m)

    def test_score_construction_and_baselines_are_not_models(self):
        a, _ = fixture()
        saved, domain = D.make_scores(a)
        np.testing.assert_array_equal(saved["test_zero_mse"], np.mean(a["test_x"].astype(float) ** 2, axis=1))
        self.assertEqual(domain["test"]["coordinates"], 6 * 48)
        result = D.analyze(saved, 0.)
        self.assertFalse(result["is_neural_reproduction"])
        self.assertEqual(set(result["bounds"]["all"]), {"mse", "rmse", "sse"})
        daily = result["untrained_baselines"]["all"]["negative_daily_mean"]
        self.assertIsNone(daily["metrics_at_0.51"])
        synthetic = result["untrained_baselines"]["synthetic"]["zero_mse"]
        self.assertIsNone(synthetic["AUC"])
        self.assertIsNone(synthetic["metrics_at_0.51"]["DR"])

    def test_guard_and_favorable_rounding_do_not_overclaim(self):
        lo, hi, y = np.array([.512, .2]), np.array([1., .6]), np.array([0, 1])
        exact = D.bound_counts(lo, hi, y, .51, 0.)
        favorable = D.bound_counts(lo, hi, y, .515, 1e-6)
        self.assertEqual(exact["minimum_FA"], 100.)
        self.assertEqual(favorable["minimum_FA"], 0.)
        self.assertEqual(favorable["maximum_DR"], 100.)
        one_class = D.bound_counts(lo[:1], hi[:1], y[:1], .51, 0.)
        self.assertIsNone(one_class["maximum_DR"])

    def test_input_hash_tampering_is_rejected_before_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = write_fixture(root)
            with patch.object(D, "METADATA", hashes):
                cases, records = D.load_pair(root)
                self.assertEqual(len(records["p00"]["array_sha256"]), 28)
                self.assertFalse(np.array_equal(cases["p00"]["test_x"], cases["p30"]["test_x"]))
                path = root / "generalized-novelty-p00/test_x.npy"
                np.save(path, np.zeros((6, 48), np.float32))
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    D.load_pair(root)

    def test_allocation_guard_precedes_output_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                D.run(root / "unused", root / "output")
            self.assertFalse((root / "output").exists())

    def test_complete_fixture_artifacts_reaudit_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = write_fixture(root / "inputs")
            env = {"SLURM_JOB_ID": "constructed-fixture", "SLURM_JOB_NODELIST": "fixture", "EXPECTED_COMMIT": "fixture"}
            with patch.object(D, "METADATA", hashes), patch.dict(os.environ, env):
                report = D.run(root / "inputs", root / "output")
                self.assertEqual(report["model_fits"], 0)
                self.assertEqual(report["input_arrays_checked"], 56)
                self.assertEqual(report, D.audit(root / "inputs", root / "output"))
                path = root / "output/result.json"
                r = json.loads(path.read_text())
                r["cases"]["p00"]["analysis"]["bounds_are_not_AUC_limits"] = False
                path.write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError, "analysis differs"):
                    D.audit(root / "inputs", root / "output")


if __name__ == "__main__":
    unittest.main()
