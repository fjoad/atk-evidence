"""Constructed fixtures for the first detector; never uses CER observations."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "studies/takiddin-2021-robust-poisoning/reproduction"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("robust_models", REPRO / "models.py")
A = load("robust_analysis", REPRO / "analyze_results.py")
with patch.dict(sys.modules, {"models": M, "analyze_results": A}):
    R = load("robust_runner", REPRO / "run_experiment.py")


def constructed_input(reverse_training_labels=False):
    generator = np.random.default_rng(5)
    train_y = np.arange(120) % 2
    test_y = np.arange(60) % 2
    train = (1 + 2 * train_y[:, None] + generator.normal(0, .1, (120, 48))).astype(np.float32)
    test = (1 + 2 * test_y[:, None] + generator.normal(0, .1, (60, 48))).astype(np.float32)
    return {
        "train_x": train, "train_true_y": train_y,
        "train_observed_y": 1 - train_y if reverse_training_labels else train_y,
        "test_x": test, "test_raw_x": test.copy(), "test_true_y": test_y,
        "test_observed_y": test_y.copy(), "test_uid": np.arange(60),
        "test_synthetic": np.arange(60) % 4 == 0,
        "test_attack": np.where(test_y == 1, 1 + np.arange(60) % 6, 0),
    }


class RobustBaselineTests(unittest.TestCase):
    def test_model_is_stock_forest_with_explicit_source_and_default_settings(self):
        actual = M.random_forest(42, 4)
        expected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
        self.assertEqual(actual.get_params(), expected.get_params())

    def test_metrics_match_library_and_distinguish_accuracy_from_balance(self):
        labels = np.array([1, 1, 0, 0, 0, 0])
        predictions = np.array([1, 0, 1, 0, 0, 0])
        scores = np.array([.9, .3, .8, .2, .1, .05])
        actual = A.metrics(labels, predictions, scores)
        self.assertEqual([actual[k] for k in ("TP", "FN", "FP", "TN")], [1, 1, 1, 3])
        self.assertAlmostEqual(actual["ACC"], 100 * accuracy_score(labels, predictions))
        self.assertEqual(actual["balanced_accuracy"], 62.5)
        self.assertAlmostEqual(actual["PR"], 100 * precision_score(labels, predictions))
        self.assertAlmostEqual(actual["DR"], 100 * recall_score(labels, predictions))
        self.assertAlmostEqual(actual["F1"], 100 * f1_score(labels, predictions))
        self.assertAlmostEqual(actual["AUC"], 100 * roc_auc_score(labels, scores))

    def test_undefined_precision_and_single_class_are_explicit(self):
        result = A.metrics([0, 1], [0, 0], [.5, .5])
        self.assertIsNone(result["PR"])
        self.assertEqual(result["F1"], 0)
        self.assertEqual(result["AUC"], 50)
        result = A.metrics([0, 0], [0, 0], [.1, .2])
        self.assertIsNone(result["DR"])
        self.assertIsNone(result["AUC"])

    def test_roc_diagnostics_match_exhaustive_tied_thresholds(self):
        labels = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([.8, .8, .5, .4, .4, .1])
        for direction in (1, -1):
            values = direction * scores
            actual = A.threshold_summary(labels, values, caps=(0., 17.6, 33.3, 100.))
            candidates = [np.inf, *np.unique(values)]
            results = [A.metrics(labels, values >= cutoff, values) for cutoff in candidates]
            self.assertEqual(actual["boundary_count"], len(candidates))
            for cap in (0., 17.6, 33.3, 100.):
                expected = max(r["DR"] for r in results if r["FA"] <= cap)
                self.assertAlmostEqual(actual["at_false_alarm_caps"][str(cap)]["DR"], expected, places=12)
            self.assertAlmostEqual(actual["best_balanced_accuracy"],
                                   max(r["balanced_accuracy"] for r in results))

    def test_argmax_uses_class_zero_on_exact_ties(self):
        inputs = constructed_input()
        data = {"labels": inputs["test_true_y"], "predictions": np.zeros(60, dtype=int),
                "probabilities": np.full((60, 2), .5), "synthetic": inputs["test_synthetic"],
                "attack": inputs["test_attack"], "negative_daily_mean": -inputs["test_raw_x"].mean(axis=1)}
        result = A.analyze_scores(data, .5)
        self.assertEqual(result["primary"]["DR"], 0)
        self.assertEqual(result["controls"]["constant_training_prior"]["DR"], 0)
        data["predictions"][0] = 1
        with self.assertRaisesRegex(ValueError, "argmax"):
            A.analyze_scores(data, .5)

    def test_positive_control_and_reload_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = R.fit_one(constructed_input(), Path(directory), seed=42, workers=1)
            self.assertEqual(report["forest"]["trees"], 100)
            self.assertGreater(report["training"]["observed_label_accuracy"], 99)
            self.assertGreater(report["analysis"]["primary"]["ACC"], 99)
            self.assertTrue(report["reload"]["identical_probabilities"])

    def test_fit_uses_corrupted_labels_not_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            report = R.fit_one(constructed_input(True), Path(directory), seed=42, workers=1)
            self.assertGreater(report["training"]["observed_label_accuracy"], 99)
            self.assertLess(report["training"]["true_label_accuracy"], 1)
            self.assertLess(report["analysis"]["primary"]["ACC"], 1)

    def test_read_only_audit_recomputes_result_and_rejects_corrupt_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = R.fit_one(constructed_input(), output, seed=42, workers=1)
            report.update(status="complete", case="fixture", code_commit="fixture",
                          scope="software fixture", poisoning_rate=0.)
            (output / "result.json").write_text(json.dumps(report))
            result = A.audit_result(output)
            self.assertEqual(result["status"], "verified")
            self.assertFalse(result["comparison_is_reproduction"])
            with (output / "predictions.npz").open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash"):
                A.audit_result(output)

    def test_unknown_or_changed_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                R.load_preparation(Path(directory))
            known = Path(directory) / "generalized-two-class-p00"
            known.mkdir()
            (known / "metadata.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "metadata"):
                R.load_preparation(known)

    def test_local_real_data_execution_is_refused(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                R.require_compute()


if __name__ == "__main__":
    unittest.main()
