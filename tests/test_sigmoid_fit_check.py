"""Small software fixtures, never experimental evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

CHECKS = Path(__file__).resolve().parents[1] / "studies/atk-2022-deep-autoencoder/checks"
sys.path.insert(0, str(CHECKS))
spec = importlib.util.spec_from_file_location("sigmoid_fit_check", CHECKS / "sigmoid_fit_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SigmoidFitTests(unittest.TestCase):
    def test_heads_differ_only_in_activation_and_have_same_initial_weights(self):
        softmax, sigmoid = [check.build_model(name) for name in ("softmax", "sigmoid")]
        self.assertEqual(check.weight_digest(softmax), check.weight_digest(sigmoid))
        self.assertEqual(softmax.count_params(), 450448)
        self.assertEqual(sigmoid.count_params(), 450448)
        for left, right in zip(softmax.layers[:-1], sigmoid.layers[:-1]):
            self.assertEqual(left.get_config(), right.get_config())
        self.assertEqual(softmax.layers[-1].activation.__name__, "softmax")
        self.assertEqual(sigmoid.layers[-1].activation.__name__, "sigmoid")

    def test_selection_is_deterministic_disjoint_and_keeps_siblings(self):
        first, original = check.select_indices(4000, 1500, 1500 * 12, "small")
        second, _ = check.select_indices(4000, 1500, 1500 * 12, "small")
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.intersect1d(first["fit"], first["calibration"]).size)
        self.assertEqual(original, 7168)
        for group in range(7):
            np.testing.assert_array_equal(first["test"][group * 1024:(group + 1) * 1024], first["test"][:1024] + 1500 * group)

    def test_calibration_cutoff_uses_only_benign_errors_and_strict_rules(self):
        errors = np.arange(100, dtype=float)
        high, low = check.calibration_cutoff(errors), check.calibration_cutoff(errors, True)
        self.assertLessEqual(np.mean(errors > high), .15)
        self.assertLessEqual(np.mean(errors < low), .15)

    def test_calibrated_metrics_and_oracle_diagnostics_are_separate(self):
        y = np.array([0, 1, 1, 1, 1, 1, 1, 0])
        scores = np.array([.1, .8, .8, .8, .8, .8, .8, .2])
        result = check.summarize(y, scores, np.array([.9, 1.]), 7)
        high = result["sampled_prepared"]["printed"]
        self.assertEqual(high["all_cutoffs_diagnostic"]["max_ACC"], 100)
        self.assertEqual(high["calibrated_metrics"]["DR"], 0)

    def test_one_update_is_finite_and_changes_sigmoid_weights(self):
        model = check.build_model("sigmoid")
        inputs = np.random.default_rng(4).normal(size=(4, 48)).astype(np.float32)
        before = check.weight_digest(model)
        loss = model.train_on_batch(inputs, inputs)
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(before, check.weight_digest(model))
        scores, bounds = check.score(model, inputs, "sigmoid")
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreaterEqual(bounds["min"], 0)
        self.assertLessEqual(bounds["max"], 1)

    def test_saved_results_match_frozen_sources_and_transferred_bytes(self):
        study = CHECKS.parent
        folder = study / "results/sigmoid_fit_20260831"
        execution = json.loads((folder / "execution.json").read_text())
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, expected in execution["transferred_sha256"].items():
            self.assertEqual(sha(folder / name), expected)
        for stage, updates in (("pilot", 4), ("small", 640)):
            result = json.loads((folder / f"{stage}.json").read_text())
            self.assertEqual(result["analysis_commit"], execution["analysis_commit"])
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["script_sha256"], sha(CHECKS / "sigmoid_fit_check.py"))
            self.assertEqual(result["contract_sha256"], sha(study / "SIGMOID_FIT_CHECK.md"))
            self.assertEqual(result["model_source_sha256"], sha(study / "reproduction/models.py"))
            for name, expected in result["helpers"].items():
                self.assertEqual(sha(CHECKS / name), expected)
            heads = list(result["models"].values())
            self.assertEqual(heads[0]["initial_weight_sha256"], heads[1]["initial_weight_sha256"])
            for head in heads:
                self.assertEqual(head["parameters"], 450448)
                self.assertEqual(head["updates"], updates)
                self.assertTrue(head["completed_requested_updates"])
                self.assertFalse(head["budget_stopped"])
                self.assertNotEqual(head["initial_weight_sha256"], head["selected_weight_sha256"])

    def test_finite_cutoff_failure_does_not_erase_learning_or_prior_open_bound(self):
        study = CHECKS.parent
        result = json.loads((study / "results/sigmoid_fit_20260831/small.json").read_text())
        for head in result["models"].values():
            for stage in ("initial", "selected"):
                for view in head[stage].values():
                    for direction in view.values():
                        cutoff = direction["all_cutoffs_diagnostic"]
                        self.assertFalse(cutoff["target_pair_not_excluded"])
                        self.assertFalse(cutoff["rounded_target_pair_not_excluded"])
        sigmoid = result["models"]["sigmoid"]
        self.assertEqual(sigmoid["selected_epoch"], len(sigmoid["epochs"]))
        self.assertLess(sigmoid["epochs"][-1]["val_loss"], sigmoid["epochs"][0]["val_loss"])
        prior = json.loads((study / "results/sigmoid_sanity_20260831/full.json").read_text())
        self.assertTrue(prior["views"]["full"]["bounds"]["printed"]["target_pair_not_excluded"])
        finding = (study / "SIGMOID_FIT_FINDING.md").read_text()
        self.assertIn("No seed-level confidence interval", finding)
        self.assertIn("long-run plateau", finding)


if __name__ == "__main__":
    unittest.main()
