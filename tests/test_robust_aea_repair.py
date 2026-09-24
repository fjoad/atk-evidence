"""Constructed interval/repair witnesses; no local research experiment."""

import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from tests.test_robust_baseline import A, load
from tests.test_robust_aea_diagnostic import D, fixture, write_fixture

with patch.dict("sys.modules", {"aea_diagnostic": D}):
    C = load("robust_aea_repair", D.STUDY / "checks/aea_repair.py")


class AEARepairTests(unittest.TestCase):
    def test_box_bounds_mse_mae_and_arbitrary_coordinate_ranges(self):
        x = np.array([[-2., .25, 2.]])
        for metric, expected in (("mse", 5 / 3), ("mae", 1.)):
            lo, hi = C.error_intervals(x, metric=metric)
            self.assertAlmostEqual(lo[0], expected)
            power = 2 if metric == "mse" else 1
            brute = [np.mean(np.abs(x - np.array(c)) ** power) for c in itertools.product((0., 1.), repeat=3)]
            self.assertEqual(hi[0], max(brute))
        lo, hi = C.error_intervals([[-1., 10.]], [2., 5.], [3., 7.])
        np.testing.assert_allclose(lo, [9.])
        np.testing.assert_allclose(hi, [20.5])

    def test_oracle_dominates_random_legal_scores_in_both_directions(self):
        generator = np.random.default_rng(13)
        y = np.arange(24) % 2
        lower = generator.random(24)
        upper = lower + generator.random(24)
        for reverse in (False, True):
            oracle, result = C.optimistic_envelope(lower, upper, y, reverse)
            self.assertTrue(result["uses_true_test_labels"])
            self.assertFalse(result["is_a_trained_or_deployable_model"])
            for _ in range(20):
                scores = lower + generator.random(24) * (upper - lower)
                if reverse:
                    scores = -scores
                self.assertLessEqual(roc_auc_score(y, scores), roc_auc_score(y, oracle) + 1e-14)
                for threshold in np.r_[-np.inf, np.unique(np.r_[scores, oracle])]:
                    actual, ideal = scores > threshold, oracle > threshold
                    self.assertLessEqual(actual[y == 1].sum(), ideal[y == 1].sum())
                    self.assertGreaterEqual(actual[y == 0].sum(), ideal[y == 0].sum())

    def test_relaxation_can_pass_for_identical_inputs_with_conflicting_labels(self):
        x = np.full((2, 48), .25)
        lo, hi = C.error_intervals(x)
        y = np.array([0, 1])
        _, upper = C.optimistic_envelope(lo, hi, y)
        self.assertEqual(upper["AUC_upper_bound"], 100.)
        self.assertEqual(roc_auc_score(y, np.mean(x ** 2, axis=1)), .5)

    def test_threshold_interval_strict_edges_and_ties(self):
        y = np.array([0, 0, 1, 1])
        lo, hi = np.array([.1, .2, .1, .2]), np.array([.4, .5, .8, 1.])
        interval = C.necessary_threshold_interval(lo, hi, y, 0., 100.)
        self.assertEqual((interval["minimum_inclusive"], interval["maximum_exclusive"]), (.2, .8))
        self.assertFalse(interval["empty"])
        self.assertEqual(int((lo[y == 0] > .2).sum()), 0)
        self.assertEqual(int((hi[y == 1] > .8).sum()), 1)
        self.assertEqual(C.scale_interval(interval)["positive_multiplier_maximum_inclusive"], 2.55)
        tied = C.necessary_threshold_interval(np.array([.5, .5]), np.array([1., .5]), np.array([0, 1]), 0., 100.)
        self.assertTrue(tied["empty"])

    def test_empty_interval_excludes_all_cutoffs_and_increasing_score_scales(self):
        y = np.array([0, 0, 1, 1])
        lo, hi = np.array([1., 1., 0., 0.]), np.array([2., 2., .5, .5])
        _, report = C.optimistic_envelope(lo, hi, y)
        _, scaled = C.optimistic_envelope(3 * lo + 2, 3 * hi + 2, y)
        self.assertEqual(report["AUC_upper_bound"], scaled["AUC_upper_bound"])
        self.assertEqual(report["at_FA_caps"]["5.2"]["DR_upper_bound"], 0.)
        self.assertTrue(C.necessary_threshold_interval(lo, hi, y, 5.2, 94.1)["empty"])
        _, reversed_bound = C.optimistic_envelope(lo, hi, y, reverse=True)
        self.assertEqual(reversed_bound["AUC_upper_bound"], 100.)

    def test_training_only_transforms_do_not_clip_or_use_test_statistics(self):
        a, _ = fixture()
        for branch in ("global_z_mse", "minmax_mse"):
            initial = C.view(a, branch)
            changed = {k: v.copy() for k, v in a.items()}
            changed["test_raw_x"][0] = 1000.
            after = C.view(changed, branch)
            np.testing.assert_array_equal(initial["location"], after["location"])
            np.testing.assert_array_equal(initial["scale"], after["scale"])
            self.assertGreater(after["test_target"].max(), 1.)
            self.assertEqual(after["test_target"].dtype, np.float32)

    def test_constant_training_feature_has_recorded_unit_scale(self):
        a, _ = fixture()
        a["train_raw_x"][:, 0] = 7.
        a["test_raw_x"][0, 0] = 11.
        result = C.view(a, "minmax_mse")
        self.assertTrue(result["zero_scale"][0])
        self.assertEqual(result["scale"][0], 1.)
        self.assertEqual(result["test_target"][0, 0], 4.)

    def test_raw_unit_box_is_the_inverse_of_the_original_scaler(self):
        a, _ = fixture()
        saved = C.view(a, "raw_unit_mse")
        np.testing.assert_array_equal(saved["test_target"], a["test_raw_x"])
        np.testing.assert_array_equal(saved["box_low"], a["scaler_mean"])
        np.testing.assert_array_equal(saved["box_high"], a["scaler_mean"] + a["scaler_scale"])

    def test_all_branches_and_shared_intervals_are_reported_without_selection(self):
        cases = {}
        for level in ("p00", "p30"):
            a, _ = fixture(level == "p30")
            cases[level] = {}
            for branch in C.BRANCHES:
                saved = C.view(a, branch)
                result = C.analyze(saved, branch, 0. if level == "p00" else .3)
                self.assertFalse(result["is_reproduction"])
                self.assertIsNone(result["groups"]["synthetic"]["favorable_envelope"])
                cases[level][branch] = {"analysis": result}
        self.assertEqual(set(C.common_intervals(cases)), set(C.BRANCHES))

    def test_compute_guard_precedes_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "unused"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                C.run(path, path, path)
            self.assertFalse(path.exists())

    def test_constructed_end_to_end_archives_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = write_fixture(root / "inputs")
            env = {"SLURM_JOB_ID": "constructed-fixture", "SLURM_JOB_NODELIST": "fixture", "EXPECTED_COMMIT": "fixture"}
            with patch.object(D, "METADATA", hashes), patch.dict(os.environ, env):
                D.run(root / "inputs", root / "prior")
                with patch.object(C, "PREVIOUS_HASH", A.digest(root / "prior/result.json")):
                    report = C.run(root / "inputs", root / "prior", root / "repair")
                    self.assertEqual(report["repair_archives_checked"], 10)
                    self.assertEqual(report, C.audit(root / "inputs", root / "prior", root / "repair"))
                    path = root / "repair/result.json"
                    r = json.loads(path.read_text())
                    r["common_cutoff_necessary_intervals"]["feature_z_mse"]["empty"] = "tampered"
                    path.write_text(json.dumps(r))
                    with self.assertRaises(AssertionError):
                        C.audit(root / "inputs", root / "prior", root / "repair")


if __name__ == "__main__":
    unittest.main()
