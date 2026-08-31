"""Hand-sized fixtures; no paper data or experimental scoring."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


CHECKS = Path(__file__).resolve().parents[1] / "studies/atk-2022-deep-autoencoder/checks"
sys.path.insert(0, str(CHECKS))
spec = importlib.util.spec_from_file_location("sigmoid_sanity", CHECKS / "sigmoid_sanity.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SigmoidSanityTests(unittest.TestCase):
    def test_pilot_retains_original_blocks_and_synthetic_tail(self):
        rows, original = check.selection(100, 750, "pilot")
        self.assertEqual(original, 448)
        self.assertEqual(len(rows), 498)
        for group in range(7):
            np.testing.assert_array_equal(rows[group * 64:(group + 1) * 64], rows[:64] + group * 100)
        self.assertTrue(np.all(rows[448:] >= 700))
        self.assertEqual(check.selection(100, 750, "full"), (None, 700))

    def test_identity_check_rejects_mislabeled_synthetic_rows(self):
        y = np.r_[np.repeat([0, 1, 1, 1, 1, 1, 1], 2), [0]]
        attacks = np.r_[np.repeat(np.arange(7), 2), [-1]]
        sources = np.r_[np.tile([10, 11], 7), [-1]]
        check.verify_identities(y, attacks, sources, 14)
        y[-1] = 1
        with self.assertRaises(AssertionError):
            check.verify_identities(y, attacks, sources, 14)

    def test_strict_cutoff_and_reversal_keep_ties_unflagged(self):
        y = np.array([0, 0, 1, 1])
        errors = np.array([0.58, 0.7, 0.58, 0.3])
        high = check.fixed_metrics(y, errors)
        low = check.fixed_metrics(y, errors, reverse=True)
        self.assertEqual((high["TP"], high["FP"], high["ACC"]), (0, 1, 25))
        self.assertEqual((low["TP"], low["FP"], low["ACC"]), (1, 0, 75))

    def test_fixed_control_auc_and_threshold_summary_are_not_an_oracle_model(self):
        y = np.array([0, 0, 1, 1])
        errors = np.array([0.1, 0.2, 0.8, 0.9])
        summary = check.control_summary(y, errors)
        self.assertEqual(summary["fixed_cutoff"]["AUC"], 100)
        self.assertEqual(summary["fixed_cutoff"]["ACC"], 100)
        self.assertEqual(check.control_summary(y, errors, reverse=True)["fixed_cutoff"]["AUC"], 0)

    def test_cube_closed_range_and_constant_are_distinct_controls(self):
        x = np.array([[-1.0, 0.25, 2.0]])
        lo, hi = check.cube_bounds(x)
        self.assertAlmostEqual(lo[0], 2 / 3)
        self.assertAlmostEqual(hi[0], (4 + 0.75**2 + 4) / 3)
        self.assertLess(lo[0], np.mean((x - 0.5)**2))

    def test_all_seven_metric_names_and_confusion_counts(self):
        summary = check.control_summary(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]))
        metrics = summary["fixed_cutoff"]
        self.assertEqual(set(metrics), {"TP", "FP", "FN", "TN", "DR", "FA", "SP", "PR", "ACC", "F1", "AUC"})
        self.assertEqual(metrics["ACC"], 75)
        self.assertAlmostEqual(metrics["PR"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
