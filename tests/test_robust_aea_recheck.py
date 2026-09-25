"""Regression checks against preserved summaries, not new research scoring."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/takiddin-2021-robust-poisoning"
spec = importlib.util.spec_from_file_location("aea_recheck", STUDY / "checks/aea_recheck.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class AEARecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((STUDY / "results/aea_repair_20260925/result.json").read_text())["cases"]

    def summary(self, level, branch):
        return M.case_summary(self.cases[level][branch]["analysis"])

    def test_each_poisoning_condition_uses_its_own_favorable_fa_cap(self):
        self.assertEqual(self.summary("p00", "feature_z_mse")["favorable_FA_cap"], 5.25)
        for branch, expected in (("feature_z_mse", 100.), ("raw_unit_mse", 100.),
                                 ("global_z_mse", 96.81547619047619)):
            summary = self.summary("p30", branch)
            self.assertEqual(summary["favorable_FA_cap"], 18.45)
            self.assertAlmostEqual(summary["all_cutoff_DR_upper_bound"], expected)

    def test_mae_free_cutoff_pass_does_not_rescue_printed_cutoff(self):
        for level, minimum in (("p00", 36.75239234449761), ("p30", 33.791866028708135)):
            summary = self.summary(level, "feature_z_mae")
            self.assertEqual(summary["all_cutoff_DR_upper_bound"], 100.)
            self.assertTrue(summary["fixed_cutoff_excluded_by_FA"])
            self.assertAlmostEqual(summary["fixed_cutoff_favorable_minimum_FA"], minimum)
            self.assertGreater(summary["free_cutoff_interval"]["minimum_inclusive"], .515)

    def test_raw_unit_p00_fails_by_detection_despite_sufficient_fa(self):
        summary = self.summary("p00", "raw_unit_mse")
        self.assertFalse(summary["fixed_cutoff_excluded_by_FA"])
        self.assertTrue(summary["fixed_cutoff_excluded_by_DR"])
        self.assertAlmostEqual(summary["fixed_cutoff_favorable_maximum_DR"], 90.0297619047619)


if __name__ == "__main__":
    unittest.main()
