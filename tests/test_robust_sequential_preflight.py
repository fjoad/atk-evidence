"""Timing/learning gates must reject partial or wrong-branch evidence."""

import json
import sys
import tempfile
from pathlib import Path
import unittest

from tests.test_robust_baseline import load
from tests.test_robust_sequential import G, ROOT

sys.path.insert(0, str(ROOT / "studies/takiddin-2021-robust-poisoning/checks"))
P = load("sequential_preflight_fixture", ROOT /
         "studies/takiddin-2021-robust-poisoning/checks/sequential_preflight.py")


class SequentialPreflightTests(unittest.TestCase):
    def test_variable_epoch_gate_uses_slower_warm_epoch_and_requires_completion(self):
        rows = [{"updates": 45, "seconds": t} for t in (100., 60., 72.)]
        self.assertEqual(P.timing_gate(rows)["projected_fit_seconds"], 3600.)
        self.assertTrue(P.timing_gate(rows)["passed"])
        rows[2]["seconds"] = 72.01
        self.assertFalse(P.timing_gate(rows)["passed"])
        rows[2]["updates"] = 44
        with self.assertRaises(ValueError):
            P.timing_gate(rows)
        with self.assertRaises(ValueError):
            P.timing_gate(rows[:2])

    def test_failed_sigmoid_cannot_be_rescued_by_a_passing_control_flag(self):
        case = {"status": "complete", "updates": 300, "learning_gate_passed": False,
                "initial_weights_sha256": "same", "gradient_group_gate_passed": True,
                "weights_finite": True, "reload_exact": True, "maximum_constrained_norm": 1.}
        result = {"status": "complete", "sigmoid_eligible_for_timing": True,
                  "branches": {"sigmoid": {"interpretation": "I-SEQ-scalar-sigmoid",
                                            "cases": {"normal": case, "reversed": case}},
                               "linear": {"gate_passed": True}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning.json"
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Sigmoid learning gate"):
                P.verify_learning(path, G.digest(path))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                P.verify_learning(path, "wrong")


if __name__ == "__main__":
    unittest.main()
