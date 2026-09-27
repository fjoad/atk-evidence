"""Hand-sized controls for collapse timing and local affine attribution."""

import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_baseline import load
from tests.test_robust_sequential import G, ROOT

with patch.dict(sys.modules, {"sequential_constructed": G}):
    T = load("sequential_trace_fixture", ROOT /
             "studies/takiddin-2021-robust-poisoning/checks/sequential_trace.py")


class SequentialTraceTests(unittest.TestCase):
    def test_initial_transient_and_persistent_zeros_are_distinct(self):
        rows = [{"step": step, "train": {"layers": {n: [0., 1., .1, .5] for n in T.STAGES}}}
                for step in range(4)]
        for row in rows:
            row["train"]["layers"]["gru_1"] = [0., 0., 0., 0.]
        rows[1]["train"]["layers"]["gru_8"] = [0., 0., 0., 0.]
        for row in rows[2:]:
            row["train"]["layers"]["classifier_hidden"] = [0., 0., 0., 0.]
        result = T.zero_events(rows)
        self.assertTrue(result["gru_1"]["initially_zero"])
        self.assertEqual(result["gru_1"]["new_zero_steps"], [])
        self.assertEqual(result["gru_8"]["new_zero_steps"], [1])
        self.assertFalse(result["gru_8"]["zero_through_observed_end_from_first_transition"])
        self.assertEqual(result["classifier_hidden"]["new_zero_steps"], [2])
        self.assertTrue(result["classifier_hidden"]["zero_through_observed_end_from_first_transition"])
        json.dumps(result, allow_nan=False)

    def test_bias_and_input_collapse_have_different_counterfactuals(self):
        x, w, b = np.array([[1.], [2.]]), np.array([[1.]]), np.array([0.])
        bias = T.affine_comparison(x, w, b, x, w, [-3.])
        self.assertEqual(bias["cases"]["after_inputs_after_parameters"]["relu_output"][3], 0.)
        self.assertEqual(bias["cases"]["after_inputs_after_weights_before_bias"]["relu_output"][3], 1.)
        self.assertEqual(bias["cases"]["after_inputs_before_weights_after_bias"]["relu_output"][3], 0.)
        self.assertEqual(bias["additive_identity_max_error"], 0.)
        inputs = T.affine_comparison(x, w, b, x*0, w, b)
        self.assertEqual(inputs["cases"]["after_inputs_after_weights_before_bias"]["relu_output"][3], 0.)
        self.assertEqual(inputs["cases"]["before_inputs_after_parameters"]["relu_output"][3], 1.)
        self.assertEqual(inputs["additive_identity_max_error"], 0.)
        json.dumps(bias, allow_nan=False)

    def test_nonfinite_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            T.stats([0., np.nan])
        with self.assertRaises(ValueError):
            T.stats([np.inf])


if __name__ == "__main__":
    unittest.main()
