"""Constructed witnesses only; no CER rows, model fitting or inference."""

import itertools
from pathlib import Path
import unittest

import numpy as np

from tests.test_robust_baseline import load

PATH = Path(__file__).resolve().parents[1] / "studies/takiddin-2021-robust-poisoning/checks/aea_geometry.py"
G = load("robust_aea_geometry", PATH)


class AEAGeometryTests(unittest.TestCase):
    def test_hand_calculated_mixed_domain_row(self):
        lower, upper = G.mse_box_bounds([[-2., .25, 2.]])
        np.testing.assert_allclose(lower, [5. / 3])
        np.testing.assert_allclose(upper, [13.5625 / 3])

    def test_projection_attains_lower_bound_in_closed_box(self):
        x = np.array([[-2., .25, 2.], [.1, .5, .9]])
        lower, _ = G.mse_box_bounds(x)
        np.testing.assert_array_equal(lower, np.mean((x - np.clip(x, 0, 1)) ** 2, axis=1))
        self.assertEqual(lower[1], 0.)

    def test_upper_bound_equals_exhaustive_corner_maximum(self):
        x = np.array([[-2., .25, 2.], [.1, .5, .9]])
        _, upper = G.mse_box_bounds(x)
        errors = [np.mean((x - np.array(corner)) ** 2, axis=1)
                  for corner in itertools.product((0., 1.), repeat=3)]
        np.testing.assert_array_equal(upper, np.max(errors, axis=0))

    def test_random_bounded_reconstructions_obey_both_bounds(self):
        generator = np.random.default_rng(20260924)
        x = generator.normal(0, 2, (64, 48))
        lower, upper = G.mse_box_bounds(x)
        for _ in range(50):
            errors = np.mean((x - generator.uniform(size=x.shape)) ** 2, axis=1)
            self.assertTrue(np.all(errors >= lower - 1e-12))
            self.assertTrue(np.all(errors <= upper + 1e-12))

    def test_standardized_fixture_cannot_reconstruct_negative_row(self):
        x = np.vstack((-np.ones(48), np.ones(48)))
        np.testing.assert_array_equal(x.mean(axis=0), np.zeros(48))
        np.testing.assert_array_equal(x.std(axis=0), np.ones(48))
        lower, upper = G.mse_box_bounds(x)
        np.testing.assert_array_equal(lower, [1., 0.])
        self.assertTrue(G.interval_decisions(lower, upper, .51)["must_alarm"][0])

    def test_range_mismatch_alone_does_not_prove_detection_failure(self):
        # A legal constant reconstruction perfectly separates this toy task.
        x = np.array([[.25] * 48, [-1.25] * 48])
        reconstruction = np.full_like(x, .25)
        np.testing.assert_array_equal(np.mean((x - reconstruction) ** 2, axis=1) > .51,
                                      [False, True])

    def test_threshold_strictness_and_unknown_middle(self):
        result = G.interval_decisions([.51, .52, .1], [.7, .8, .51], .51)
        np.testing.assert_array_equal(result["must_alarm"], [False, True, False])
        np.testing.assert_array_equal(result["must_be_benign"], [False, False, True])
        np.testing.assert_array_equal(result["unresolved"], [True, False, False])

    def test_favorable_rounding_can_remove_a_literal_threshold_certificate(self):
        lower, upper = np.array([.512]), np.array([1.])
        self.assertTrue(G.interval_decisions(lower, upper, .51)["must_alarm"][0])
        self.assertFalse(G.interval_decisions(lower, upper, .515)["must_alarm"][0])

    def test_rmse_and_sum_error_thresholds_are_not_mse_thresholds(self):
        mse = .3
        self.assertFalse(mse > .51)
        self.assertTrue(np.sqrt(mse) > .51)
        self.assertTrue(48 * mse > .51)

    def test_classification_loss_does_not_identify_reconstruction(self):
        x, y = np.array([.25, .75]), np.array([0., 1.])
        complement = 1 - x
        logits = 40 * (x - .5)
        compensated_logits = -40 * (complement - .5)
        np.testing.assert_array_equal(logits, compensated_logits)
        loss = np.mean(np.logaddexp(0., logits) - y * logits)
        self.assertLess(loss, .0001)
        self.assertEqual(np.mean((x - x) ** 2), 0.)
        self.assertEqual(np.mean((x - complement) ** 2), .25)

    def test_invalid_arrays_and_intervals_are_rejected(self):
        for x in ([], [[]], [1., 2.], [[np.nan]], [[np.inf]]):
            with self.assertRaises(ValueError):
                G.mse_box_bounds(x)
        for lo, hi, threshold in (([1], [0], .51), ([-1], [0], .51),
                                   ([0], [1, 2], .51), ([0], [1], np.nan),
                                   ([0], [1], -.1), ([], [], .51)):
            with self.assertRaises(ValueError):
                G.interval_decisions(lo, hi, threshold)


if __name__ == "__main__":
    unittest.main()
