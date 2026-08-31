"""Hand-sized software fixtures, not scientific experiments."""

import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score


SCRIPT = Path(__file__).resolve().parents[1] / "studies/atk-2022-deep-autoencoder/checks/post_anchor_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("post_anchor_diagnostics", SCRIPT)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class PostAnchorTests(unittest.TestCase):
    def test_exact_two_coordinate_extrema(self):
        x = np.array([[0, 0], [1, 0], [2, -1], [-1, -1]], dtype=float)
        lower, upper, uniform, energy = diag.simplex_bounds(x)
        np.testing.assert_allclose(lower, [0.25, 0, 1, 2.25])
        np.testing.assert_allclose(upper, [0.5, 1, 4, 2.5])
        np.testing.assert_allclose(uniform, np.mean((x - 0.5) ** 2, axis=1))
        np.testing.assert_allclose(energy, np.mean(x ** 2, axis=1))

    def test_dense_simplex_grid_agrees_with_extrema(self):
        x = np.array([[0.4, 0.7], [-0.2, 0.2], [1.2, -0.6]])
        lower, upper, _, _ = diag.simplex_bounds(x)
        t = np.linspace(0, 1, 1001)
        reconstructions = np.stack((t, 1 - t), axis=1)
        scores = np.mean((x[:, None, :] - reconstructions[None, :, :]) ** 2, axis=2)
        np.testing.assert_allclose(scores.min(axis=1), lower, atol=1e-6)
        np.testing.assert_allclose(scores.max(axis=1), upper)

    def test_domain_contains_random_softmax_outputs(self):
        rng = np.random.default_rng(5)
        x, logits = rng.normal(size=(20, 48)), rng.normal(size=(20, 48))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        r = exp / exp.sum(axis=1, keepdims=True)
        lower, upper, uniform, _ = diag.simplex_bounds(x)
        scores = np.mean((x - r) ** 2, axis=1)
        self.assertTrue(np.all(lower <= scores))
        self.assertTrue(np.all(scores <= upper))
        self.assertTrue(np.all(lower <= uniform))
        self.assertTrue(np.all(uniform <= upper))

    def test_single_coordinate_and_invalid_input(self):
        lower, upper, _, _ = diag.simplex_bounds([[3], [-2]])
        np.testing.assert_allclose(lower, [4, 9])
        np.testing.assert_allclose(lower, upper)
        with self.assertRaises(ValueError):
            diag.simplex_bounds([[np.nan, 1]])

    def test_oracle_matches_brute_force_all_endpoint_assignments(self):
        y = np.array([0, 0, 1, 1])
        lower, upper = np.array([0.2, 0.5, 0.1, 0.3]), np.array([0.9, 0.7, 0.6, 0.8])
        for reverse in (False, True):
            best_acc, best_auc = 0, 0
            for picks in itertools.product((0, 1), repeat=4):
                score = np.where(picks, upper, lower) * (-1 if reverse else 1)
                best_auc = max(best_auc, roc_auc_score(y, score) * 100)
                for t in np.r_[-np.inf, np.unique(score)]:
                    pred = score > t
                    acc = 50 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
                    best_acc = max(best_acc, acc)
            actual = diag.oracle_envelope(y, lower, upper, reverse=reverse)
            self.assertAlmostEqual(actual["max_ACC"], best_acc)
            self.assertAlmostEqual(actual["max_AUC"], best_auc)

    def test_oracle_ties_use_strict_greater_rule(self):
        y = np.array([0, 0, 1, 1])
        endpoint = np.full(4, 0.58)
        result = diag.oracle_envelope(y, endpoint, endpoint)
        self.assertEqual(result["max_ACC"], 50)
        self.assertEqual(result["max_AUC"], 50)
        self.assertEqual(result["at_printed_threshold"], {"max_DR": 0.0, "min_FA": 0.0})
        self.assertFalse(result["target_pair_not_excluded"])

    def test_reported_oracle_cutoff_uses_strict_greater_semantics(self):
        y = np.array([0, 0, 1, 1])
        lower, upper = np.array([0.2, 0.4, 0.1, 0.3]), np.array([0.9, 0.8, 0.7, 0.7])
        result = diag.oracle_envelope(y, lower, upper)
        selected = result["at_FA_cap"]["15.0"]
        self.assertEqual(selected["max_DR"], 100 * np.mean(upper[y == 1] > selected["threshold"]))
        self.assertEqual(selected["FA"], 100 * np.mean(lower[y == 0] > selected["threshold"]))

    def test_relaxation_never_worse_than_feasible_scores(self):
        rng = np.random.default_rng(8)
        y = np.tile([0, 1], 20)
        lower = rng.uniform(0, 1, len(y))
        upper = lower + rng.uniform(0, 1, len(y))
        actual = lower + rng.uniform(size=len(y)) * (upper - lower)
        bound = diag.oracle_envelope(y, lower, upper)
        self.assertGreaterEqual(bound["max_AUC"], 100 * roc_auc_score(y, actual))
        for t in np.r_[-np.inf, actual]:
            pred = actual > t
            self.assertGreaterEqual(bound["max_ACC"] + 1e-10, 50 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))

    def test_customer_pairing_cancels_identical_methods(self):
        ids = np.repeat([10, 20, 30], [2, 1, 3])
        values = np.linspace(0, 1, 42)
        result = diag.paired_customer_statistics({"trained": values, "zero": values.copy()}, ids, bootstraps=100)
        comparison = result["comparisons"]["zero"]
        self.assertEqual(result["customers"], 3)
        self.assertEqual(comparison["original_ACC_gain_pp"], 0)
        self.assertEqual(comparison["original_ACC_gain_95CI_pp"], [0, 0])
        self.assertTrue(comparison["within_predeclared_plus_minus_1pp"])
        for row in comparison["per_attack"]:
            self.assertEqual(row["source_pair_win_gain_95CI_pp"], [0, 0])

    def test_customer_intervals_detect_known_full_advantage(self):
        ids = np.array([1, 1, 2, 3])
        trained = np.r_[np.zeros(4), np.ones(24)]
        control = 1 - trained
        result = diag.paired_customer_statistics({"trained": trained, "zero": control}, ids, bootstraps=100)
        comparison = result["comparisons"]["zero"]
        self.assertEqual(comparison["original_ACC_gain_pp"], 100)
        self.assertEqual(comparison["original_ACC_gain_95CI_pp"], [100, 100])
        self.assertFalse(comparison["within_predeclared_plus_minus_1pp"])

    def test_energy_bands_can_detect_signal_hidden_by_equal_energy(self):
        labels = np.tile([0, 1], 20)
        energy = np.ones(40)
        result = diag.energy_band_rankings(labels, energy, labels.astype(float))
        self.assertEqual(result["pair_weighted_within_bin_AUC"]["energy"], 50)
        self.assertEqual(result["pair_weighted_within_bin_AUC"]["trained"], 100)

    def test_decision_changes_preserve_beneficial_and_harmful(self):
        result = diag.changed_decisions(np.array([0, 1, 0, 1]),
                                       np.array([0, 1, 1, 0]), np.array([1, 1, 0, 0]))
        self.assertEqual(result, {"rows": 4, "changed": 2, "beneficial": 1, "harmful": 1})

    def test_figures_render_from_hand_sized_fixture(self):
        y = np.tile([0, 1], 4)
        bounds = diag.oracle_envelope(y, np.zeros(8), np.ones(8))
        values = np.linspace(0, 1, 28)
        customer = diag.paired_customer_statistics({"trained": values, "zero": values}, np.array([1, 2, 3, 4]), bootstraps=10)
        record = {"bounds": {"full": {"printed": bounds}, "original": {"printed": bounds}},
                  "metrics": {"full": {"trained": {"FA": 20, "DR": 50}}},
                  "customer_statistics": customer}
        with tempfile.TemporaryDirectory() as temp:
            diag.make_figures(record, Path(temp))
            self.assertTrue((Path(temp) / "output-domain-envelope.svg").is_file())
            self.assertTrue((Path(temp) / "useful-work-by-attack.svg").is_file())


if __name__ == "__main__":
    unittest.main()
