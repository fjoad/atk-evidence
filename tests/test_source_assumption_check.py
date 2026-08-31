"""Hand-sized fixtures only; no paper data or experimental scoring."""

import itertools
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "studies/atk-2022-deep-autoencoder/checks"))
import source_assumption_check as check


class SourceAssumptionTests(unittest.TestCase):
    def test_streamed_moments_match_direct_values_without_mutation(self):
        x = np.arange(240, dtype=float).reshape(80, 3)
        before = x.copy()
        mean, second = check.moments(x)
        np.testing.assert_allclose(mean, x.mean(axis=0))
        np.testing.assert_allclose(second, (x*x).mean(axis=0))
        np.testing.assert_array_equal(x, before)

    def test_alternative_statistics_recover_full_population_moments(self):
        rng = np.random.default_rng(24)
        benign = rng.normal(10, 3, (40, 4))
        attacks = rng.normal(4, 2, (240, 4))
        together = np.concatenate([benign, attacks])
        scalers = check.alternative_scalers(together.mean(0), together.std(0),
                                           benign.mean(0), (benign**2).mean(0))
        self.assertAlmostEqual(scalers["joint_scalar"]["mean"], together.mean())
        self.assertAlmostEqual(scalers["joint_scalar"]["scale"], together.std())
        np.testing.assert_allclose(scalers["benign"]["scale"], benign.std(0))
        np.testing.assert_allclose(scalers["malicious"]["mean"], attacks.mean(0))
        np.testing.assert_allclose(scalers["malicious"]["scale"], attacks.std(0))
        y = np.concatenate([np.zeros(40), np.ones(240)])
        z = check.transform(together, y, scalers, "separate_class_feature_softmax")
        np.testing.assert_allclose(z[y == 0].mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(z[y == 1].mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(z[y == 1].std(0), 1, atol=1e-14)

    def test_cube_extrema_match_vertices_and_projection(self):
        x = np.array([[-2, .4, 2], [0, 1, .5], [1, 1, 1]], dtype=float)
        lower, upper = check.cube_bounds(x)
        vertices = np.array(list(itertools.product([0, 1], repeat=3)))
        distances = ((x[:, None, :] - vertices[None, :, :])**2).mean(2)
        np.testing.assert_allclose(upper, distances.max(1))
        np.testing.assert_allclose(lower, [5/3, 0, 0])
        sl, su, _, _ = check.simplex_bounds(x)
        self.assertTrue(np.all(lower <= sl + 1e-12))
        self.assertTrue(np.all(upper >= su - 1e-12))

    def test_pilot_keeps_every_attack_sibling(self):
        rows = check.selected_rows(1000, "pilot").reshape(7, -1)
        self.assertEqual(rows.shape, (7, 64))
        for group in range(7):
            np.testing.assert_array_equal(rows[group] - group*1000, rows[0])
        np.testing.assert_array_equal(check.selected_rows(3, "full"), np.arange(21))

    def test_positive_score_transforms_preserve_roc_and_threshold_decisions(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        mse = np.array([0, .25, 1, 1, 4, 9])
        fpr, tpr, _ = roc_curve(y, mse, drop_intermediate=False)
        for converted in (48*mse, np.sqrt(mse)):
            a, b, _ = roc_curve(y, converted, drop_intermediate=False)
            np.testing.assert_array_equal(a, fpr)
            np.testing.assert_array_equal(b, tpr)
            self.assertEqual(roc_auc_score(y, converted), roc_auc_score(y, mse))
        np.testing.assert_array_equal(mse > 1, 48*mse > 48)
        np.testing.assert_array_equal(mse > 1, np.sqrt(mse) > 1)

    def test_bad_moments_fail_closed(self):
        with self.assertRaises(ValueError):
            check.scales_from_moments(np.array([3.0]), np.array([2.0]))
        with self.assertRaises(ValueError):
            check.moments(np.array([[np.nan, 2.0]]))

    def test_saved_records_match_the_frozen_sources_and_transfer_hashes(self):
        study = Path(check.__file__).resolve().parent.parent
        results = study / "results/source_assumption_20260831"
        execution = json.loads((results / "execution.json").read_text())
        for stage in ("pilot", "full"):
            raw = (results / f"{stage}.json").read_bytes()
            record = json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), execution[f"{stage}_sha256"])
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["script_sha256"], check.digest(Path(check.__file__)))
            self.assertEqual(record["contract_sha256"], check.digest(study / "SOURCE_ASSUMPTION_CHECK.md"))
            self.assertEqual(record["helper_sha256"], check.digest(study / "checks/post_anchor_diagnostics.py"))
            self.assertEqual(set(record["branches"]), set(check.BRANCHES))
        self.assertTrue(record["reference_original_bound_reproduced"])
        self.assertEqual(record["rows"], 5255369)

    def test_discussion_detection_limits_round_up_from_the_record(self):
        study = Path(check.__file__).resolve().parent.parent
        record = json.loads((study / "results/source_assumption_20260831/full.json").read_text())
        finding = (study / "SOURCE_ASSUMPTION_FINDING.md").read_text()
        for branch in record["branches"].values():
            limit = branch["printed"]["at_printed_threshold"]["max_DR"]
            self.assertIn(f"{math.ceil(100 * limit) / 100:.2f}%", finding)
        self.assertIn("NOT a new Table III result", finding)
        self.assertIn("Stop here for discussion", finding)


if __name__ == "__main__":
    unittest.main()
