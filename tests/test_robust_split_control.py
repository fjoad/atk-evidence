"""Constructed-input checks of the split/resampling experiment's pairing."""

from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_baseline import A, R
from tests.test_robust_preparation import P


PATH = Path(__file__).resolve().parents[1] / "studies/takiddin-2021-robust-poisoning/checks/split_resampling.py"
SPEC = importlib.util.spec_from_file_location("robust_split_control", PATH)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
with patch.dict(sys.modules, {"prepare_data": P, "run_experiment": R, "analyze_results": A}):
    SPEC.loader.exec_module(C)


class SplitControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, meter, day = P.fixture()
        arrays, _ = P.prepare(x, meter, day, mode="two-class", rate=0.)
        cls.train, cls.test = [
            P.Rows(**{f.name: arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)]
                      for f in fields(P.Rows)}) for prefix in ("train", "test")
        ]

    def test_originals_are_recovered_without_regenerating_attacks(self):
        trains, evaluation, common, population, info = C.design(self.train, self.test)
        self.assertEqual(info["original_pool_rows"], 560 * 7)
        self.assertEqual(info["source_days"], 560)
        self.assertEqual(len(population), 20)
        np.testing.assert_array_equal(trains["B"].uid, self.train.uid[~self.train.synthetic])
        np.testing.assert_array_equal(evaluation.x, self.test.x[~self.test.synthetic])
        self.assertFalse(common.synthetic.any())
        self.assertTrue(set(common.uid) <= set(evaluation.uid))

    def test_grouped_training_keeps_all_siblings_and_excludes_common_days(self):
        trains, evaluation, common, _, info = C.design(self.train, self.test)
        grouped = trains["C"]
        self.assertEqual(info["day_training_groups"], 373)
        self.assertEqual(info["heldout_day_groups"], 187)
        self.assertEqual(len(grouped), 373 * 7)
        _, counts = np.unique(grouped.source_id, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(373, 7))
        self.assertEqual(np.intersect1d(grouped.source_id, common.source_id).size, 0)
        heldout = ~np.isin(evaluation.source_id, grouped.source_id)
        np.testing.assert_array_equal(common.uid, evaluation.uid[heldout])
        self.assertEqual(info, C.design(self.train, self.test)[-1])

    def test_incomplete_attack_families_are_rejected(self):
        original = np.flatnonzero(~self.test.synthetic)[0]
        selected = np.arange(len(self.test))
        selected = selected[selected != original]
        with self.assertRaisesRegex(ValueError, "seven"):
            C.design(self.train, self.test.take(selected))

    def test_lineage_checks_both_parent_ids_and_source_day_overlap(self):
        trains, evaluation, common, _, _ = C.design(self.train, self.test)
        for arm in ("B", "C"):
            balanced, _ = P.balance(trains[arm], seed=P.library_seed(C.SEED, 202))
            test = evaluation if arm == "B" else common
            result = C.verify_lineage(balanced, test, disjoint_days=arm == "C")
            self.assertEqual(result["synthetic_parent_crossings"], 0)
            corrupted = balanced.take(np.arange(len(balanced)))
            first_synthetic = np.flatnonzero(corrupted.synthetic)[0]
            corrupted.parent_b[first_synthetic] = test.uid[0]
            with self.assertRaisesRegex(ValueError, "parent"):
                C.verify_lineage(corrupted, test, disjoint_days=arm == "C")
        x = np.ones((1, 48), dtype=np.float32)
        benign = P.original_rows(x, np.array([1]), np.array([195]), 0)
        attack = P.original_rows(x * .5, np.array([1]), np.array([195]), 1)
        self.assertEqual(C.verify_lineage(benign, attack, disjoint_days=False)["shared_source_days"], 1)
        with self.assertRaisesRegex(ValueError, "Source-day"):
            C.verify_lineage(benign, attack, disjoint_days=True)

    def test_poisoning_preserves_raw_evaluation_and_fits_scale_to_training(self):
        trains, _, common, population, _ = C.design(self.train, self.test)
        changed_customers = []
        for arm in ("B", "C"):
            balanced, _ = P.balance(trains[arm], seed=P.library_seed(C.SEED, 202))
            clean, _ = C.inputs_for(balanced, common, population, 0., grouped=arm == "C")
            altered, info = C.inputs_for(balanced, common, population, .3, grouped=arm == "C")
            for key in ("test_raw_x", "test_true_y", "test_uid", "train_raw_x", "scaler_mean", "scaler_scale"):
                np.testing.assert_array_equal(clean[key], altered[key])
            np.testing.assert_allclose(clean["scaler_mean"], clean["train_raw_x"].mean(axis=0, dtype=np.float64))
            flipped = altered["train_true_y"] != altered["train_observed_y"]
            self.assertEqual(int(flipped.sum()), info["poisoning"]["changed_rows"])
            changed_customers.append(set(altered["train_meter"][flipped]))
        self.assertEqual(changed_customers[0], changed_customers[1])
        self.assertEqual(len(changed_customers[0]), 6)

    def test_score_selection_is_by_identity_and_preserves_requested_order(self):
        scores = {"uid": np.array([30, 10, 20]), "labels": np.array([1, 0, 1]),
                  "probabilities": np.array([[.1, .9], [.8, .2], [.4, .6]])}
        result = C.select_scores(scores, np.array([20, 10]))
        np.testing.assert_array_equal(result["uid"], [20, 10])
        np.testing.assert_array_equal(result["probabilities"], [[.4, .6], [.8, .2]])
        with self.assertRaises(KeyError):
            C.select_scores(scores, np.array([40]))
        scores["uid"][1] = 30
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C.select_scores(scores, np.array([30]))

    def test_incomplete_control_cannot_be_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"status": "started"}))
            with self.assertRaisesRegex(ValueError, "did not finish"):
                C.audit(root, root)

    def test_real_control_is_refused_without_compute_allocation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                C.run(path, path, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
