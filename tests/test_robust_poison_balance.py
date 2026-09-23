"""Constructed software fixtures only; never load research observations."""

from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from imblearn.over_sampling import ADASYN

from tests.test_robust_baseline import A, R, load
from tests.test_robust_preparation import P
from tests.test_robust_split_control import C

PATH = Path(__file__).resolve().parents[1] / "studies/takiddin-2021-robust-poisoning/checks/poison_balance.py"
with patch.dict("sys.modules", {"prepare_data": P, "run_experiment": R,
                               "analyze_results": A, "split_resampling": C}):
    D = load("robust_poison_balance", PATH)


class PoisonBalanceTests(unittest.TestCase):
    def test_version_preflight_resolves_the_numpy_import_alias(self):
        self.assertEqual(D.versions(), D.EXPECTED_VERSIONS)
        self.assertEqual(D.versions()["numpy"], np.__version__)

    def test_incompatible_runtime_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(D, "require_compute"), patch.object(D, "versions", return_value={}):
            output = Path(directory) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "frozen main CPU"):
                D.run(output, output, output)
            self.assertFalse(output.exists())

    @classmethod
    def setUpClass(cls):
        x, meter, day = P.fixture()
        arrays, _ = P.prepare(x, meter, day, mode="two-class", rate=0.)
        rows = [P.Rows(**{f.name: arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)]
                         for f in fields(P.Rows)}) for prefix in ("train", "test")]
        trains, cls.evaluation, _, cls.population, _ = C.design(*rows)
        cls.originals = trains["B"]
        balanced, _ = P.balance(cls.originals, seed=P.library_seed(D.SEED, 202))
        cls.before, cls.after, cls.metadata = {}, {}, {}
        for rate in (0., .3):
            cls.before[rate], _ = C.inputs_for(balanced, cls.evaluation, cls.population, rate)
            cls.after[rate], cls.metadata[rate] = D.prepare_control(cls.originals, cls.evaluation, cls.population, rate)

    def test_zero_poisoning_matches_every_original_control_array(self):
        self.assertEqual(self.before[0.].keys(), self.after[0.].keys())
        for name in self.before[0.]:
            np.testing.assert_array_equal(self.before[0.][name], self.after[0.][name], err_msg=name)

    def test_matches_uninstrumented_adasyn_on_corrupted_labels(self):
        corrupted, _ = P.poison(self.originals, {}, mode="two-class", scope="generalized",
                               rate=.3, population=self.population, seed=D.SEED)
        sampler = ADASYN(sampling_strategy={0: int(np.count_nonzero(corrupted.observed_y))},
                         random_state=P.library_seed(D.SEED, 202), n_neighbors=5)
        x, y = sampler.fit_resample(corrupted.x, corrupted.observed_y)
        np.testing.assert_array_equal(x, self.after[.3]["train_raw_x"])
        np.testing.assert_array_equal(y, self.after[.3]["train_observed_y"])
        np.testing.assert_array_equal(self.originals.true_y, self.originals.observed_y)

    def test_original_labels_evaluation_lineage_and_scaling_are_verified(self):
        info = D.verify_inputs(self.after[.3], self.before[.3])
        self.assertEqual(info["evaluation_rows"], len(self.evaluation))
        self.assertTrue(info["evaluation_unchanged"])
        self.assertGreater(info["unknown_synthetic_truth"], 0)
        self.assertEqual(info["synthetic_parent_attack_counts_0_1_2"],
                         self.metadata[.3]["resampling"]["synthetic_parent_attack_counts_0_1_2"])
        self.assertEqual(sum(info["synthetic_parent_attack_counts_0_1_2"]),
                         self.after[.3]["train_synthetic"].sum())
        self.assertFalse(self.after[.3]["test_synthetic"].any())

    def test_unknown_truth_cannot_be_reported_as_benign(self):
        arrays = {key: value.copy() for key, value in self.after[.3].items()}
        unknown = np.flatnonzero(arrays["train_true_y"] == -1)[0]
        arrays["train_true_y"][unknown] = 0
        with self.assertRaises(AssertionError):
            D.verify_inputs(arrays, self.before[.3])

    def test_evaluation_cannot_enter_synthetic_ancestry(self):
        arrays = {key: value.copy() for key, value in self.after[.3].items()}
        synthetic = np.flatnonzero(arrays["train_synthetic"])[0]
        arrays["train_parent_b"][synthetic] = arrays["test_uid"][0]
        with self.assertRaisesRegex(ValueError, "parent"):
            D.verify_inputs(arrays, self.before[.3])

    def test_changed_originals_and_evaluation_are_rejected(self):
        for name in ("train_observed_y", "test_true_y", "test_raw_x"):
            arrays = {key: value.copy() for key, value in self.after[.3].items()}
            arrays[name][0] += 1
            with self.subTest(name=name), self.assertRaises(AssertionError):
                D.verify_inputs(arrays, self.before[.3])

    def test_sampler_rejects_synthetic_inputs(self):
        synthetic = self.originals.take(np.arange(len(self.originals)))
        synthetic.synthetic[0] = True
        with self.assertRaisesRegex(ValueError, "original"):
            D.observed_balance(synthetic, 42)

    def test_fixture_fit_reload_and_comparison_with_unknown_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = D.fit_control(self.after[.3], output)
            self.assertIsNone(report["training"]["true_label_accuracy"])
            self.assertTrue(report["reload"]["identical_probabilities"])
            self.assertGreater(report["training"]["observed_label_accuracy"], 0)
            comparison = D.comparison(report, {"B-p00": report, "B-p30": report})
            self.assertTrue(all(v == 0 for v in comparison["D_minus_B_metric_points"].values()))
            self.assertEqual(comparison["D_minus_B_common_cap_DR_points"], {"17.6": 0., "33.3": 0.})
            record = {**report, "status": "complete", "case": "fixture-only", "code_commit": "fixture",
                      "scope": "constructed fixture, never paper evidence", "poisoning_rate": .3}
            P.save_arrays(output / "inputs", self.after[.3], self.metadata[.3])
            record.update(input_metadata_sha256=A.digest(output / "inputs/metadata.json"),
                          preparation=self.metadata[.3],
                          input_audit=D.verify_inputs(self.after[.3], self.before[.3]),
                          zero_poison_guard={"all_arrays_identical": True, "arrays": len(self.after[.3]), "new_fits": 0},
                          comparison=comparison)
            (output / "result.json").write_text(json.dumps(record))
            self.assertEqual(A.audit_result(output)["status"], "verified")
            with patch.object(D, "references", return_value=({"B-p00": report, "B-p30": report},
                                                             {"B-p30": self.before[.3]})):
                self.assertEqual(D.audit(output, output)["status"], "verified")
                record["training"]["true_label_accuracy"] = 0.
                (output / "result.json").write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "unknown labels"):
                    D.audit(output, output)

    def test_persistence_rejects_changed_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            P.save_arrays(root, self.after[.3], self.metadata[.3])
            expected = A.digest(root / "metadata.json")
            arrays, _ = D.read_inputs(root, expected)
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], self.after[.3][name])
            np.save(root / "train_true_y.npy", np.zeros(3))
            with self.assertRaisesRegex(ValueError, "array changed"):
                D.read_inputs(root, expected)

    def test_real_control_requires_compute_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            output = Path(directory) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                D.run(output, output, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
