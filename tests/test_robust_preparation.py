"""Software fixtures for Paper 3; no actual consumption data or model fitting."""

from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "studies/takiddin-2021-robust-poisoning/reproduction"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = load("robust_source_data", REPRO / "download_data.py")
with patch.dict(sys.modules, {"download_data": D}):
    P = load("robust_preparation", REPRO / "prepare_data.py")


class RobustPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.meter, cls.day = P.fixture()

    def test_customer_selection_is_nested_deterministic_and_residential(self):
        mapping = {i: 1 if i < 30 else 2 for i in range(1, 40)}
        small = P.select_customers(mapping, 8, 42)
        large = P.select_customers(mapping, 20, 42)
        np.testing.assert_array_equal(small, P.select_customers(mapping, 8, 42))
        self.assertTrue(set(small) <= set(large) <= set(range(1, 30)))
        with self.assertRaises(ValueError):
            P.select_customers(mapping, 3000, 42)

    def test_allocation_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation.csv"
            path.write_text("ID,Code\n1,1\n1,1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                D.read_allocation(path)

    def test_strict_parser_rejects_missing_duplicate_and_extra_slots(self):
        rows = [[1, day * 100 + slot, float(slot)]
                for day in range(1, 5) for slot in range(1, 49)]
        rows.remove([1, 248, 48.0])
        rows.extend([[1, 301, 1.0], [1, 449, 49.0], [1, 450, 50.0]])
        frame = pd.DataFrame(rows, columns=["meter_id", "day_time", "kwh"])
        values, meter, day, info = P.profiles(frame)
        self.assertEqual(values.shape, (1, 48))
        self.assertEqual(info["excluded_days"], 3)
        self.assertEqual(info["days_with_duplicate_slots"], 1)
        np.testing.assert_array_equal(day, [1])
        np.testing.assert_array_equal(values[0], np.arange(1, 49))
        frame.loc[0, "kwh"] = -1
        with self.assertRaises(ValueError):
            P.profiles(frame)

    def test_zip_parser_keeps_days_across_file_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            records = [f"1 {19500 + t} {t}\n" for t in range(1, 49)]
            # An unrelated meter with an invalid value is excluded before validation.
            records.append("2 19500 -9\n")
            for name, lines in zip(D.FILES, np.array_split(records, 6)):
                with zipfile.ZipFile(raw / name, "w") as out:
                    out.writestr(name.removesuffix(".zip"), "".join(lines))
            x, _, days, info = P.read_profiles(raw, np.array([1]), 195, 195, chunk_rows=7)
            np.testing.assert_array_equal(x[0], np.arange(1, 49))
            np.testing.assert_array_equal(days, [195])
            self.assertEqual(info["scanned_rows"], 49)

    def test_six_attack_formulas_and_order(self):
        x = np.tile(np.arange(1, 49, dtype=np.float32), (3, 1))
        meter = np.array([1, 1, 2])
        attacked = P.attack_arrays(x, meter, seed=22)
        self.assertEqual(set(attacked), set(range(1, 7)))
        np.testing.assert_allclose(attacked[1][0] / x[0], attacked[1][1] / x[1])
        self.assertFalse(np.array_equal(attacked[1][0], attacked[1][2]))
        for attack in (1, 2):
            self.assertTrue(np.all(attacked[attack] >= .1 * x - 1e-6))
            self.assertTrue(np.all(attacked[attack] <= .8 * x + 1e-6))
        for row in attacked[3]:
            zeros = np.flatnonzero(row == 0)
            self.assertTrue(8 <= len(zeros) <= 48)
            np.testing.assert_array_equal(zeros, np.arange(zeros[0], zeros[-1] + 1))
        np.testing.assert_array_equal(attacked[4], np.full_like(x, 24.5))
        self.assertTrue(np.all((attacked[5] >= 2.45) & (attacked[5] <= 19.6)))
        np.testing.assert_array_equal(attacked[6], x[:, ::-1])
        for key, value in attacked.items():
            np.testing.assert_array_equal(value, P.attack_arrays(x, meter, seed=22)[key])

    def test_alternative_attack_completions_are_explicit(self):
        global_alpha = P.attack_arrays(self.x, self.meter, seed=5, alpha_scope="global")[1]
        ratios = global_alpha / self.x
        np.testing.assert_allclose(ratios, np.full_like(ratios, ratios[0, 0]), rtol=1e-6)
        clipped = P.attack_arrays(self.x, self.meter, seed=5, bypass="start-first-clip")[3]
        self.assertTrue(np.all(np.count_nonzero(clipped == 0, axis=1) >= 6))
        with self.assertRaises(ValueError):
            P.attack_arrays(self.x, self.meter, seed=5, bypass="unknown")

    def test_adasyn_matches_stock_library_and_every_parent_interpolation(self):
        b = P.original_rows(self.x, self.meter, self.day, 0)
        a = [P.original_rows(values, self.meter, self.day, key)
             for key, values in P.attack_arrays(self.x, self.meter, seed=1).items()]
        pool = P.join(b, *a)
        actual, info = P.balance(pool, seed=31)
        expected_x, expected_y = ADASYN(sampling_strategy={0: len(self.x) * 6},
                                         random_state=31, n_neighbors=5).fit_resample(pool.x, pool.true_y)
        np.testing.assert_array_equal(actual.x, expected_x)
        np.testing.assert_array_equal(actual.true_y, expected_y)
        lookup = {int(uid): i for i, uid in enumerate(pool.uid)}
        syn = actual.synthetic
        left = np.array([lookup[int(uid)] for uid in actual.parent_a[syn]])
        right = np.array([lookup[int(uid)] for uid in actual.parent_b[syn]])
        rebuilt = (pool.x[left] + actual.mix[syn, None] * (pool.x[right] - pool.x[left])).astype(np.float32)
        np.testing.assert_array_equal(rebuilt, actual.x[syn])
        self.assertTrue(info["all_interpolations_verified"])
        self.assertTrue(np.all(actual.meter[syn] == -1))
        self.assertTrue(np.all(pool.true_y[left] == 0))

    def test_generalized_label_poison_is_nested_and_does_not_change_features(self):
        masks = []
        original = None
        for rate in (0., .1, .2, .3):
            arrays, record = P.prepare(self.x, self.meter, self.day, mode="two-class", rate=rate)
            if original is None:
                original = arrays
            for key in ("train_raw_x", "train_x", "test_x", "test_true_y", "test_uid"):
                np.testing.assert_array_equal(arrays[key], original[key])
            mask = arrays["train_true_y"] != arrays["train_observed_y"]
            self.assertTrue(np.all(arrays["train_true_y"][mask] == 1))
            masks.append(set(arrays["train_uid"][mask]))
            self.assertEqual(record["poisoning"]["selected_customer_count"], int(rate * 20))
            self.assertEqual(record["shared_original_row_identities"], 0)
        self.assertTrue(masks[0] <= masks[1] <= masks[2] <= masks[3])

    def test_novelty_replacement_keeps_test_pool_and_records_overlap(self):
        clean, _ = P.prepare(self.x, self.meter, self.day, mode="novelty", rate=0)
        poisoned, record = P.prepare(self.x, self.meter, self.day, mode="novelty", rate=.3)
        for key in ("test_raw_x", "test_true_y", "test_uid"):
            np.testing.assert_array_equal(clean[key], poisoned[key])
        self.assertEqual(len(clean["train_x"]), len(poisoned["train_x"]))
        self.assertTrue(np.all(poisoned["train_observed_y"] == 0))
        changed = poisoned["train_true_y"] == 1
        self.assertEqual(int(changed.sum()), record["poisoning"]["changed_rows"])
        self.assertEqual(record["shared_original_row_identities"], int(changed.sum()))
        self.assertTrue(np.all(poisoned["train_attack"][changed] > 0))
        self.assertFalse(np.array_equal(clean["scaler_mean"], poisoned["scaler_mean"]))

    def test_scaling_uses_only_actual_training_inputs(self):
        arrays, _ = P.prepare(self.x, self.meter, self.day, mode="novelty", rate=.3)
        scaler = StandardScaler().fit(arrays["train_raw_x"].astype(np.float64))
        np.testing.assert_array_equal(arrays["scaler_mean"], scaler.mean_)
        np.testing.assert_array_equal(arrays["test_x"], scaler.transform(arrays["test_raw_x"]).astype(np.float32))
        self.assertFalse(np.allclose(arrays["test_x"].mean(axis=0), 0, atol=.01))

    def test_customer_specific_poison_denominators(self):
        for mode in ("novelty", "two-class"):
            arrays, record = P.prepare(self.x, self.meter, self.day, mode=mode,
                                       scope="customer-specific", customer=1, rate=.3)
            self.assertEqual(record["poisoning"]["changed_rows"], int(.3 * record["training_rows"]))
            for part in ("train", "test"):
                self.assertTrue(np.all(arrays[part + "_meter"][~arrays[part + "_synthetic"]] == 1))
        _, alternate = P.prepare(self.x, self.meter, self.day, mode="two-class",
                                  scope="customer-specific", customer=1, rate=.3,
                                  denominator="malicious-training")
        self.assertEqual(alternate["poisoning"]["changed_rows"], int(.3 * alternate["poisoning"]["eligible_rows"]))

    def test_serialization_preserves_arrays_and_refuses_overwrite(self):
        arrays, record = P.prepare(self.x, self.meter, self.day, mode="two-class")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            saved = P.save_arrays(output, arrays, record)
            for name, value in arrays.items():
                np.testing.assert_array_equal(value, np.load(output / (name + ".npy"), allow_pickle=False))
                self.assertEqual(saved["files"][name + ".npy"]["sha256"],
                                 D.hashes(output / (name + ".npy"))["sha256"])
            self.assertEqual(json.loads((output / "metadata.json").read_text()), saved)
            with self.assertRaises(FileExistsError):
                P.save_arrays(output, arrays, record)

    def test_real_preparation_requires_compute_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                P.require_compute()


if __name__ == "__main__":
    unittest.main()
