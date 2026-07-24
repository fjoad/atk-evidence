from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paper_literal_data import (
    load_prepared_sgcc,
    prepare_sgcc_paper_literal,
    save_prepared_sgcc,
)


def tiny_sgcc_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(24):
        base = float(index)
        rows.append(
            {
                "CONS_NO": f"B{index:02d}",
                "FLAG": 0,
                "2014/1/10": base + 3.0,
                "2014/1/1": base,
                "2014/1/3": base + 2.0,
                "2014/1/2": base + 1.0,
            }
        )
    # Place malicious examples among benign points so fixture-scale ADASYN has
    # majority neighbours rather than an isolated minority-only cluster.
    for index, base in enumerate([1.5, 5.5, 9.5, 13.5]):
        rows.append(
            {
                "CONS_NO": f"M{index:02d}",
                "FLAG": 1,
                "2014/1/10": base + 3.0,
                "2014/1/1": base,
                "2014/1/3": base + 2.0,
                "2014/1/2": base + 1.0,
            }
        )
    rows.append(
        {
            "CONS_NO": "DROP_ALL_MISSING",
            "FLAG": 0,
            "2014/1/10": np.nan,
            "2014/1/1": np.nan,
            "2014/1/3": np.nan,
            "2014/1/2": np.nan,
        }
    )
    # Exercise an unresolved leading edge and one bounded interpolation gap.
    rows[0]["2014/1/1"] = np.nan
    rows[0]["2014/1/3"] = np.nan
    return pd.DataFrame(rows)


def wide_sgcc_fixture(days: int = 96) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2014-01-01", periods=days, freq="D")
    for label, count, prefix in ((0, 18, "B"), (1, 6, "M")):
        for customer_index in range(count):
            values = (
                1.0
                + 0.03 * customer_index
                + 0.2 * label
                + np.arange(days, dtype=np.float64) / 100.0
            )
            row: dict[str, object] = {
                "CONS_NO": f"{prefix}{customer_index:02d}",
                "FLAG": label,
            }
            row.update(
                {
                    f"{date.year}/{date.month}/{date.day}": float(value)
                    for date, value in zip(dates, values, strict=True)
                }
            )
            rows.append(row)
    rows[0]["2014/1/1"] = np.nan
    rows[0]["2014/1/20"] = np.nan
    return pd.DataFrame(rows)


class PaperLiteralDataTests(unittest.TestCase):
    def prepare(self):
        return prepare_sgcc_paper_literal(
            tiny_sgcc_fixture(),
            data_seed=20260721,
            validation_fraction=0.15,
            adasyn_neighbors=1,
            expected_feature_count=4,
        )

    def test_chronology_imputation_scaling_and_drop(self) -> None:
        prepared = self.prepare()
        self.assertEqual(
            prepared.dates.astype(str).tolist(),
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-10"],
        )
        counts = prepared.metadata["counts"]
        self.assertEqual(counts["dropped_fully_missing"], 1)
        self.assertEqual(counts["dropped_customer_ids"], ["DROP_ALL_MISSING"])
        self.assertEqual(counts["benign"], 24)
        self.assertEqual(counts["malicious"], 4)

        original_test_count = counts["anomaly_test_original"]
        original_values = np.concatenate(
            [
                prepared.anomaly_train.values,
                prepared.anomaly_validation.values,
                prepared.anomaly_test.values[:original_test_count],
            ]
        )
        np.testing.assert_allclose(original_values.mean(axis=0), 0.0, atol=2e-6)
        np.testing.assert_allclose(original_values.std(axis=0), 1.0, atol=2e-6)

        partitions = [
            prepared.anomaly_train,
            prepared.anomaly_validation,
            prepared.anomaly_test,
        ]
        b0_scaled = next(
            partition.values[np.flatnonzero(partition.sample_ids == "B00")[0]]
            for partition in partitions
            if "B00" in partition.sample_ids
        )
        b0_completed = b0_scaled * prepared.scaler_scale + prepared.scaler_mean
        self.assertAlmostEqual(
            b0_completed[0], float(prepared.imputation_fallback[0]), places=5
        )
        self.assertAlmostEqual(b0_completed[2], 2.0, places=5)

    def test_paper_literal_split_and_adasyn_order(self) -> None:
        prepared = self.prepare()
        counts = prepared.metadata["counts"]
        self.assertEqual(counts["anomaly_b1_total"], 16)
        self.assertEqual(counts["anomaly_b2"], 8)
        self.assertEqual(
            counts["anomaly_train"] + counts["anomaly_validation"], 16
        )
        self.assertTrue((prepared.anomaly_train.labels == 0).all())
        self.assertTrue((prepared.anomaly_validation.labels == 0).all())
        self.assertEqual(
            prepared.anomaly_test.is_synthetic.sum(),
            prepared.metadata["adasyn"]["anomaly_test"]["generated"],
        )
        self.assertEqual(
            prepared.supervised_train.is_synthetic.sum()
            + prepared.supervised_test.is_synthetic.sum(),
            prepared.metadata["adasyn"]["supervised_before_split"]["generated"],
        )
        self.assertGreater(
            prepared.metadata["adasyn"]["supervised_before_split"]["generated"], 0
        )
        anomaly_train_ids = set(prepared.anomaly_train.sample_ids)
        anomaly_validation_ids = set(prepared.anomaly_validation.sample_ids)
        original_test_ids = set(
            prepared.anomaly_test.sample_ids[: counts["anomaly_test_original"]]
        )
        self.assertTrue(anomaly_train_ids.isdisjoint(anomaly_validation_ids))
        self.assertTrue(anomaly_train_ids.isdisjoint(original_test_ids))
        self.assertTrue(anomaly_validation_ids.isdisjoint(original_test_ids))
        self.assertTrue(
            set(prepared.supervised_train.sample_ids).isdisjoint(
                prepared.supervised_test.sample_ids
            )
        )

    def test_preparation_is_deterministic(self) -> None:
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first.metadata, second.metadata)
        for name in (
            "anomaly_train",
            "anomaly_validation",
            "anomaly_test",
            "supervised_train",
            "supervised_test",
        ):
            first_partition = getattr(first, name)
            second_partition = getattr(second, name)
            np.testing.assert_array_equal(
                first_partition.values, second_partition.values
            )
            np.testing.assert_array_equal(
                first_partition.sample_ids, second_partition.sample_ids
            )

    def test_corrected_resampling_keeps_test_rows_untouched(self) -> None:
        prepared = prepare_sgcc_paper_literal(
            tiny_sgcc_fixture(),
            data_seed=20260721,
            validation_fraction=0.15,
            adasyn_neighbors=1,
            expected_feature_count=4,
            scaling="train_benign_only",
            anomaly_adasyn="none",
            supervised_adasyn="customer_split_then_train_only",
        )
        self.assertFalse(prepared.anomaly_test.is_synthetic.any())
        self.assertFalse(prepared.supervised_test.is_synthetic.any())
        self.assertGreater(prepared.supervised_train.is_synthetic.sum(), 0)
        self.assertEqual(
            prepared.metadata["adasyn"]["anomaly_test"]["reason"],
            "disabled_by_branch",
        )
        self.assertEqual(
            prepared.metadata["preprocessing"]["scaling_details"]["fit_population"],
            "anomaly_train_benign_only",
        )
        train_original = set(
            prepared.supervised_train.sample_ids[
                ~prepared.supervised_train.is_synthetic
            ]
        )
        test_original = set(prepared.supervised_test.sample_ids)
        self.assertTrue(train_original.isdisjoint(test_original))
        np.testing.assert_allclose(
            prepared.anomaly_train.values.mean(axis=0), 0.0, atol=2e-6
        )
        np.testing.assert_allclose(
            prepared.anomaly_train.values.std(axis=0), 1.0, atol=2e-6
        )

    def test_all_registered_scaling_interpretations_execute(self) -> None:
        for branch in (
            "joint_featurewise",
            "per_class_featurewise",
            "per_profile",
            "train_benign_only",
        ):
            with self.subTest(branch=branch):
                prepared = prepare_sgcc_paper_literal(
                    tiny_sgcc_fixture(),
                    data_seed=20260721,
                    validation_fraction=0.15,
                    adasyn_neighbors=1,
                    expected_feature_count=4,
                    scaling=branch,
                )
                self.assertEqual(
                    prepared.metadata["config"]["scaling_branch"], branch
                )
                self.assertTrue(np.isfinite(prepared.anomaly_test.values).all())

    def test_cache_round_trip_and_checksum(self) -> None:
        prepared = self.prepare()
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "data" / "derived" / "sgcc-paper-literal"
            npz_path, manifest_path = save_prepared_sgcc(prepared, prefix)
            restored = load_prepared_sgcc(prefix)
            self.assertTrue(npz_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(restored.metadata, prepared.metadata)
            np.testing.assert_array_equal(
                restored.anomaly_test.values, prepared.anomaly_test.values
            )
            with npz_path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_prepared_sgcc(prefix)

    def test_wrong_feature_count_and_duplicate_ids_rejected(self) -> None:
        fixture = tiny_sgcc_fixture()
        with self.assertRaisesRegex(ValueError, "expected 1034"):
            prepare_sgcc_paper_literal(fixture, adasyn_neighbors=1)
        fixture.loc[1, "CONS_NO"] = fixture.loc[0, "CONS_NO"]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_sgcc_paper_literal(
                fixture,
                expected_feature_count=4,
                adasyn_neighbors=1,
            )

    def test_all_sgcc_representation_branches_have_declared_sample_units(
        self,
    ) -> None:
        frame = wide_sgcc_fixture()
        expected = {
            "full_1034": (96, 1),
            "windows_48_nonoverlap": (48, 2),
            "windows_48_rolling": (48, 49),
            "first_48": (48, 1),
            "last_48": (48, 1),
            "binned_mean_48": (48, 1),
        }
        for branch, (features, samples_per_customer) in expected.items():
            with self.subTest(branch=branch):
                prepared = prepare_sgcc_paper_literal(
                    frame,
                    expected_feature_count=96,
                    adasyn_neighbors=2,
                    representation=branch,
                    missing="zero_fill",
                    anomaly_adasyn="none",
                    supervised_adasyn="customer_split_then_train_only",
                )
                self.assertEqual(prepared.dates.size, features)
                self.assertEqual(
                    prepared.metadata["preprocessing"]["representation_details"][
                        "samples_per_customer"
                    ],
                    samples_per_customer,
                )
                self.assertEqual(
                    prepared.metadata["counts"]["retained"],
                    24 * samples_per_customer,
                )
                train_sources = {
                    item.split("::", 1)[0]
                    for item in prepared.anomaly_train.sample_ids
                }
                validation_sources = {
                    item.split("::", 1)[0]
                    for item in prepared.anomaly_validation.sample_ids
                }
                benign_test_sources = {
                    item.split("::", 1)[0]
                    for item in prepared.anomaly_test.sample_ids[
                        prepared.anomaly_test.labels == 0
                    ]
                }
                self.assertTrue(train_sources.isdisjoint(validation_sources))
                self.assertTrue(train_sources.isdisjoint(benign_test_sources))
                self.assertTrue(
                    validation_sources.isdisjoint(benign_test_sources)
                )

    def test_all_sgcc_missing_data_branches_execute_and_record_drops(self) -> None:
        frame = wide_sgcc_fixture()
        for branch in (
            "drop_incomplete",
            "zero_fill",
            "interpolate_edge_median",
            "customer_mean",
        ):
            with self.subTest(branch=branch):
                prepared = prepare_sgcc_paper_literal(
                    frame,
                    expected_feature_count=96,
                    adasyn_neighbors=2,
                    representation="first_48",
                    missing=branch,
                    anomaly_adasyn="none",
                    supervised_adasyn="customer_split_then_train_only",
                )
                self.assertEqual(
                    prepared.metadata["config"]["missing_branch"], branch
                )
                self.assertTrue(np.isfinite(prepared.anomaly_train.values).all())
                expected_drop = 1 if branch == "drop_incomplete" else 0
                self.assertEqual(
                    prepared.metadata["counts"]["dropped_incomplete"],
                    expected_drop,
                )

    def test_row_random_window_split_is_explicitly_not_customer_disjoint(
        self,
    ) -> None:
        prepared = prepare_sgcc_paper_literal(
            wide_sgcc_fixture(),
            expected_feature_count=96,
            adasyn_neighbors=2,
            representation="windows_48_rolling",
            missing="zero_fill",
            split_unit="row_random",
            anomaly_adasyn="none",
            supervised_adasyn="customer_split_then_train_only",
        )
        b1_sources = {
            item.split("::", 1)[0]
            for item in np.concatenate(
                [
                    prepared.anomaly_train.sample_ids,
                    prepared.anomaly_validation.sample_ids,
                ]
            )
        }
        b2_sources = {
            item.split("::", 1)[0]
            for item in prepared.anomaly_test.sample_ids[
                prepared.anomaly_test.labels == 0
            ]
        }
        self.assertFalse(b1_sources.isdisjoint(b2_sources))
        self.assertEqual(
            prepared.metadata["config"]["split_unit_branch"], "row_random"
        )


if __name__ == "__main__":
    unittest.main()
