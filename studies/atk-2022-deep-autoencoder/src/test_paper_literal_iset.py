from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from paper_literal_iset import (
    ALLOCATION_FILENAME,
    ARCHIVE_FILENAMES,
    HALF_HOUR_COLUMNS,
    SCIENCEDB_ALLOCATION_BRANCH,
    SCIENCEDB_ALLOCATION_FILENAME,
    _resolve_attack_seed,
    _select_meter_population,
    load_prepared_iset,
    prepare_iset_paper_literal,
    save_prepared_iset,
    verify_authorized_iset_files,
)
from paper_literal_runner import build_threshold_population


def prepared_profile_fixture(
    *,
    residential_meters: int = 12,
    days: int = 4,
    include_nonresidential: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, int]] = []
    total_meters = residential_meters + (6 if include_nonresidential else 0)
    slots = np.arange(48, dtype=np.float64)
    for meter_index in range(total_meters):
        meter_id = 1000 + meter_index
        residential = meter_index < residential_meters
        allocation_rows.append(
            {"meter_id": meter_id, "allocation_code": 1 if residential else 2}
        )
        for day in range(1, days + 1):
            # A mostly level profile keeps attack 4 near benign examples, so
            # tiny-fixture ADASYN has informative majority neighbours.
            level = 1.0 + 0.035 * meter_index + 0.012 * day
            values = level + 0.025 * np.sin((slots + meter_index) / 7.0)
            row: dict[str, object] = {
                "meter_id": meter_id,
                "day_number": day,
                "source_ref": f"fixture-{meter_index % 6 + 1}",
            }
            row.update(
                {
                    column: float(value)
                    for column, value in zip(HALF_HOUR_COLUMNS, values, strict=True)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(allocation_rows)


def prepare_fixture():
    profiles, allocation = prepared_profile_fixture(include_nonresidential=True)
    return prepare_iset_paper_literal(
        profiles,
        allocation_source=allocation,
        data_seed=20260721,
        validation_fraction=0.20,
        adasyn_neighbors=2,
        table_v_samples=7,
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _cer_text_for_profiles(frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for row in frame.itertuples(index=False):
        for slot, column in enumerate(HALF_HOUR_COLUMNS, start=1):
            lines.append(
                f"{row.meter_id} {int(row.day_number) * 100 + slot} "
                f"{getattr(row, column):.9f}\n"
            )
    return "".join(lines)


def write_archive_fixture(
    directory: Path,
    *,
    allocation_filename: str = ALLOCATION_FILENAME,
) -> tuple[list[Path], Path, dict[str, str]]:
    profiles, allocation = prepared_profile_fixture(
        residential_meters=12,
        days=2,
        include_nonresidential=True,
    )
    archive_paths: list[Path] = []
    # Distribute whole meters across archives.  This verifies that the loader
    # is not relying on one concatenated in-memory raw-reading frame.
    for file_index, filename in enumerate(ARCHIVE_FILENAMES, start=1):
        meter_ids = sorted(profiles["meter_id"].unique())
        selected_meters = meter_ids[file_index - 1 :: 6]
        selected = profiles.loc[profiles["meter_id"].isin(selected_meters)]
        path = directory / filename
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(filename.removesuffix(".zip"), _cer_text_for_profiles(selected))
        archive_paths.append(path)
    allocation_path = directory / allocation_filename
    allocation.to_csv(
        allocation_path,
        sep="," if allocation_path.suffix == ".csv" else "\t",
        index=False,
    )
    checksums = {
        path.name: _md5(path) for path in [*archive_paths, allocation_path]
    }
    return archive_paths, allocation_path, checksums


class PaperLiteralIsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_fixture()

    def test_a06_meter_disjointness_attack_provenance_and_shapes(self) -> None:
        prepared = self.prepared
        meter_partitions = prepared.metadata["meter_partitions"]
        train = set(meter_partitions["anomaly_train"])
        validation = set(meter_partitions["anomaly_validation"])
        heldout = set(meter_partitions["b2"])
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(heldout))
        self.assertTrue(validation.isdisjoint(heldout))
        self.assertEqual(train | validation, set(meter_partitions["b1"]))

        original_count = prepared.metadata["counts"]["anomaly_test_original"]
        original = prepared.anomaly_test
        original_malicious = np.flatnonzero(
            (original.labels[:original_count] == 1)
            & ~original.is_synthetic[:original_count]
        )
        self.assertEqual(set(original.meter_ids[original_malicious]), heldout)
        self.assertTrue((prepared.anomaly_train.labels == 0).all())
        self.assertTrue((prepared.anomaly_validation.labels == 0).all())
        for partition in (
            prepared.anomaly_train,
            prepared.anomaly_validation,
            prepared.anomaly_test,
            prepared.supervised_train,
            prepared.supervised_test,
        ):
            self.assertEqual(partition.values.shape[1], 48)
            self.assertEqual(partition.values.dtype, np.float32)

        original_attacks = original.attack_ids[:original_count]
        for attack_id in range(1, 7):
            self.assertEqual(
                int(np.count_nonzero(original_attacks == attack_id)),
                prepared.metadata["counts"]["malicious_profiles_per_attack"],
            )
        self.assertEqual(
            int(prepared.anomaly_test.is_synthetic.sum()),
            prepared.metadata["adasyn"]["anomaly_test"]["generated"],
        )
        self.assertTrue(
            (prepared.anomaly_test.attack_ids[prepared.anomaly_test.is_synthetic] == -1).all()
        )

    def test_joint_scaling_and_paper_ordered_class_counts(self) -> None:
        prepared = self.prepared
        original_count = prepared.metadata["counts"]["anomaly_test_original"]
        joint_original = np.concatenate(
            [
                prepared.anomaly_train.values,
                prepared.anomaly_validation.values,
                prepared.anomaly_test.values[:original_count],
            ]
        )
        np.testing.assert_allclose(joint_original.mean(axis=0), 0.0, atol=2e-5)
        np.testing.assert_allclose(joint_original.std(axis=0), 1.0, atol=2e-5)

        anomaly_counts = np.bincount(prepared.anomaly_test.labels, minlength=2)
        supervised_counts = np.bincount(
            np.concatenate(
                [prepared.supervised_train.labels, prepared.supervised_test.labels]
            ),
            minlength=2,
        )
        np.testing.assert_array_equal(
            anomaly_counts,
            prepared.metadata["adasyn"]["anomaly_test"]["counts_after"],
        )
        np.testing.assert_array_equal(
            supervised_counts,
            prepared.metadata["adasyn"]["supervised_before_split"]["counts_after"],
        )
        self.assertEqual(
            prepared.supervised_train.labels.size
            + prepared.supervised_test.labels.size,
            prepared.metadata["counts"]["supervised_after_adasyn"],
        )
        self.assertGreater(prepared.anomaly_test.is_synthetic.sum(), 0)
        self.assertGreater(
            prepared.supervised_train.is_synthetic.sum()
            + prepared.supervised_test.is_synthetic.sum(),
            0,
        )

    def test_deterministic_distinct_attack_streams_and_table_v_reuse(self) -> None:
        first = self.prepared
        second = prepare_fixture()
        self.assertEqual(first.metadata, second.metadata)
        for attack_id in range(1, 7):
            first_benign, first_attack = first.table_v_pair(attack_id)
            second_benign, second_attack = second.table_v_pair(attack_id)
            self.assertIs(first_benign, first.table_v_benign)
            np.testing.assert_array_equal(
                first_benign.sample_ids, first.table_v_benign.sample_ids
            )
            np.testing.assert_array_equal(
                first_attack.source_profile_ids, first_benign.source_profile_ids
            )
            np.testing.assert_array_equal(first_attack.values, second_attack.values)
            np.testing.assert_array_equal(first_benign.values, second_benign.values)
            self.assertTrue((first_attack.attack_ids == attack_id).all())
            self.assertFalse(first_attack.is_synthetic.any())
        self.assertFalse(
            np.array_equal(first.table_v_attacks[0].values, first.table_v_attacks[1].values)
        )

        # Attack 6 is reversal before feature-wise standardization.  Undo the
        # scaler to verify the current attack implementation was used.
        benign_raw = (
            first.table_v_benign.values * first.scaler_scale + first.scaler_mean
        )
        attack_6_raw = (
            first.table_v_attacks[5].values * first.scaler_scale + first.scaler_mean
        )
        np.testing.assert_allclose(attack_6_raw, benign_raw[:, ::-1], atol=2e-6)

    def test_b1_threshold_population_retains_iset_attack_provenance(self) -> None:
        population = build_threshold_population(
            self.prepared,
            branch="b1_generated_attacks",
            seed=20260721,
            validation_fraction=0.20,
        )
        benign_count = self.prepared.anomaly_validation.labels.size
        self.assertEqual(
            np.bincount(population.labels, minlength=2).tolist(),
            [benign_count, benign_count * 6],
        )
        self.assertEqual(
            population.values.shape[1],
            48,
        )
        np.testing.assert_array_equal(
            population.test_partition.sample_ids,
            self.prepared.anomaly_test.sample_ids,
        )
        self.assertEqual(
            population.metadata["derivation"],
            "six_printed_attacks_from_b1_validation",
        )

    def test_all_customer_attack_population_matches_source_paragraph(self) -> None:
        profiles, allocation = prepared_profile_fixture()
        prepared = prepare_iset_paper_literal(
            profiles,
            allocation_source=allocation,
            data_seed=20260721,
            validation_fraction=0.20,
            adasyn_neighbors=2,
            table_v_samples=7,
            attack_population="all_customer_m",
        )
        counts = prepared.metadata["counts"]
        self.assertEqual(
            counts["malicious_population_profiles"],
            counts["benign_profiles"] * 6,
        )
        self.assertEqual(
            counts["malicious_profiles"],
            counts["anomaly_b2_benign"] * 6,
        )
        self.assertEqual(
            prepared.metadata["preprocessing"]["malicious_source"],
            "all_customer_m",
        )
        original_supervised = counts["supervised_before_adasyn"]
        self.assertEqual(
            original_supervised,
            counts["benign_profiles"] + counts["malicious_population_profiles"],
        )

    def test_corrected_policy_keeps_meter_disjoint_test_rows_untouched(self) -> None:
        profiles, allocation = prepared_profile_fixture()
        prepared = prepare_iset_paper_literal(
            profiles,
            allocation_source=allocation,
            data_seed=20260721,
            validation_fraction=0.20,
            adasyn_neighbors=2,
            table_v_samples=7,
            attack_population="all_customer_m",
            scaling="train_benign_only",
            anomaly_adasyn="none",
            supervised_adasyn="customer_split_then_train_only",
        )
        self.assertFalse(prepared.anomaly_test.is_synthetic.any())
        self.assertFalse(prepared.supervised_test.is_synthetic.any())
        self.assertGreater(prepared.supervised_train.is_synthetic.sum(), 0)
        train_meters = set(
            prepared.supervised_train.meter_ids[
                ~prepared.supervised_train.is_synthetic
            ]
        )
        test_meters = set(prepared.supervised_test.meter_ids)
        self.assertTrue(train_meters.isdisjoint(test_meters))
        self.assertEqual(
            prepared.metadata["adasyn"]["anomaly_test"]["reason"],
            "disabled_by_branch",
        )
        self.assertEqual(
            prepared.metadata["preprocessing"]["scaling_details"]["fit_population"],
            "anomaly_train_benign_only",
        )
        np.testing.assert_allclose(
            prepared.anomaly_train.values.mean(axis=0), 0.0, atol=2e-5
        )
        np.testing.assert_allclose(
            prepared.anomaly_train.values.std(axis=0), 1.0, atol=2e-5
        )

    def test_all_registered_scaling_interpretations_execute(self) -> None:
        profiles, allocation = prepared_profile_fixture()
        for branch in (
            "joint_featurewise",
            "per_class_featurewise",
            "per_profile",
            "train_benign_only",
        ):
            with self.subTest(branch=branch):
                prepared = prepare_iset_paper_literal(
                    profiles,
                    allocation_source=allocation,
                    data_seed=20260721,
                    validation_fraction=0.20,
                    adasyn_neighbors=2,
                    table_v_samples=3,
                    scaling=branch,
                )
                self.assertEqual(
                    prepared.metadata["config"]["scaling_branch"], branch
                )
                self.assertTrue(np.isfinite(prepared.anomaly_test.values).all())

    def test_registered_attack_scope_and_transform_branches_execute(self) -> None:
        profiles, allocation = prepared_profile_fixture()
        for scope in (
            "per_profile",
            "per_customer_matrix",
            "per_generated_dataset",
        ):
            with self.subTest(scope=scope):
                prepared = prepare_iset_paper_literal(
                    profiles,
                    allocation_source=allocation,
                    data_seed=20260721,
                    validation_fraction=0.20,
                    adasyn_neighbors=2,
                    table_v_samples=1_000,
                    attack1_scope=scope,
                    attack2_granularity="per_hour_pair",
                    attack3_interval="printed_start_wrap",
                    attack_hour_mapping="two_slots_per_hour",
                )
                config = prepared.metadata["config"]
                self.assertEqual(config["attack1_scope"], scope)
                self.assertEqual(
                    config["attack2_granularity"], "per_hour_pair"
                )
                self.assertEqual(
                    config["attack3_interval"], "printed_start_wrap"
                )
                self.assertEqual(
                    config["attack_hour_mapping"], "two_slots_per_hour"
                )

                benign_raw = (
                    prepared.table_v_benign.values * prepared.scaler_scale
                    + prepared.scaler_mean
                )
                attack1_raw = (
                    prepared.table_v_attacks[0].values * prepared.scaler_scale
                    + prepared.scaler_mean
                )
                factors = np.mean(attack1_raw / benign_raw, axis=1)
                if scope == "per_generated_dataset":
                    np.testing.assert_allclose(factors, factors[0], atol=2e-6)
                elif scope == "per_customer_matrix":
                    for meter_id in np.unique(prepared.table_v_benign.meter_ids):
                        selected = (
                            prepared.table_v_benign.meter_ids == meter_id
                        )
                        np.testing.assert_allclose(
                            factors[selected],
                            factors[selected][0],
                            atol=2e-6,
                        )

                attack2_raw = (
                    prepared.table_v_attacks[1].values * prepared.scaler_scale
                    + prepared.scaler_mean
                )
                pair_factors = attack2_raw / benign_raw
                np.testing.assert_allclose(
                    pair_factors[:, 0::2],
                    pair_factors[:, 1::2],
                    atol=2e-6,
                )

    def test_seeded_3000_meter_population_is_exact_and_deterministic(self) -> None:
        profiles = pd.DataFrame(
            {"meter_id": np.arange(1_000, 4_001, dtype=np.int64)}
        )
        first, eligible_first, selected_first = _select_meter_population(
            profiles,
            branch="seeded_3000",
            seed=20260721,
        )
        second, eligible_second, selected_second = _select_meter_population(
            profiles,
            branch="seeded_3000",
            seed=20260721,
        )
        self.assertEqual(eligible_first.size, 3_001)
        self.assertEqual(selected_first.size, 3_000)
        self.assertEqual(first["meter_id"].astype(str).nunique(), 3_000)
        np.testing.assert_array_equal(eligible_first, eligible_second)
        np.testing.assert_array_equal(selected_first, selected_second)
        pd.testing.assert_frame_equal(first, second)

    def test_row_random_iset_split_is_profile_disjoint_but_not_meter_disjoint(
        self,
    ) -> None:
        profiles, allocation = prepared_profile_fixture(
            residential_meters=12,
            days=8,
        )
        prepared = prepare_iset_paper_literal(
            profiles,
            allocation_source=allocation,
            data_seed=20260721,
            validation_fraction=0.20,
            adasyn_neighbors=2,
            table_v_samples=7,
            attack_population="all_customer_m",
            split_unit="row_random",
            anomaly_adasyn="none",
        )
        train_sources = set(prepared.anomaly_train.source_profile_ids)
        validation_sources = set(prepared.anomaly_validation.source_profile_ids)
        test_sources = set(
            prepared.anomaly_test.source_profile_ids[
                prepared.anomaly_test.labels == 0
            ]
        )
        self.assertTrue(train_sources.isdisjoint(validation_sources))
        self.assertTrue(train_sources.isdisjoint(test_sources))
        self.assertTrue(validation_sources.isdisjoint(test_sources))
        train_meters = set(prepared.anomaly_train.meter_ids)
        test_meters = set(
            prepared.anomaly_test.meter_ids[
                prepared.anomaly_test.labels == 0
            ]
        )
        self.assertFalse(train_meters.isdisjoint(test_meters))
        self.assertEqual(
            prepared.metadata["config"]["split_unit_branch"],
            "row_random",
        )

    def test_all_attack_regeneration_seed_schedules_are_deterministic(
        self,
    ) -> None:
        fixed, fixed_metadata = _resolve_attack_seed(
            data_seed=20260721,
            attack_seed=None,
            attack_regeneration="fixed_per_data_seed",
            model_seed=None,
            experiment_index=None,
        )
        by_model, model_metadata = _resolve_attack_seed(
            data_seed=20260721,
            attack_seed=None,
            attack_regeneration="regenerate_per_model_seed",
            model_seed=11,
            experiment_index=None,
        )
        by_experiment, experiment_metadata = _resolve_attack_seed(
            data_seed=20260721,
            attack_seed=None,
            attack_regeneration="regenerate_per_experiment",
            model_seed=11,
            experiment_index=4,
        )
        repeated, _ = _resolve_attack_seed(
            data_seed=20260721,
            attack_seed=None,
            attack_regeneration="regenerate_per_experiment",
            model_seed=11,
            experiment_index=4,
        )
        self.assertEqual(fixed, 20260721)
        self.assertEqual(by_model, 11)
        self.assertEqual(by_experiment, repeated)
        self.assertEqual(fixed_metadata["source"], "data_seed")
        self.assertEqual(model_metadata["source"], "model_seed")
        self.assertEqual(experiment_metadata["experiment_index"], 4)
        self.assertEqual(len({fixed, by_model, by_experiment}), 3)
        with self.assertRaisesRegex(ValueError, "requires an explicit model_seed"):
            _resolve_attack_seed(
                data_seed=20260721,
                attack_seed=None,
                attack_regeneration="regenerate_per_model_seed",
                model_seed=None,
                experiment_index=None,
            )

    def test_table_iv_nested_sizes_and_checksum_cache_round_trip(self) -> None:
        prepared = self.prepared
        half = prepared.table_iv_subset("half")
        three_quarter = prepared.table_iv_subset("three_quarter")
        full = prepared.table_iv_subset("full")
        counts = prepared.metadata["counts"]
        self.assertEqual(half.labels.size, counts["table_iv_half"])
        self.assertEqual(
            three_quarter.labels.size, counts["table_iv_three_quarter"]
        )
        self.assertEqual(full.labels.size, counts["table_iv_full"])
        np.testing.assert_array_equal(
            half.sample_ids, three_quarter.sample_ids[: half.labels.size]
        )
        np.testing.assert_array_equal(
            three_quarter.sample_ids, full.sample_ids[: three_quarter.labels.size]
        )
        self.assertTrue((full.labels == 0).all())

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "data" / "derived" / "iset-paper-literal"
            npz_path, manifest_path = save_prepared_iset(prepared, prefix)
            restored = load_prepared_iset(prefix)
            self.assertTrue(npz_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(restored.metadata, prepared.metadata)
            np.testing.assert_array_equal(
                restored.anomaly_test.values, prepared.anomaly_test.values
            )
            np.testing.assert_array_equal(
                restored.table_v_attacks[3].source_profile_ids,
                prepared.table_v_attacks[3].source_profile_ids,
            )
            with npz_path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_prepared_iset(prefix)

    def test_archive_route_selects_code_one_and_enforces_checksum_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archives, allocation, checksums = write_archive_fixture(directory)
            verified = verify_authorized_iset_files(
                archives, allocation, expected_md5=checksums
            )
            self.assertEqual(set(verified), set(checksums))
            prepared = prepare_iset_paper_literal(
                archive_paths=archives,
                allocation_source=allocation,
                expected_md5=checksums,
                data_seed=20260721,
                validation_fraction=0.20,
                adasyn_neighbors=2,
                table_v_samples=5,
                chunksize=97,
                shard_count=4,
                scratch_dir=directory,
            )
            self.assertEqual(prepared.metadata["counts"]["residential_meters"], 12)
            self.assertEqual(prepared.metadata["counts"]["benign_profiles"], 24)
            self.assertEqual(
                prepared.metadata["residential_selection"]["method"],
                "official_allocation_code_1",
            )
            self.assertTrue(
                prepared.metadata["source"]["route"].startswith("checksum_gated")
            )

            with self.assertRaisesRegex(ValueError, "exactly six"):
                verify_authorized_iset_files(
                    archives[:-1], allocation, expected_md5=checksums
                )
            absent = directory / ARCHIVE_FILENAMES[0]
            original = archives[0]
            original.rename(absent.with_suffix(".missing"))
            with self.assertRaises(FileNotFoundError):
                verify_authorized_iset_files(
                    archives, allocation, expected_md5=checksums
                )
            absent.with_suffix(".missing").rename(original)
            invalid = dict(checksums)
            invalid[ARCHIVE_FILENAMES[0]] = "0" * 32
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_authorized_iset_files(
                    archives, allocation, expected_md5=invalid
                )

    def test_sciencedb_allocation_branch_is_explicit_and_filename_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archives, allocation, checksums = write_archive_fixture(
                directory,
                allocation_filename=SCIENCEDB_ALLOCATION_FILENAME,
            )
            verified = verify_authorized_iset_files(
                archives,
                allocation,
                allocation_branch=SCIENCEDB_ALLOCATION_BRANCH,
                expected_md5=checksums,
            )
            self.assertEqual(set(verified), set(checksums))
            with self.assertRaisesRegex(ValueError, "must be named"):
                verify_authorized_iset_files(
                    archives,
                    directory / ALLOCATION_FILENAME,
                    allocation_branch=SCIENCEDB_ALLOCATION_BRANCH,
                    expected_md5=checksums,
                )


if __name__ == "__main__":
    unittest.main()
