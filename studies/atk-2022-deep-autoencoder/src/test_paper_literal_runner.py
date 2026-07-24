"""Fast tests for the resumable Paper 1 experiment runner."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from paper_literal_data import DataPartition, SgccPaperLiteralData
from paper_literal_runner import (
    Contract,
    ExecutionResult,
    UnsupportedExperimentError,
    build_threshold_population,
    execute_neural,
    reject_unsupported_scope,
    resolve_models,
    run_one,
)


def _partition(values, labels, prefix: str) -> DataPartition:
    matrix = np.asarray(values, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int8)
    return DataPartition(
        values=matrix,
        labels=label_array,
        sample_ids=np.asarray([f"{prefix}{i}" for i in range(len(matrix))]),
        is_synthetic=np.zeros(len(matrix), dtype=bool),
    )


def _prepared() -> SgccPaperLiteralData:
    benign_train = _partition(
        [[-1, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        [0, 0, 0, 0],
        "AT",
    )
    benign_validation = _partition(
        [[-0.5, 0.5, 1.5, 2.5], [1.5, 2.5, 3.5, 4.5]],
        [0, 0],
        "AV",
    )
    anomaly_test = _partition(
        [[0, 0, 0, 0], [1, 1, 1, 1], [8, 8, 8, 8], [9, 9, 9, 9]],
        [0, 0, 1, 1],
        "AX",
    )
    supervised_train = _partition(
        [[i, i + 1, i + 2, i + 3] for i in range(12)],
        [0, 1] * 6,
        "ST",
    )
    supervised_test = _partition(
        [[0, 1, 2, 3], [1, 2, 3, 4], [8, 9, 10, 11], [9, 10, 11, 12]],
        [0, 0, 1, 1],
        "SX",
    )
    return SgccPaperLiteralData(
        dates=np.arange(4).astype("datetime64[D]"),
        imputation_fallback=np.zeros(4),
        scaler_mean=np.zeros(4),
        scaler_scale=np.ones(4),
        anomaly_train=benign_train,
        anomaly_validation=benign_validation,
        anomaly_test=anomaly_test,
        supervised_train=supervised_train,
        supervised_test=supervised_test,
        metadata={
            "partition_id_sha256": {"fixture": "partitions"},
            "transformation_sha256": {"fixture": "transformations"},
            "counts": {"retained": 16},
        },
    )


def _contract(root: Path) -> Contract:
    raw = {
        "run": {
            "mode": "exploratory_paper_literal",
            "data_seed": 20260721,
            "model_seeds": [11, 22, 33],
            "max_epochs": 3,
            "batch_size": 4,
            "warmup_epochs": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0001,
            "validation_fraction_within_train": 0.25,
            "supervised_svm_max_samples": 30,
            "one_class_svm_max_samples": 12,
        },
        "data": {"adasyn_neighbors": 1},
        "thresholds": {
            "arima": 0.58,
            "one_class_svm": 0.45,
            "fc_sae": 0.58,
            "lstm_sae": 0.61,
            "fc_vae": 0.43,
            "lstm_vae": 0.47,
            "lstm_aea": 0.51,
        },
        "table_1": {
            "fc_sae": {"encoder_widths": [4, 3, 2, 1]},
            "lstm_sae": {"encoder_widths": [3, 2]},
            "fc_vae": {"encoder_widths": [4, 3, 2, 1]},
            "lstm_vae": {"encoder_widths": [3, 2]},
            "lstm_aea": {"encoder_widths": [4, 3, 2]},
            "supervised_feed_forward": {"encoder_widths": [4] * 5},
            "supervised_lstm": {"encoder_widths": [3] * 4},
        },
    }
    path = root / "contract.toml"
    path.write_text("# fixture contract\n", encoding="utf-8")
    return Contract(path=path, raw=raw, sha256="contract-fixture")


def _successful_result() -> ExecutionResult:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    predictions = np.asarray([0, 0, 1, 1], dtype=np.int8)
    return ExecutionResult(
        scores={"primary": scores},
        predictions={"primary": predictions},
        labels=np.asarray([0, 0, 1, 1], dtype=np.int8),
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        is_synthetic=np.asarray([False, False, True, True]),
        history={"series": {"loss": [1.0, 0.5]}},
        metrics={"primary": {"tp": 2, "fp": 0, "tn": 2, "fn": 0}},
        fit_seconds=1.25,
        score_seconds=0.25,
        metadata={"test": True},
    )


class RunnerPersistenceTests(unittest.TestCase):
    def test_completed_attempt_is_checksum_verified_before_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _contract(root)
            prepared = _prepared()
            verification = {"actual_sha256": "data-fixture", "verified": True}
            executor = mock.Mock(return_value=_successful_result())

            first = run_one(
                output=root / "results",
                model_name="fc_sae",
                seed=11,
                prepared=prepared,
                contract=contract,
                verification=verification,
                data_prep_seconds=0.75,
                executor=executor,
            )
            self.assertEqual(first.status, "complete")
            with np.load(first.attempt_dir / "arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {
                        "labels",
                        "sample_ids",
                        "is_synthetic",
                        "score__primary",
                        "prediction__primary",
                    },
                )
                np.testing.assert_array_equal(arrays["sample_ids"], ["a", "b", "c", "d"])
            result = json.loads((first.attempt_dir / "result.json").read_text())
            self.assertEqual(result["timings"]["data_prep_seconds"], 0.75)
            self.assertEqual(result["timings"]["fit_seconds"], 1.25)
            self.assertEqual(result["timings"]["score_seconds"], 0.25)

            second = run_one(
                output=root / "results",
                model_name="fc_sae",
                seed=11,
                prepared=prepared,
                contract=contract,
                verification=verification,
                data_prep_seconds=0.75,
                executor=executor,
            )
            self.assertEqual(second.status, "skipped_complete")
            self.assertEqual(executor.call_count, 1)

            # A corrupted artifact is not a completed immutable run and must
            # not be silently trusted on resume.
            with (first.attempt_dir / "arrays.npz").open("ab") as handle:
                handle.write(b"tamper")
            third = run_one(
                output=root / "results",
                model_name="fc_sae",
                seed=11,
                prepared=prepared,
                contract=contract,
                verification=verification,
                data_prep_seconds=0.75,
                executor=executor,
            )
            self.assertEqual(third.status, "complete")
            self.assertNotEqual(first.attempt_dir, third.attempt_dir)
            self.assertEqual(executor.call_count, 2)

    def test_force_appends_and_failure_does_not_count_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = {
                "output": root / "results",
                "model_name": "arima",
                "seed": 22,
                "prepared": _prepared(),
                "contract": _contract(root),
                "verification": {"actual_sha256": "data-fixture", "verified": True},
                "data_prep_seconds": 0.1,
            }
            failed = run_one(
                **kwargs,
                executor=mock.Mock(side_effect=RuntimeError("deliberate failure")),
            )
            self.assertEqual(failed.status, "failed")
            failure_manifest = json.loads((failed.attempt_dir / "manifest.json").read_text())
            self.assertEqual(failure_manifest["status"], "failed")
            self.assertFalse((failed.attempt_dir / "arrays.npz").exists())

            succeeded = run_one(**kwargs, executor=mock.Mock(return_value=_successful_result()))
            self.assertEqual(succeeded.status, "complete")
            forced = run_one(
                **kwargs,
                force=True,
                executor=mock.Mock(return_value=_successful_result()),
            )
            self.assertEqual(forced.status, "complete")
            self.assertNotEqual(succeeded.attempt_dir, forced.attempt_dir)
            attempts = list((succeeded.attempt_dir.parent).iterdir())
            self.assertEqual(len(attempts), 3)


class RunnerExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract = _contract(self.root)
        self.prepared = _prepared()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _callback_factory(contract):
        del contract
        return [object()], types.SimpleNamespace(epoch_seconds=[0.01, 0.02])

    def test_vae_fits_without_target_and_preserves_both_score_branches(self) -> None:
        class FakeModel:
            def __init__(self):
                self.fit_kwargs = None

            def fit(self, **kwargs):
                self.fit_kwargs = kwargs
                return types.SimpleNamespace(history={"loss": [2.0, 1.0], "val_loss": [2.2, 1.2]})

            def count_params(self):
                return 123

        model = FakeModel()
        bundle = types.SimpleNamespace(
            model=model,
            anomaly_scores=lambda values, batch_size: {
                "reconstruction_mse": np.asarray([0.1, 0.2, 0.8, 0.9]),
                "mse_plus_kl_surrogate": np.asarray([0.2, 0.3, 1.0, 1.1]),
            },
        )
        builder = mock.Mock(return_value=bundle)
        result = execute_neural(
            "fc_vae",
            self.prepared,
            self.contract,
            11,
            model_builder=builder,
            callback_factory=self._callback_factory,
        )
        self.assertIsNone(model.fit_kwargs["y"])
        self.assertIsNone(model.fit_kwargs["validation_data"][1])
        self.assertEqual(
            set(result.scores),
            {"reconstruction_mse", "mse_plus_kl_surrogate"},
        )
        self.assertEqual(set(result.metrics), set(result.scores))
        self.assertEqual(result.history["epoch_seconds"], [0.01, 0.02])

    def test_source_vae_probability_uses_low_score_anomaly_orientation(self) -> None:
        class FakeModel:
            def fit(self, **kwargs):
                self.fit_kwargs = kwargs
                return types.SimpleNamespace(history={"loss": [1.0]})

            def count_params(self):
                return 321

        model = FakeModel()
        bundle = types.SimpleNamespace(
            model=model,
            vae_reconstruction_probability=lambda *args, **kwargs: {
                "reconstruction_probability": np.asarray([0.9, 0.8, 0.2, 0.1])
            },
        )
        builder = mock.Mock(return_value=bundle)
        result = execute_neural(
            "fc_vae",
            self.prepared,
            self.contract,
            11,
            model_builder=builder,
            callback_factory=self._callback_factory,
            model_overrides={
                "architecture_contract": "paper_source_v2",
                "vae_score": "prob_fixed_var_mc10",
            },
        )
        self.assertEqual(set(result.scores), {"prob_fixed_var_mc10"})
        np.testing.assert_array_equal(
            result.predictions["prob_fixed_var_mc10"],
            [0, 0, 1, 1],
        )
        self.assertEqual(result.metrics["prob_fixed_var_mc10"]["tp"], 2)
        self.assertEqual(result.metadata["positive_if"], "lower")
        passed_config = builder.call_args.args[2]
        self.assertEqual(passed_config["vae_score"], "prob_fixed_var_mc10")

    def test_threshold_population_branches_preserve_final_test_identity(self) -> None:
        generated = build_threshold_population(
            self.prepared,
            branch="b1_generated_attacks",
            seed=20260721,
            validation_fraction=0.25,
        )
        self.assertEqual(
            np.bincount(generated.labels, minlength=2).tolist(),
            [2, 12],
        )
        np.testing.assert_array_equal(
            generated.test_partition.sample_ids,
            self.prepared.anomaly_test.sample_ids,
        )
        repeated = build_threshold_population(
            self.prepared,
            branch="b1_generated_attacks",
            seed=20260721,
            validation_fraction=0.25,
        )
        np.testing.assert_array_equal(generated.values, repeated.values)

        carved = build_threshold_population(
            self.prepared,
            branch="b2_validation_carveout",
            seed=11,
            validation_fraction=0.25,
        )
        self.assertEqual(
            np.bincount(carved.labels, minlength=2).tolist(),
            [1, 1],
        )
        self.assertEqual(
            np.bincount(carved.test_partition.labels, minlength=2).tolist(),
            [1, 1],
        )
        self.assertTrue(
            set(carved.sample_ids).isdisjoint(
                set(carved.test_partition.sample_ids)
            )
        )

    def test_dataset_specific_threshold_is_derived_and_test_carveout_removed(self) -> None:
        class FakeModel:
            def fit(self, **kwargs):
                return types.SimpleNamespace(history={"loss": [1.0]})

            def count_params(self):
                return 42

        def anomaly_scores(values, batch_size):
            del batch_size
            return {
                "reconstruction_mse": np.mean(values, axis=1),
            }

        result = execute_neural(
            "fc_sae",
            self.prepared,
            self.contract,
            11,
            model_builder=mock.Mock(
                return_value=types.SimpleNamespace(
                    model=FakeModel(),
                    anomaly_scores=anomaly_scores,
                )
            ),
            callback_factory=self._callback_factory,
            threshold_rule="threshold_iqr_median",
            threshold_scope="dataset_specific",
            validation_labels="b2_validation_carveout",
        )
        self.assertEqual(result.labels.size, 2)
        selection = result.metadata["threshold_selection"]["reconstruction_mse"]
        self.assertEqual(selection["rule"], "threshold_iqr_median")
        self.assertEqual(selection["scope"], "dataset_specific")
        self.assertGreater(selection["finite_roc_thresholds"], 0)
        self.assertEqual(
            result.metadata["threshold_population"]["test_samples_after_carveout"],
            2,
        )

    def test_sgcc_iset_transfer_requires_and_records_external_threshold(self) -> None:
        class FakeModel:
            def fit(self, **kwargs):
                return types.SimpleNamespace(history={"loss": [1.0]})

            def count_params(self):
                return 42

        bundle = types.SimpleNamespace(
            model=FakeModel(),
            anomaly_scores=lambda values, batch_size: {
                "reconstruction_mse": np.mean(values, axis=1)
            },
        )
        kwargs = {
            "model_name": "fc_sae",
            "prepared": self.prepared,
            "contract": self.contract,
            "seed": 11,
            "model_builder": mock.Mock(return_value=bundle),
            "callback_factory": self._callback_factory,
            "threshold_rule": "threshold_iqr_midpoint",
            "threshold_scope": "iset_transferred",
            "validation_labels": "b1_generated_attacks",
        }
        with self.assertRaisesRegex(ValueError, "transferred"):
            execute_neural(**kwargs)
        result = execute_neural(
            **kwargs,
            transferred_thresholds={"reconstruction_mse": 4.5},
        )
        self.assertEqual(
            result.metadata["score_thresholds"]["reconstruction_mse"],
            4.5,
        )
        selection = result.metadata["threshold_selection"]["reconstruction_mse"]
        self.assertEqual(
            selection["details"]["source"],
            "frozen_iset_transfer_artifact",
        )

    def test_printed_constant_cannot_claim_dataset_specific_derivation(self) -> None:
        class FakeModel:
            def fit(self, **kwargs):
                return types.SimpleNamespace(history={"loss": [1.0]})

            def count_params(self):
                return 42

        with self.assertRaisesRegex(ValueError, "incompatible"):
            execute_neural(
                "fc_sae",
                self.prepared,
                self.contract,
                11,
                model_builder=mock.Mock(
                    return_value=types.SimpleNamespace(
                        model=FakeModel(),
                        anomaly_scores=lambda values, batch_size: {
                            "reconstruction_mse": np.mean(values, axis=1)
                        },
                    )
                ),
                callback_factory=self._callback_factory,
                threshold_rule="printed_constant",
                threshold_scope="dataset_specific",
                validation_labels="printed_threshold_no_derivation",
            )

    def test_all_anomaly_validation_policies_change_actual_fit_semantics(self) -> None:
        expected = {
            "none_fixed_epochs": (1, 6, False),
            "holdout_no_refit": (1, 4, True),
            "holdout_refit_b1": (2, 6, False),
            "crossval_refit_b1": (6, 6, False),
        }
        for policy, (
            expected_fits,
            final_train_rows,
            final_has_validation,
        ) in expected.items():
            with self.subTest(policy=policy):
                fit_calls = []

                class FakeModel:
                    def fit(self, **kwargs):
                        fit_calls.append(kwargs)
                        epochs = int(kwargs["epochs"])
                        history = {"loss": [3.0, 2.0, 2.5][:epochs]}
                        if "validation_data" in kwargs:
                            history["val_loss"] = [3.0, 1.0, 2.0][:epochs]
                        return types.SimpleNamespace(history=history)

                    def count_params(self):
                        return 42

                def builder(*args, **kwargs):
                    del args, kwargs
                    return types.SimpleNamespace(
                        model=FakeModel(),
                        anomaly_scores=lambda values, batch_size: {
                            "reconstruction_mse": np.mean(values, axis=1)
                        },
                    )

                result = execute_neural(
                    "fc_sae",
                    self.prepared,
                    self.contract,
                    11,
                    model_builder=builder,
                    callback_factory=self._callback_factory,
                    fixed_callback_factory=self._callback_factory,
                    validation_policy=policy,
                )
                self.assertEqual(len(fit_calls), expected_fits)
                self.assertEqual(fit_calls[-1]["x"].shape[0], final_train_rows)
                self.assertEqual(
                    "validation_data" in fit_calls[-1],
                    final_has_validation,
                )
                self.assertEqual(
                    result.metadata["training_policy"]["policy"],
                    policy,
                )
                if policy in {"holdout_refit_b1", "crossval_refit_b1"}:
                    self.assertTrue(
                        result.metadata["training_policy"]["refit"]
                    )
                    self.assertEqual(fit_calls[-1]["epochs"], 2)

    def test_supervised_cross_validation_is_stratified_then_refit(self) -> None:
        fit_calls = []

        class FakeModel:
            def fit(self, **kwargs):
                fit_calls.append(kwargs)
                history = {"loss": [2.0, 1.0], "val_loss": [2.0, 1.0]}
                if "validation_data" not in kwargs:
                    history.pop("val_loss")
                return types.SimpleNamespace(history=history)

            def predict(self, values, batch_size, verbose):
                del batch_size, verbose
                probability = (np.mean(values, axis=1) > 5).astype(float)
                return np.column_stack([1.0 - probability, probability])

            def count_params(self):
                return 24

        result = execute_neural(
            "supervised_feed_forward",
            self.prepared,
            self.contract,
            11,
            model_builder=lambda *args, **kwargs: types.SimpleNamespace(
                model=FakeModel()
            ),
            callback_factory=self._callback_factory,
            fixed_callback_factory=self._callback_factory,
            validation_policy="crossval_refit_b1",
        )
        self.assertEqual(len(fit_calls), 6)
        for call in fit_calls[:-1]:
            validation_labels = call["validation_data"][1]
            self.assertEqual(set(np.asarray(validation_labels).tolist()), {0, 1})
        self.assertEqual(fit_calls[-1]["x"].shape[0], 12)
        self.assertNotIn("validation_data", fit_calls[-1])
        self.assertEqual(
            result.metadata["training_policy"]["folds"],
            5,
        )

    def test_supervised_feed_forward_uses_standard_classifier_decision(self) -> None:
        class FakeModel:
            def fit(self, **kwargs):
                self.fit_kwargs = kwargs
                return types.SimpleNamespace(history={"loss": [1.0]})

            def predict(self, values, batch_size, verbose):
                del values, batch_size, verbose
                return np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])

            def count_params(self):
                return 50

        model = FakeModel()
        result = execute_neural(
            "supervised_feed_forward",
            self.prepared,
            self.contract,
            11,
            model_builder=mock.Mock(return_value=types.SimpleNamespace(model=model)),
            callback_factory=self._callback_factory,
        )
        np.testing.assert_array_equal(
            result.predictions["positive_class_probability"], [0, 0, 1, 1]
        )
        self.assertEqual(result.metrics["positive_class_probability"]["tp"], 2)
        self.assertTrue(model.fit_kwargs["validation_data"][0].shape[0] > 0)

    def test_scope_and_model_selection_are_explicit(self) -> None:
        reject_unsupported_scope(2, "sgcc")
        for table in (3, 4, 5):
            with self.assertRaisesRegex(UnsupportedExperimentError, "seven"):
                reject_unsupported_scope(table, "cer")
        self.assertEqual(
            resolve_models(["FC-SAE,NB", "multi-class-svm"]),
            ["fc_sae", "naive_bayes", "multiclass_svm"],
        )


if __name__ == "__main__":
    unittest.main()
