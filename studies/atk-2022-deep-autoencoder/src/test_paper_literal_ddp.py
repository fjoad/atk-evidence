"""Focused deterministic tests for the production Table-II DDP runner."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from paper_literal_ddp import (
    _all_training_partitions,
    balanced_shard_bounds,
    build_parser,
    ddp_loss_scale,
    distributed_fingerprint,
    epoch_permutation,
    inference_batch_size,
    shard_indices,
    timing_payload,
    weighted_mean,
)
from paper_literal_data import DataPartition, SgccPaperLiteralData
from paper_literal_runner import _persist_attempt, _verified_completed_attempt
from paper_literal_runner import Contract, _keras_callbacks


class ShardingTests(unittest.TestCase):
    def test_real_cardinalities_cover_every_index_once(self) -> None:
        cardinalities = (21960, 3876, 44036, 7772, 44035, 7771, 3, 1, 0)
        for count in cardinalities:
            with self.subTest(count=count):
                indices = np.arange(count, dtype=np.int64)
                shards = [shard_indices(indices, 4, rank) for rank in range(4)]
                joined = np.concatenate(shards) if count else np.empty(0, dtype=np.int64)
                np.testing.assert_array_equal(joined, indices)
                sizes = [len(shard) for shard in shards]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_zero_rank_and_bounds_validation(self) -> None:
        self.assertEqual([balanced_shard_bounds(3, 4, rank) for rank in range(4)],
                         [(0, 1), (1, 2), (2, 3), (3, 3)])
        with self.assertRaises(ValueError):
            balanced_shard_bounds(1, 0, 0)
        with self.assertRaises(ValueError):
            balanced_shard_bounds(1, 4, 4)

    def test_global_batch_shards_and_loss_scales(self) -> None:
        for count, expected_sizes in ((512, [128] * 4), (456, [114] * 4),
                                      (292, [73] * 4), (3, [1, 1, 1, 0])):
            indices = np.arange(count)
            sizes = [len(shard_indices(indices, 4, rank)) for rank in range(4)]
            self.assertEqual(sizes, expected_sizes)
            scales = [ddp_loss_scale(size, count, 4) for size in sizes]
            self.assertAlmostEqual(sum(scales) / 4.0, 1.0)


class AggregationAndShuffleTests(unittest.TestCase):
    def test_sample_weighted_mean_is_not_mean_of_rank_means(self) -> None:
        parts = [(1.0, 3), (5.0, 1)]
        self.assertEqual(weighted_mean(parts), 2.0)
        self.assertNotEqual(weighted_mean(parts), np.mean([1.0, 5.0]))

    def test_epoch_shuffle_is_reproducible_complete_and_epoch_specific(self) -> None:
        first = epoch_permutation(1031, 11, 0)
        repeat = epoch_permutation(1031, 11, 0)
        later = epoch_permutation(1031, 11, 1)
        np.testing.assert_array_equal(first, repeat)
        np.testing.assert_array_equal(np.sort(first), np.arange(1031))
        self.assertFalse(np.array_equal(first, later))

    def test_weighted_ddp_gradient_equals_global_three_sample_gradient(self) -> None:
        x = np.asarray([1.0, 2.0, 4.0])
        y = np.asarray([0.5, -1.0, 2.0])
        weight = 0.25
        global_gradient = np.mean(2.0 * (weight * x - y) * x)
        rank_gradients = []
        for rank in range(4):
            shard = shard_indices(np.arange(3), 4, rank)
            if len(shard):
                local_gradient = np.mean(
                    2.0 * (weight * x[shard] - y[shard]) * x[shard]
                )
            else:
                local_gradient = 0.0
            rank_gradients.append(
                ddp_loss_scale(len(shard), 3, 4) * local_gradient
            )
        self.assertAlmostEqual(float(np.mean(rank_gradients)), float(global_gradient))

    def test_inference_batch_stays_within_one_rank_training_envelope(self) -> None:
        self.assertEqual(inference_batch_size(512, 4), 128)
        with self.assertRaises(ValueError):
            inference_batch_size(513, 4)

    def test_timing_payload_does_not_double_count_data_preparation(self) -> None:
        complete = timing_payload(
            data_prep_seconds=30.0,
            run_seconds=120.0,
            fit_seconds=100.0,
            score_seconds=20.0,
        )
        self.assertEqual(complete["end_to_end_seconds"], 150.0)
        failed = timing_payload(
            data_prep_seconds=30.0, run_seconds=7.0, failed=True
        )
        self.assertEqual(failed["elapsed_until_failure_seconds"], 7.0)
        self.assertEqual(failed["end_to_end_seconds"], 37.0)

    def test_all_training_refit_population_and_cli_branches_are_explicit(self) -> None:
        def partition(rows, labels, prefix):
            values = np.asarray(rows, dtype=np.float32)
            return DataPartition(
                values=values,
                labels=np.asarray(labels, dtype=np.int8),
                sample_ids=np.asarray(
                    [f"{prefix}{index}" for index in range(len(values))]
                ),
                is_synthetic=np.zeros(len(values), dtype=bool),
            )

        anomaly_train = partition([[0, 1], [1, 2]], [0, 0], "AT")
        anomaly_validation = partition([[2, 3]], [0], "AV")
        anomaly_test = partition([[0, 0], [9, 9]], [0, 1], "AX")
        supervised_train = partition(
            [[0, 1], [1, 2], [8, 9], [9, 10]],
            [0, 0, 1, 1],
            "ST",
        )
        prepared = SgccPaperLiteralData(
            dates=np.arange(2).astype("datetime64[D]"),
            imputation_fallback=np.zeros(2),
            scaler_mean=np.zeros(2),
            scaler_scale=np.ones(2),
            anomaly_train=anomaly_train,
            anomaly_validation=anomaly_validation,
            anomaly_test=anomaly_test,
            supervised_train=supervised_train,
            supervised_test=anomaly_test,
            metadata={},
        )
        anomaly_all = _all_training_partitions("fc_sae", prepared)
        self.assertEqual(anomaly_all.train_x.shape[0], 3)
        self.assertEqual(anomaly_all.validation_x.shape[0], 0)
        supervised_all = _all_training_partitions(
            "supervised_feed_forward",
            prepared,
            supervised_head="softmax2_categorical",
        )
        self.assertEqual(supervised_all.train_x.shape[0], 4)
        self.assertEqual(supervised_all.train_y.shape, (4,))
        sigmoid_all = _all_training_partitions(
            "supervised_feed_forward",
            prepared,
            supervised_head="sigmoid1_binary",
        )
        self.assertEqual(sigmoid_all.train_y.shape, (4, 1))
        with self.assertRaisesRegex(ValueError, "unsupported supervised head"):
            _all_training_partitions(
                "supervised_lstm",
                prepared,
                supervised_head="invented",
            )
        args = build_parser().parse_args(
            [
                "--model",
                "fc_sae",
                "--seed",
                "11",
                "--data",
                "data.csv",
                "--output",
                "runs",
                "--validation-policy",
                "crossval_refit_b1",
                "--threshold-rule",
                "threshold_iqr_midpoint",
                "--threshold-scope",
                "dataset_specific",
                "--validation-labels",
                "b2_validation_carveout",
                "--branch-id",
                "paper-branch-id",
            ]
        )
        self.assertEqual(args.validation_policy, "crossval_refit_b1")
        self.assertEqual(args.threshold_scope, "dataset_specific")
        self.assertEqual(args.branch_id, "paper-branch-id")


class StoppingTests(unittest.TestCase):
    def test_existing_callback_warmup_patience_and_restore_semantics(self) -> None:
        run = {
            "early_stopping_min_delta": 0.0001,
            "early_stopping_patience": 4,
            "warmup_epochs": 2,
        }
        contract = Contract(
            path=Path("fixture.toml"), raw={"run": run}, sha256="fixture"
        )

        class FakeModel:
            def __init__(self) -> None:
                self.stop_training = False
                self.weight = 0

            def get_weights(self):
                return [self.weight]

            def set_weights(self, weights):
                self.weight = weights[0]

        model = FakeModel()
        callbacks, _ = _keras_callbacks(contract)
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        stop_epoch = None
        validation_losses = [9.0, 8.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for epoch, loss in enumerate(validation_losses):
            model.weight = epoch
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
                callback.on_epoch_end(epoch, {"val_loss": loss})
            if model.stop_training:
                stop_epoch = epoch
                break
        for callback in callbacks:
            callback.on_train_end()
        self.assertEqual(stop_epoch, 8)
        self.assertEqual(model.weight, 4)


class FingerprintAndPersistenceTests(unittest.TestCase):
    def test_execution_spec_changes_fingerprint(self) -> None:
        base = {"model": "lstm_sae", "seed": 11, "contract": "fixture"}
        spec = {
            "implementation_source_sha256": "source-a",
            "world_size": 4,
            "global_batch_size": 512,
            "cardinalities": {"train": 21960, "validation": 3876},
        }
        first, payload = distributed_fingerprint(
            base_payload=base, execution_spec=spec
        )
        repeat, _ = distributed_fingerprint(base_payload=base, execution_spec=spec)
        changed, _ = distributed_fingerprint(
            base_payload=base, execution_spec={**spec, "world_size": 2}
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, changed)
        self.assertEqual(payload["distributed_execution"], spec)

    def test_only_complete_checksum_valid_attempt_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logical = Path(temporary) / "table_2" / "sgcc" / "lstm_sae" / "seed_11"
            failed = _persist_attempt(
                logical,
                status="failed",
                fingerprint="ddp-fingerprint",
                fingerprint_payload={"distributed_execution": {"world_size": 4}},
                metadata={"status": "failed"},
                history={},
                result_summary={"status": "failed"},
                arrays=None,
            )
            self.assertIsNone(
                _verified_completed_attempt(logical, "ddp-fingerprint")
            )
            complete = _persist_attempt(
                logical,
                status="complete",
                fingerprint="ddp-fingerprint",
                fingerprint_payload={"distributed_execution": {"world_size": 4}},
                metadata={"status": "complete"},
                history={"epochs_completed": 1},
                result_summary={"status": "complete"},
                arrays=None,
            )
            self.assertEqual(
                _verified_completed_attempt(logical, "ddp-fingerprint"), complete
            )
            (complete / "result.json").write_text("tampered", encoding="utf-8")
            self.assertIsNone(
                _verified_completed_attempt(logical, "ddp-fingerprint")
            )
            self.assertTrue(failed.exists())


if __name__ == "__main__":
    unittest.main()
