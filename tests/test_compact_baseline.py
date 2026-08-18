from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = (
    ROOT / "studies/atk-2022-deep-autoencoder/reproduction"
)
sys.path.insert(0, str(REPRODUCTION))

import analyze_results  # noqa: E402
import models  # noqa: E402
import prepare_data  # noqa: E402
import run_experiment  # noqa: E402


class CompactBaselineTests(unittest.TestCase):
    def test_complete_metric_audit_exactly_searches_all_thresholds(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        reported = {
            "DR": 100.0,
            "FA": 0.0,
            "SP": 100.0,
            "PR": 100.0,
            "ACC": 100.0,
            "F1": 100.0,
            "AUC": 100.0,
        }
        audit = analyze_results.closest_reported_metric_vector(
            labels,
            scores,
            direction="higher",
            reported=reported,
        )
        self.assertEqual(audit["minimum_maximum_absolute_gap"], 0.0)
        self.assertEqual(audit["metrics"], reported)
        # Four distinct scores plus sklearn's all-negative boundary.
        self.assertEqual(audit["threshold_candidates"], 5)

    def test_sgcc_representation_contrasts_are_exactly_48_wide(self) -> None:
        values = np.arange(2 * 1_034, dtype=np.float32).reshape(2, 1_034)
        first = prepare_data.represent_sgcc(values, "first_48")
        last = prepare_data.represent_sgcc(values, "last_48")
        binned = prepare_data.represent_sgcc(values, "binned_mean_48")
        self.assertTrue(np.array_equal(first, values[:, :48]))
        self.assertTrue(np.array_equal(last, values[:, -48:]))
        self.assertEqual(binned.shape, (2, 48))
        self.assertAlmostEqual(
            float(binned[0, 0]), float(values[0, :22].mean())
        )

    def test_sgcc_last48_completion_preserves_printed_order_and_width(self) -> None:
        rng = np.random.default_rng(11)
        dates = pd.date_range("2014-01-01", periods=1_034, freq="D")
        values = rng.lognormal(size=(30, dates.size)).astype(np.float32)
        values[0, 10:12] = np.nan
        values[1, :3] = np.nan
        frame = pd.DataFrame(
            values,
            columns=[f"{date.year}/{date.month}/{date.day}" for date in dates],
        )
        frame.insert(0, "FLAG", np.array([0] * 20 + [1] * 10, dtype=np.int8))
        frame.insert(0, "CONS_NO", [f"customer-{index}" for index in range(30)])
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            prepare_data, "digest", return_value=prepare_data.SGCC_SHA256
        ), patch.object(prepare_data.pd, "read_csv", return_value=frame):
            output = Path(temporary)
            metadata = prepare_data.prepare_sgcc(
                output, seed=11, mode="tiny", adasyn_neighbors=1
            )
            self.assertEqual(metadata["method"], "I-SGCC-LAST48")
            self.assertEqual(np.load(output / "x_train.npy").shape, (13, 48))
            self.assertEqual(np.load(output / "test_original_x.npy").shape[1], 48)
            self.assertEqual(np.load(output / "x_test.npy").shape[1], 48)
            supervised_y = np.load(output / "supervised_y.npy")
            class_counts = np.bincount(supervised_y)
            self.assertEqual(int(class_counts[0]), 20)
            self.assertGreater(int(class_counts[1]), 10)
            self.assertTrue(np.isfinite(np.load(output / "x_train.npy")).all())
            args = argparse.Namespace(
                data=output,
                output=output / "results",
                seed=11,
                score_batch=8,
            )
            metadata_path = output / "metadata.json"
            run_experiment.run_naive_bayes(
                args, metadata=metadata, metadata_path=metadata_path
            )
            result_path = next((output / "results/table_2").rglob("result.json"))
            audit = analyze_results.audit_scores(result_path)
            self.assertIn("closest_reported_operating_point", audit)
            self.assertEqual(
                audit["effective_eligibility"],
                "exploratory_interpretation_I-SGCC-LAST48",
            )
            successes, _ = analyze_results.load_attempts(output / "results")
            tables = analyze_results.aggregate(successes)
            self.assertEqual(len(tables["table_2_summary"]), 1)
            self.assertEqual(tables["table_3_summary"], [])
            self.assertEqual(tables["table_4_summary"], [])

    def test_fc_sae_runtime_matches_frozen_table_i_replay(self) -> None:
        model = models.build_fc_sae(seed=11, learning_rate=0.001)
        dense = [
            row["units"]
            for row in models.layer_inventory(model)
            if row["class"] == "Dense"
        ]
        self.assertEqual(
            dense,
            [400, 300, 200, 100, 100, 200, 300, 400, 48],
        )
        self.assertEqual(model.count_params(), 450_448)

    def test_linear_output_control_changes_only_the_final_activation(self) -> None:
        paper = models.layer_inventory(models.build_fc_sae(seed=11))
        control = models.layer_inventory(
            models.build_fc_sae(seed=11, output_activation="linear")
        )
        paper_dense = [row for row in paper if row["class"] == "Dense"]
        control_dense = [row for row in control if row["class"] == "Dense"]
        self.assertEqual(
            [row["units"] for row in paper_dense],
            [row["units"] for row in control_dense],
        )
        self.assertEqual(
            [row["activation"] for row in paper_dense[:-1]],
            [row["activation"] for row in control_dense[:-1]],
        )
        self.assertEqual(paper_dense[-1]["activation"], "softmax")
        self.assertEqual(control_dense[-1]["activation"], "linear")
        self.assertEqual(
            analyze_results.effective_eligibility(
                {"configuration": {"output_activation": "linear"}}
            ),
            "exploratory_control_C-OUTPUT-LINEAR",
        )

    def test_frozen_attack_repairs_and_scaler_are_direct(self) -> None:
        benign = np.arange(1, 1 + 4 * 48, dtype=np.float32).reshape(4, 48)
        meters = np.array([10, 10, 20, 20], dtype=np.int32)
        blocks = {
            attack_id: values
            for attack_id, _, values in prepare_data.attack_blocks(
                benign, meters, seed=11
            )
        }
        self.assertEqual(set(blocks), set(range(1, 7)))
        ratio = blocks[1] / benign
        self.assertTrue(np.allclose(ratio[0], ratio[1]))
        self.assertTrue(np.allclose(ratio[2], ratio[3]))
        self.assertFalse(np.allclose(ratio[0], ratio[2]))
        self.assertGreater(np.unique(blocks[2] / benign).size, 2)
        zero_counts = np.count_nonzero(blocks[3] == 0, axis=1)
        self.assertTrue(np.all(zero_counts >= 8))
        self.assertTrue(np.all(zero_counts <= 48))
        self.assertTrue(
            np.allclose(blocks[4], benign.mean(axis=1, keepdims=True))
        )
        self.assertTrue(np.array_equal(blocks[6], benign[:, ::-1]))

        mean, scale = prepare_data.joint_scaler(benign, meters, seed=11)
        expected = np.concatenate([benign, *blocks.values()], axis=0)
        self.assertTrue(np.allclose(mean, expected.mean(axis=0)))
        self.assertTrue(np.allclose(scale, expected.std(axis=0)))

    def test_paper_metric_formulas_use_balanced_accuracy(self) -> None:
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        scores = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
        result = run_experiment.confusion_metrics(
            labels, scores, threshold=0.5
        )
        self.assertEqual((result["TP"], result["FN"]), (1, 1))
        self.assertEqual((result["FP"], result["TN"]), (1, 1))
        self.assertEqual(result["DR"], 50)
        self.assertEqual(result["FA"], 50)
        self.assertEqual(result["ACC"], 50)
        self.assertEqual(result["PR"], 50)
        self.assertEqual(result["F1"], 50)

    def test_supervised_split_is_exact_seeded_and_deterministic(self) -> None:
        left = run_experiment.exact_random_train_mask(101, seed=11)
        repeated = run_experiment.exact_random_train_mask(101, seed=11)
        other = run_experiment.exact_random_train_mask(101, seed=22)
        self.assertEqual(int(left.sum()), (2 * 101) // 3)
        self.assertTrue(np.array_equal(left, repeated))
        self.assertFalse(np.array_equal(left, other))

    def test_binary_svm_margin_is_oriented_toward_malicious(self) -> None:
        margins = np.array([-2.0, 0.5, 3.0])
        self.assertTrue(
            np.array_equal(
                run_experiment.svm_attack_margin(margins, np.array([0, 1])),
                margins,
            )
        )

    def test_adasyn_audit_uses_resampled_attack_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, run = root / "data", root / "run"
            data.mkdir()
            run.mkdir()
            np.save(run / "scores.npy", np.array([0.2, 0.8], dtype=np.float32))
            np.save(run / "labels.npy", np.array([0, 1], dtype=np.int8))
            np.save(run / "test_global_row.npy", np.array([2, 3], dtype=np.int64))
            np.save(data / "test_attack_id.npy", np.array([0, 0, 1, 2], dtype=np.int8))
            np.save(data / "test_original_attack_id.npy", np.array([0, 0], dtype=np.int8))
            _, attacks, _, _ = analyze_results.audit_attempt_arrays(
                run, data, {"task": "anomaly", "test_view": "adasyn"}
            )
            self.assertEqual(attacks.tolist(), [1, 2])

    def test_table_ii_audit_uses_canonical_csv_after_metadata_correction(self) -> None:
        attempt = {
            "configuration": {"model": "supervised_feed_forward"},
            "reported_table_2": {"ACC": 90.25},
        }
        target = analyze_results.canonical_reported(attempt)
        self.assertIsNotNone(target)
        self.assertEqual(target["DR"], 91)
        self.assertEqual(target["ACC"], 91)
        self.assertEqual(target["F1"], 90.5)

    def test_runner_table_ii_targets_equal_source_transcription(self) -> None:
        for model, expected in run_experiment.REPORTED_TABLE_2.items():
            attempt = {
                "configuration": {"model": model},
                "reported_table_2": expected,
            }
            self.assertEqual(
                analyze_results.canonical_reported(attempt),
                {key: float(value) for key, value in expected.items()},
            )

    def test_naive_bayes_route_uses_complete_b_plus_m(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "results"
            data.mkdir()
            np.save(data / "benign.npy", rng.normal(0, 1, (12, 48)).astype("float32"))
            for attack_id in range(1, 7):
                np.save(
                    data / f"attack_{attack_id}.npy",
                    rng.normal(attack_id, 1, (12, 48)).astype("float32"),
                )
            metadata = {"status": "complete", "files": {}}
            metadata_path = data / "metadata.json"
            metadata_path.write_text(json.dumps(metadata))
            args = argparse.Namespace(
                data=data,
                output=output,
                seed=11,
                score_batch=7,
            )
            self.assertEqual(
                run_experiment.run_naive_bayes(
                    args, metadata=metadata, metadata_path=metadata_path
                ),
                0,
            )
            results = list(output.rglob("result.json"))
            self.assertEqual(len(results), 1)
            result = json.loads(results[0].read_text())
            self.assertEqual(result["configuration"]["task"], "supervised")
            self.assertEqual(result["data"]["counts"]["total"], 84)
            self.assertEqual(result["data"]["counts"]["train"], 56)
            self.assertEqual(result["data"]["counts"]["test"], 28)
            self.assertEqual(
                sum(result["data"]["counts"]["train_by_class"]), 56
            )
            self.assertEqual(
                sum(result["data"]["counts"]["test_by_class"]), 28
            )
            self.assertEqual(
                analyze_results.effective_eligibility(result),
                "exploratory_interpretation_I-SUPERVISED-ADASYN-NONE",
            )
            audit = analyze_results.audit_scores(results[0])
            self.assertIsNone(audit["trained_vs_zero_reconstruction"])
            self.assertEqual(
                sum(
                    row["profiles"]
                    for row in audit["table_v_heldout_benign_interpretation"]
                ),
                int(np.count_nonzero(np.load(results[0].parent / "labels.npy"))),
            )
            self.assertIn("closest_reported_operating_point", audit)

    def test_every_remaining_table_iii_model_builds_and_scores(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(size=(2, 48)).astype("float32")
        expected_output = {
            "lstm_sae": (2, 48),
            "fc_vae": (2, 48),
            "lstm_vae": (2, 48),
            "lstm_aea": (2, 48),
            "supervised_feed_forward": (2, 2),
            "supervised_lstm": (2, 1),
        }
        for name, shape in expected_output.items():
            with self.subTest(model=name):
                model = models.build_model(name, seed=11)
                self.assertEqual(tuple(model(values, training=False).shape), shape)
                inventory = models.layer_inventory(model)
                self.assertEqual(len(inventory), len(model.layers))
                self.assertTrue(all("output_shape" in row for row in inventory))

    def test_reported_targets_cover_every_proposed_model_and_table_cell(self) -> None:
        proposed = {"fc_sae", "lstm_sae", "fc_vae", "lstm_vae", "lstm_aea"}
        self.assertEqual(set(run_experiment.REPORTED_TABLE_4), proposed)
        self.assertEqual(set(run_experiment.REPORTED_TABLE_5), proposed)
        for model in proposed:
            self.assertEqual(
                set(run_experiment.REPORTED_TABLE_4[model]),
                {"half", "three_quarter", "full"},
            )
            self.assertEqual(set(run_experiment.REPORTED_TABLE_5[model]), set(range(1, 7)))

    def test_vae_probability_completion_is_low_when_error_is_large(self) -> None:
        class ZeroModel:
            def __call__(self, values: np.ndarray, training: bool = False) -> np.ndarray:
                del training
                return np.zeros_like(values)

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "scores.npy"
            values = np.array([[0.0] * 48, [2.0] * 48], dtype=np.float32)
            scores, _ = run_experiment.score_mse(
                ZeroModel(),
                values,
                target,
                batch_size=2,
                score_kind="reconstruction_probability",
            )
            self.assertGreater(float(scores[0]), float(scores[1]))
            self.assertAlmostEqual(float(scores[0]), 1.0)

    def test_failed_scoring_reuses_preserved_weights_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            run = root / "run"
            data.mkdir()
            run.mkdir()
            metadata_path = data / "metadata.json"
            metadata_path.write_text(json.dumps({"status": "complete"}))
            values = np.arange(4 * 48, dtype=np.float32).reshape(4, 48) / 100
            np.save(data / "test_original_x.npy", values)
            np.save(
                data / "test_original_y.npy",
                np.array([0, 0, 1, 1], dtype=np.int8),
            )
            model = models.build_lstm_vae(seed=11, learning_rate=0.01)
            model.save_weights(run / "model.weights.h5")
            configuration = {
                "method": "I-ADASYN-NONE-ISET-LSTM-VAE",
                "model": "lstm_vae",
                "seed": 11,
                "learning_rate": 0.01,
                "score_batch": 2,
                "test_view": "original",
                "threshold": 0.47,
                "anomaly_direction": "lower",
                "data_metadata_sha256": run_experiment.sha256(metadata_path),
            }
            (run / "failure.json").write_text(
                json.dumps(
                    {
                        "configuration": configuration,
                        "git_commit": "training-commit",
                        "elapsed_seconds": 10,
                    }
                )
            )

            self.assertEqual(run_experiment.recover_failed_scoring(run, data), 0)
            recovery = json.loads((run / "score_recovery.json").read_text())
            self.assertEqual(recovery["kind"], "operational_score_recovery")
            self.assertEqual(recovery["training_git_commit"], "training-commit")
            self.assertEqual(np.load(run / "scores.npy").shape, (4,))
            self.assertFalse((run / "history.json").exists())

    def test_untrained_sanity_respects_scoring_batch(self) -> None:
        class ZeroModel:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def __call__(self, values: np.ndarray, training: bool = False) -> np.ndarray:
                del training
                self.batch_sizes.append(values.shape[0])
                return np.zeros_like(values)

        model = ZeroModel()
        values = np.ones((14, 48), dtype=np.float32)
        labels = np.array([0] * 7 + [1] * 7, dtype=np.int8)
        result = run_experiment.score_untrained_sample(
            model,
            values,
            labels,
            threshold=0.5,
            batch_size=3,
        )
        self.assertEqual(result["rows"], 14)
        self.assertEqual(sum(model.batch_sizes), 14)
        self.assertLessEqual(max(model.batch_sizes), 3)

    def test_classical_breadth_routes_preserve_their_explicit_caps(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            benign = rng.normal(0, 1, (24, 48)).astype("float32")
            np.save(data / "benign.npy", benign)
            for attack_id in range(1, 7):
                np.save(
                    data / f"attack_{attack_id}.npy",
                    rng.normal(attack_id / 4, 1, (24, 48)).astype("float32"),
                )
            np.save(data / "x_train.npy", benign[:16])
            test_x = np.concatenate(
                [benign[16:], np.load(data / "attack_1.npy")[:8]]
            )
            np.save(data / "test_original_x.npy", test_x)
            np.save(
                data / "test_original_y.npy",
                np.array([0] * 8 + [1] * 8, dtype=np.int8),
            )
            metadata = {"status": "complete", "source_nodes": {}}
            metadata_path = data / "metadata.json"
            metadata_path.write_text(json.dumps(metadata))
            for name in ("arima", "one_class_svm", "multiclass_svm"):
                args = argparse.Namespace(
                    model=name,
                    data=data,
                    output=root / "results",
                    seed=11,
                    score_batch=7,
                    one_class_svm_train_cap=8,
                    multiclass_svm_train_cap=21,
                    svm_test_cap=12,
                )
                with self.subTest(model=name):
                    self.assertEqual(
                        run_experiment.run_classical_benchmark(
                            args, metadata=metadata, metadata_path=metadata_path
                        ),
                        0,
                    )
            one_class = json.loads(
                next((root / "results/table_3/one_class_svm").rglob("result.json")).read_text()
            )
            self.assertEqual(one_class["data"]["train_rows_used"], 8)
            self.assertEqual(one_class["data"]["test_rows_used"], 12)

    def test_analysis_never_merges_different_execution_configs(self) -> None:
        def attempt(seed: int, batch: int, group: str) -> dict[str, object]:
            config = {
                "configuration_id": group,
                "method": "I-ADASYN-NONE-ISET-FC-SAE",
                "model": "fc_sae",
                "seed": seed,
                "train_fraction": "full",
                "test_view": "original",
                "table_v": False,
                "batch_size": batch,
                "epochs_max": 100,
            }
            metrics = {name: 50.0 for name in analyze_results.METRICS}
            return {
                "configuration": config,
                "eligibility": "exploratory_interpretation_I-ADASYN-NONE",
                "metrics": metrics,
                "reported_table_3": {
                    name: 80.0 for name in analyze_results.METRICS
                },
                "reported_table_4": {"training_minutes": 137, "ACC": 83},
                "timing_seconds": {
                    "fit": 60.0,
                    "score_table_3": 1.0,
                    "total": 62.0,
                },
                "table_v": None,
                "_path": f"result-{seed}-{batch}.json",
            }

        tables = analyze_results.aggregate(
            [attempt(11, 512, "group-a"), attempt(22, 32, "group-b")]
        )
        self.assertEqual(len(tables["table_3_summary"]), 2)
        self.assertEqual(len(tables["table_4_summary"]), 2)
        self.assertEqual(
            {row["batch_size"] for row in tables["table_3_summary"]},
            {32, 512},
        )

    def test_attempt_ids_include_every_recorded_execution_choice(self) -> None:
        left = {"model": "fc_sae", "batch_size": 512, "seed": 11}
        right = {"model": "fc_sae", "batch_size": 32, "seed": 11}
        control = {**left, "output_activation": "linear"}
        self.assertEqual(run_experiment.stable_id(left), run_experiment.stable_id(left))
        self.assertNotEqual(run_experiment.stable_id(left), run_experiment.stable_id(right))
        self.assertNotEqual(run_experiment.stable_id(left), run_experiment.stable_id(control))

    def test_csv_writer_preserves_fields_present_in_only_some_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.csv"
            analyze_results.write_csv(path, [{"a": 1}, {"a": 2, "b": 3}])
            lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "a,b")
        self.assertEqual(lines[1:], ["1,", "2,3"])


if __name__ == "__main__":
    unittest.main()
