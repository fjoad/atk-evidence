from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


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
