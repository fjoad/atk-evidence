"""Fixtures for the bounded LSTM-SAE anchor-promotion decision."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/atk-2022-deep-autoencoder"
CHECKS = STUDY / "checks"
sys.path.insert(0, str(CHECKS))
spec = importlib.util.spec_from_file_location(
    "lstm_sae_h200_cost", CHECKS / "lstm_sae_h200_cost.py"
)
cost = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cost
spec.loader.exec_module(cost)


class LstmSaeAnchorPromotionTests(unittest.TestCase):
    def test_preserved_a16_projection_is_recalculated_exactly(self) -> None:
        projection = cost.cost_projection(
            slowest_epoch_seconds=218.0,
            score_seconds=3.9136646389961243,
        )
        self.assertAlmostEqual(projection["minimum_10_epoch_hours"], 42.79023332265118)
        self.assertAlmostEqual(
            projection["worst_case_100_epoch_hours"], 417.14251374867655
        )
        self.assertFalse(projection["passes_100_epoch_gate"])
        record = json.loads(
            (STUDY / "results/lstm_sae_anchor_cost_20260901.json").read_text()
        )
        self.assertEqual(
            projection["projected_full_epoch_seconds"],
            record["projection"]["projected_full_epoch_seconds"],
        )
        self.assertEqual(
            projection["projected_full_score_seconds"],
            record["projection"]["projected_full_score_seconds"],
        )

    def test_decision_stability_rule_accepts_identical_scores(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        result = cost.stability_summary(
            labels, {batch: scores.copy() for batch in cost.BATCHES}
        )
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["maximums"]["printed_cutoff_label_changes"], 0)
        self.assertEqual(
            result["maximums"]["fixed_threshold_transfer_label_changes"], 0
        )

    def test_hardware_wrapper_cannot_launch_a_full_anchor(self) -> None:
        wrapper = (CHECKS / "run_lstm_sae_h200_cost.sbatch").read_text()
        script = (CHECKS / "lstm_sae_h200_cost.py").read_text()
        self.assertIn("#SBATCH -p gpu-H200", wrapper)
        self.assertIn("#SBATCH --gres=gpu:1", wrapper)
        self.assertIn("#SBATCH --time=02:00:00", wrapper)
        self.assertIn("--batches 256 128 64 32", wrapper)
        self.assertNotIn("--epochs", wrapper)
        self.assertNotIn("FULL_SCORE_ROWS,", wrapper)
        self.assertNotIn("epochs=100", script)
        self.assertNotIn("fc_vae", wrapper)
        self.assertNotIn("lstm_vae", wrapper)
        self.assertNotIn("lstm_aea", wrapper)

    def test_contract_keeps_fabrication_and_implausibility_separate(self) -> None:
        decision = (
            ROOT
            / "docs/decisions/2026-09-01-implausibility-and-fabrication-boundary.md"
        ).read_text()
        contract = (STUDY / "LSTM_SAE_ANCHOR_PROMOTION.md").read_text()
        self.assertIn("highly implausible within", decision)
        self.assertIn("additional forensic evidence", decision)
        self.assertIn("original absolute `1e-6` all-score gate remains", contract)
        self.assertIn("If any gate fails", contract)


if __name__ == "__main__":
    unittest.main()
