"""Software fixtures for the no-training recurrent score recovery."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/atk-2022-deep-autoencoder"
CHECKS = STUDY / "checks"
spec = importlib.util.spec_from_file_location(
    "remaining_score_recovery", CHECKS / "remaining_score_recovery.py"
)
recovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recovery
spec.loader.exec_module(recovery)


class RemainingScoreRecoveryTests(unittest.TestCase):
    def test_source_bindings_match_the_committed_feasibility_record(self) -> None:
        record = json.loads(recovery.SOURCE_RECORD.read_text())
        self.assertEqual(recovery.digest(recovery.SOURCE_RECORD), recovery.SOURCE_RECORD_SHA256)
        jobs = {job["model"]: job for job in record["jobs"]}
        for model, expected in recovery.ATTEMPTS.items():
            job = jobs[model]
            self.assertEqual(job["attempt_id"], expected["attempt_id"])
            self.assertEqual(
                job["artifacts_sha256"]["config.json"], expected["config_sha256"]
            )
            self.assertEqual(
                job["artifacts_sha256"]["failure.json"], expected["failure_sha256"]
            )
            self.assertEqual(
                job["artifacts_sha256"]["model.weights.h5"], expected["weights_sha256"]
            )
            self.assertEqual(
                job["artifacts_sha256"]["selection.npz"], expected["selection_sha256"]
            )

    def test_source_implementation_hashes_remain_exact(self) -> None:
        for name, expected in recovery.SOURCE_IMPLEMENTATION_SHA256.items():
            self.assertEqual(recovery.digest(recovery.REPRODUCTION / name), expected)

    def test_envelope_preserves_strict_cutoff_semantics(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        for scores, direction in (
            (np.array([0.1, 0.2, 0.8, 0.9]), "higher"),
            (np.array([0.9, 0.8, 0.2, 0.1]), "lower"),
        ):
            envelope = recovery.score_envelope(labels, scores, direction)
            self.assertEqual(envelope["AUC"], 100.0)
            selected = envelope["at_FA_cap"]["15.0"]
            self.assertEqual(selected["DR"], 100.0)
            self.assertEqual(selected["FA"], 0.0)
            self.assertEqual(selected["metrics"]["TP"], 2)
            self.assertEqual(selected["metrics"]["FP"], 0)
            predictions = recovery.strict_predictions(
                scores, selected["strict_cutoff"], direction
            )
            np.testing.assert_array_equal(predictions, labels.astype(bool))

    def test_numerical_difference_keeps_scale_and_exact_counts_separate(self) -> None:
        left = np.array([0.0, 1.0, 2.0, 3.0])
        right = np.array([0.0, 1.0, 2.0 + 2e-6, 3.0])
        result = recovery.numerical_difference(left, right)
        self.assertAlmostEqual(result["maximum_absolute"], 2e-6)
        self.assertAlmostEqual(result["maximum_over_score_range"], 2e-6 / 3.0)
        self.assertEqual(result["exactly_equal_rows"], 3)
        self.assertEqual(result["rows"], 4)

    def test_ties_remain_unflagged(self) -> None:
        scores = np.array([0.4, 0.5, 0.6])
        np.testing.assert_array_equal(
            recovery.strict_predictions(scores, 0.5, "higher"),
            np.array([False, False, True]),
        )
        np.testing.assert_array_equal(
            recovery.strict_predictions(scores, 0.5, "lower"),
            np.array([True, False, False]),
        )

    def test_wrapper_is_pilot_only_and_has_no_training_route(self) -> None:
        script = (CHECKS / "remaining_score_recovery.py").read_text()
        wrapper = (CHECKS / "run_remaining_score_recovery.sbatch").read_text()
        self.assertNotIn(".fit(", script)
        self.assertNotIn("train_on_batch", script)
        self.assertNotIn("--epochs", wrapper)
        self.assertIn("#SBATCH --time=00:30:00", wrapper)
        self.assertIn("--batches 256 128 64 32", wrapper)
        self.assertIn("lstm_sae)", wrapper)
        self.assertIn("lstm_vae)", wrapper)
        self.assertNotIn("fc_vae)", wrapper)
        self.assertNotIn("lstm_aea)", wrapper)


if __name__ == "__main__":
    unittest.main()
