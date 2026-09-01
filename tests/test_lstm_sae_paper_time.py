import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "studies/atk-2022-deep-autoencoder/checks/lstm_sae_paper_time.py"
)
WRAPPER = (
    ROOT
    / "studies/atk-2022-deep-autoencoder/checks/run_lstm_sae_paper_time.sbatch"
)
CONTRACT = (
    ROOT / "studies/atk-2022-deep-autoencoder/PAPER_TIME_BUDGET_CONTRACT.md"
)


def load_module():
    os.environ.setdefault("KERAS_BACKEND", "torch")
    spec = importlib.util.spec_from_file_location("lstm_sae_paper_time", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LstmSaePaperTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_contract_freezes_paper_time_and_one_contemporaneous_gpu(self):
        text = CONTRACT.read_text()
        self.assertIn("183 minutes", text)
        self.assertIn("exactly one V100-16GB", text)
        self.assertIn("Do not say that every undocumented implementation is impossible", text)

    def test_wrapper_cannot_scale_out_or_use_newer_gpu(self):
        wrapper = WRAPPER.read_text()
        self.assertIn("#SBATCH --gres=gpu:v100_16GB:1", wrapper)
        self.assertNotIn("H200", wrapper)
        self.assertNotIn("A100", wrapper)
        self.assertIn("#SBATCH --time=06:00:00", wrapper)
        self.assertIn("EXPECTED_COMMIT", wrapper)

    def test_time_budget_is_exactly_183_minutes(self):
        self.assertEqual(self.module.FIT_SECONDS_LIMIT, 10_980)
        self.assertEqual(self.module.BATCH_SIZE, 32)
        self.assertEqual(self.module.FULL_FIT_ROWS, 1_500_523)
        self.assertEqual(self.module.FULL_SCORE_ROWS, 8_884_989)

    def test_complete_threshold_envelope_finds_reachable_corner(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        scores = np.asarray([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
        envelope = self.module.score_envelope(labels, scores, direction="higher")
        self.assertEqual(envelope["threshold_candidates"], 9)
        self.assertTrue(envelope["reported_corner_reached"])
        self.assertAlmostEqual(envelope["AUC"], 100.0)

    def test_complete_threshold_envelope_preserves_failed_corner(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.3, 0.5, 0.7, 0.0, 0.2, 0.4, 0.6])
        envelope = self.module.score_envelope(labels, scores, direction="higher")
        self.assertFalse(envelope["reported_corner_reached"])
        self.assertFalse(envelope["rounded_corner_reached"])


if __name__ == "__main__":
    unittest.main()
