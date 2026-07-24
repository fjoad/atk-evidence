from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = (
    REPO_ROOT
    / "studies/atk-2022-deep-autoencoder/analyze_results.py"
)
SPEC = importlib.util.spec_from_file_location("paper1_score_sanity", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoreSanityTests(unittest.TestCase):
    def test_oracle_threshold_diagnostic_separates_simple_scores(self) -> None:
        diagnostic = MODULE._score_diagnostic(
            np.asarray([0, 0, 1, 1], dtype=np.int8),
            np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        )
        self.assertEqual(diagnostic["auc"], 1.0)
        self.assertEqual(diagnostic["oracle_test_balanced_accuracy"], 1.0)
        self.assertEqual(diagnostic["n"], 4)

    def test_verified_selection_prefers_ddp_over_later_single_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "fc_sae/seed_11/attempts"
            for name, distributed in (("a-ddp", True), ("z-local", False)):
                attempt = attempts / name
                attempt.mkdir(parents=True)
                (attempt / "arrays.npz").write_bytes(b"arrays")
                (attempt / "metadata.json").write_text(
                    json.dumps(
                        {
                            "execution": (
                                {"distributed_execution": {}}
                                if distributed
                                else {}
                            )
                        }
                    ),
                    encoding="utf-8",
                )
                (attempt / "result.json").write_text("{}", encoding="utf-8")
                artifacts = {
                    filename: hashlib.sha256(
                        (attempt / filename).read_bytes()
                    ).hexdigest()
                    for filename in ("arrays.npz", "metadata.json", "result.json")
                }
                (attempt / "manifest.json").write_text(
                    json.dumps({"status": "complete", "artifacts": artifacts}),
                    encoding="utf-8",
                )
            selected = MODULE._verified_latest_attempts(root, ("fc_sae",))
            self.assertEqual([attempt.name for attempt in selected], ["a-ddp"])

    def test_iset_compact_scores_bind_to_checksum_cache_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "iset.npz"
            values = np.asarray(
                [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]],
                dtype=np.float32,
            )
            labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
            sample_ids = np.asarray(["a", "b", "c", "d"])
            synthetic = np.zeros(4, dtype=bool)
            np.savez(
                cache,
                anomaly_test_values=values,
                anomaly_test_labels=labels,
                anomaly_test_sample_ids=sample_ids,
                anomaly_test_is_synthetic=synthetic,
                supervised_test_labels=labels,
                supervised_test_sample_ids=sample_ids,
                supervised_test_is_synthetic=synthetic,
            )

            attempt = root / "runs/fc_sae/seed_11/attempts/a"
            attempt.mkdir(parents=True)
            np.savez(
                attempt / "arrays.npz",
                score__reconstruction_mse=np.asarray([0.1, 0.2, 0.8, 0.9]),
                prediction__reconstruction_mse=labels,
            )
            (attempt / "metadata.json").write_text(
                json.dumps({"environment": {"slurm_job_id": "1"}}),
                encoding="utf-8",
            )
            (attempt / "result.json").write_text(
                json.dumps(
                    {
                        "model": "fc_sae",
                        "seed": 11,
                        "metrics": {"reconstruction_mse": {"auc": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            artifacts = {
                filename: hashlib.sha256((attempt / filename).read_bytes()).hexdigest()
                for filename in ("arrays.npz", "metadata.json", "result.json")
            }
            (attempt / "manifest.json").write_text(
                json.dumps({"status": "complete", "artifacts": artifacts}),
                encoding="utf-8",
            )

            document = MODULE.analyze(
                root / "runs", cache, ("fc_sae",), dataset="iset"
            )
            record = document["score_records"][0]
            self.assertEqual(record["execution_branch"], "panther_single_gpu")
            self.assertEqual(
                record["subsets"]["paper_primary_all_rows"]["auc"], 1.0
            )
            self.assertIn("input_energy_diagnostic", record)


if __name__ == "__main__":
    unittest.main()
