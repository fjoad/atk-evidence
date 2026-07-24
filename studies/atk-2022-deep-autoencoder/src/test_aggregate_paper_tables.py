"""Fixture tests for deterministic Paper 1 result aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aggregate_paper_tables import (
    BLOCKED,
    EXPECTED_SGCC_SHA256,
    ORDINARY_EXECUTION_BRANCH,
    PANTHER_DDP_EXECUTION_BRANCH,
    PANTHER_DDP_IMPLEMENTATION,
    PANTHER_DDP_SOURCE,
    VAE_DIAGNOSTIC_BRANCH,
    _canonical_json_bytes,
    aggregate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(value: float, false_alarm: float | None = None) -> dict[str, object]:
    fa = 1.0 - value if false_alarm is None else false_alarm
    tp = int(round(100 * value))
    fp = int(round(100 * fa))
    tn = 100 - fp
    fn = 100 - tp
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    balanced_accuracy = (value + specificity) / 2.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    # Scores in the fixture are binary; their AUC equals balanced accuracy.
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "dr": value,
        "fa": fa,
        "sp": specificity,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "auc": balanced_accuracy,
        "threshold": 0.5,
        "positive_if": "higher",
        "n": 200,
        "positives": 100,
        "negatives": 100,
    }


class AggregationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = self.root / "immutable-runs"
        self.runner.mkdir()
        self.reported = self.root / "reported"
        self.output = self.root / "output"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "fixture_runner.py").write_text(
            "# immutable fixture runner\n", encoding="utf-8"
        )
        (self.source / PANTHER_DDP_SOURCE).write_text(
            "# immutable DDP fixture runner\n", encoding="utf-8"
        )
        self.config = self.root / "contract.toml"
        self.config.write_text(
            """[run]
model_seeds = [11, 22, 33]
batch_size = 512

[thresholds]
arima = 0.58
one_class_svm = 0.45
fc_sae = 0.58
lstm_sae = 0.61
fc_vae = 0.43
lstm_vae = 0.47
lstm_aea = 0.51

[table_1.fc_sae]
layers_total = 8
encoder_widths = [400, 300, 200, 100]
optimizer = "adam"
dropout = 0.4
hidden_activation = "sigmoid"
output_activation = "softmax"

[table_1.fc_vae]
layers_total = 8
encoder_widths = [500, 400, 300, 100]
optimizer = "adam"
dropout = 0.4
hidden_activation = "relu"
output_activation = "softmax"
""",
            encoding="utf-8",
        )
        self.contract_sha = _sha256(self.config)
        self._reported_fixtures()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _reported_fixtures(self) -> None:
        table_1_fields = [
            "model",
            "layers_total",
            "encoder_widths",
            "optimizer",
            "dropout",
            "hidden_activation",
            "output_activation",
            "printed_page",
            "pdf_page",
        ]
        _write_csv(
            self.reported / "table_1.csv",
            table_1_fields,
            [
                {
                    "model": "FC-SAE",
                    "layers_total": 8,
                    "encoder_widths": "400;300;200;100",
                    "optimizer": "Adam",
                    "dropout": 0.4,
                    "hidden_activation": "Sigmoid",
                    "output_activation": "Softmax",
                    "printed_page": 1,
                    "pdf_page": 1,
                },
                {
                    "model": "FC-VAE",
                    "layers_total": 8,
                    "encoder_widths": "500;400;300;100",
                    "optimizer": "Adam",
                    "dropout": 0.4,
                    "hidden_activation": "ReLU",
                    "output_activation": "Softmax",
                    "printed_page": 1,
                    "pdf_page": 1,
                },
            ],
        )
        targets = []
        for model in ("FC-SAE", "FC-VAE", "Naive Bayes"):
            targets.append(
                {
                    "model": model,
                    "DR": 90,
                    "FA": 10,
                    "SP": 90,
                    "PR": 90,
                    "ACC": 90,
                    "F1": 90,
                    "AUC": 90,
                    "printed_page": 2,
                    "pdf_page": 2,
                }
            )
        _write_csv(
            self.reported / "table_2.csv",
            [
                "model",
                "DR",
                "FA",
                "SP",
                "PR",
                "ACC",
                "F1",
                "AUC",
                "printed_page",
                "pdf_page",
            ],
            targets,
        )
        _write_csv(
            self.reported / "table_3.csv",
            [
                "model",
                "DR",
                "FA",
                "SP",
                "PR",
                "ACC",
                "F1",
                "AUC",
                "printed_page",
                "pdf_page",
            ],
            [targets[0]],
        )
        _write_csv(
            self.reported / "table_4.csv",
            [
                "model",
                "metric",
                "half_train",
                "three_quarter_train",
                "full_train",
                "unit",
                "printed_page",
                "pdf_page",
            ],
            [
                {
                    "model": "FC-SAE",
                    "metric": "ACC",
                    "half_train": 70,
                    "three_quarter_train": 80,
                    "full_train": 90,
                    "unit": "percent",
                    "printed_page": 2,
                    "pdf_page": 2,
                }
            ],
        )
        _write_csv(
            self.reported / "table_5.csv",
            [
                "model",
                "metric",
                "attack_1",
                "attack_2",
                "attack_3",
                "attack_4",
                "attack_5",
                "attack_6",
                "average",
                "printed_page",
                "pdf_page",
            ],
            [
                {
                    "model": "FC-SAE",
                    "metric": "DR",
                    "attack_1": 1,
                    "attack_2": 2,
                    "attack_3": 3,
                    "attack_4": 4,
                    "attack_5": 5,
                    "attack_6": 6,
                    "average": 3.5,
                    "printed_page": 2,
                    "pdf_page": 2,
                }
            ],
        )

    def _attempt(
        self,
        model: str,
        seed: int,
        attempt_id: str,
        *,
        status: str = "complete",
        metrics: dict[str, dict[str, object]] | None = None,
        array_metrics: dict[str, dict[str, object]] | None = None,
        tamper: bool = False,
        execution_branch: str | None = None,
        ddp_source_sha256: str | None = None,
        metadata_git_commit: str | None = None,
        slurm_job_id: str = "123456",
    ) -> Path:
        directory = (
            self.runner
            / "table_2"
            / "sgcc"
            / model
            / f"seed_{seed}"
            / "attempts"
            / attempt_id
        )
        directory.mkdir(parents=True)
        source_hash = _sha256(self.source / "fixture_runner.py")
        payload = {
            "schema_version": 1,
            "study": "atk-2022-deep-autoencoder",
            "table": 2,
            "dataset": "SGCC",
            "model": model,
            "seed": seed,
            "contract_sha256": self.contract_sha,
            "source_sha256": EXPECTED_SGCC_SHA256,
            "partition_id_sha256": {"fixture": "partition"},
            "transformation_sha256": {"fixture": "transform"},
            "source_code_sha256": {"fixture_runner.py": source_hash},
            "model_config": (
                self._contract_document()["table_1"].get(model)
                if model not in {"naive_bayes", "arima", "one_class_svm", "multiclass_svm"}
                else None
            ),
            "run_config": self._contract_document()["run"],
            "threshold": (
                self._contract_document()["thresholds"].get(model)
                if model
                in {
                    "arima",
                    "one_class_svm",
                    "fc_sae",
                    "lstm_sae",
                    "fc_vae",
                    "lstm_vae",
                    "lstm_aea",
                }
                else (0.0 if model == "multiclass_svm" else 0.5)
            ),
        }
        if execution_branch is None:
            execution_branch = (
                ORDINARY_EXECUTION_BRANCH
                if model
                in {"naive_bayes", "arima", "one_class_svm", "multiclass_svm"}
                else PANTHER_DDP_EXECUTION_BRANCH
            )
        distributed_execution: dict[str, object] | None = None
        if execution_branch == PANTHER_DDP_EXECUTION_BRANCH:
            git_commit = "a" * 40
            distributed_execution = {
                "implementation": PANTHER_DDP_IMPLEMENTATION,
                "implementation_source_sha256": (
                    ddp_source_sha256 or _sha256(self.source / PANTHER_DDP_SOURCE)
                ),
                "world_size": 4,
                "global_batch_size": 512,
                "rank_zero_inference_batch_size": 128,
                "thread_environment": {
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                },
                "gpu_inventory": [
                    {
                        "rank": rank,
                        "local_rank": rank,
                        "name": "Tesla V100-PCIE-16GB",
                        "total_memory_bytes": 16 * 1024**3,
                    }
                    for rank in range(4)
                ],
                "git_commit": git_commit,
            }
            payload["distributed_execution"] = distributed_execution
        if metrics is not None:
            metrics = {
                branch: {**branch_metrics, "threshold": payload["threshold"]}
                for branch, branch_metrics in metrics.items()
            }
        if array_metrics is not None:
            array_metrics = {
                branch: {**branch_metrics, "threshold": payload["threshold"]}
                for branch, branch_metrics in array_metrics.items()
            }
        timing = {
            "fit_seconds": float(seed),
            "score_seconds": seed / 10.0,
            "run_seconds": float(seed),
        }
        if status == "complete":
            result = {
                "status": status,
                "model": model,
                "seed": seed,
                "metrics": metrics or {},
                "timings": timing,
                "score_names": sorted((metrics or {}).keys()),
                "n_test": 200,
            }
            history = {
                "epochs_completed": 2,
                "epoch_seconds": [seed / 4.0, seed / 4.0],
            }
        else:
            result = {
                "status": status,
                "model": model,
                "seed": seed,
                "timings": {"elapsed_until_failure_seconds": 1.0},
                "error": {"type": "RuntimeError", "message": "fixture failure"},
            }
            history = {}
        metadata = {
            "status": status,
            "model": model,
            "seed": seed,
            "fingerprint": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
            "contract": {"sha256": self.contract_sha},
            "data_verification": {
                "verified": True,
                "actual_sha256": EXPECTED_SGCC_SHA256,
                "expected_sha256": EXPECTED_SGCC_SHA256,
            },
            "prepared_data_metadata": {
                "partition_id_sha256": payload["partition_id_sha256"],
                "transformation_sha256": payload["transformation_sha256"],
            },
        }
        if distributed_execution is not None:
            metadata["environment"] = {"distributed_world_size": 4}
            metadata["execution_provenance"] = {
                "slurm_job_id": slurm_job_id,
                "git_commit": metadata_git_commit or "a" * 40,
            }
            if status == "complete":
                metadata["execution"] = {
                    "distributed_execution": distributed_execution
                }
        artifacts = {
            "metadata.json": metadata,
            "history.json": history,
            "result.json": result,
        }
        if status == "complete":
            metric_arrays = array_metrics or metrics or {}
            labels = np.concatenate(
                [np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8)]
            )
            arrays: dict[str, np.ndarray] = {
                "labels": labels,
                "sample_ids": np.asarray([f"sample-{index}" for index in range(200)]),
                "is_synthetic": np.zeros(200, dtype=bool),
            }
            for branch, branch_metrics in metric_arrays.items():
                tp = int(branch_metrics["tp"])
                fp = int(branch_metrics["fp"])
                scores = np.concatenate(
                    [
                        np.full(fp, 0.9),
                        np.full(100 - fp, 0.1),
                        np.full(tp, 0.9),
                        np.full(100 - tp, 0.1),
                    ]
                )
                arrays[f"score__{branch}"] = scores
                arrays[f"prediction__{branch}"] = (scores >= 0.5).astype(np.int8)
            with (directory / "arrays.npz").open("wb") as handle:
                np.savez_compressed(handle, **arrays)
        for filename, value in artifacts.items():
            (directory / filename).write_bytes(_canonical_json_bytes(value))
        artifact_names = list(artifacts)
        if status == "complete":
            artifact_names.append("arrays.npz")
        manifest = {
            "schema_version": 1,
            "status": status,
            "fingerprint": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
            "fingerprint_payload": payload,
            "artifacts": {
                filename: _sha256(directory / filename) for filename in artifact_names
            },
            "attempt_id": attempt_id,
        }
        (directory / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
        if tamper:
            with (directory / "arrays.npz").open("ab") as handle:
                handle.write(b"tampered")
        return directory

    def _contract_document(self) -> dict[str, object]:
        import tomllib

        return tomllib.loads(self.config.read_text(encoding="utf-8"))

    def _populate_attempts(self) -> None:
        self._attempt(
            "fc_sae",
            11,
            "20260721T010000.000000Z-aaaaaaaaaaaa",
            metrics={"reconstruction_mse": _metric(0.4, 0.6)},
        )
        self._attempt(
            "fc_sae",
            11,
            "20260721T020000.000000Z-bbbbbbbbbbbb",
            metrics={"reconstruction_mse": _metric(0.90, 0.10)},
        )
        self._attempt(
            "fc_sae",
            22,
            "20260721T030000.000000Z-cccccccccccc",
            metrics={"reconstruction_mse": _metric(0.91, 0.09)},
        )
        self._attempt(
            "fc_sae",
            33,
            "20260721T040000.000000Z-dddddddddddd",
            status="failed",
        )
        self._attempt(
            "fc_sae",
            33,
            "20260721T050000.000000Z-eeeeeeeeeeee",
            metrics={"reconstruction_mse": _metric(0.90, 0.10)},
            tamper=True,
        )
        self._attempt(
            "fc_vae",
            11,
            "20260721T060000.000000Z-ffffffffffff",
            metrics={
                "reconstruction_mse": _metric(0.90, 0.10),
                VAE_DIAGNOSTIC_BRANCH: _metric(0.50, 0.50),
            },
        )
        self._attempt(
            "naive_bayes",
            11,
            "20260721T070000.000000Z-abababababab",
            metrics={"positive_class_probability": _metric(0.90, 0.10)},
        )

    def test_aggregation_selects_first_observation_and_retains_failures(self) -> None:
        self._populate_attempts()
        document = aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )

        individuals = _read_csv(self.output / "table_2_individual.csv")
        selected_fc_sae = [
            row
            for row in individuals
            if row["internal_model"] == "fc_sae"
            and row["selected"] == "true"
            and row["score_role"] == "primary"
        ]
        self.assertEqual({row["seed"] for row in selected_fc_sae}, {"11", "22"})
        self.assertEqual(
            next(row for row in selected_fc_sae if row["seed"] == "11")["attempt_id"],
            "20260721T010000.000000Z-aaaaaaaaaaaa",
        )
        self.assertTrue(
            all(
                row["execution_branch"] == PANTHER_DDP_EXECUTION_BRANCH
                for row in selected_fc_sae
            )
        )
        failed = next(row for row in individuals if row["attempt_status"] == "failed")
        self.assertEqual(failed["error_message"], "fixture failure")
        invalid = next(
            row
            for row in individuals
            if row["attempt_id"] == "20260721T050000.000000Z-eeeeeeeeeeee"
        )
        self.assertEqual(invalid["verification_status"], "INVALID")
        self.assertEqual(invalid["selected"], "false")

        table_2 = _read_csv(self.output / "table_2_reproduction.csv")
        fc_sae = next(
            row
            for row in table_2
            if row["internal_model"] == "fc_sae" and row["score_role"] == "primary"
        )
        self.assertEqual(fc_sae["selected_seed_count"], "2")
        self.assertEqual(fc_sae["close_seed_count"], "1")
        self.assertEqual(fc_sae["status"], "NOT_CLOSE_MATCH")
        self.assertEqual(float(fc_sae["DR_mean"]), 65.5)
        self.assertAlmostEqual(float(fc_sae["fit_seconds_mean"]), 16.5)

        vae_rows = [row for row in table_2 if row["internal_model"] == "fc_vae"]
        self.assertEqual(len(vae_rows), 2)
        diagnostic = next(row for row in vae_rows if row["score_role"] != "primary")
        self.assertEqual(diagnostic["score_branch"], VAE_DIAGNOSTIC_BRANCH)
        self.assertEqual(diagnostic["status"], "DIAGNOSTIC_ONLY")
        naive_bayes = next(row for row in table_2 if row["internal_model"] == "naive_bayes")
        self.assertEqual(naive_bayes["selected_seed_count"], "1")
        self.assertEqual(naive_bayes["status"], "INSUFFICIENT_VALID_SEEDS")

        self.assertEqual(document["attempts"]["discovered"], 7)
        self.assertEqual(len(document["attempts"]["selected"]), 4)
        self.assertEqual(len(document["attempts"]["failures_and_invalid"]), 2)
        self.assertIn(
            PANTHER_DDP_SOURCE,
            document["provenance"]["current_runner_source_code_sha256"],
        )

    def test_ordinary_neural_attempt_cannot_compete_with_panther_ddp(self) -> None:
        self._attempt(
            "fc_sae",
            11,
            "20260721T010000.000000Z-aaaaaaaaaaaa",
            metrics={"reconstruction_mse": _metric(0.99, 0.01)},
            execution_branch=ORDINARY_EXECUTION_BRANCH,
        )
        self._attempt(
            "fc_sae",
            11,
            "20260721T020000.000000Z-bbbbbbbbbbbb",
            metrics={"reconstruction_mse": _metric(0.40, 0.60)},
            execution_branch=PANTHER_DDP_EXECUTION_BRANCH,
        )
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        rows = _read_csv(self.output / "table_2_individual.csv")
        ordinary = next(
            row for row in rows if row["attempt_id"].endswith("aaaaaaaaaaaa")
        )
        distributed = next(
            row for row in rows if row["attempt_id"].endswith("bbbbbbbbbbbb")
        )
        self.assertEqual(ordinary["verification_status"], "VERIFIED_NONMATCHING")
        self.assertEqual(ordinary["selected"], "false")
        self.assertIn("expected 'panther_four_v100_ddp'", ordinary["verification_detail"])
        self.assertEqual(distributed["verification_status"], "VERIFIED_MATCHING")
        self.assertEqual(distributed["selected"], "true")
        self.assertEqual(distributed["slurm_job_id"], "123456")

    def test_panther_ddp_source_hash_and_provenance_are_verified(self) -> None:
        self._attempt(
            "fc_sae",
            11,
            "20260721T010000.000000Z-aaaaaaaaaaaa",
            metrics={"reconstruction_mse": _metric(0.90, 0.10)},
            ddp_source_sha256="b" * 64,
        )
        self._attempt(
            "fc_sae",
            22,
            "20260721T020000.000000Z-bbbbbbbbbbbb",
            metrics={"reconstruction_mse": _metric(0.90, 0.10)},
            metadata_git_commit="b" * 40,
        )
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        rows = _read_csv(self.output / "table_2_individual.csv")
        source_mismatch = next(row for row in rows if row["seed"] == "11")
        provenance_mismatch = next(row for row in rows if row["seed"] == "22")
        self.assertEqual(
            source_mismatch["verification_status"], "VERIFIED_NONMATCHING"
        )
        self.assertIn(
            "DDP runner source hash does not match current file",
            source_mismatch["verification_detail"],
        )
        self.assertEqual(provenance_mismatch["verification_status"], "INVALID")
        self.assertIn(
            "metadata and fingerprint Git commits disagree",
            provenance_mismatch["verification_detail"],
        )
        self.assertEqual(source_mismatch["selected"], "false")
        self.assertEqual(provenance_mismatch["selected"], "false")

    def test_blocked_tables_preserve_targets_and_list_seven_files(self) -> None:
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        table_3 = _read_csv(self.output / "table_3_reproduction.csv")
        self.assertEqual(table_3[0]["DR"], "90")
        self.assertEqual(table_3[0]["reproduction_DR"], BLOCKED)
        self.assertEqual(table_3[0]["status"], BLOCKED)
        self.assertEqual(table_3[0]["missing_file_count"], "7")
        self.assertEqual(table_3[0]["missing_file_gate"].count("|md5="), 7)

        table_4 = _read_csv(self.output / "table_4_reproduction.csv")
        self.assertEqual(table_4[0]["full_train"], "90")
        self.assertEqual(table_4[0]["reproduction_full_train"], BLOCKED)
        table_5 = _read_csv(self.output / "table_5_reproduction.csv")
        self.assertEqual(table_5[0]["attack_6"], "6")
        self.assertEqual(table_5[0]["reproduction_attack_6"], BLOCKED)

        table_1 = _read_csv(self.output / "table_1_reconstructed.csv")
        self.assertTrue(all(row["status"] == "MATCH_CONFIG" for row in table_1))
        document = json.loads((self.output / "paper_1_results.json").read_text())
        self.assertEqual(document["exact_data_gate"]["required_file_count"], 7)
        self.assertEqual(len(document["exact_data_gate"]["required_files"]), 7)

    def test_self_consistent_manifest_cannot_hide_metric_array_disagreement(self) -> None:
        self._attempt(
            "fc_sae",
            11,
            "20260721T010000.000000Z-aaaaaaaaaaaa",
            metrics={"reconstruction_mse": _metric(0.90, 0.10)},
            array_metrics={"reconstruction_mse": _metric(0.50, 0.50)},
        )
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        individual = _read_csv(self.output / "table_2_individual.csv")[0]
        self.assertEqual(individual["verification_status"], "INVALID")
        self.assertEqual(individual["selected"], "false")
        self.assertIn("disagrees with arrays.npz", individual["verification_detail"])

    def test_same_inputs_produce_byte_identical_outputs(self) -> None:
        self._populate_attempts()
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        first = {
            path.name: path.read_bytes() for path in sorted(self.output.iterdir())
        }
        aggregate(
            runner_root=self.runner,
            reported_dir=self.reported,
            config_path=self.config,
            output_dir=self.output,
            source_dir=self.source,
        )
        second = {
            path.name: path.read_bytes() for path in sorted(self.output.iterdir())
        }
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
