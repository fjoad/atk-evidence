from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "studies" / "takiddin-2021-robust-poisoning"
MODULE_PATH = STUDY_DIR / "checks" / "source_table_audit.py"
RESULT_PATH = STUDY_DIR / "results" / "source_table_audit_20260902.json"

SPEC = importlib.util.spec_from_file_location("robust_poisoning_source_table_audit", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class RobustPoisoningSourceAuditTests(unittest.TestCase):
    def test_reported_table_coverage_is_exact(self) -> None:
        expected_rows = {"table_3": 49, "table_4": 49, "table_5": 21}
        for table_id, row_count in expected_rows.items():
            table = AUDIT.load_table(table_id)
            self.assertEqual(sum(len(metrics) for metrics in table.values()), row_count)

    def test_synthetic_consistent_row_passes_every_identity(self) -> None:
        row = {
            "DR": 80.0,
            "FA": 20.0,
            "SP": 80.0,
            "PR": 80.0,
            "ACC": 80.0,
            "F1": 80.0,
            "AUC": 80.0,
        }
        result = AUDIT.audit_row(row)
        for name in (
            "specificity",
            "f1",
            "balanced_accuracy",
            "balanced_precision",
            "any_prevalence_screen",
        ):
            self.assertTrue(result[name]["pass"], name)

    def test_synthetic_inconsistent_precision_fails_balance(self) -> None:
        row = {
            "DR": 90.0,
            "FA": 20.0,
            "SP": 80.0,
            "PR": 60.0,
            "ACC": 85.0,
            "F1": 72.0,
            "AUC": 80.0,
        }
        result = AUDIT.audit_row(row)
        self.assertFalse(result["balanced_precision"]["pass"])
        self.assertFalse(result["any_prevalence_screen"]["pass"])

    def test_prose_calculations_cover_every_declared_comparison(self) -> None:
        tables = {
            table_id: AUDIT.load_table(table_id)
            for table_id in AUDIT.EXPECTED_MODELS
        }
        prose = AUDIT.prose_calculations(tables)
        self.assertEqual(
            set(prose["per_model_dr_drop_from_p0_percentage_points"]),
            {"table_3", "table_4", "table_5"},
        )
        self.assertEqual(
            set(prose["mean_stepwise_dr_drop_percentage_points"]),
            {"table_3", "table_4"},
        )
        self.assertEqual(
            set(prose["p30_sequential_minus_baseline_percentage_points"]),
            {"aea", "ensemble_averaging"},
        )
        for comparison in prose["p30_sequential_minus_baseline_percentage_points"].values():
            self.assertEqual(set(comparison), set(AUDIT.METRICS))

    def test_preserved_result_matches_fresh_calculation(self) -> None:
        if not RESULT_PATH.exists():
            self.skipTest("source audit result has not been generated yet")
        preserved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        fresh = AUDIT.build_audit()
        for result in (preserved, fresh):
            result.pop("created_utc")
        self.assertEqual(preserved, fresh)


if __name__ == "__main__":
    unittest.main()
