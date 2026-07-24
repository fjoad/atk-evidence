from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.verify_data import CER_OFFICIAL_MD5, CER_SCIENCEDB_MD5, digest, verify


class VerifyDataTests(unittest.TestCase):
    def test_cer_gate_includes_consumption_and_residential_allocation(self) -> None:
        self.assertEqual(len(CER_SCIENCEDB_MD5), 7)
        self.assertIn(
            "SME_and_Residential_allocations.csv", CER_SCIENCEDB_MD5
        )
        self.assertEqual(len(CER_OFFICIAL_MD5), 7)
        self.assertIn("SME and Residential allocations.tab", CER_OFFICIAL_MD5)

    def test_digest_matches_standard_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.bin"
            path.write_bytes(b"atk-evidence")
            self.assertEqual(digest(path, "sha256"), hashlib.sha256(b"atk-evidence").hexdigest())

    def test_verify_reports_verified_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.bin"
            path.write_bytes(b"evidence")
            expected = hashlib.md5(b"evidence").hexdigest()
            verified = verify("test", {path: ("md5", expected)})
            mismatched = verify("test", {path: ("md5", "0" * 32)})
            self.assertEqual(verified[0]["status"], "verified")
            self.assertEqual(mismatched[0]["status"], "mismatch")

    def test_verify_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing.bin"
            record = verify("test", {path: ("sha256", "0" * 64)})[0]
            self.assertEqual(record["status"], "missing")


if __name__ == "__main__":
    unittest.main()
