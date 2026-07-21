from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StudyRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        with (REPO_ROOT / "studies/registry.toml").open("rb") as handle:
            self.registry = tomllib.load(handle)

    def test_study_ids_and_sequences_are_unique(self) -> None:
        studies = self.registry["studies"]
        self.assertEqual(len({study["id"] for study in studies}), len(studies))
        self.assertEqual(len({study["sequence"] for study in studies}), len(studies))

    def test_registered_paths_exist(self) -> None:
        for study in self.registry["studies"]:
            self.assertTrue((REPO_ROOT / study["path"]).is_dir())
            self.assertTrue((REPO_ROOT / study["report_path"]).is_dir())

    def test_exactly_one_study_is_primary(self) -> None:
        self.assertEqual(sum(bool(study["primary"]) for study in self.registry["studies"]), 1)


if __name__ == "__main__":
    unittest.main()

