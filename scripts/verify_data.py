#!/usr/bin/env python3
"""Verify every currently registered Study 1 raw-data artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SGCC = {
    REPO_ROOT / "data/raw/sgcc-verified/data.csv": (
        "sha256",
        "99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7",
    )
}
CER_ARCHIVE_MD5 = {
    "File1.txt.zip": "00203f66f3f5e5201b20ed160b787684",
    "File2.txt.zip": "5e3af1474d3c8976e2e1e0f8c1969507",
    "File3.txt.zip": "b537785f8b37cb3e89103600d39da8ff",
    "File4.txt.zip": "53ec9e70c1610b74ae72417cc010a0c3",
    "File5.txt.zip": "6f8c7c9dfba3bbfbff0e5f1703e122fc",
    "File6.txt.zip": "c0a435d0359974f23ce434b5e838e251",
}
CER_OFFICIAL_MD5 = {
    **CER_ARCHIVE_MD5,
    "SME and Residential allocations.tab": "124c10711ab1e7c52cb7317c8f69e42e",
}
CER_SCIENCEDB_MD5 = {
    **CER_ARCHIVE_MD5,
    "SME_and_Residential_allocations.csv": "89263f89253cf56b857079986ae73096",
}
CER_OFFICIAL = {
    REPO_ROOT / "data/raw/cer-authorized" / filename: ("md5", expected)
    for filename, expected in CER_OFFICIAL_MD5.items()
}
CER_SCIENCEDB = {
    REPO_ROOT / "data/raw/cer-sciencedb" / filename: ("md5", expected)
    for filename, expected in CER_SCIENCEDB_MD5.items()
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(group: str, artifacts: dict[Path, tuple[str, str]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path, (algorithm, expected) in artifacts.items():
        if not path.is_file():
            records.append({"group": group, "path": str(path), "status": "missing"})
            continue
        actual = digest(path, algorithm)
        records.append(
            {
                "group": group,
                "path": str(path),
                "algorithm": algorithm,
                "expected": expected,
                "actual": actual,
                "status": "verified" if actual == expected else "mismatch",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail if any registered file is missing")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    sgcc_records = verify("SGCC", SGCC)
    official_records = verify("CER official", CER_OFFICIAL)
    sciencedb_records = verify("CER ScienceDB", CER_SCIENCEDB)
    records = sgcc_records + official_records + sciencedb_records
    official_ready = all(
        record["status"] == "verified" for record in official_records
    )
    sciencedb_ready = all(
        record["status"] == "verified" for record in sciencedb_records
    )
    cer_ready = official_ready or sciencedb_ready
    if args.json:
        print(
            json.dumps(
                {
                    "records": records,
                    "branches": {
                        "official-tab-v1": official_ready,
                        "sciencedb-csv-semantic-equivalence-v1": sciencedb_ready,
                    },
                },
                indent=2,
            )
        )
    else:
        for record in records:
            print(f"{record['status']:8} {record['group']:8} {record['path']}")
        print(
            "\nCER/ISET ready branch: "
            + (
                "official-tab-v1"
                if official_ready
                else (
                    "sciencedb-csv-semantic-equivalence-v1"
                    if sciencedb_ready
                    else "none"
                )
            )
        )
        if not cer_ready:
            print(
                "\nCER/ISET requires the six exact archives and the declared "
                "allocation branch; follow the study DATA_SOURCES.md."
            )

    mismatches = any(record["status"] == "mismatch" for record in records)
    sgcc_ready = all(record["status"] == "verified" for record in sgcc_records)
    return 1 if mismatches or (args.strict and not (sgcc_ready and cer_ready)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
