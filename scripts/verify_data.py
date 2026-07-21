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
CER_MD5 = {
    "File1.txt.zip": "00203f66f3f5e5201b20ed160b787684",
    "File2.txt.zip": "5e3af1474d3c8976e2e1e0f8c1969507",
    "File3.txt.zip": "b537785f8b37cb3e89103600d39da8ff",
    "File4.txt.zip": "53ec9e70c1610b74ae72417cc010a0c3",
    "File5.txt.zip": "6f8c7c9dfba3bbfbff0e5f1703e122fc",
    "File6.txt.zip": "c0a435d0359974f23ce434b5e838e251",
}
CER = {
    REPO_ROOT / "data/raw/cer-authorized" / filename: ("md5", expected)
    for filename, expected in CER_MD5.items()
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

    records = verify("SGCC", SGCC) + verify("CER/ISET", CER)
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            print(f"{record['status']:8} {record['group']:8} {record['path']}")
        if any(record["group"] == "CER/ISET" and record["status"] == "missing" for record in records):
            print("\nCER/ISET requires approved ISSDA access; follow docs/GETTING_STARTED.md.")

    mismatches = any(record["status"] == "mismatch" for record in records)
    missing = any(record["status"] == "missing" for record in records)
    return 1 if mismatches or (args.strict and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())

