#!/usr/bin/env python3
"""Acquire or verify the two datasets named by Paper 1.

ISET/CER is access controlled by ISSDA. After access is approved, set
``ISSDA_API_TOKEN`` and run ``--download-official``. The token is never
printed or stored. The already available ScienceDB branch is accepted only
when all six consumption archives match the official ISSDA MD5 values and its
allocation CSV matches the separately frozen semantic checksum.

Raw files remain under ``data/raw`` and are ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
CER_OFFICIAL_DIR = REPO / "data/raw/cer-authorized"
CER_SCIENCEDB_DIR = REPO / "data/raw/cer-sciencedb"
SGCC_PATH = REPO / "data/raw/sgcc-verified/data.csv"

CER_FILES = {
    "File1.txt.zip": (806, 101_978_611, "00203f66f3f5e5201b20ed160b787684"),
    "File2.txt.zip": (805, 102_197_028, "5e3af1474d3c8976e2e1e0f8c1969507"),
    "File3.txt.zip": (804, 101_624_145, "b537785f8b37cb3e89103600d39da8ff"),
    "File4.txt.zip": (802, 102_401_577, "53ec9e70c1610b74ae72417cc010a0c3"),
    "File5.txt.zip": (803, 102_257_883, "6f8c7c9dfba3bbfbff0e5f1703e122fc"),
    "File6.txt.zip": (807, 147_826_765, "c0a435d0359974f23ce434b5e838e251"),
}
OFFICIAL_ALLOCATION = (
    "SME and Residential allocations.tab",
    808,
    196_316,
    "124c10711ab1e7c52cb7317c8f69e42e",
)
SCIENCEDB_ALLOCATION = (
    "SME_and_Residential_allocations.csv",
    112_589,
    "89263f89253cf56b857079986ae73096",
    "96298be047f34ba91fe281c899b440d2b28747b4f102af6f239dbbd93dd354d4",
)
SGCC_SHA256 = "99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_file(
    path: Path,
    *,
    algorithm: str,
    expected: str,
    expected_bytes: int | None = None,
) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    actual = digest(path, algorithm)
    size = path.stat().st_size
    status = "verified"
    if actual != expected or (expected_bytes is not None and size != expected_bytes):
        status = "mismatch"
    return {
        "path": str(path),
        "bytes": size,
        "expected_bytes": expected_bytes,
        "algorithm": algorithm,
        "expected": expected,
        "actual": actual,
        "status": status,
    }


def verify_zip(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                item for item in archive.infolist()
                if not item.is_dir() and not Path(item.filename).name.startswith(".")
            ]
            corrupt = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        return {"status": "invalid", "error": str(exc)}
    if len(members) != 1 or corrupt is not None:
        return {
            "status": "invalid",
            "members": [item.filename for item in members],
            "corrupt_member": corrupt,
        }
    return {
        "status": "verified",
        "member": members[0].filename,
        "uncompressed_bytes": members[0].file_size,
    }


def verify_cer_directory(
    directory: Path,
    *,
    allocation: tuple[str, int, str] | tuple[str, int, str, str],
    branch: str,
) -> dict[str, object]:
    records: dict[str, object] = {}
    for filename, (_, size, md5) in CER_FILES.items():
        path = directory / filename
        record = verify_file(
            path, algorithm="md5", expected=md5, expected_bytes=size
        )
        if record["status"] == "verified":
            record["zip"] = verify_zip(path)
            if record["zip"]["status"] != "verified":
                record["status"] = "invalid"
        records[filename] = record

    allocation_name, allocation_size, allocation_md5 = allocation[:3]
    allocation_path = directory / allocation_name
    allocation_record = verify_file(
        allocation_path,
        algorithm="md5",
        expected=allocation_md5,
        expected_bytes=allocation_size,
    )
    if len(allocation) == 4 and allocation_record["status"] == "verified":
        allocation_record["sha256"] = digest(allocation_path, "sha256")
        if allocation_record["sha256"] != allocation[3]:
            allocation_record["status"] = "mismatch"
    records[allocation_name] = allocation_record
    ready = all(record["status"] == "verified" for record in records.values())
    return {"branch": branch, "ready": ready, "files": records}


def verify_cer() -> dict[str, object]:
    official = verify_cer_directory(
        CER_OFFICIAL_DIR,
        allocation=(
            OFFICIAL_ALLOCATION[0],
            OFFICIAL_ALLOCATION[2],
            OFFICIAL_ALLOCATION[3],
        ),
        branch="official-tab-v1",
    )
    sciencedb = verify_cer_directory(
        CER_SCIENCEDB_DIR,
        allocation=SCIENCEDB_ALLOCATION,
        branch="sciencedb-csv-semantic-equivalence-v1",
    )
    selected = (
        official["branch"] if official["ready"]
        else sciencedb["branch"] if sciencedb["ready"]
        else None
    )
    return {
        "dataset": "ISET/CER",
        "official_record": "https://doi.org/10.7929/ISSDA/BX59EU",
        "alternate_record": "https://doi.org/10.57760/sciencedb.17619",
        "ready": selected is not None,
        "selected_branch": selected,
        "branches": [official, sciencedb],
    }


def verify_sgcc() -> dict[str, object]:
    record = verify_file(
        SGCC_PATH, algorithm="sha256", expected=SGCC_SHA256
    )
    return {
        "dataset": "SGCC",
        "source": "https://github.com/henryRDlab/ElectricityTheftDetection/",
        "ready": record["status"] == "verified",
        "file": record,
    }


def download_official_cer() -> None:
    token = os.environ.get("ISSDA_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ISSDA_API_TOKEN is not set. Request access through "
            "https://doi.org/10.7929/ISSDA/BX59EU first."
        )
    CER_OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    downloads = {
        **{
            name: (file_id, size, md5)
            for name, (file_id, size, md5) in CER_FILES.items()
        },
        OFFICIAL_ALLOCATION[0]: (
            OFFICIAL_ALLOCATION[1],
            OFFICIAL_ALLOCATION[2],
            OFFICIAL_ALLOCATION[3],
        ),
    }
    for filename, (file_id, expected_bytes, expected_md5) in downloads.items():
        target = CER_OFFICIAL_DIR / filename
        current = verify_file(
            target,
            algorithm="md5",
            expected=expected_md5,
            expected_bytes=expected_bytes,
        )
        if current["status"] == "verified":
            print(f"verified existing {filename}")
            continue
        if target.exists():
            raise RuntimeError(f"refusing to overwrite mismatched {target}")
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            f"https://issda.ucd.ie/api/access/datafile/{file_id}",
            headers={"X-Dataverse-key": token},
        )
        try:
            with urllib.request.urlopen(request) as response, partial.open("wb") as out:
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    out.write(chunk)
        except (OSError, urllib.error.HTTPError) as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"failed to download {filename}: {exc}") from exc
        verified = verify_file(
            partial,
            algorithm="md5",
            expected=expected_md5,
            expected_bytes=expected_bytes,
        )
        if verified["status"] != "verified":
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded {filename} failed its checksum gate")
        partial.replace(target)
        print(f"downloaded and verified {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("iset", "sgcc", "all"), nargs="?", default="all")
    parser.add_argument("--download-official", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.download_official:
            if args.dataset not in {"iset", "all"}:
                parser.error("--download-official applies only to ISET/CER")
            download_official_cer()
        results = []
        if args.dataset in {"iset", "all"}:
            results.append(verify_cer())
        if args.dataset in {"sgcc", "all"}:
            results.append(verify_sgcc())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {"results": results, "ready": all(item["ready"] for item in results)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in results:
            branch = item.get("selected_branch", "")
            print(f"{item['dataset']}: {'READY' if item['ready'] else 'BLOCKED'} {branch}")
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
