#!/usr/bin/env python3
"""Verify Paper 3's CER bytes; optionally download official files with access."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = ROOT / "data/raw/cer-sciencedb"
OFFICIAL = "https://doi.org/10.7929/ISSDA/BX59EU"
METADATA_URL = "https://issda.ucd.ie/api/datasets/:persistentId/?persistentId=doi:10.7929/ISSDA/BX59EU"
FILES = {
    "File1.txt.zip": (806, 101978611, "00203f66f3f5e5201b20ed160b787684"),
    "File2.txt.zip": (805, 102197028, "5e3af1474d3c8976e2e1e0f8c1969507"),
    "File3.txt.zip": (804, 101624145, "b537785f8b37cb3e89103600d39da8ff"),
    "File4.txt.zip": (802, 102401577, "53ec9e70c1610b74ae72417cc010a0c3"),
    "File5.txt.zip": (803, 102257883, "6f8c7c9dfba3bbfbff0e5f1703e122fc"),
    "File6.txt.zip": (807, 147826765, "c0a435d0359974f23ce434b5e838e251"),
}
ALLOCATION_CSV = "SME_and_Residential_allocations.csv"
ALLOCATION_SHA256 = "96298be047f34ba91fe281c899b440d2b28747b4f102af6f239dbbd93dd354d4"
ALLOCATION_TAB = "SME and Residential allocations.tab"
ALLOCATION_TAB_MD5 = "124c10711ab1e7c52cb7317c8f69e42e"


def hashes(path: Path) -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            for value in digests.values():
                value.update(block)
    return {name: value.hexdigest() for name, value in digests.items()}


def read_allocation(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix == ".tab" else ",")
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        if not {"id", "code"} <= columns.keys():
            raise ValueError("Allocation requires ID and Code columns")
        result: dict[int, int] = {}
        for row in reader:
            meter, category = int(row[columns["id"]]), int(row[columns["code"]])
            if meter in result or category not in (1, 2, 3):
                raise ValueError("Duplicate meter or invalid allocation category")
            result[meter] = category
    return result


def verify(raw: Path, *, online: bool = False, check_crc: bool = False) -> dict:
    if online:
        with urllib.request.urlopen(METADATA_URL, timeout=30) as response:
            metadata = json.load(response)
        remote = {row["dataFile"]["filename"]: row["dataFile"]
                  for row in metadata["data"]["latestVersion"]["files"]}
        for name, (_, size, md5) in FILES.items():
            if remote[name]["filesize"] != size or remote[name]["checksum"]["value"] != md5:
                raise ValueError(f"Official metadata differs from recorded identity: {name}")
    records = []
    for name, (_, size, md5) in FILES.items():
        path = raw / name
        if not path.is_file():
            records.append({"filename": name, "status": "missing"})
            continue
        record = {"filename": name, "bytes": path.stat().st_size, **hashes(path)}
        record["status"] = "verified" if record["bytes"] == size and record["md5"] == md5 else "mismatch"
        if record["status"] == "verified":
            with zipfile.ZipFile(path) as archive:
                members = [x for x in archive.infolist() if not x.is_dir()]
                if len(members) != 1 or not members[0].filename.endswith(".txt"):
                    raise ValueError(f"Unexpected archive structure: {name}")
                record["member"] = members[0].filename
                record["uncompressed_bytes"] = members[0].file_size
                if check_crc and archive.testzip() is not None:
                    raise ValueError(f"Archive CRC failed: {name}")
        records.append(record)
    allocation = raw / ALLOCATION_TAB
    branch = "official-tab"
    if not allocation.is_file():
        allocation = raw / ALLOCATION_CSV
        branch = "verified-consumption-with-crosschecked-allocation-csv"
    allocation_record = {"filename": allocation.name, "status": "missing"}
    if allocation.is_file():
        actual = hashes(allocation)
        valid = (actual["md5"] == ALLOCATION_TAB_MD5 if branch == "official-tab"
                 else actual["sha256"] == ALLOCATION_SHA256)
        allocation_record.update(actual)
        allocation_record["status"] = "verified" if valid else "mismatch"
        if valid:
            mapping = read_allocation(allocation)
            allocation_record["rows"] = len(mapping)
            allocation_record["category_counts"] = {
                str(code): sum(value == code for value in mapping.values()) for code in (1, 2, 3)
            }
            if len(mapping) != 6445 or allocation_record["category_counts"]["1"] != 4225:
                raise ValueError("Verified allocation has unexpected cardinality")
    records.append(allocation_record)
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "official_record": OFFICIAL, "official_metadata_url": METADATA_URL,
        "online_metadata_checked": online, "zip_crc_checked": check_crc,
        "alternate_record": "https://doi.org/10.57760/sciencedb.17619",
        "source_branch": branch,
        "ready": all(x["status"] == "verified" for x in records), "files": records,
        "scope": "source byte identity; paper's exact 3000-customer subset is unknown",
    }


def download_official(raw: Path) -> None:
    token = os.environ.get("ISSDA_API_TOKEN")
    if not token:
        raise ValueError(f"Approved ISSDA access and ISSDA_API_TOKEN are required: {OFFICIAL}")
    raw.mkdir(parents=True, exist_ok=True)
    expected = {**FILES, ALLOCATION_TAB: (808, 196316, ALLOCATION_TAB_MD5)}
    for name, (file_id, size, md5) in expected.items():
        target = raw / name
        if target.exists():
            if target.stat().st_size != size or hashes(target)["md5"] != md5:
                raise ValueError(f"Refusing to overwrite mismatched source: {name}")
            continue
        partial = raw / (name + ".part")
        request = urllib.request.Request(
            f"https://issda.ucd.ie/api/access/datafile/{file_id}",
            headers={"X-Dataverse-key": token},
        )
        with partial.open("xb") as out, urllib.request.urlopen(request, timeout=60) as response:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                out.write(block)
        if partial.stat().st_size != size or hashes(partial)["md5"] != md5:
            raise ValueError(f"Download failed identity check; partial preserved: {name}")
        partial.rename(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--check-crc", action="store_true")
    parser.add_argument("--download-official", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download_official:
        download_official(args.raw)
    result = verify(args.raw, online=args.online, check_crc=args.check_crc)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as out:
            out.write(encoded)
    print(encoded, end="")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
