"""Download authorized CER files from the official ISSDA Dataverse.

Set ISSDA_API_TOKEN to a token belonging to an account whose access request has
been approved. The token is read from the environment and is never printed.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


FILES = {
    806: ("File1.txt.zip", "00203f66f3f5e5201b20ed160b787684"),
    805: ("File2.txt.zip", "5e3af1474d3c8976e2e1e0f8c1969507"),
    804: ("File3.txt.zip", "b537785f8b37cb3e89103600d39da8ff"),
    802: ("File4.txt.zip", "53ec9e70c1610b74ae72417cc010a0c3"),
    803: ("File5.txt.zip", "6f8c7c9dfba3bbfbff0e5f1703e122fc"),
    807: ("File6.txt.zip", "c0a435d0359974f23ce434b5e838e251"),
    808: (
        "SME and Residential allocations.tab",
        "124c10711ab1e7c52cb7317c8f69e42e",
    ),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    token = os.environ.get("ISSDA_API_TOKEN")
    if not token:
        print(
            "ISSDA_API_TOKEN is not set. Request access at "
            "https://doi.org/10.7929/ISSDA/BX59EU and create a Dataverse API token.",
            file=sys.stderr,
        )
        return 2

    repository_root = Path(__file__).resolve().parents[3]
    destination = repository_root / "data" / "raw" / "cer-authorized"
    destination.mkdir(parents=True, exist_ok=True)
    for file_id, (filename, expected_md5) in FILES.items():
        target = destination / filename
        if target.exists() and md5(target) == expected_md5:
            print(f"verified existing {filename}")
            continue
        if target.exists():
            print(
                f"checksum mismatch for existing {filename}; refusing to overwrite it",
                file=sys.stderr,
            )
            return 1
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            f"https://issda.ucd.ie/api/access/datafile/{file_id}",
            headers={"X-Dataverse-key": token},
        )
        try:
            with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            print(f"failed {filename}: HTTP {exc.code}", file=sys.stderr)
            if exc.code in (401, 403):
                print(
                    "Confirm that the ISSDA access request is approved and the API token is current.",
                    file=sys.stderr,
                )
            return 1
        actual_md5 = md5(partial)
        if actual_md5 != expected_md5:
            partial.unlink(missing_ok=True)
            print(
                f"checksum mismatch for {filename}: {actual_md5} != {expected_md5}",
                file=sys.stderr,
            )
            return 1
        partial.replace(target)
        print(f"downloaded and verified {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
