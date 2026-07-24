#!/usr/bin/env python3
"""Download and verify one of Paper 1's two named datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path.insert(0, str(STUDY / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=("sgcc", "iset"),
        help="SGCC is public; ISET requires an approved ISSDA API token",
    )
    args = parser.parse_args()
    if args.dataset == "sgcc":
        return subprocess.run(
            ["bash", str(REPO / "scripts/acquire_sgcc.sh")],
            cwd=REPO,
            check=False,
        ).returncode

    from download_cer import main as download_iset

    return download_iset()


if __name__ == "__main__":
    raise SystemExit(main())
