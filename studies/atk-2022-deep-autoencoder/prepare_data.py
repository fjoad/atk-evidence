#!/usr/bin/env python3
"""Verify raw data, generate the paper's attacks, and freeze the final splits."""

from __future__ import annotations

import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "src"))

from prepare_paper_data import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
