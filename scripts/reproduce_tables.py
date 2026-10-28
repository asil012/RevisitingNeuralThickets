#!/usr/bin/env python3
"""Regenerate the 15 active paper tables as portable CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_thickets.analysis.tables import write_tables  # noqa: E402
from neural_thickets.data import Bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced/tables")
    args = parser.parse_args()
    bundle = Bundle.default()
    for path in write_tables(bundle, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
