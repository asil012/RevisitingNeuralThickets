#!/usr/bin/env python3
"""Regenerate the six active paper figures from bundled data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_thickets.data import Bundle  # noqa: E402
from neural_thickets.figures import write_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced/figures")
    args = parser.parse_args()
    bundle = Bundle.default()
    for path in write_figures(bundle, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
