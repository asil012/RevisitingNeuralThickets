#!/usr/bin/env python3
"""Regenerate all active paper figures and tables on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_thickets.analysis.tables import write_tables  # noqa: E402
from neural_thickets.data import Bundle  # noqa: E402
from neural_thickets.figures import write_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    bundle = Bundle.default()
    figures = write_figures(bundle, args.output_dir / "figures")
    tables = write_tables(bundle, args.output_dir / "tables")
    print(f"Regenerated {len(figures)} figures and {len(tables)} tables in {args.output_dir}")


if __name__ == "__main__":
    main()
