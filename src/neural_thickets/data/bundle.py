"""Load the compact data used by the paper figures and tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class Bundle:
    """Access to the score-only paper data."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    @classmethod
    def default(cls) -> Bundle:
        return cls(Path(__file__).resolve().parents[3] / "data")

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        assert path.is_relative_to(self.root), name
        return path

    def json(self, name: str) -> dict:
        with self.path(name).open() as stream:
            return json.load(stream)

    def csv(self, name: str) -> list[dict[str, str]]:
        with self.path(name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.path(name), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
