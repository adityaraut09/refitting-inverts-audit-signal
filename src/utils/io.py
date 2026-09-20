"""Disk cache helpers.

Embedding caches live under cache/, which is gitignored: they are large and
are regenerated deterministically from the dataset and the frozen encoder.
CSV results live under results/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache"


def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.npz"


def load_npz_cache(key: str) -> dict[str, np.ndarray] | None:
    """Return the cached arrays for `key`, or None if not cached."""
    path = cache_path(key)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def save_npz_cache(key: str, **arrays: np.ndarray) -> None:
    np.savez_compressed(cache_path(key), **arrays)


def append_csv_row(path: Path, row: dict) -> None:
    """Append one result row to a CSV, writing the header on first write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
