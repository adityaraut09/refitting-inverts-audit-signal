"""Raw (pre-embedding) and embedded preference-pair data structures."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawPairs:
    """A batch of raw text preference pairs pulled from one source.

    Parallel lists indexed 0..n-1: example i is
    `(prompts[i], chosen[i], rejected[i])`, with the annotator having said
    chosen[i] > rejected[i]. Embedding (src/data/embed.py, implemented later)
    turns this into the difference-feature PreferenceDataset used by the
    rest of the pipeline.
    """

    prompts: list[str]
    chosen: list[str]
    rejected: list[str]
    meta: list[dict] = field(default_factory=list)
    source: str = "unknown"

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if not (len(self.chosen) == len(self.rejected) == n):
            raise ValueError("prompts/chosen/rejected must have equal length")
        if not self.meta:
            self.meta = [{} for _ in range(n)]
        elif len(self.meta) != n:
            raise ValueError("meta must have one entry per example (or be empty)")

    def __len__(self) -> int:
        return len(self.prompts)


@dataclass
class PreferenceDataset:
    """Embedded preference pairs: the object every downstream stage consumes.

    Z is fixed once (from the frozen embedder, see src/data/embed.py) and
    never changes across attack/detect/remediate trials. Only `s` (and
    `flip_mask`) change when an attack flips labels — see `flipped()`.
    """

    Z: np.ndarray          # (n, d) difference features in CLEAN orientation
    s: np.ndarray          # (n,)   labels in {+1,-1}; clean = +1
    flip_mask: np.ndarray  # (n,)   bool ground truth (eval only; never given to detectors)
    meta: dict             # source/embedder provenance + per-example meta

    def __post_init__(self) -> None:
        n = self.Z.shape[0]
        if self.s.shape != (n,):
            raise ValueError(f"s must have shape ({n},), got {self.s.shape}")
        if self.flip_mask.shape != (n,):
            raise ValueError(f"flip_mask must have shape ({n},), got {self.flip_mask.shape}")

    @property
    def n(self) -> int:
        return self.Z.shape[0]

    @property
    def d(self) -> int:
        return self.Z.shape[1]

    def flipped(self, indices) -> "PreferenceDataset":
        """Return a copy with `s` negated (and `flip_mask` set) at `indices`.

        `Z` is shared (not copied) with the original — it is label-independent
        (flipping a label doesn't change the Hessian) and never mutated.
        """
        idx = np.asarray(indices, dtype=int)
        s = self.s.copy()
        flip_mask = self.flip_mask.copy()
        s[idx] *= -1
        flip_mask[idx] = True
        return PreferenceDataset(Z=self.Z, s=s, flip_mask=flip_mask, meta=self.meta)

    def subset(self, indices) -> "PreferenceDataset":
        """Return a copy restricted to `indices` (e.g. a train/held-out split
        or a cross-fit fold). Unlike `flipped()`, this copies `Z`."""
        idx = np.asarray(indices, dtype=int)
        return PreferenceDataset(
            Z=self.Z[idx], s=self.s[idx], flip_mask=self.flip_mask[idx], meta=self.meta
        )
