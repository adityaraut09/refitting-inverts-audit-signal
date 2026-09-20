"""Detector protocol. A detector scores every
example's suspicion of being poisoned, using only the (possibly poisoned)
dataset and the head fit on it — never the ground-truth `flip_mask`, which
is reserved for evaluation (`src/metrics/`)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class Detector(Protocol):
    name: str

    def score(self, ds: PreferenceDataset, head: BTHead) -> np.ndarray:
        """Per-example suspicion scores (n,), higher = more likely poisoned.
        Must not read `ds.flip_mask`."""
        ...
