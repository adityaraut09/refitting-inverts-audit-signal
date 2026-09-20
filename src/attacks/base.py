"""Attack protocol. Every attack picks which
labels to flip; only `PreferenceDataset.flipped(indices)` (never the attack
itself) actually mutates labels."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class Attack(Protocol):
    name: str

    def select_flips(
        self,
        ds: PreferenceDataset,
        budget: int,
        rng: np.random.Generator,
        head: BTHead | None = None,
    ) -> np.ndarray:
        """Return an int array of indices to flip, length <= budget.
        Deterministic given `rng`. Must not read `ds.flip_mask`."""
        ...
