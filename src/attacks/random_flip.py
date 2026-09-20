"""Uniform random label flips — the control attack. Any real attack should
beat this at the same budget; if it doesn't, it isn't
exploiting anything about the data."""
from __future__ import annotations

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class RandomFlipAttack:
    name = "random_flip"

    def select_flips(
        self,
        ds: PreferenceDataset,
        budget: int,
        rng: np.random.Generator,
        head: BTHead | None = None,
    ) -> np.ndarray:
        budget = min(budget, ds.n)
        return rng.choice(ds.n, size=budget, replace=False)
