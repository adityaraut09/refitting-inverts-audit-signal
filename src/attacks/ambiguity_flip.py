"""Ambiguity-targeted flip attack — flips low-margin (alpha=1) or
high-margin (alpha=0) pairs, controlled by alpha. Intermediate alpha
values slide a fixed-size window across the margin-sorted order."""
from __future__ import annotations

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class AmbiguityFlipAttack:
    name = "ambiguity_flip"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def select_flips(
        self,
        ds: PreferenceDataset,
        budget: int,
        rng: np.random.Generator,
        head: BTHead | None = None,
    ) -> np.ndarray:
        budget = min(budget, ds.n)
        if head is None:
            head = BTHead().fit(ds)
        margins = np.abs(head.margins(ds))
        order = np.argsort(margins)
        n = ds.n
        max_start = n - budget
        start = round((1.0 - self.alpha) * max_start)
        return order[start:start + budget]