"""Ensemble detector — averages the rank of each detector, per Chapter 4:
robust to different detectors' scores living on different scales."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class EnsembleDetector:
    name = "ensemble"

    def __init__(self, detectors: list):
        self.detectors = detectors

    def score(self, ds: PreferenceDataset, head: BTHead | None = None) -> np.ndarray:
        ranks = []
        for det in self.detectors:
            s = det.score(ds, head)
            ranks.append(rankdata(s))
        return np.mean(ranks, axis=0)