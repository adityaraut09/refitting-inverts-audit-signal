"""Remove-and-refit remediation: drop the top-k
highest-scoring (most suspicious) examples and refit a fresh BTHead on
what's left. This is "using the detector" — the counterfactual is leaving
the poisoned head as-is."""
from __future__ import annotations

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


def remove_refit(ds: PreferenceDataset, scores: np.ndarray, k: int, lam: float) -> BTHead:
    k = min(k, ds.n)
    flagged = np.argsort(-np.asarray(scores, dtype=float))[:k]
    keep = np.setdiff1d(np.arange(ds.n), flagged, assume_unique=False)
    return BTHead(lam=lam).fit(ds.subset(keep))
