"""Cross-fit agreement — trains a model on all folds except one, then
checks whether the held-out model agrees with each example's observed
label. Confident disagreement from a model that never saw the example
is suspicious. The Confident Learning idea behind cleanlab."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class CrossFitDetector:
    name = "cross_fit"

    def __init__(self, n_folds: int = 5, lam: float = 1e-2):
        self.n_folds = n_folds
        self.lam = lam

    def score(self, ds: PreferenceDataset, head: BTHead | None = None) -> np.ndarray:
        n = ds.n
        fold_ids = np.arange(n) % self.n_folds
        scores = np.zeros(n)
        for fold in range(self.n_folds):
            test_idx = np.where(fold_ids == fold)[0]
            train_idx = np.where(fold_ids != fold)[0]
            fold_head = BTHead(lam=self.lam).fit(ds.subset(train_idx))
            test_ds = ds.subset(test_idx)
            margins = test_ds.s * (test_ds.Z @ fold_head.theta)
            scores[test_idx] = 1 - expit(margins)
        return scores