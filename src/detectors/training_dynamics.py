"""Training-dynamics / AUM (cheap version) — uses the final fitted margin
as a proxy for average margin over training, per Chapter 4's cheap-version
note. Low margin is suspicious."""
from __future__ import annotations

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


class TrainingDynamicsDetector:
    name = "training_dynamics"

    def score(self, ds: PreferenceDataset, head: BTHead | None = None) -> np.ndarray:
        if head is None:
            head = BTHead().fit(ds)
        return -head.margins(ds)