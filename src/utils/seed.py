"""Every stochastic call in this repo takes an explicit np.random.Generator,
obtained from here, so a trial is fully reproducible from one seed."""
from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
