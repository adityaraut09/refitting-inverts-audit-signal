"""Self-influence and leverage detectors.

Both use the fitted BTHead's closed-form curvature: `Hinv = H(theta_hat)^-1`,
and `z_i^T Hinv z_i`, the leverage of example i in that geometry.

`self_influence_i = sigma(-m_i)^2 * (z_i^T Hinv z_i)` — gradient-based: large
when the model *disagrees* with the observed label (m_i < 0, i.e. the fitted
reward ranks the "rejected" side above the observed "chosen" side) on a
high-leverage example.

`leverage_i = sigma'(m_i) * (z_i^T Hinv z_i)` — curvature-only, Cook's-
distance style: high for high-leverage examples regardless of agreement.
`sigma'` is even in `m_i`, so this variant is label-independent (flipping a
label doesn't change the Hessian) — it flags "structurally influential"
examples whether or not they're actually poisoned, unlike `self_influence`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


def _leverage(ds: PreferenceDataset, head: BTHead) -> np.ndarray:
    Hinv = head.hinv(ds)
    return np.einsum("ij,jk,ik->i", ds.Z, Hinv, ds.Z)


class SelfInfluenceDetector:
    name = "self_influence"

    def score(self, ds: PreferenceDataset, head: BTHead) -> np.ndarray:
        m = head.margins(ds)
        return (expit(-m) ** 2) * _leverage(ds, head)


class LeverageDetector:
    name = "leverage"

    def score(self, ds: PreferenceDataset, head: BTHead) -> np.ndarray:
        m = head.margins(ds)
        w = expit(m) * expit(-m)  # sigma'(m_i)
        return w * _leverage(ds, head)
