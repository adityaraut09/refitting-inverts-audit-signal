"""Directional stress test on the preference labels.

Flipping example i shifts the fitted gradient by a term proportional to
`s_i * z_i`. To push the fitted theta away from a target direction `v`, flip
the `budget` examples maximizing the signed alignment `s_i * <z_i, v>`: the
ones currently carrying the strongest evidence *for* `v`. Removing that
evidence, and replacing it with its opposite, has the largest effect on the
component of theta along `v`.

With no explicit `target_dir` the rule uses the top eigenvector of
`Sigma = Z^T Z / n`, sign-fixed by `canonical_sign`. That default carries no
semantic target: it reverses a mutually aligned block along the direction the
features vary in most. It is a stress test along one direction and is not
claimed to be worst case, to maximise damage, or to deny service. The opposite
tail of the same axis is a different attack and is reported separately.

Selection reads the observed labels `s`, so this rule is adaptive rather than
blind, and guarantees proved for independent or blind label noise do not
transfer to it.
"""
from __future__ import annotations

import numpy as np

from ..data.types import PreferenceDataset
from ..model.bt_head import BTHead


def canonical_sign(v: np.ndarray) -> np.ndarray:
    """Fix the sign of a direction so the attack is well defined.

    An eigenvector is only determined up to sign, and `select_flips` takes the
    rows with the *largest* signed alignment, so `v` and `-v` select opposite
    tails: on this data they share no rows at all. Whatever sign LAPACK happens
    to return is therefore not a specification.

    The rule: let j be the coordinate of largest absolute value, with numpy's
    smallest-index tie-break, and negate `v` if `v[j] < 0`. This is invariant to
    the solver's sign choice and deterministic given the input. It fixes one of
    the two tails, and makes no claim about which tail is preferable for an
    attacker; the opposite tail is measured separately.
    """
    v = np.asarray(v, dtype=float)
    j = int(np.argmax(np.abs(v)))
    return -v if v[j] < 0 else v.copy()


class GradientFlipAttack:
    name = "gradient_flip"

    def __init__(self, target_dir: np.ndarray | None = None):
        self.target_dir = target_dir

    def _validate_target_dir(self, v: np.ndarray, d: int) -> np.ndarray:
        """Check a caller-supplied direction before it reaches the selection.

        A silently malformed direction would produce a plausible-looking but
        meaningless reversal set, so these are hard errors rather than warnings.
        """
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError(
                f"target_dir must be one-dimensional, got shape {v.shape}")
        if v.shape[0] != d:
            raise ValueError(
                f"target_dir has length {v.shape[0]}, expected {d} to match ds.d")
        if not np.all(np.isfinite(v)):
            raise ValueError("target_dir must be finite; found nan or inf")
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("target_dir must have nonzero norm")
        return v / norm

    def _resolve_target_dir(self, ds: PreferenceDataset) -> np.ndarray:
        if self.target_dir is not None:
            # An explicitly supplied direction is used verbatim after validation.
            # Its sign is the caller's choice and carries meaning, so it is
            # never canonicalized.
            return self._validate_target_dir(self.target_dir, ds.d)
        cov = ds.Z.T @ ds.Z / ds.n
        eigvals, eigvecs = np.linalg.eigh(cov)
        return canonical_sign(eigvecs[:, -1])  # top eigenvector, sign fixed

    def select_flips(
        self,
        ds: PreferenceDataset,
        budget: int,
        rng: np.random.Generator,
        head: BTHead | None = None,
    ) -> np.ndarray:
        budget = min(budget, ds.n)
        v = self._resolve_target_dir(ds)
        alignment = ds.s * (ds.Z @ v)
        order = np.argsort(-alignment)  # most positively aligned with v first
        return order[:budget]
