"""The directional attack's default direction must not depend on an eigensolver's
arbitrary sign choice.

`GradientFlipAttack` selects the rows with the largest signed alignment
`s_i <z_i, v>`, so `v` and `-v` pick opposite tails of the same distribution. On
the real pool those two sets are disjoint. Before `canonical_sign` the default
direction was whatever `np.linalg.eigh` returned, which varied from split to
split, so the attack was not fully specified. These tests pin the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.attacks.gradient_flip import GradientFlipAttack, canonical_sign
from src.data.types import PreferenceDataset


def _ds(n=60, d=8, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    Z[:, 0] *= 4.0                      # a clearly dominant direction
    s = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return PreferenceDataset(Z=Z, s=s, flip_mask=np.zeros(n, dtype=bool), meta={})


# --------------------------- the canonical rule ----------------------------
def test_canonical_sign_is_invariant_to_input_sign_flip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(size=12)
        assert np.array_equal(canonical_sign(v), canonical_sign(-v))


def test_canonical_sign_makes_largest_coordinate_positive():
    rng = np.random.default_rng(2)
    for _ in range(20):
        v = rng.normal(size=12)
        c = canonical_sign(v)
        assert c[int(np.argmax(np.abs(c)))] > 0


def test_canonical_sign_preserves_the_axis_and_the_norm():
    v = np.array([0.3, -0.9, 0.1])
    c = canonical_sign(v)
    assert np.isclose(abs(float(c @ v)), float(v @ v))     # parallel or antiparallel
    assert np.isclose(np.linalg.norm(c), np.linalg.norm(v))


def test_canonical_sign_ties_go_to_the_smallest_index():
    # equal magnitudes: numpy's argmax picks index 0, whose sign then decides
    assert canonical_sign(np.array([-1.0, 1.0]))[0] > 0
    assert np.array_equal(canonical_sign(np.array([1.0, -1.0])),
                          np.array([1.0, -1.0]))


# ----------------------- the default attack direction ----------------------
def test_default_direction_is_canonically_signed():
    ds = _ds()
    v = GradientFlipAttack()._resolve_target_dir(ds)
    assert v[int(np.argmax(np.abs(v)))] > 0


def test_default_direction_survives_a_sign_flipped_eigensolver(monkeypatch):
    """Simulate a solver that returns the opposite sign and check nothing moves."""
    ds = _ds()
    baseline_v = GradientFlipAttack()._resolve_target_dir(ds).copy()
    baseline_sel = GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(0))

    real_eigh = np.linalg.eigh

    def flipped_eigh(a):
        w, u = real_eigh(a)
        return w, -u                       # every eigenvector sign inverted

    monkeypatch.setattr(np.linalg, "eigh", flipped_eigh)
    assert np.allclose(GradientFlipAttack()._resolve_target_dir(ds), baseline_v)
    assert np.array_equal(
        GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(0)),
        baseline_sel)


def test_selected_mask_is_deterministic_across_calls_and_rngs():
    ds = _ds()
    a = GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(0))
    b = GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(0))
    c = GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(999))
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)            # the rule reads no randomness


def test_the_two_orientations_really_do_select_different_rows():
    """Guards the premise: if +v and -v agreed, the sign would not matter."""
    ds = _ds()
    v = GradientFlipAttack()._resolve_target_dir(ds)
    plus = set(GradientFlipAttack(target_dir=v).select_flips(
        ds, 9, np.random.default_rng(0)).tolist())
    minus = set(GradientFlipAttack(target_dir=-v).select_flips(
        ds, 9, np.random.default_rng(0)).tolist())
    assert plus != minus
    assert not (plus & minus)              # disjoint tails


def test_an_explicit_target_dir_is_not_canonicalized():
    """The override must keep the caller's sign, or the orientation audit could
    not have reproduced the pre-fix behaviour."""
    ds = _ds()
    v = np.array([-1.0] + [0.0] * (ds.d - 1))
    got = GradientFlipAttack(target_dir=v)._resolve_target_dir(ds)
    assert got[0] == pytest.approx(-1.0)


def test_default_matches_an_explicit_canonical_direction():
    ds = _ds()
    v = GradientFlipAttack()._resolve_target_dir(ds)
    assert np.array_equal(
        GradientFlipAttack().select_flips(ds, 9, np.random.default_rng(0)),
        GradientFlipAttack(target_dir=v).select_flips(ds, 9, np.random.default_rng(0)))


# ----------------------- explicit target_dir validation --------------------
# A malformed direction would still produce a plausible-looking reversal set,
# so each of these must fail loudly rather than silently mis-select rows.
def test_zero_target_dir_is_rejected():
    ds = _ds()
    with pytest.raises(ValueError, match="nonzero norm"):
        GradientFlipAttack(target_dir=np.zeros(ds.d)).select_flips(
            ds, 9, np.random.default_rng(0))


def test_non_finite_target_dir_is_rejected():
    ds = _ds()
    for bad in (np.nan, np.inf, -np.inf):
        v = np.ones(ds.d)
        v[2] = bad
        with pytest.raises(ValueError, match="finite"):
            GradientFlipAttack(target_dir=v).select_flips(
                ds, 9, np.random.default_rng(0))


def test_wrong_length_target_dir_is_rejected():
    ds = _ds()
    for n in (ds.d - 1, ds.d + 1):
        with pytest.raises(ValueError, match="expected"):
            GradientFlipAttack(target_dir=np.ones(n)).select_flips(
                ds, 9, np.random.default_rng(0))


def test_non_one_dimensional_target_dir_is_rejected():
    ds = _ds()
    for shape in ((1, ds.d), (ds.d, 1), (2, 2, 2)):
        with pytest.raises(ValueError, match="one-dimensional"):
            GradientFlipAttack(target_dir=np.ones(shape)).select_flips(
                ds, 9, np.random.default_rng(0))


def test_a_valid_target_dir_is_normalised_and_still_accepted():
    ds = _ds()
    v = np.zeros(ds.d)
    v[1] = -7.5                                  # unnormalised, negative
    got = GradientFlipAttack(target_dir=v)._resolve_target_dir(ds)
    assert np.isclose(np.linalg.norm(got), 1.0)
    assert got[1] == pytest.approx(-1.0)         # sign preserved, not canonicalized
