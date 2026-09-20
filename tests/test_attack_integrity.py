"""Row-identity invariants for the attack pipeline.

These pin the properties the forensic pass on gradient_flip relied on:
an attack must flip exactly its budget, flip exactly the rows it selected,
leave every other row untouched, and reverse the effective feature
`s_i * z_i` exactly once. Also checks that `subset()` preserves row order,
since attack indices are local to whatever subset they were chosen on.
"""
import numpy as np
import pytest

from src.attacks.ambiguity_flip import AmbiguityFlipAttack
from src.attacks.gradient_flip import GradientFlipAttack
from src.attacks.random_flip import RandomFlipAttack
from src.data.embed import make_synthetic_dataset

ATTACKS = [RandomFlipAttack(), GradientFlipAttack(), AmbiguityFlipAttack()]
IDS = [a.name for a in ATTACKS]


def _dataset(n=120, d=8, seed=0):
    return make_synthetic_dataset(n=n, d=d, separation=1.5, seed=seed)


@pytest.mark.parametrize("attack", ATTACKS, ids=IDS)
def test_selection_is_unique_and_budget_sized(attack):
    ds = _dataset()
    budget = int(0.15 * ds.n)
    sel = attack.select_flips(ds, budget, np.random.default_rng(0))

    assert len(sel) == budget
    assert len(np.unique(sel)) == budget
    assert sel.min() >= 0 and sel.max() < ds.n


@pytest.mark.parametrize("attack", ATTACKS, ids=IDS)
def test_flip_mask_marks_exactly_the_selected_rows(attack):
    ds = _dataset()
    budget = int(0.15 * ds.n)
    sel = attack.select_flips(ds, budget, np.random.default_rng(0))
    poisoned = ds.flipped(sel)

    assert int(poisoned.flip_mask.sum()) == budget
    assert np.array_equal(np.flatnonzero(poisoned.flip_mask), np.sort(sel))
    assert poisoned.n == ds.n  # no row lost or duplicated


@pytest.mark.parametrize("attack", ATTACKS, ids=IDS)
def test_no_unintended_row_is_changed(attack):
    ds = _dataset()
    budget = int(0.15 * ds.n)
    sel = attack.select_flips(ds, budget, np.random.default_rng(0))
    poisoned = ds.flipped(sel)

    changed = np.flatnonzero(poisoned.s != ds.s)
    assert np.array_equal(changed, np.sort(sel))
    assert np.array_equal(poisoned.Z, ds.Z)  # attacks relabel, they do not edit features


@pytest.mark.parametrize("attack", ATTACKS, ids=IDS)
def test_effective_feature_reverses_exactly_once(attack):
    """s_i * z_i must negate on selected rows and be untouched elsewhere.
    Negating both s and Z would cancel; negating neither is a no-op."""
    ds = _dataset()
    budget = int(0.15 * ds.n)
    sel = attack.select_flips(ds, budget, np.random.default_rng(0))
    poisoned = ds.flipped(sel)

    sel_mask = np.zeros(ds.n, dtype=bool)
    sel_mask[sel] = True
    eff_clean = ds.s[:, None] * ds.Z
    eff_att = poisoned.s[:, None] * poisoned.Z

    assert np.abs(eff_att[sel_mask] + eff_clean[sel_mask]).max() == 0.0
    assert np.abs(eff_att[~sel_mask] - eff_clean[~sel_mask]).max() == 0.0


def test_subset_preserves_row_order():
    """Attack indices are local to the subset they were chosen on, so subset()
    must be a plain order-preserving gather. Row identity is tracked with an
    external parallel array, never by writing an id into Z or s, so nothing
    here can perturb a norm, a Hessian or a detector."""
    ds = _dataset(n=60, d=4)
    source_ids = np.arange(ds.n)
    take = np.random.default_rng(0).permutation(ds.n)[:25]

    kept = ds.subset(take)
    kept_ids = source_ids[take]  # parallel index op only

    assert kept.n == len(kept_ids)
    assert np.array_equal(kept.Z, ds.Z[take])
    assert np.array_equal(kept.s, ds.s[take])
    assert np.array_equal(kept.flip_mask, ds.flip_mask[take])
    # every retained row sits at the position its external id says it does
    for local, sid in enumerate(kept_ids):
        assert np.array_equal(kept.Z[local], ds.Z[sid])


def test_zero_norm_filter_preserves_row_order():
    ds = _dataset(n=60, d=4)
    ds.Z[[3, 17, 41]] = 0.0  # plant zero-norm rows like the real HH-RLHF duplicates
    source_ids = np.arange(ds.n)

    valid = np.where(np.linalg.norm(ds.Z, axis=1) > 1e-8)[0]
    kept = ds.subset(valid)
    kept_ids = source_ids[valid]

    assert kept.n == ds.n - 3
    assert not np.isin([3, 17, 41], kept_ids).any()
    assert np.array_equal(kept_ids, np.sort(kept_ids))  # order preserved
    for local, sid in enumerate(kept_ids):
        assert np.array_equal(kept.Z[local], ds.Z[sid])


def test_external_id_tracking_survives_split_filter_and_attack():
    """The full pipeline shape used on real data: split, zero-norm filter,
    attack, flip_mask. Row identity is carried only in an external array."""
    ds = _dataset(n=200, d=6)
    ds.Z[[5, 40, 77, 120]] = 0.0
    source_ids = np.arange(ds.n)

    idx = np.random.default_rng(0).permutation(ds.n)
    n_ho = int(ds.n * 0.3)
    tr = ds.subset(idx[n_ho:])
    tr_ids = source_ids[idx[n_ho:]]

    valid = np.where(np.linalg.norm(tr.Z, axis=1) > 1e-8)[0]
    trc = tr.subset(valid)
    clean_ids = tr_ids[valid]

    budget = int(0.15 * trc.n)
    sel = GradientFlipAttack().select_flips(trc, budget, np.random.default_rng(0))
    poisoned = trc.flipped(sel)
    sel_ids = clean_ids[sel]

    assert len(clean_ids) == trc.n
    assert len(np.unique(clean_ids)) == trc.n
    assert len(np.unique(sel_ids)) == budget
    assert np.array_equal(np.flatnonzero(poisoned.flip_mask), np.sort(sel))
    assert np.array_equal(np.sort(clean_ids[poisoned.flip_mask]), np.sort(sel_ids))
    # no planted zero-norm row survived, by external id
    assert not np.isin([5, 40, 77, 120], clean_ids).any()
