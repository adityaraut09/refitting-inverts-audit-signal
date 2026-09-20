"""Pooled-replication tests: independence, leakage, orientation, ties, determinism.

These guard the properties the pooled replication claims. They are the reason a
reader can believe the pools really are disjoint and that no test row influenced
a hyperparameter.

Most of these read the HH-RLHF corpus and embed it; see DATA.md. Run
``pytest tests/ --ignore=tests/test_pooled.py`` for the data-free subset.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.types import PreferenceDataset
from src.pooled import core as sc
from src.pooled import data as sd
from src.pooled.core import margin_matched_honest, select_lam_on_dev

FULL_SPEC = [("n3000", 3000, 5), ("n1000", 1000, 5), ("n300", 300, 5)]


# ---------------------------------------------------------------- independence
def test_pools_are_pairwise_disjoint():
    pools = sd.allocate_pools(43835, FULL_SPEC)
    assert len(pools) == 15
    sd.verify_disjoint(pools)          # raises on any overlap
    allidx = np.concatenate([pools[k] for k in sorted(pools)])
    assert len(allidx) == len(np.unique(allidx)) == 21500


def test_pool_sizes_match_the_preregistered_spec():
    pools = sd.allocate_pools(43835, FULL_SPEC)
    for scale, n, k in FULL_SPEC:
        for i in range(k):
            assert len(pools[f"{scale}_pool{i}"]) == n


def test_pool_allocation_is_deterministic():
    a = sd.allocate_pools(43835, FULL_SPEC)
    b = sd.allocate_pools(43835, FULL_SPEC)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_allocation_refuses_to_oversubscribe():
    with pytest.raises(ValueError):
        sd.allocate_pools(1000, [("big", 600, 5)])


def test_train_and_test_come_from_different_files():
    tr = sd.load_helpful_base("train")
    te = sd.load_helpful_base("test")
    assert tr.file_sha256 != te.file_sha256
    assert len(tr) == 43835 and len(te) == 2354
    # a pool index can never address a test row: they index different corpora
    assert tr.source.endswith("/train") and te.source.endswith("/test")


# ------------------------------------------------------- partitions and basis
def test_train_dev_disjoint_and_exhaustive():
    p = sc.prepare("n300", 0, "resp", 64)
    assert p.n_train + p.n_dev == 300          # resp has no zero-norm rows here
    assert p.n_train > 0 and p.n_dev > 0


def test_basis_is_built_from_training_rows_only():
    """The projection must be recomputable from the training block alone."""
    p = sc.prepare("n1000", 1, "resp", 32)
    raw = sd.load_helpful_base("train")
    pools = sd.allocate_pools(len(raw), FULL_SPEC)
    Z = sd.embed_block(raw, pools["n1000_pool1"], "resp")
    keep = sd.nonzero_mask(Z)
    rng = np.random.default_rng(0)
    perm = rng.permutation(int(keep.sum()))
    n_dev = int(round(sc.DEV_FRAC * int(keep.sum())))
    tr_i = perm[n_dev:]
    V = sc.train_basis(Z[keep][tr_i], 32)
    assert np.allclose(np.abs(p.basis), np.abs(V))


def test_projection_at_full_dim_is_a_rotation():
    p = sc.prepare("n300", 2, "resp", 384)
    assert np.abs(p.basis.T @ p.basis - np.eye(384)).max() < 1e-8
    s = np.ones(p.n_train)
    h_proj = sc.fit(p.Ztr, s, 1e-3)
    h_raw = sc.fit(p.Ztr @ p.basis.T, s, 1e-3)
    assert sc.data_logloss(p.Ztr, s, h_proj.theta) == pytest.approx(
        sc.data_logloss(p.Ztr @ p.basis.T, s, h_raw.theta), abs=1e-6)


def test_zero_norm_filter_uses_the_original_space():
    p = sc.prepare("n300", 3, "cat", 32)
    # cat has zero-difference rows; they must be gone before projection
    assert p.n_zero_train > 0
    assert p.n_train + p.n_dev == 300 - p.n_zero_train


# ------------------------------------------------------------- no test leakage
def test_lambda_selection_ignores_the_test_block():
    """Corrupting the test features must not change the selected lambda."""
    p = sc.prepare("n1000", 0, "resp", 64)
    lam_a = select_lam_on_dev(p)
    corrupted = sc.Prepared(
        Ztr=p.Ztr, Zdev=p.Zdev, Zte=p.Zte * 0.0 + 999.0, dim=p.dim,
        n_train=p.n_train, n_dev=p.n_dev, n_test=p.n_test,
        n_zero_train=p.n_zero_train, n_zero_test=p.n_zero_test, basis=p.basis)
    assert select_lam_on_dev(corrupted) == lam_a


def test_lambda_selection_responds_to_dev():
    """Sanity: the selector is not a constant function."""
    p = sc.prepare("n1000", 0, "resp", 64)
    lam_a = select_lam_on_dev(p)
    shuffled = sc.Prepared(
        Ztr=p.Ztr, Zdev=np.roll(p.Zdev, 1, axis=1), Zte=p.Zte, dim=p.dim,
        n_train=p.n_train, n_dev=p.n_dev, n_test=p.n_test,
        n_zero_train=p.n_zero_train, n_zero_test=p.n_zero_test, basis=p.basis)
    assert select_lam_on_dev(shuffled) in sc.LAMBDA_GRID
    assert lam_a in sc.LAMBDA_GRID


def test_data_logloss_excludes_the_ridge_penalty():
    """BTHead.loss adds 0.5*lam*||theta||^2; a log-2 comparison must not."""
    p = sc.prepare("n300", 0, "resp", 64)
    s = np.ones(p.n_train)
    lam = 1e-1
    h = sc.fit(p.Ztr, s, lam)
    ds = PreferenceDataset(Z=p.Ztr, s=s, flip_mask=np.zeros(p.n_train, bool), meta={})
    penalty = 0.5 * lam * float(h.theta @ h.theta)
    assert h.loss(ds) == pytest.approx(sc.data_logloss(p.Ztr, s, h.theta) + penalty, abs=1e-9)
    assert penalty > 1e-6           # the two really do differ here


# ---------------------------------------------------------------- orientation
def test_score_orientation_higher_is_more_suspicious():
    """A reversed label must raise the final-margin score for that row."""
    p = sc.prepare("n300", 0, "resp", 64)
    ds = sc.make_ds(p.Ztr)
    h = sc.fit(p.Ztr, ds.s, 1e-3)
    base = sc.score_final_margin(ds, h)
    flipped = sc.score_final_margin(ds.flipped([0, 1, 2]), h)
    for i in (0, 1, 2):
        assert flipped[i] == pytest.approx(-base[i])
        assert flipped[i] > base[i] or base[i] > 0
    assert np.array_equal(flipped[3:], base[3:])


def test_auroc_above_half_means_reversed_score_higher():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    mask = np.array([True, True, False, False])
    assert sc.auroc(scores, mask) == pytest.approx(1.0)
    assert sc.auroc(-scores, mask) == pytest.approx(0.0)


# --------------------------------------------------------------- tie handling
def test_auroc_is_exactly_half_on_constant_scores():
    scores = np.zeros(10)
    mask = np.array([True] * 3 + [False] * 7)
    assert sc.auroc(scores, mask) == pytest.approx(0.5)


def test_precision_at_k_is_tie_order_dependent_and_is_flagged():
    """Documents a real weakness of the repo metric, which the paper reports.

    ``precision_at_k`` sorts with an unstable argsort and does not average
    over ties, so on a constant score vector its value depends on position.
    ``detection_metrics`` therefore records ``n_tied_at_cut`` so such a value
    can be recognized as untrustworthy.
    """
    n = 10
    scores = np.zeros(n)
    front = np.array([True, True] + [False] * 8)
    m = sc.detection_metrics(scores, front, 2)
    assert m["n_tied_at_cut"] == n          # everything ties at the cut
    assert m["auroc"] == pytest.approx(0.5)  # AUROC is still honest


def test_interior_band_reproduces_zero_precision_at_nonchance_auroc():
    """The interior-band mechanism, as a unit test on synthetic ranks.

    Positives placed strictly in the interior give precision@k == 0 while
    AUROC stays well away from 0.5, with no ties involved.
    """
    n, k = 100, 10
    scores = np.arange(n, dtype=float)[::-1]   # distinct, rank 0 = highest
    mask = np.zeros(n, dtype=bool)
    mask[40:60] = True                          # interior band only
    m = sc.detection_metrics(scores, mask, k)
    assert m["n_tied_at_cut"] == 1
    assert m["prec_at_b"] == pytest.approx(0.0)
    assert 0.4 < m["auroc"] < 0.6               # interior band -> near chance
    mask2 = np.zeros(n, dtype=bool)
    mask2[20:40] = True                         # above median, still interior
    m2 = sc.detection_metrics(scores, mask2, k)
    assert m2["prec_at_b"] == pytest.approx(0.0)
    assert m2["auroc"] > 0.65                   # non-chance with zero precision


# -------------------------------------------------------------- determinism
def test_flip_selection_is_deterministic_per_seed():
    p = sc.prepare("n300", 0, "resp", 64)
    ds = sc.make_ds(p.Ztr)
    a = sc.select_flips("directional", ds, 20, 1e-3, seed=0)
    b = sc.select_flips("directional", ds, 20, 1e-3, seed=0)
    assert np.array_equal(a, b)


def test_flip_mask_marks_exactly_the_selected_rows():
    p = sc.prepare("n300", 0, "resp", 64)
    ds = sc.make_ds(p.Ztr)
    idx = sc.select_flips("directional", ds, 20, 1e-3, seed=0)
    dsp = ds.flipped(idx)
    assert dsp.flip_mask.sum() == len(np.unique(idx)) == 20
    assert np.array_equal(np.where(dsp.flip_mask)[0], np.sort(np.unique(idx)))
    assert np.array_equal(dsp.s[dsp.flip_mask], -ds.s[dsp.flip_mask])
    assert np.array_equal(dsp.Z, ds.Z)          # features never change


def test_attack_does_not_read_ground_truth():
    """A second attack call on an already-poisoned set must not consult the mask."""
    p = sc.prepare("n300", 0, "resp", 64)
    ds = sc.make_ds(p.Ztr)
    idx = sc.select_flips("directional", ds, 20, 1e-3, seed=0)
    dsp = ds.flipped(idx)
    hidden = sc.make_ds(p.Ztr)
    hidden.s = dsp.s.copy()                     # same labels, empty mask
    a = sc.select_flips("directional", dsp, 20, 1e-3, seed=1)
    b = sc.select_flips("directional", hidden, 20, 1e-3, seed=1)
    assert np.array_equal(a, b)


# ------------------------------------------------------------- control sanity
def test_margin_matched_control_excludes_the_detector_picks():
    abs_m = np.arange(20, dtype=float)
    targets = np.array([2, 3, 4])
    honest = np.arange(20)
    picked = margin_matched_honest(abs_m, targets, honest, 3)
    assert len(picked) == 3
    assert not set(picked) & set(targets.tolist())


def test_margin_matched_control_picks_similar_margins():
    abs_m = np.array([0.0, 0.01, 5.0, 5.01, 10.0, 10.01])
    targets = np.array([0, 2, 4])
    honest = np.arange(6)
    picked = margin_matched_honest(abs_m, targets, honest, 3)
    assert sorted(picked.tolist()) == [1, 3, 5]
