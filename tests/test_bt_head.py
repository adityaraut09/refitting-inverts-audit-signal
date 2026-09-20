import numpy as np

from src.data.embed import make_synthetic_dataset
from src.model.bt_head import BTHead


def test_fit_converges():
    ds = make_synthetic_dataset(n=200, d=8, separation=2.0, seed=0)
    head = BTHead(lam=1e-3).fit(ds)
    assert np.linalg.norm(head.grad(ds)) < 1e-6


def test_fit_reduces_loss_below_untrained_baseline():
    ds = make_synthetic_dataset(n=200, d=8, separation=2.0, seed=1)
    untrained = BTHead(lam=1e-3)
    untrained.theta_ = np.zeros(ds.d)
    loss_before = untrained.loss(ds)
    assert np.isclose(loss_before, np.log(2.0), atol=1e-6)  # sigma(0) = 0.5 exactly

    head = BTHead(lam=1e-3).fit(ds)
    assert head.loss(ds) < loss_before


def test_hessian_is_label_independent():
    """Tilt identity: sigma' is even, so H(theta) must not depend on s."""
    rng = np.random.default_rng(0)
    ds = make_synthetic_dataset(n=100, d=6, separation=1.0, seed=2)
    head = BTHead(lam=0.1)
    head.theta_ = rng.normal(size=6)

    H_clean = head.hessian(ds)
    flips = rng.choice(ds.n, size=30, replace=False)
    ds_flipped = ds.flipped(flips)
    H_flipped = head.hessian(ds_flipped)

    np.testing.assert_allclose(H_clean, H_flipped, atol=1e-10)


def test_exact_tilt_identity_on_loss():
    """L_flip(theta) = L_clean(theta) + <theta, Delta_S>, Delta_S = mean_i in S z_i
    (over all n examples), since flips start from an all-clean (s=+1) dataset."""
    rng = np.random.default_rng(3)
    ds = make_synthetic_dataset(n=50, d=4, separation=1.0, seed=4)
    head = BTHead(lam=0.05)
    theta = rng.normal(size=4)

    loss_clean, _ = head._loss_and_grad(theta, ds)
    flips = rng.choice(ds.n, size=12, replace=False)
    ds_flipped = ds.flipped(flips)
    loss_flip, _ = head._loss_and_grad(theta, ds_flipped)

    delta_S = ds.Z[flips].sum(axis=0) / ds.n
    assert np.isclose(loss_flip, loss_clean + theta @ delta_S, atol=1e-10)


def test_higher_separation_is_easier_to_learn():
    hard = make_synthetic_dataset(n=300, d=10, separation=0.1, seed=5)
    easy = make_synthetic_dataset(n=300, d=10, separation=3.0, seed=5)

    head_hard = BTHead(lam=1e-3).fit(hard)
    head_easy = BTHead(lam=1e-3).fit(easy)

    assert head_easy.accuracy(easy) > head_hard.accuracy(hard)


def test_subset_and_flipped_do_not_mutate_original():
    ds = make_synthetic_dataset(n=20, d=3, separation=1.0, seed=6)
    Z_before = ds.Z.copy()
    s_before = ds.s.copy()

    sub = ds.subset([0, 1, 2])
    assert sub.n == 3
    flipped = ds.flipped([0, 1])

    np.testing.assert_array_equal(ds.Z, Z_before)
    np.testing.assert_array_equal(ds.s, s_before)
    assert flipped.s[0] == -1 and ds.s[0] == 1
