import numpy as np

from src.attacks.gradient_flip import GradientFlipAttack
from src.attacks.random_flip import RandomFlipAttack
from src.data.embed import make_synthetic_dataset
from src.model.bt_head import BTHead


def test_random_flip_respects_budget_and_is_deterministic():
    ds = make_synthetic_dataset(n=100, d=6, separation=1.0, seed=0)
    attack = RandomFlipAttack()
    flips_a = attack.select_flips(ds, budget=15, rng=np.random.default_rng(42))
    flips_b = attack.select_flips(ds, budget=15, rng=np.random.default_rng(42))
    assert len(flips_a) == 15
    assert len(set(flips_a)) == 15  # no duplicates
    np.testing.assert_array_equal(flips_a, flips_b)


def test_random_flip_budget_capped_at_n():
    ds = make_synthetic_dataset(n=20, d=4, separation=1.0, seed=0)
    attack = RandomFlipAttack()
    flips = attack.select_flips(ds, budget=1000, rng=np.random.default_rng(0))
    assert len(flips) == ds.n


def test_gradient_flip_respects_budget_and_is_deterministic():
    ds = make_synthetic_dataset(n=100, d=6, separation=1.0, seed=1)
    attack = GradientFlipAttack()
    flips_a = attack.select_flips(ds, budget=10, rng=np.random.default_rng(0))
    flips_b = attack.select_flips(ds, budget=10, rng=np.random.default_rng(0))
    assert len(flips_a) == 10
    np.testing.assert_array_equal(flips_a, flips_b)  # no randomness involved at all


def test_gradient_flip_targets_examples_most_aligned_with_v():
    ds = make_synthetic_dataset(n=200, d=8, separation=1.0, seed=2)
    v = np.zeros(8)
    v[0] = 1.0  # known target direction
    attack = GradientFlipAttack(target_dir=v)
    flips = attack.select_flips(ds, budget=20, rng=np.random.default_rng(0))

    alignment = ds.s * (ds.Z @ v)
    selected_alignment = alignment[flips]
    unselected_alignment = np.delete(alignment, flips)
    assert selected_alignment.min() >= unselected_alignment.max()


def test_gradient_flip_hurts_more_than_random_at_equal_budget():
    """The core 'does the attack work' sanity check: at the same budget, the
    targeted attack should corrupt theta at least as much as the random
    control on average, on data where the target direction carries signal."""
    ds = make_synthetic_dataset(n=400, d=10, separation=2.0, seed=3)
    idx = np.random.default_rng(3).permutation(ds.n)
    holdout, train = ds.subset(idx[:100]), ds.subset(idx[100:])

    clean_head = BTHead(lam=1e-3).fit(train)
    budget = int(0.15 * train.n)

    random_flips = RandomFlipAttack().select_flips(train, budget, np.random.default_rng(3))
    gradient_flips = GradientFlipAttack().select_flips(train, budget, np.random.default_rng(3))

    random_head = BTHead(lam=1e-3).fit(train.flipped(random_flips))
    gradient_head = BTHead(lam=1e-3).fit(train.flipped(gradient_flips))

    random_drift = np.linalg.norm(random_head.theta - clean_head.theta)
    gradient_drift = np.linalg.norm(gradient_head.theta - clean_head.theta)
    assert gradient_drift > random_drift

    assert gradient_head.accuracy(holdout) <= random_head.accuracy(holdout)
