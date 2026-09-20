import numpy as np

from src.attacks.gradient_flip import GradientFlipAttack
from src.data.embed import make_synthetic_dataset
from src.detectors.self_influence import LeverageDetector, SelfInfluenceDetector
from src.metrics.detection import auroc
from src.model.bt_head import BTHead
from src.remediation.remove_refit import remove_refit


def test_self_influence_flags_planted_flips_auroc_above_0_75():
    ds = make_synthetic_dataset(n=400, d=10, separation=1.5, seed=0)
    budget = int(0.15 * ds.n)
    flips = GradientFlipAttack().select_flips(ds, budget, np.random.default_rng(0))
    poisoned = ds.flipped(flips)

    head = BTHead(lam=1e-2).fit(poisoned)
    scores = SelfInfluenceDetector().score(poisoned, head)

    assert scores.shape == (poisoned.n,)
    assert auroc(scores, poisoned.flip_mask) > 0.75


def test_leverage_detector_runs_and_is_finite():
    ds = make_synthetic_dataset(n=100, d=6, separation=1.0, seed=1)
    head = BTHead(lam=1e-2).fit(ds)
    scores = LeverageDetector().score(ds, head)
    assert scores.shape == (ds.n,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)  # sigma'(m) >= 0 and leverage >= 0 (quadratic form, Hinv PD)


def test_remove_refit_recovers_theta_closer_to_clean_than_leaving_poison():
    ds = make_synthetic_dataset(n=400, d=10, separation=1.5, seed=2)
    clean_head = BTHead(lam=1e-2).fit(ds)

    budget = int(0.15 * ds.n)
    flips = GradientFlipAttack().select_flips(ds, budget, np.random.default_rng(2))
    poisoned = ds.flipped(flips)
    poisoned_head = BTHead(lam=1e-2).fit(poisoned)

    scores = SelfInfluenceDetector().score(poisoned, poisoned_head)
    remediated_head = remove_refit(poisoned, scores, k=budget, lam=1e-2)

    drift_poisoned = np.linalg.norm(poisoned_head.theta - clean_head.theta)
    drift_remediated = np.linalg.norm(remediated_head.theta - clean_head.theta)
    assert drift_remediated < drift_poisoned
