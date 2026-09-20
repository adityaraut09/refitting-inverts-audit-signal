"""Detector-side score directions, checked against each detector's spec.

The metric-direction test in test_metrics.py only pins the metrics. These
pin the detectors: every one must return a score that *increases* with
suspicion, so a below-chance AUROC is a statement about the detector's
behaviour and not about a reversed formula.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from src.data.embed import make_synthetic_dataset
from src.detectors.cross_fit import CrossFitDetector
from src.detectors.ensemble import EnsembleDetector
from src.detectors.self_influence import SelfInfluenceDetector
from src.detectors.training_dynamics import TrainingDynamicsDetector
from src.model.bt_head import BTHead

LAM = 1e-2


def _head_with_theta(theta):
    head = BTHead(lam=LAM)
    head.theta_ = np.asarray(theta, dtype=float)
    return head


# --------------------------------------------------------------------------
# training_dynamics: higher score == more negative signed margin
# --------------------------------------------------------------------------
def test_training_dynamics_score_is_negated_margin():
    ds = make_synthetic_dataset(n=40, d=5, separation=1.5, seed=0)
    head = BTHead(lam=LAM).fit(ds)

    scores = TrainingDynamicsDetector().score(ds, head)
    margins = head.margins(ds)

    assert np.allclose(scores, -margins)
    # strictly decreasing in the signed margin: rank order is exactly reversed
    assert np.array_equal(np.argsort(scores), np.argsort(-margins))


def test_training_dynamics_ranks_most_negative_margin_highest():
    ds = make_synthetic_dataset(n=6, d=2, separation=1.0, seed=1)
    head = _head_with_theta([1.0, 0.0])
    ds.Z[:] = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0],
                        [-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
    ds.s[:] = 1.0  # margins are then just Z[:, 0]: 3, 2, 1, -1, -2, -3

    scores = TrainingDynamicsDetector().score(ds, head)

    assert int(np.argmax(scores)) == 5  # margin -3, the strongest disagreement
    assert int(np.argmin(scores)) == 0  # margin +3, the strongest agreement


# --------------------------------------------------------------------------
# cross_fit: higher score == stronger held-out disagreement
# --------------------------------------------------------------------------
def test_cross_fit_score_is_strictly_decreasing_in_out_of_fold_margin():
    ds = make_synthetic_dataset(n=60, d=5, separation=1.5, seed=2)
    det = CrossFitDetector(n_folds=5, lam=LAM)
    scores = det.score(ds)

    # recompute the out-of-fold margins with the detector's own fold logic
    fold_ids = np.arange(ds.n) % det.n_folds
    oof_margin = np.zeros(ds.n)
    for fold in range(det.n_folds):
        test_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        fold_head = BTHead(lam=det.lam).fit(ds.subset(train_idx))
        test_ds = ds.subset(test_idx)
        oof_margin[test_idx] = test_ds.s * (test_ds.Z @ fold_head.theta)

    assert np.allclose(scores, expit(-oof_margin))
    assert np.array_equal(np.argsort(scores), np.argsort(-oof_margin))


def test_cross_fit_score_rises_when_a_single_label_is_flipped():
    """Flipping one row's label makes the held-out model disagree with it,
    which must push that row's score up."""
    ds = make_synthetic_dataset(n=60, d=5, separation=2.0, seed=3)
    det = CrossFitDetector(n_folds=5, lam=LAM)
    before = det.score(ds)

    row = 7
    after = det.score(ds.flipped([row]))

    assert after[row] > before[row]
    assert after[row] > 0.5 > before[row]


# --------------------------------------------------------------------------
# self_influence: higher score == larger (disagreement x leverage)
# --------------------------------------------------------------------------
def test_self_influence_equals_disagreement_times_leverage():
    ds = make_synthetic_dataset(n=40, d=5, separation=1.5, seed=4)
    head = BTHead(lam=LAM).fit(ds)

    scores = SelfInfluenceDetector().score(ds, head)

    m = head.margins(ds)
    disagreement = expit(-m) ** 2
    leverage = np.einsum("ij,jk,ik->i", ds.Z, head.hinv(ds), ds.Z)

    assert np.allclose(scores, disagreement * leverage)
    assert np.all(leverage > 0)  # Hinv is positive definite


def test_self_influence_factors_are_monotone_in_the_right_direction():
    """Composed direction: the score rises as the margin falls (more
    disagreement) with leverage held, and rises with leverage with
    disagreement held."""
    m = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    disagreement = expit(-m) ** 2
    assert np.all(np.diff(disagreement) < 0)  # strictly decreasing in m

    leverage = np.array([1.0, 2.0, 5.0, 10.0])
    fixed_disagreement = 0.25
    assert np.all(np.diff(fixed_disagreement * leverage) > 0)

    fixed_leverage = 3.0
    assert np.all(np.diff(disagreement * fixed_leverage) < 0)


# --------------------------------------------------------------------------
# ensemble: larger component ranks == larger ensemble score
# --------------------------------------------------------------------------
class _StubDetector:
    def __init__(self, scores):
        self.name = "stub"
        self._scores = np.asarray(scores, dtype=float)

    def score(self, ds, head=None):
        return self._scores


def test_ensemble_averages_component_ranks():
    ds = make_synthetic_dataset(n=4, d=2, separation=1.0, seed=5)
    a = _StubDetector([0.1, 0.2, 0.3, 0.4])
    b = _StubDetector([10.0, 20.0, 30.0, 40.0])  # different scale, same order

    scores = EnsembleDetector([a, b]).score(ds)

    assert np.allclose(scores, rankdata([0.1, 0.2, 0.3, 0.4]))
    assert np.all(np.diff(scores) > 0)  # agreeing components keep the order


def test_ensemble_score_rises_when_a_component_score_rises():
    ds = make_synthetic_dataset(n=4, d=2, separation=1.0, seed=6)
    other = _StubDetector([1.0, 1.0, 1.0, 1.0])

    low = EnsembleDetector([_StubDetector([0.1, 0.2, 0.3, 0.4]), other]).score(ds)
    # promote row 0 from smallest to largest in the first component
    high = EnsembleDetector([_StubDetector([0.9, 0.2, 0.3, 0.4]), other]).score(ds)

    assert high[0] > low[0]
