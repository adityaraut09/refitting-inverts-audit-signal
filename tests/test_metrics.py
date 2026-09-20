import numpy as np

from src.metrics.detection import auprc, auroc, precision_at_k, recall_at_k


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    flip_mask = np.array([False, False, True, True, False])
    assert auroc(scores, flip_mask) == 1.0


def test_auroc_perfectly_wrong():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    flip_mask = np.array([False, False, True, True, False])
    assert auroc(scores, flip_mask) == 0.0


def test_auroc_ties_score_half():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    flip_mask = np.array([True, True, False, False])
    assert np.isclose(auroc(scores, flip_mask), 0.5)


def test_auroc_nan_when_no_positives_or_negatives():
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([False, False])))
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([True, True])))


def test_auprc_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    flip_mask = np.array([True, True, False, False])
    assert auprc(scores, flip_mask) == 1.0


def test_precision_recall_at_k():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    flip_mask = np.array([True, False, True, False, False])
    assert precision_at_k(scores, flip_mask, k=2) == 1.0  # top-2 are both true positives
    assert recall_at_k(scores, flip_mask, k=2) == 1.0  # both true positives captured
    assert precision_at_k(scores, flip_mask, k=5) == 2 / 5
    assert recall_at_k(scores, flip_mask, k=1) == 0.5


def test_precision_at_k_caps_at_n():
    scores = np.array([0.5, 0.4])
    flip_mask = np.array([True, False])
    assert precision_at_k(scores, flip_mask, k=100) == 0.5


def test_metric_direction_is_higher_score_more_suspicious():
    """All three metrics agree that a *higher* score means more suspicious.

    Pins the sign convention so a below-chance AUROC can be read as a real
    result about the detector rather than a flipped comparison in the metric.
    """
    # Every positive outranks every negative, no ties.
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    flip_mask = np.array([True, True, True, False, False, False])
    n_pos = int(flip_mask.sum())

    assert auroc(scores, flip_mask) == 1.0
    assert precision_at_k(scores, flip_mask, k=n_pos) == 1.0
    assert recall_at_k(scores, flip_mask, k=n_pos) == 1.0

    # Negating sends every positive below every negative, so AUROC inverts.
    assert auroc(-scores, flip_mask) == 0.0
