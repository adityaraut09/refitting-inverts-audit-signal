"""Detection-quality metrics. The only place
`flip_mask` should be read outside test/notebook evaluation code — never
pass it to a `Detector.score`."""
from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, flip_mask: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U / n_pos*n_neg), no external
    dependency: P(score of a random positive > a random negative), ties
    counted as a half-win."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(flip_mask, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_per_value = np.zeros(len(counts))
    np.add.at(sum_ranks_per_value, inverse, ranks)
    avg_rank = (sum_ranks_per_value / counts)[inverse]

    u = avg_rank[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(scores: np.ndarray, flip_mask: np.ndarray) -> float:
    """Average precision: mean precision@k evaluated at each true positive's
    rank, sweeping the threshold down through all scores (no interpolation)."""
    y = np.asarray(flip_mask, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float))
    y_sorted = y[order]
    precision_at_each_rank = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float(precision_at_each_rank[y_sorted].sum() / n_pos)


def precision_at_k(scores: np.ndarray, flip_mask: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    if k <= 0:
        return float("nan")
    top_k = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float(np.mean(np.asarray(flip_mask, dtype=bool)[top_k]))


def recall_at_k(scores: np.ndarray, flip_mask: np.ndarray, k: int) -> float:
    y = np.asarray(flip_mask, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, len(scores))
    top_k = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float(y[top_k].sum() / n_pos)
