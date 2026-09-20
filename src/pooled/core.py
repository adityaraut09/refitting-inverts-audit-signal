"""Pooled-replication core: partitions, projections, heads, scores.

The attacks, detectors and detection metrics are imported from the rest of
``src`` rather than reimplemented, so that a pooled-replication number and a
benchmark number mean the same thing.

Three points of care:

*   ``BTHead.loss`` includes the ridge penalty ``0.5*lam*||theta||^2``. A
    comparison against log 2 must use the unpenalized data term, so
    :func:`data_logloss` is used for every reported predictive number.
*   ``CrossFitDetector`` and ``TrainingDynamicsDetector`` both default to
    ``lam=1e-2``, which is the wrong scale for 384-dimensional real features.
    Every construction here passes ``lam`` explicitly.
*   Dimensionality reduction uses the top right singular vectors of the
    *training* features only. It is an orthonormal projection, so at
    ``dim == d`` it is a pure rotation and ridge Bradley-Terry is invariant to
    it, which ``tests/test_pooled.py`` asserts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import data as sd
from ..attacks.ambiguity_flip import AmbiguityFlipAttack
from ..attacks.gradient_flip import GradientFlipAttack
from ..attacks.random_flip import RandomFlipAttack
from ..data.types import PreferenceDataset
from ..detectors.cross_fit import CrossFitDetector
from ..detectors.self_influence import SelfInfluenceDetector
from ..metrics.detection import auroc, auprc, precision_at_k, recall_at_k
from ..model.bt_head import BTHead

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = sd.EVIDENCE

UNINF_LOSS = float(np.log(2.0))  # 0.6931471805599453
LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
DIMS = [32, 64, 128, 256, 384]
PREVALENCE = 0.15
DEV_FRAC = 0.30


# --------------------------------------------------------------------------
# predictive quantities
# --------------------------------------------------------------------------
def data_logloss(Z: np.ndarray, s: np.ndarray, theta: np.ndarray) -> float:
    """Mean logistic loss with NO ridge penalty. Comparable to log 2."""
    m = s * (Z @ theta)
    return float(np.mean(np.logaddexp(0.0, -m)))


def accuracy(Z: np.ndarray, s: np.ndarray, theta: np.ndarray) -> float:
    return float(np.mean(s * (Z @ theta) > 0))


def hessian_cond(Z: np.ndarray, s: np.ndarray, theta: np.ndarray, lam: float) -> float:
    m = s * (Z @ theta)
    w = expit(m) * expit(-m)
    H = (Z * w[:, None]).T @ Z / Z.shape[0] + lam * np.eye(Z.shape[1])
    ev = np.linalg.eigvalsh(H)
    return float(ev.max() / max(ev.min(), 1e-300))


# --------------------------------------------------------------------------
# partitions and projection
# --------------------------------------------------------------------------
@dataclass
class Prepared:
    Ztr: np.ndarray
    Zdev: np.ndarray
    Zte: np.ndarray
    dim: int
    n_train: int
    n_dev: int
    n_test: int
    n_zero_train: int
    n_zero_test: int
    basis: np.ndarray


def train_basis(Ztr_raw: np.ndarray, dim: int) -> np.ndarray:
    """Top-``dim`` right singular vectors of the training features.

    Uses training rows only. ``Z`` carries no labels, so this cannot leak a
    label; it can only leak training-row geometry, which is permitted.
    """
    # eigh on the Gram matrix is cheaper than a full SVD and numerically fine
    G = Ztr_raw.T @ Ztr_raw
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1][:dim]
    return evecs[:, order]


def prepare(scale: str, pool: int, convention: str, dim: int,
            split_seed: int = 0) -> Prepared:
    """Build train/dev/test feature blocks for one configuration.

    Zero-difference rows are removed in the original 384-dimensional space,
    matching the benchmark pipeline's filter, and only then projected.
    """
    train_raw = sd.load_helpful_base("train")
    test_raw = sd.load_helpful_base("test")
    pools = sd.allocate_pools(len(train_raw), _POOL_SPEC)
    idx = pools[f"{scale}_pool{pool}"]

    Zp = sd.embed_block(train_raw, idx, convention)
    keep_p = sd.nonzero_mask(Zp)
    Zp_ok = Zp[keep_p]

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(Zp_ok.shape[0])
    n_dev = int(round(DEV_FRAC * Zp_ok.shape[0]))
    dev_i, tr_i = perm[:n_dev], perm[n_dev:]

    Zte_all = sd.embed_block(test_raw, np.arange(len(test_raw)), convention)
    keep_t = sd.nonzero_mask(Zte_all)
    Zte_ok = Zte_all[keep_t]

    V = train_basis(Zp_ok[tr_i], dim)
    return Prepared(
        Ztr=Zp_ok[tr_i] @ V,
        Zdev=Zp_ok[dev_i] @ V,
        Zte=Zte_ok @ V,
        dim=dim,
        n_train=len(tr_i),
        n_dev=len(dev_i),
        n_test=Zte_ok.shape[0],
        n_zero_train=int((~keep_p).sum()),
        n_zero_test=int((~keep_t).sum()),
        basis=V,
    )


_POOL_SPEC = [("n3000", 3000, 5), ("n1000", 1000, 5), ("n300", 300, 5)]


def make_ds(Z: np.ndarray) -> PreferenceDataset:
    """Clean-orientation dataset: every label +1, nothing flipped yet."""
    n = Z.shape[0]
    return PreferenceDataset(
        Z=Z, s=np.ones(n), flip_mask=np.zeros(n, dtype=bool), meta={"pipeline": "pooled"}
    )


def fit(Z: np.ndarray, s: np.ndarray, lam: float) -> BTHead:
    ds = PreferenceDataset(Z=Z, s=s, flip_mask=np.zeros(Z.shape[0], dtype=bool), meta={})
    return BTHead(lam=lam).fit(ds)


# --------------------------------------------------------------------------
# attacks
# --------------------------------------------------------------------------
def make_attack(kind: str, alpha: float = 1.0):
    if kind == "directional":
        return GradientFlipAttack()
    if kind == "margin_targeted":
        return AmbiguityFlipAttack(alpha=alpha)
    if kind == "random":
        return RandomFlipAttack()
    raise ValueError(f"unknown attack {kind!r}")


def select_flips(kind: str, ds_clean: PreferenceDataset, budget: int,
                 lam: float, seed: int, alpha: float = 1.0) -> np.ndarray:
    """Choose which rows to reverse.

    The margin-targeted rule needs a head; it is given one fit at the
    configuration's own lambda, never the attack class's 1e-2 default.
    """
    rng = np.random.default_rng(seed)
    atk = make_attack(kind, alpha=alpha)
    head = None
    if kind == "margin_targeted":
        head = BTHead(lam=lam).fit(ds_clean)
    return atk.select_flips(ds_clean, budget=budget, rng=rng, head=head)


# --------------------------------------------------------------------------
# detector scores, all oriented so that higher means more suspicious
# --------------------------------------------------------------------------
def score_final_margin(ds: PreferenceDataset, head: BTHead) -> np.ndarray:
    """``-m_i`` under the given head. The paper's headline score."""
    return -head.margins(ds)


def score_self_influence(ds: PreferenceDataset, head: BTHead) -> np.ndarray:
    return SelfInfluenceDetector().score(ds, head)


def score_cross_fit(ds: PreferenceDataset, lam: float, n_folds: int = 5) -> np.ndarray:
    return CrossFitDetector(n_folds=n_folds, lam=lam).score(ds)


def detection_metrics(scores: np.ndarray, mask: np.ndarray, budget: int,
                      prefix: str = "") -> dict:
    """AUROC/AUPRC plus budget-depth precision, recall and lift.

    ``precision_at_k`` and ``recall_at_k`` in ``src`` use an unstable argsort
    and so are tie-order dependent. ``n_tied_at_cut`` records how many rows
    share the cut score, which is what makes such a value untrustworthy.
    """
    prev = float(mask.mean())
    p = precision_at_k(scores, mask, budget)
    order = np.argsort(-scores)
    cut = scores[order[min(budget, len(scores)) - 1]]
    n_tied = int(np.sum(scores == cut))
    return {
        f"{prefix}auroc": float(auroc(scores, mask)),
        f"{prefix}auprc": float(auprc(scores, mask)),
        f"{prefix}prec_at_b": float(p),
        f"{prefix}recall_at_b": float(recall_at_k(scores, mask, budget)),
        f"{prefix}prec_lift": float(p - prev) if np.isfinite(p) else float("nan"),
        f"{prefix}n_tied_at_cut": n_tied,
        f"{prefix}n_hits_at_b": int(round(p * budget)) if np.isfinite(p) else -1,
    }


def absorption(ds_poisoned: PreferenceDataset, head: BTHead) -> dict:
    """Signed-margin absorption: do reversed rows end up better fit?"""
    m = head.margins(ds_poisoned)
    f = ds_poisoned.flip_mask
    return {
        "marg_mean_reversed": float(m[f].mean()) if f.any() else float("nan"),
        "marg_mean_retained": float(m[~f].mean()) if (~f).any() else float("nan"),
        "marg_excess_reversed": float(m[f].mean() - m[~f].mean())
        if f.any() and (~f).any() else float("nan"),
        "frac_reversed_positive": float((m[f] > 0).mean()) if f.any() else float("nan"),
    }


def select_lam_on_dev(prep: Prepared) -> float:
    """Pick lambda by clean-label dev log-loss. Never touches test."""
    s_tr = np.ones(prep.n_train)
    s_dev = np.ones(prep.n_dev)
    best, best_l = np.inf, LAMBDA_GRID[0]
    for lam in LAMBDA_GRID:
        h = fit(prep.Ztr, s_tr, lam)
        ll = data_logloss(prep.Zdev, s_dev, h.theta)
        if ll < best:
            best, best_l = ll, lam
    return best_l


def margin_matched_honest(abs_m: np.ndarray, targets: np.ndarray,
                          honest: np.ndarray, k: int) -> np.ndarray:
    """Greedy nearest-|margin| matching of k honest rows to the target rows.

    The detector's own picks are excluded from the candidate pool. Without
    that exclusion the arm is vacuous whenever the detector selects only
    honest rows: the nearest honest row to an honest target is the target
    itself, so the "matched" set comes back equal to the detector's set and
    the two arms report identical numbers.
    """
    pool = [i for i in honest if i not in set(targets[:k].tolist())]
    picked: list[int] = []
    for t in targets[:k]:
        if not pool:
            break
        j = int(np.argmin([abs(abs_m[c] - abs_m[t]) for c in pool]))
        picked.append(pool.pop(j))
    return np.array(picked, dtype=int)
