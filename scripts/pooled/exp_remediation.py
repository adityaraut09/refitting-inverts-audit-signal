"""Remediation with seven equal-size arms, scored on untouched test data.

Localization quality and repair quality are kept separate throughout. An arm
that finds poison well can still leave the model worse, and the table reports
both rather than letting one stand in for the other.

Arms, all deleting exactly ``k = B`` rows except ``none`` (0) and
``oracle_correct`` (0, it repairs instead of deleting):

``none``            fit on the poisoned set, no intervention
``detector``        delete the top-B rows ranked by the deployable score
``random``          delete B uniformly random rows, averaged over repetitions
``margin_matched``  delete B *honest* rows matched to the detector's picks by
                    absolute margin, isolating the cost of deleting rows that
                    look like the detector's targets
``oracle_delete``   delete exactly the B reversed rows
``oracle_correct``  flip exactly the B reversed labels back
``matched_clean``   the same rows the detector kept, but with original labels,
                    which separates "lost data" from "removed poison"

Lambda is selected per pool on dev with clean labels, before any arm is run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from src.pooled import core as sc
from src.pooled.core import margin_matched_honest, select_lam_on_dev

OUT = sc.EVIDENCE / "x_remediation_arms.csv"
RANDOM_REPS = 10
ATTACK = "margin_targeted"
ALPHA = 1.0


def bernoulli_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))


def fit_arm(prep: sc.Prepared, keep: np.ndarray, s_vals: np.ndarray, lam: float):
    return sc.fit(prep.Ztr[keep], s_vals[keep], lam)


def run_pool(conv: str, scale: str, pool: int, dim: int = 384) -> list[dict]:
    prep = sc.prepare(scale, pool, conv, dim)
    lam = select_lam_on_dev(prep)
    ds_clean = sc.make_ds(prep.Ztr)
    B = int(sc.PREVALENCE * prep.n_train)

    idx = sc.select_flips(ATTACK, ds_clean, B, lam, seed=pool, alpha=ALPHA)
    ds_p = ds_clean.flipped(idx)
    F = ds_p.flip_mask
    s_p, s_clean = ds_p.s, ds_clean.s
    n = prep.n_train
    all_i = np.arange(n)

    h_pois = sc.fit(prep.Ztr, s_p, lam)
    h_clean = sc.fit(prep.Ztr, s_clean, lam)

    score = sc.score_final_margin(ds_p, h_pois)
    order = np.argsort(-score)
    D = order[:B]
    abs_m = np.abs(h_pois.margins(ds_p))
    honest = all_i[~F]

    s_te = np.ones(prep.n_test)
    p_clean_te = expit(prep.Zte @ h_clean.theta)
    rank_clean_te = prep.Zte @ h_clean.theta

    # localization quality, reported separately from repair quality
    loc = sc.detection_metrics(score, F, B, prefix="loc_")

    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "none": (all_i, s_p),
        "detector": (np.setdiff1d(all_i, D), s_p),
        "margin_matched": (
            np.setdiff1d(all_i, margin_matched_honest(abs_m, D, honest, B)), s_p),
        "oracle_delete": (all_i[~F], s_p),
        "oracle_correct": (all_i, s_clean),
        "matched_clean": (np.setdiff1d(all_i, D), s_clean),
    }

    rows = []
    for arm, (keep, s_use) in arms.items():
        h = fit_arm(prep, keep, s_use, lam)
        rows.append(_row(conv, scale, pool, dim, lam, B, arm, prep, h, keep, s_use,
                         F, h_clean, p_clean_te, rank_clean_te, s_te, loc, rep=-1))

    # random deletion, averaged over repetitions
    for r in range(RANDOM_REPS):
        rng = np.random.default_rng(7_000_000 + 1000 * pool + r)
        R = rng.choice(n, size=B, replace=False)
        keep = np.setdiff1d(all_i, R)
        h = fit_arm(prep, keep, s_p, lam)
        rows.append(_row(conv, scale, pool, dim, lam, B, "random", prep, h, keep, s_p,
                         F, h_clean, p_clean_te, rank_clean_te, s_te, loc, rep=r))
    return rows


def _row(conv, scale, pool, dim, lam, B, arm, prep, h, keep, s_use, F,
         h_clean, p_clean_te, rank_clean_te, s_te, loc, rep) -> dict:
    # a row is still poisoned if it survived AND its used label is reversed
    kept_mask = np.zeros(prep.n_train, dtype=bool)
    kept_mask[keep] = True
    remaining_pois = int(np.sum(kept_mask & F & (s_use == -1)))
    p_arm = expit(prep.Zte @ h.theta)
    row = {
        "convention": conv, "scale": scale, "pool": pool, "dim": dim, "lam": lam,
        "attack": ATTACK, "alpha": ALPHA, "budget": B, "arm": arm, "rep": rep,
        "n_train": prep.n_train, "n_kept": int(len(keep)), "n_deleted": int(prep.n_train - len(keep)),
        "n_test": prep.n_test,
        "remaining_poisoned": remaining_pois,
        "test_logloss": sc.data_logloss(prep.Zte, s_te, h.theta),
        "test_acc": sc.accuracy(prep.Zte, s_te, h.theta),
        "kl_from_clean": bernoulli_kl(p_clean_te, p_arm),
        "mean_abs_pred_div": float(np.mean(np.abs(p_clean_te - p_arm))),
        "theta_dist_from_clean": float(np.linalg.norm(h.theta - h_clean.theta)),
        "rank_agree_with_clean": float(spearmanr(rank_clean_te, prep.Zte @ h.theta).statistic),
        "uninf_loss": sc.UNINF_LOSS,
    }
    row.update(loc)
    return row


def main() -> int:
    t0 = time.time()
    rows = []
    for conv in ["cat", "resp"]:
        for scale in ["n300", "n1000", "n3000"]:
            for pool in range(5):
                rows.extend(run_pool(conv, scale, pool))
            print(f"  {conv}/{scale}: done ({len(rows)} rows, {time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(sc.REPO)}  rows={len(df)}  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
