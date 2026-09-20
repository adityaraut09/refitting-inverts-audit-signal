"""Equal-budget review policies, including an orientation-agnostic one.

The deployable score is the final margin under the poisoned refit. The
benchmark shows that score ranks reversed comparisons below chance under the directional
attack and above chance under the margin-targeted attack, so an auditor who
must commit to one tail can be wrong either way. The two-tailed policy spends
the *same total budget* split across both tails and never consults labels to
choose a direction, so it is deployable in a way that ``bottom`` is not.

``bottom`` is reported only as an oracle diagnostic: choosing it requires
already knowing the ranking is inverted.

Every policy at a given budget reviews exactly that many rows. Two
post-review repairs are scored on the untouched test split: deleting the
reviewed rows, and correcting the reversed labels found among them.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pooled import core as sc
from src.pooled.core import select_lam_on_dev

OUT = sc.EVIDENCE / "x_twotailed_policies.csv"
RANDOM_REPS = 10
BUDGET_MULTS = [0.5, 1.0, 2.0]


def policies(score: np.ndarray, budget: int, n: int, seed: int) -> dict[str, list[np.ndarray]]:
    order = np.argsort(-score)          # most suspicious first
    top = order[:budget]
    bottom = order[::-1][:budget]
    n_top = budget // 2
    n_bot = budget - n_top
    two = np.concatenate([order[:n_top], order[::-1][:n_bot]])
    out: dict[str, list[np.ndarray]] = {
        "top": [top],
        "bottom_oracle_diagnostic": [bottom],
        "two_tailed": [two],
        "random": [],
    }
    for r in range(RANDOM_REPS):
        rng = np.random.default_rng(9_000_000 + 1000 * seed + r)
        out["random"].append(rng.choice(n, size=budget, replace=False))
    return out


def run_pool(conv: str, scale: str, pool: int, attack: str, dim: int = 384) -> list[dict]:
    prep = sc.prepare(scale, pool, conv, dim)
    lam = select_lam_on_dev(prep)
    ds_clean = sc.make_ds(prep.Ztr)
    B_atk = int(sc.PREVALENCE * prep.n_train)
    idx = sc.select_flips(attack, ds_clean, B_atk, lam, seed=pool, alpha=1.0)
    ds_p = ds_clean.flipped(idx)
    F = ds_p.flip_mask
    n = prep.n_train
    all_i = np.arange(n)
    s_te = np.ones(prep.n_test)

    h_pois = sc.fit(prep.Ztr, ds_p.s, lam)
    h_clean = sc.fit(prep.Ztr, ds_clean.s, lam)
    score = sc.score_final_margin(ds_p, h_pois)
    score_auroc = sc.detection_metrics(score, F, B_atk)["auroc"]
    base_ll = sc.data_logloss(prep.Zte, s_te, h_pois.theta)
    clean_ll = sc.data_logloss(prep.Zte, s_te, h_clean.theta)

    rows = []
    for mult in BUDGET_MULTS:
        budget = max(1, int(round(mult * B_atk)))
        for name, sel_list in policies(score, budget, n, pool).items():
            for rep, sel in enumerate(sel_list):
                reviewed = np.zeros(n, dtype=bool)
                reviewed[sel] = True
                recovered = int(np.sum(reviewed & F))

                # repair 1: delete every reviewed row
                keep = all_i[~reviewed]
                h_del = sc.fit(prep.Ztr[keep], ds_p.s[keep], lam)
                # repair 2: correct the reversed labels found by the review
                s_fix = ds_p.s.copy()
                s_fix[reviewed & F] *= -1
                h_fix = sc.fit(prep.Ztr, s_fix, lam)

                rows.append({
                    "convention": conv, "scale": scale, "pool": pool, "dim": dim,
                    "lam": lam, "attack": attack, "n_train": n, "n_test": prep.n_test,
                    "attack_budget": B_atk, "budget_mult": mult, "review_budget": budget,
                    "policy": name, "rep": rep if len(sel_list) > 1 else -1,
                    "n_flips": int(F.sum()), "prevalence": float(F.mean()),
                    "score_auroc": score_auroc,
                    "recovered": recovered,
                    "precision": recovered / budget,
                    "recall": recovered / max(int(F.sum()), 1),
                    "precision_lift": recovered / budget - float(F.mean()),
                    "remaining_poisoned_after_delete": int(np.sum(~reviewed & F)),
                    "test_logloss_no_action": base_ll,
                    "test_logloss_after_delete": sc.data_logloss(prep.Zte, s_te, h_del.theta),
                    "test_logloss_after_correct": sc.data_logloss(prep.Zte, s_te, h_fix.theta),
                    "test_logloss_clean_oracle": clean_ll,
                    "delete_minus_no_action": sc.data_logloss(prep.Zte, s_te, h_del.theta) - base_ll,
                    "correct_minus_no_action": sc.data_logloss(prep.Zte, s_te, h_fix.theta) - base_ll,
                })
    return rows


def main() -> int:
    t0 = time.time()
    rows = []
    for conv in ["cat", "resp"]:
        for scale in ["n300", "n1000", "n3000"]:
            for attack in ["directional", "margin_targeted"]:
                for pool in range(5):
                    rows.extend(run_pool(conv, scale, pool, attack))
                print(f"  {conv}/{scale}/{attack}: done "
                      f"({len(rows)} rows, {time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(sc.REPO)}  rows={len(df)}  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
