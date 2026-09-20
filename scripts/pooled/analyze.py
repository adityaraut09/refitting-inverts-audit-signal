"""Adjudicate the preregistered questions from the stored evidence.

Every hyperparameter choice made here is made on dev log-loss. The test
column is read only after a choice is fixed, and never compared across
candidates. Each question prints its preregistered decision rule and the
verdict that rule produces, whether or not it favors the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pooled import core as sc

EV = sc.EVIDENCE
LOG2 = sc.UNINF_LOSS
PRIMARY_SCALE = "n3000"
PRIMARY_CONV = "cat"


def dev_select(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Pick, within each group in ``by``, the (dim, lam) with lowest mean dev
    log-loss over pools. Dev only: the test column plays no part."""
    g = (df.groupby(by + ["dim", "lam"], as_index=False)
           .agg(dev=("clean_dev_logloss", "mean")))
    return g.loc[g.groupby(by).dev.idxmin()].reset_index(drop=True)


def paired_bootstrap(diff: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple:
    """Percentile interval for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def q1_q2(grid: pd.DataFrame, out: dict) -> None:
    print("=" * 78)
    print("Q1  inversion across independent pools, on the untouched test split")
    print("    rule: reproduced if cleanhead AUROC>0.5 AND refit AUROC<0.5 in >=4 of 5 pools")
    print("=" * 78)

    sel = dev_select(grid, ["convention", "scale"])
    sel.to_csv(EV / "q_dev_selected_configs.csv", index=False)
    rows = []
    for _, r in sel.iterrows():
        sub = grid[(grid.convention == r.convention) & (grid.scale == r.scale)
                   & (grid.dim == r.dim) & (grid.lam == r.lam)]
        n_cross = int(((sub.fm_cleanhead_auroc > 0.5) & (sub.fm_refit_auroc < 0.5)).sum())
        verdict = ("reproduced" if n_cross >= 4 else
                   "failed to reproduce" if n_cross <= 2 else "inconclusive")
        rows.append({
            "convention": r.convention, "scale": r.scale, "dim": int(r.dim), "lam": r.lam,
            "n_train_mean": sub.n_train.mean(), "n_over_d_mean": sub.n_over_d.mean(),
            "fm_cleanhead_auroc_mean": sub.fm_cleanhead_auroc.mean(),
            "fm_cleanhead_auroc_sd": sub.fm_cleanhead_auroc.std(ddof=1),
            "fm_refit_auroc_mean": sub.fm_refit_auroc.mean(),
            "fm_refit_auroc_sd": sub.fm_refit_auroc.std(ddof=1),
            "auroc_diff_mean": (sub.fm_cleanhead_auroc - sub.fm_refit_auroc).mean(),
            "si_refit_auroc_mean": sub.si_refit_auroc.mean(),
            "n_pools_crossing": n_cross, "n_pools": len(sub),
            "clean_test_logloss_mean": sub.clean_test_logloss.mean(),
            "clean_test_acc_mean": sub.clean_test_acc.mean(),
            "beats_log2": bool(sub.clean_test_logloss.mean() < LOG2),
            "marg_excess_reversed_mean": sub.refit_marg_excess_reversed.mean(),
            "inverted_separation_mean": sub.inverted_separation_fm_refit.mean(),
            "verdict": verdict,
        })
    q1 = pd.DataFrame(rows)
    q1.to_csv(EV / "q1_pools_by_config.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(q1[["convention", "scale", "dim", "lam", "n_train_mean", "n_over_d_mean",
                  "fm_cleanhead_auroc_mean", "fm_refit_auroc_mean", "n_pools_crossing",
                  "clean_test_logloss_mean", "beats_log2", "verdict"]].to_string(index=False))
    out["q1"] = q1.to_dict("records")

    print()
    print("=" * 78)
    print("Q2  the N/d ladder: does the inversion survive N>d?")
    print("    rule: survives at n3000 if refit AUROC<0.5 in >=4 of 5 pools")
    print("=" * 78)
    lad = []
    for (conv, scale, dim), sub_all in grid.groupby(["convention", "scale", "dim"]):
        best_lam = (sub_all.groupby("lam").clean_dev_logloss.mean().idxmin())
        sub = sub_all[sub_all.lam == best_lam]
        lad.append({
            "convention": conv, "scale": scale, "dim": int(dim), "lam_dev_sel": best_lam,
            "n_train_mean": sub.n_train.mean(), "n_over_d": sub.n_over_d.mean(),
            "regime": ("N<d" if sub.n_over_d.mean() < 1 else
                       "N~d" if sub.n_over_d.mean() < 2 else "N>d"),
            "train_acc": sub.clean_train_acc.mean(),
            "train_logloss": sub.clean_train_logloss.mean(),
            "test_logloss": sub.clean_test_logloss.mean(),
            "test_acc": sub.clean_test_acc.mean(),
            "train_test_gap": (sub.clean_test_logloss - sub.clean_train_logloss).mean(),
            "beats_log2": bool(sub.clean_test_logloss.mean() < LOG2),
            "hessian_cond": sub.clean_hessian_cond.mean(),
            "fm_cleanhead_auroc": sub.fm_cleanhead_auroc.mean(),
            "fm_refit_auroc": sub.fm_refit_auroc.mean(),
            "fm_refit_auroc_sd": sub.fm_refit_auroc.std(ddof=1),
            "n_pools_refit_below_half": int((sub.fm_refit_auroc < 0.5).sum()),
            "marg_excess_reversed": sub.refit_marg_excess_reversed.mean(),
            "theta_dist": sub.theta_dist.mean(),
        })
    ladder = pd.DataFrame(lad).sort_values(["convention", "scale", "dim"])
    ladder.to_csv(EV / "q2_nd_ladder.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(ladder[["convention", "scale", "dim", "n_over_d", "regime", "lam_dev_sel",
                      "train_acc", "test_logloss", "beats_log2", "fm_cleanhead_auroc",
                      "fm_refit_auroc", "n_pools_refit_below_half",
                      "marg_excess_reversed"]].to_string(index=False))
    out["q2"] = ladder.to_dict("records")


def q2b(grid: pd.DataFrame, out: dict) -> None:
    """Isolate the effect of N by holding dim and lambda fixed across scales.

    In the dev-selected ladder both lambda and dim move with scale, so an N
    effect there is confounded. This slice fixes dim=384 and lambda=1e-3 and
    varies only the amount of training data.
    """
    print()
    print("=" * 78)
    print("Q2b  effect of N alone: dim=384 and lam=1e-3 held fixed across scales")
    print("=" * 78)
    sub = grid[(grid.dim == 384) & (grid.lam == 1e-3)]
    rows = []
    for (conv, scale), s in sub.groupby(["convention", "scale"]):
        rows.append({
            "convention": conv, "scale": scale, "dim": 384, "lam": 1e-3,
            "n_train": s.n_train.mean(), "n_over_d": s.n_over_d.mean(),
            "clean_train_acc": s.clean_train_acc.mean(),
            "clean_test_logloss": s.clean_test_logloss.mean(),
            "clean_test_acc": s.clean_test_acc.mean(),
            "beats_log2": bool(s.clean_test_logloss.mean() < LOG2),
            "n_pools_below_log2": int((s.clean_test_logloss < LOG2).sum()),
            "oracle_cleanhead_auroc": s.fm_cleanhead_auroc.mean(),
            "oracle_cleanhead_auroc_sd": s.fm_cleanhead_auroc.std(ddof=1),
            "n_pools_cleanhead_above_half": int((s.fm_cleanhead_auroc > 0.5).sum()),
            "refit_auroc": s.fm_refit_auroc.mean(),
            "refit_auroc_sd": s.fm_refit_auroc.std(ddof=1),
            "n_pools_refit_below_half": int((s.fm_refit_auroc < 0.5).sum()),
            "inverted_separation": s.inverted_separation_fm_refit.mean(),
            "marg_excess_reversed": s.refit_marg_excess_reversed.mean(),
            "frac_reversed_positive": s.refit_frac_reversed_positive.mean(),
            "hessian_cond": s.clean_hessian_cond.mean(),
        })
    q = pd.DataFrame(rows).sort_values(["convention", "n_train"])
    q.to_csv(EV / "q2b_effect_of_N.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(q[["convention", "scale", "n_train", "n_over_d", "clean_train_acc",
                 "clean_test_logloss", "beats_log2", "oracle_cleanhead_auroc",
                 "n_pools_cleanhead_above_half", "refit_auroc",
                 "n_pools_refit_below_half", "marg_excess_reversed"]].to_string(index=False))
    out["q2b"] = q.to_dict("records")


def q3(grid: pd.DataFrame, out: dict) -> None:
    print()
    print("=" * 78)
    print("Q3  is there a useful clean head, selected without test access?")
    print("    rule: dev-selected config qualifies only if clean TEST logloss < log 2")
    print(f"    log 2 = {LOG2:.6f}")
    print("=" * 78)
    sel = dev_select(grid, ["convention", "scale"])
    rows = []
    for _, r in sel.iterrows():
        sub = grid[(grid.convention == r.convention) & (grid.scale == r.scale)
                   & (grid.dim == r.dim) & (grid.lam == r.lam)]
        rows.append({
            "convention": r.convention, "scale": r.scale, "dim": int(r.dim), "lam": r.lam,
            "dev_logloss": sub.clean_dev_logloss.mean(),
            "test_logloss": sub.clean_test_logloss.mean(),
            "test_logloss_sd": sub.clean_test_logloss.std(ddof=1),
            "test_acc": sub.clean_test_acc.mean(),
            "n_pools_below_log2": int((sub.clean_test_logloss < LOG2).sum()),
            "qualifies": bool(sub.clean_test_logloss.mean() < LOG2),
            "fm_refit_auroc": sub.fm_refit_auroc.mean(),
            "fm_cleanhead_auroc": sub.fm_cleanhead_auroc.mean(),
            "n_pools_crossing": int(((sub.fm_cleanhead_auroc > 0.5)
                                     & (sub.fm_refit_auroc < 0.5)).sum()),
        })
    q = pd.DataFrame(rows)
    q.to_csv(EV / "q3_cleanhead_validity.csv", index=False)
    print(q.to_string(index=False))
    out["q3"] = q.to_dict("records")


def q4(out: dict) -> None:
    print()
    print("=" * 78)
    print("Q4  remediation arms against no action, on the untouched test split")
    print("    rule: beats no action if paired mean improvement>0 and bootstrap 95% excludes 0")
    print("=" * 78)
    rem = pd.read_csv(EV / "x_remediation_arms.csv")
    # collapse the random arm's repetitions to one value per pool first
    rem = (rem.groupby(["convention", "scale", "pool", "arm"], as_index=False)
              .agg({c: "mean" for c in rem.columns
                    if c not in ("convention", "scale", "pool", "arm", "attack")}))
    rows = []
    for (conv, scale), sub in rem.groupby(["convention", "scale"]):
        base = sub[sub.arm == "none"].set_index("pool")
        for arm, a in sub.groupby("arm"):
            a = a.set_index("pool")
            common = base.index.intersection(a.index)
            # positive = improvement (loss went down relative to no action)
            d = (base.loc[common, "test_logloss"] - a.loc[common, "test_logloss"]).values
            m, lo, hi = paired_bootstrap(d)
            rows.append({
                "convention": conv, "scale": scale, "arm": arm, "n_pools": len(common),
                "remaining_poisoned": a.loc[common, "remaining_poisoned"].mean(),
                "test_logloss": a.loc[common, "test_logloss"].mean(),
                "test_acc": a.loc[common, "test_acc"].mean(),
                "kl_from_clean": a.loc[common, "kl_from_clean"].mean(),
                "theta_dist_from_clean": a.loc[common, "theta_dist_from_clean"].mean(),
                "rank_agree_with_clean": a.loc[common, "rank_agree_with_clean"].mean(),
                "improve_vs_none": m, "ci_lo": lo, "ci_hi": hi,
                "n_pools_improved": int((d > 0).sum()),
                "beats_none": bool(m > 0 and lo > 0),
                "loc_auroc": a.loc[common, "loc_auroc"].mean(),
                "loc_prec_at_b": a.loc[common, "loc_prec_at_b"].mean(),
            })
    q = pd.DataFrame(rows).sort_values(["convention", "scale", "arm"])
    q.to_csv(EV / "q4_remediation_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(q[["convention", "scale", "arm", "remaining_poisoned", "test_logloss",
                 "improve_vs_none", "ci_lo", "ci_hi", "n_pools_improved", "beats_none",
                 "rank_agree_with_clean"]].to_string(index=False))
    out["q4"] = q.to_dict("records")


def q5(out: dict) -> None:
    print()
    print("=" * 78)
    print("Q5  equal-budget review policies")
    print("    rule: two-tailed helps if it recovers more than top-B in >=4 of 5 pools")
    print("=" * 78)
    tt = pd.read_csv(EV / "x_twotailed_policies.csv")
    tt = (tt.groupby(["convention", "scale", "attack", "budget_mult", "policy", "pool"],
                     as_index=False)
            .agg({c: "mean" for c in tt.columns if c not in
                  ("convention", "scale", "attack", "budget_mult", "policy", "pool", "rep")}))
    rows = []
    for (conv, scale, atk, mult), sub in tt.groupby(
            ["convention", "scale", "attack", "budget_mult"]):
        top = sub[sub.policy == "top"].set_index("pool")
        for pol, p in sub.groupby("policy"):
            p = p.set_index("pool")
            common = top.index.intersection(p.index)
            better = int((p.loc[common, "recovered"] > top.loc[common, "recovered"]).sum())
            rows.append({
                "convention": conv, "scale": scale, "attack": atk, "budget_mult": mult,
                "policy": pol, "review_budget": p.loc[common, "review_budget"].mean(),
                "score_auroc": p.loc[common, "score_auroc"].mean(),
                "recovered": p.loc[common, "recovered"].mean(),
                "precision": p.loc[common, "precision"].mean(),
                "recall": p.loc[common, "recall"].mean(),
                "precision_lift": p.loc[common, "precision_lift"].mean(),
                "n_pools_beat_top": better,
                "delete_minus_no_action": p.loc[common, "delete_minus_no_action"].mean(),
                "correct_minus_no_action": p.loc[common, "correct_minus_no_action"].mean(),
            })
    q = pd.DataFrame(rows).sort_values(["convention", "scale", "attack", "budget_mult", "policy"])
    q.to_csv(EV / "q5_twotailed_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(q[q.budget_mult == 1.0][
            ["convention", "scale", "attack", "policy", "review_budget", "score_auroc",
             "recovered", "precision", "precision_lift", "n_pools_beat_top",
             "correct_minus_no_action"]].to_string(index=False))
    out["q5"] = q.to_dict("records")


def main() -> int:
    grid = pd.read_csv(EV / "x_grid_cells.csv")
    print(f"grid cells: {len(grid)}  conventions={sorted(grid.convention.unique())} "
          f"scales={sorted(grid.scale.unique())} dims={sorted(grid.dim.unique())} "
          f"lams={sorted(grid.lam.unique())}")
    nc = int((grid.clean_grad_norm > 1e-4).sum() + (grid.pois_grad_norm > 1e-4).sum())
    print(f"non-converged fits (||grad||>1e-4): {nc} of {2*len(grid)}")
    out: dict = {"non_converged_fits": nc, "n_grid_cells": int(len(grid)), "log2": LOG2}
    q1_q2(grid, out)
    q2b(grid, out)
    q3(grid, out)
    q4(out)
    q5(out)
    (EV / "adjudication_values.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {(EV / 'adjudication_values.json').relative_to(sc.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
