"""Build E4 (remediation) over split seeds 0-19.

The stored `results/remediation_counterfactual.csv` covers a single split and
has no generator in this repository; it is kept only as the seed-0 reproduction
gate below. This script rebuilds E4 over 20 splits under the benchmark
protocol, which is read off `repeated_split_stability.py` rather than invented
here:

    pool          one fixed 300-comparison HH-RLHF helpful-base sample, seed 0
    split         30 percent held out per seed; the holdout is NOT zero-norm
                  filtered, which is the existing convention and the reason
                  8 holdout rows have margin 0 for every theta
    filter        zero-norm rows removed from the training side only
    lambda        1e-4
    attack        margin-targeted, alpha = 1 (lowest reference margin first),
                  selected under a head fit on the filtered original data
    budget        B = int(0.15 * n_filtered)
    ranking       training dynamics, the negative final signed margin, scored
                  under the poisoned refit
    depths        5, 10, 15, 20, 35, 50, plus k = B for that split

Four heads are compared on the identical holdout rows for every (seed, k):

    full clean        fit on all filtered rows with original labels
    no action         fit on all filtered rows with B labels reversed
    remediated        fit on the rows surviving deletion, labels as poisoned
    matched control   fit on the SAME surviving row indices, original labels

The matched control is what separates the cost of having less data from the cost
of the reversals deletion failed to remove. Both are computed from one index set
so the comparison is paired by construction.

Stage 1 is a reproduction gate against the stored seed-0 CSV. If a column that
the paper depends on does not reproduce, the script reports the mismatch and
exits without writing an aggregate, so no paper claim can be built on a protocol
that does not match the recorded one.

Intervals reported are repeated-split stability summaries over 20 splits of one
pool. They are not confidence intervals and must not be described as such.
Nothing in `results/` is written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest

from src.attacks.ambiguity_flip import AmbiguityFlipAttack
from src.data.embed import embed_pairs
from src.data.loaders import load_preference_subset
from src.detectors.training_dynamics import TrainingDynamicsDetector
from src.model.bt_head import BTHead
from src.utils.seed import set_seed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

LAM = 1e-4
SEEDS = range(20)
BASE_DEPTHS = (5, 10, 15, 20, 35, 50)
OUT = REPO / "evidence" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
STORED = REPO / "results" / "remediation_counterfactual.csv"

td_det = TrainingDynamicsDetector()

print("loading the fixed 300-comparison pool")
pairs = load_preference_subset("hh-rlhf-helpful", n=300, seed=0)
real_ds = embed_pairs(pairs, embedder="sentence-transformer")
print(f"pool n={real_ds.n} d={real_ds.d}")


def split(seed):
    """Identical split rule to scripts/repeated_split_stability.py."""
    idx = set_seed(seed).permutation(real_ds.n)
    n_ho = int(real_ds.n * 0.3)
    ho_ds, tr_ds = real_ds.subset(idx[:n_ho]), real_ds.subset(idx[n_ho:])
    zmask = np.linalg.norm(tr_ds.Z, axis=1) <= 1e-8
    trc = tr_ds.subset(np.where(~zmask)[0])
    return ho_ds, trc, int(0.15 * trc.n)


def correct(head, ds):
    """Row-level correctness under the accuracy convention of BTHead.accuracy,
    which is margin > 0. A zero-norm row has margin 0 and counts incorrect."""
    return head.margins(ds) > 0


def plain_logloss(head, ds):
    """Unpenalized mean logistic loss. BTHead.loss adds the ridge penalty,
    which is a property of the fit and not of held-out predictive quality, so
    the penalty is excluded here. Stated in the paper."""
    return float(np.mean(np.logaddexp(0.0, -head.margins(ds))))


def mcnemar(a_correct, b_correct):
    """Exact two-sided McNemar on paired predictions within one holdout.

    b01 = b correct and a incorrect, b10 = a correct and b incorrect, matching
    the orientation recorded in the stored CSV. Pairing is within a single
    split; predictions from different splits are never pooled, because the
    splits share one source pool and their holdouts overlap.
    """
    b01 = int(np.sum(b_correct & ~a_correct))
    b10 = int(np.sum(a_correct & ~b_correct))
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(b10, n, 0.5, alternative="two-sided").pvalue)
    return b01, b10, p


rows = []
t0 = time.time()
for seed in SEEDS:
    ho_ds, trc, budget = split(seed)
    depths = sorted(set(BASE_DEPTHS) | {budget})

    head_clean = BTHead(lam=LAM).fit(trc)                      # full clean
    sel = AmbiguityFlipAttack(alpha=1.0).select_flips(trc, budget, set_seed(seed),
                                                      head=head_clean)
    pois = trc.flipped(sel)
    head_unsafe = BTHead(lam=LAM).fit(pois)                    # no action
    scores = td_det.score(pois, head_unsafe)

    c_clean, c_unsafe = correct(head_clean, ho_ds), correct(head_unsafe, ho_ds)
    for k in depths:
        kk = min(k, pois.n)
        flagged = np.argsort(-scores)[:kk]
        keep = np.setdiff1d(np.arange(pois.n), flagged)
        head_rem = BTHead(lam=LAM).fit(pois.subset(keep))      # remediated
        head_mat = BTHead(lam=LAM).fit(trc.subset(keep))       # matched control

        c_rem, c_mat = correct(head_rem, ho_ds), correct(head_mat, ho_ds)
        removed_poison = int(pois.flip_mask[flagged].sum())
        b01, b10, p = mcnemar(c_rem, c_mat)
        mr, mm = head_rem.margins(ho_ds), head_mat.margins(ho_ds)
        rows.append({
            "seed": seed, "k": k, "k_applied": kk, "is_budget_depth": k == budget,
            "n_train": trc.n, "n_holdout": ho_ds.n, "budget": budget,
            "n_retained": len(keep),
            "poisoned_removed": removed_poison, "clean_removed": kk - removed_poison,
            "poisoned_surviving": budget - removed_poison,
            "precision_of_removal": removed_poison / kk,
            "acc_full_clean": float(c_clean.mean()), "acc_unsafe": float(c_unsafe.mean()),
            "acc_remediated": float(c_rem.mean()), "acc_matched_clean_cf": float(c_mat.mean()),
            "logloss_full_clean": plain_logloss(head_clean, ho_ds),
            "logloss_unsafe": plain_logloss(head_unsafe, ho_ds),
            "logloss_remediated": plain_logloss(head_rem, ho_ds),
            "logloss_matched_clean_cf": plain_logloss(head_mat, ho_ds),
            "acc_gap_remed_minus_matched": float(c_rem.mean() - c_mat.mean()),
            "acc_gap_remed_minus_unsafe": float(c_rem.mean() - c_unsafe.mean()),
            "composition_cost": float(c_clean.mean() - c_mat.mean()),
            "pred_disagree_vs_matched": int(np.sum(c_rem != c_mat)),
            "pred_disagree_vs_unsafe": int(np.sum(c_rem != c_unsafe)),
            "pred_disagree_vs_oracle": int(np.sum(c_rem != c_clean)),
            "mcnemar_b01": b01, "mcnemar_b10": b10,
            "mcnemar_discordant_b01_b10": f"{b01}/{b10}", "mcnemar_p": p,
            "theta_dist_to_matched": float(np.linalg.norm(head_rem.theta - head_mat.theta)),
            "theta_dist_to_full_oracle": float(np.linalg.norm(head_rem.theta - head_clean.theta)),
            "theta_dist_matched_to_full_oracle": float(
                np.linalg.norm(head_mat.theta - head_clean.theta)),
            "holdout_margin_L2_to_matched": float(np.linalg.norm(mr - mm)),
            "holdout_margin_maxabs_to_matched": float(np.max(np.abs(mr - mm))),
            "holdout_prob_L1_to_matched": float(np.mean(np.abs(expit(mr) - expit(mm)))),
        })
    print(f"seed {seed:2d} ok  n={trc.n:3d} B={budget:2d} depths={depths}", flush=True)

per = pd.DataFrame(rows)
print(f"\n{len(per)} (seed, k) cells in {time.time() - t0:.1f}s")

# ===================== stage 1: reproduction gate, seed 0 ====================
print("\n=== reproduction gate against results/remediation_counterfactual.csv (seed 0) ===")
stored = pd.read_csv(STORED)
mine0 = per[per.seed == 0].set_index("k")
gate_rows, hard_fail = [], []

# Columns the paper depends on must match exactly. The auxiliary geometry
# columns are checked too but their definitions were not recorded anywhere, so
# a mismatch there is reported without blocking.
PAPER_COLS = ["acc_unsafe", "acc_remediated", "acc_matched_clean_cf",
              "poisoned_removed", "clean_removed", "precision_of_removal",
              "pred_disagree_vs_matched", "pred_disagree_vs_unsafe",
              "pred_disagree_vs_oracle", "mcnemar_discordant_b01_b10", "mcnemar_p"]
AUX_COLS = ["theta_dist_to_matched", "theta_dist_to_full_oracle",
            "theta_dist_matched_to_full_oracle", "holdout_margin_L2_to_matched",
            "holdout_margin_maxabs_to_matched", "holdout_prob_L1_to_matched",
            "acc_gap_remed_minus_matched"]
RENAME = {"acc_full_clean": "acc_full_clean_oracle"}

for _, sr in stored.iterrows():
    k = int(sr.k)
    if k not in mine0.index:
        hard_fail.append(f"k={k} present in the stored CSV but not recomputed")
        continue
    mr_ = mine0.loc[k]
    for col in PAPER_COLS + AUX_COLS + ["acc_full_clean_oracle"]:
        mycol = {v: kk for kk, v in RENAME.items()}.get(col, col)
        if col not in stored.columns or mycol not in mine0.columns:
            continue
        sv, mv = sr[col], mr_[mycol]
        if isinstance(sv, str) or isinstance(mv, str):
            match, diff = str(sv) == str(mv), None
        else:
            diff = abs(float(sv) - float(mv))
            match = diff <= 1e-9
        tier = "paper" if col in PAPER_COLS or col == "acc_full_clean_oracle" else "auxiliary"
        gate_rows.append({"k": k, "column": col, "tier": tier, "stored": sv,
                          "recomputed": mv, "abs_diff": diff, "match": bool(match)})
        if not match and tier == "paper":
            hard_fail.append(f"k={k} {col}: stored {sv} vs recomputed {mv}")

gates = pd.DataFrame(gate_rows)
gates.to_csv(OUT / "e4_reproduction_gate.csv", index=False)
for tier in ("paper", "auxiliary"):
    g = gates[gates.tier == tier]
    print(f"  {tier:10s}: {int(g.match.sum())}/{len(g)} columns match")
    for _, r in g[~g.match].iterrows():
        print(f"     k={int(r.k):2d} {r.column}: stored {r.stored} vs recomputed {r.recomputed}")

if hard_fail:
    print("\nREPRODUCTION FAILED on paper-relevant columns:")
    for h in hard_fail:
        print("  -", h)
    sys.exit("ABORT: seed 0 did not reproduce; no aggregate written and no paper claim built")
print("\n  seed 0 reproduces on every paper-relevant column")

# ============================ stage 2: aggregate =============================
per.to_csv(OUT / "e4_per_seed_20splits.csv", index=False)

VAL = ["acc_remediated", "acc_matched_clean_cf", "acc_unsafe", "acc_full_clean",
       "logloss_remediated", "logloss_matched_clean_cf", "logloss_unsafe",
       "logloss_full_clean", "precision_of_removal", "poisoned_removed",
       "poisoned_surviving", "acc_gap_remed_minus_matched",
       "acc_gap_remed_minus_unsafe", "composition_cost", "pred_disagree_vs_matched",
       "mcnemar_p"]
agg = per.groupby("k", as_index=False).agg(
    n_seeds=("seed", "nunique"),
    **{f"{c}_{s}": (c, s) for c in VAL for s in ("mean", "std")})
# direction counts, which is what a repeated-split design can actually support
d = per.groupby("k")
agg["remed_better_than_unsafe"] = d.apply(
    lambda g: int((g.acc_remediated > g.acc_unsafe).sum()), include_groups=False).values
agg["remed_worse_than_unsafe"] = d.apply(
    lambda g: int((g.acc_remediated < g.acc_unsafe).sum()), include_groups=False).values
agg["remed_better_than_matched"] = d.apply(
    lambda g: int((g.acc_remediated > g.acc_matched_clean_cf).sum()), include_groups=False).values
agg["remed_worse_than_matched"] = d.apply(
    lambda g: int((g.acc_remediated < g.acc_matched_clean_cf).sum()), include_groups=False).values
agg["mcnemar_p_below_05"] = d.apply(
    lambda g: int((g.mcnemar_p < 0.05).sum()), include_groups=False).values
agg.to_csv(OUT / "e4_summary_20splits.csv", index=False)

# the budget-matched depth, which differs per split, aggregated on its own
bud = per[per.is_budget_depth]
budget_row = {
    "n_seeds": int(bud.seed.nunique()),
    "budget_min": int(bud.budget.min()), "budget_max": int(bud.budget.max()),
    **{f"{c}_{s}": float(getattr(bud[c], s)()) for c in VAL for s in ("mean", "std")},
    "remed_better_than_unsafe": int((bud.acc_remediated > bud.acc_unsafe).sum()),
    "remed_worse_than_unsafe": int((bud.acc_remediated < bud.acc_unsafe).sum()),
    "remed_better_than_matched": int((bud.acc_remediated > bud.acc_matched_clean_cf).sum()),
    "remed_worse_than_matched": int((bud.acc_remediated < bud.acc_matched_clean_cf).sum()),
    "mcnemar_p_below_05": int((bud.mcnemar_p < 0.05).sum()),
}
pd.DataFrame([budget_row]).to_csv(OUT / "e4_budget_depth_20splits.csv", index=False)

meta = {
    "label": "DRAFT2_E4_REBUILD",
    "protocol_source": "scripts/repeated_split_stability.py split rule and budget; "
                       "top-k deletion by the final-margin score under the poisoned refit",
    "seeds": list(SEEDS), "lam": LAM, "depths_base": list(BASE_DEPTHS),
    "depths_note": "k = B for that split is added, so the grid has 7 depths per seed",
    "holdout_filtered": False,
    "holdout_note": "the holdout is not zero-norm filtered, matching the existing "
                    "convention; 8 such rows have margin 0 for every theta and are "
                    "always counted incorrect",
    "logloss_definition": "unpenalized mean logistic loss on the holdout",
    "mcnemar": "exact two-sided binomial on paired holdout predictions within a "
               "single split; splits are never pooled",
    "interval_type": "repeated-split stability summary over 20 splits of one pool, "
                     "NOT a confidence interval",
    "seed0_reproduces_paper_columns": True,
    "writes_to_results": False,
}
(OUT / "e4_rebuild_meta.json").write_text(json.dumps(meta, indent=2))

print("\n=== E4 over 20 splits, by deletion depth ===")
show = ["k", "n_seeds", "acc_remediated_mean", "acc_matched_clean_cf_mean",
        "acc_unsafe_mean", "acc_full_clean_mean", "remed_better_than_unsafe",
        "remed_worse_than_unsafe", "precision_of_removal_mean", "mcnemar_p_below_05"]
print(agg[show].round(4).to_string(index=False))
print("\n=== at the budget-matched depth (k = B, 26 or 27) ===")
for key in ("acc_remediated_mean", "acc_matched_clean_cf_mean", "acc_unsafe_mean",
            "acc_full_clean_mean", "composition_cost_mean",
            "acc_gap_remed_minus_matched_mean", "precision_of_removal_mean",
            "poisoned_surviving_mean"):
    print(f"  {key:38s} {budget_row[key]:+.4f}")
print(f"  remediated better than no action  {budget_row['remed_better_than_unsafe']}/20 runs")
print(f"  remediated worse  than no action  {budget_row['remed_worse_than_unsafe']}/20 runs")
print(f"  remediated worse  than matched    {budget_row['remed_worse_than_matched']}/20 runs")
print(f"  McNemar p < 0.05                  {budget_row['mcnemar_p_below_05']}/20 runs")
print(f"\nwrote 5 files to {OUT.relative_to(REPO)}/")
