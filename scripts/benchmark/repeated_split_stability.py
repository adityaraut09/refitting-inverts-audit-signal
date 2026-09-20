"""Repeated-split stability for the real-data experiments.

Re-runs E1, E3 and the gradient_flip three-stage diagnostic across split
seeds 0..19, holding the 300-example source pool, lam=1e-4, the filtering
rule, budget = int(0.15 * filtered_n), the detector definitions, the
corrected cross_fit configuration and the attack definitions all fixed.

The runs reuse one 300-example pool, so the intervals reported here are
repeated-split stability intervals, not independent population confidence
intervals.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from src.attacks.ambiguity_flip import AmbiguityFlipAttack
from src.attacks.gradient_flip import GradientFlipAttack
from src.attacks.random_flip import RandomFlipAttack
from src.data.embed import embed_pairs
from src.data.loaders import load_preference_subset
from src.detectors.cross_fit import CrossFitDetector
from src.detectors.ensemble import EnsembleDetector
from src.detectors.self_influence import SelfInfluenceDetector
from src.detectors.training_dynamics import TrainingDynamicsDetector
from src.metrics.detection import auroc, precision_at_k
from src.model.bt_head import BTHead
from src.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[2]

LAM = 1e-4
N_FOLDS = 5
SEEDS = range(20)
RESULTS = REPO / "results"

si_det = SelfInfluenceDetector()
td_det = TrainingDynamicsDetector()
cf_det = CrossFitDetector(lam=LAM)          # corrected configuration

pairs = load_preference_subset("hh-rlhf-helpful", n=300, seed=0)
real_ds = embed_pairs(pairs, embedder="sentence-transformer")   # fixed pool


def cross_fit_scores(fit_ds, score_ds):
    """CrossFitDetector.score with the fitting and scoring datasets separable.
    Identical folds, lam, model and 1 - sigma(m) score."""
    n = score_ds.n
    folds = np.arange(n) % N_FOLDS
    s = np.zeros(n)
    for f in range(N_FOLDS):
        te, tr_ = np.where(folds == f)[0], np.where(folds != f)[0]
        fh = BTHead(lam=LAM).fit(fit_ds.subset(tr_))
        t = score_ds.subset(te)
        s[te] = 1 - expit(t.s * (t.Z @ fh.theta))
    return s


def ens(components):
    return np.mean([rankdata(c) for c in components], axis=0)


e1_rows, e3_rows, stage_rows, cohort_rows, meta_rows = [], [], [], [], []
completed, failed = [], []
t_start = time.time()

for seed in SEEDS:
    t0 = time.time()
    try:
        idx = set_seed(seed).permutation(real_ds.n)
        n_ho = int(real_ds.n * 0.3)
        ho_ds, tr_ds = real_ds.subset(idx[:n_ho]), real_ds.subset(idx[n_ho:])
        zmask = np.linalg.norm(tr_ds.Z, axis=1) <= 1e-8
        trc = tr_ds.subset(np.where(~zmask)[0])
        budget = int(0.15 * trc.n)
        head_clean = BTHead(lam=LAM).fit(trc)

        meta_rows.append({"seed": seed, "n_pre_filter": tr_ds.n,
                          "zero_norm_removed": int(zmask.sum()),
                          "n_filtered": trc.n, "budget": budget})

        # ---- E1 -----------------------------------------------------------
        attacks = {
            "random_flip": RandomFlipAttack().select_flips(trc, budget, set_seed(seed)),
            "gradient_flip": GradientFlipAttack().select_flips(trc, budget, set_seed(seed)),
            "ambiguity_flip": AmbiguityFlipAttack(alpha=1.0).select_flips(
                trc, budget, set_seed(seed), head=head_clean),
        }
        poisoned_sets = {}
        for aname, sel in attacks.items():
            p = trc.flipped(sel)
            h = BTHead(lam=LAM).fit(p)
            poisoned_sets[aname] = (p, h, sel)
            comp = {"self_influence": si_det.score(p, h),
                    "cross_fit": cross_fit_scores(p, p),
                    "training_dynamics": td_det.score(p, h)}
            comp["ensemble"] = ens([comp["self_influence"], comp["cross_fit"],
                                    comp["training_dynamics"]])
            if seed == 0 and aname == "random_flip":   # one-off agreement check
                prod = EnsembleDetector([si_det, cf_det, td_det]).score(p, h)
                assert np.allclose(comp["ensemble"], prod, atol=1e-12), "ensemble mismatch"
                assert np.allclose(comp["cross_fit"], cf_det.score(p), atol=1e-12), "cross_fit mismatch"
            for dname, s in comp.items():
                e1_rows.append({"seed": seed, "attack": aname, "detector": dname,
                                "auroc": auroc(s, p.flip_mask),
                                "precision_at_budget": precision_at_k(s, p.flip_mask, budget)})

        # ---- E3 -----------------------------------------------------------
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sel_a = AmbiguityFlipAttack(alpha=a).select_flips(
                trc, budget, set_seed(seed), head=head_clean)
            pa = trc.flipped(sel_a)
            ha = BTHead(lam=LAM).fit(pa)
            for dname, s in [("self_influence", si_det.score(pa, ha)),
                             ("cross_fit", cross_fit_scores(pa, pa)),
                             ("training_dynamics", td_det.score(pa, ha))]:
                e3_rows.append({"seed": seed, "alpha": a, "detector": dname,
                                "auroc": auroc(s, pa.flip_mask)})

        # ---- gradient_flip three stages ------------------------------------
        p, hp, sel = poisoned_sets["gradient_flip"]
        cohort = np.zeros(trc.n, dtype=bool); cohort[sel] = True

        def decomp(dset, head):
            head._hinv_cache = None
            m = head.margins(dset)
            dis = expit(-m) ** 2
            lev = np.einsum("ij,jk,ik->i", dset.Z, head.hinv(dset), dset.Z)
            head._hinv_cache = None
            return m, dis, lev, dis * lev

        m0, d0, l0, s0 = decomp(trc, head_clean)
        m1, d1, l1, s1 = decomp(p, head_clean)
        m2, d2, l2, s2 = decomp(p, hp)
        stages = {
            "stage0_pre_attack": {"self_influence": s0, "training_dynamics": td_det.score(trc, head_clean),
                                  "cross_fit": cross_fit_scores(trc, trc)},
            "stage1_flipped_clean_models": {"self_influence": s1, "training_dynamics": td_det.score(p, head_clean),
                                            "cross_fit": cross_fit_scores(trc, p)},
            "stage2_poisoned_refit": {"self_influence": s2, "training_dynamics": td_det.score(p, hp),
                                      "cross_fit": cross_fit_scores(p, p)},
        }
        for st, comp in stages.items():
            comp["ensemble"] = ens([comp["self_influence"], comp["cross_fit"], comp["training_dynamics"]])
            for dname, s in comp.items():
                stage_rows.append({"seed": seed, "stage": st, "detector": dname,
                                   "auroc": auroc(s, cohort),
                                   "precision_at_budget": precision_at_k(s, cohort, budget),
                                   "mean_flipped": float(s[cohort].mean()),
                                   "mean_unflipped": float(s[~cohort].mean())})
        for st, (m, dd) in {"stage0_pre_attack": (m0, d0), "stage1_flipped_clean_models": (m1, d1),
                            "stage2_poisoned_refit": (m2, d2)}.items():
            cohort_rows.append({"seed": seed, "stage": st,
                                "mean_signed_margin_flipped": float(m[cohort].mean()),
                                "mean_signed_margin_unflipped": float(m[~cohort].mean()),
                                "median_signed_margin_flipped": float(np.median(m[cohort])),
                                "median_signed_margin_unflipped": float(np.median(m[~cohort])),
                                "mean_disagreement_flipped": float(dd[cohort].mean()),
                                "mean_disagreement_unflipped": float(dd[~cohort].mean())})
        completed.append(seed)
        print(f"seed {seed:2d} ok  n={trc.n:3d} budget={budget:2d}  ({time.time()-t0:.1f}s)", flush=True)
    except Exception as exc:                                  # preserve partial work
        failed.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        print(f"seed {seed:2d} FAILED: {type(exc).__name__}: {exc}", flush=True)

print(f"\ncompleted {len(completed)}/{len(list(SEEDS))} seeds in {time.time()-t_start:.1f}s")
if failed:
    print("failures:", failed)

e1 = pd.DataFrame(e1_rows); e3 = pd.DataFrame(e3_rows)
stg = pd.DataFrame(stage_rows); coh = pd.DataFrame(cohort_rows); mta = pd.DataFrame(meta_rows)
for df, name in [(e1, "e1"), (e3, "e3"), (stg, "stages"), (coh, "cohort"), (mta, "meta")]:
    df.to_csv(RESULTS / f"repeated_split_{name}.csv", index=False)


def agg(df, keys, valcols):
    out = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        for v in valcols:
            x = g[v].to_numpy(dtype=float)
            row = dict(zip(keys, k))
            row.update({"metric": v, "n_seeds": len(x), "mean": x.mean(),
                        "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                        "median": float(np.median(x)), "min": x.min(), "max": x.max(),
                        "p2_5": float(np.quantile(x, 0.025)), "p97_5": float(np.quantile(x, 0.975))})
            out.append(row)
    return pd.DataFrame(out)


summ = pd.concat([
    agg(e1, ["attack", "detector"], ["auroc", "precision_at_budget"]).assign(block="E1"),
    agg(e3, ["alpha", "detector"], ["auroc"]).assign(block="E3"),
    agg(stg, ["stage", "detector"], ["auroc", "precision_at_budget", "mean_flipped", "mean_unflipped"]).assign(block="stages"),
    agg(coh, ["stage"], ["mean_signed_margin_flipped", "mean_signed_margin_unflipped",
                         "mean_disagreement_flipped", "mean_disagreement_unflipped"]).assign(block="cohort"),
], ignore_index=True)
summ["interval_type"] = "repeated-split stability interval (same 300-example pool, not an independent CI)"
summ.to_csv(RESULTS / "repeated_split_summary.csv", index=False)

# per-seed E3 trend direction, reported per seed rather than off the mean curve
tr_rows = []
for (sd, det), g in e3.groupby(["seed", "detector"]):
    g = g.sort_values("alpha")
    rho = spearmanr(g.alpha, g.auroc).statistic
    tr_rows.append({"seed": sd, "detector": det, "spearman_alpha_vs_auroc": rho,
                    "direction": "increasing" if rho > 0 else ("decreasing" if rho < 0 else "flat"),
                    "auroc_at_alpha0": float(g[g.alpha == 0.0].auroc.iloc[0]),
                    "auroc_at_alpha1": float(g[g.alpha == 1.0].auroc.iloc[0])})
trend = pd.DataFrame(tr_rows)
trend.to_csv(RESULTS / "repeated_split_e3_trend.csv", index=False)
print("\nE3 per-seed trend direction counts:")
print(trend.groupby(["detector", "direction"]).size().to_string())
print("\nwrote repeated_split_{e1,e3,stages,cohort,meta,summary,e3_trend}.csv")
