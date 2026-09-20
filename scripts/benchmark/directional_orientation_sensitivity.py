"""Is the directional stress test's conclusion orientation-dependent?

`GradientFlipAttack` defaults its target direction to the top eigenvector of
`Z^T Z / n`, taken as `np.linalg.eigh(cov)[1][:, -1]`. An eigenvector is defined
only up to sign, and the attack selects the rows maximising `s_i <z_i, v>`, so
replacing `v` with `-v` selects the opposite tail of the alignment distribution.
The headline result therefore rests on an attack whose orientation was never
specified.

This script answers that, read-only with respect to `results/`. Three arms
over the same 20 Family-A splits, at the identical budget:

    raw        v = eigh(cov)[1][:, -1] exactly as the stored runs used it
    canon+     v with its largest-|.| coordinate forced positive
    canon-     the negation of canon+

The canonical rule, stated exactly: let j = argmax_i |v_i|, breaking ties by the
smallest index (numpy's argmax convention). If v_j < 0, replace v by -v. The
result is invariant to the sign an eigensolver happens to return, and is
deterministic given cov. It is not invariant to a change of basis, and it says
nothing about which tail is "the" attack; it only fixes one.

Arm `raw` doubles as a reproduction gate against the stored
`repeated_split_e1.csv` and `repeated_split_stages.csv` gradient_flip rows. If it
does not reproduce, nothing else is interpreted.

Recorded per split and arm: selected row indices, Jaccard against the other
arms, the two leading eigenvalues and the relative eigengap, poisoned-head
held-out accuracy and loss, parameter displacement, reversed/retained signed
margins at all three stages, AUROC and precision at budget for all four scores
at all three stages, and the disagreement and leverage summaries needed to test
the absorption interpretation.

Everything reported is a repeated-split stability summary over one 300-comparison
pool. None of it is a population confidence interval.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from src.attacks.gradient_flip import GradientFlipAttack
from src.data.embed import embed_pairs
from src.data.loaders import load_preference_subset
from src.detectors.self_influence import SelfInfluenceDetector
from src.detectors.training_dynamics import TrainingDynamicsDetector
from src.metrics.detection import auroc, precision_at_k
from src.model.bt_head import BTHead
from src.utils.seed import set_seed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

LAM = 1e-4
N_FOLDS = 5
SEEDS = range(20)
DETS = ("self_influence", "cross_fit", "training_dynamics", "ensemble")
STAGES = ("pre_attack", "clean_head", "poisoned_refit")
OUT = REPO / "evidence" / "benchmark" / "directional_orientation"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results"

si_det = SelfInfluenceDetector()
td_det = TrainingDynamicsDetector()

print("loading the fixed 300-comparison pool")
pairs = load_preference_subset("hh-rlhf-helpful", n=300, seed=0)
real_ds = embed_pairs(pairs, embedder="sentence-transformer")


def canonical(v: np.ndarray) -> np.ndarray:
    """Force the largest-magnitude coordinate positive. Ties go to the smallest
    index, which is numpy's argmax convention."""
    j = int(np.argmax(np.abs(v)))
    return -v if v[j] < 0 else v.copy()


def split(seed):
    idx = set_seed(seed).permutation(real_ds.n)
    n_ho = int(real_ds.n * 0.3)
    ho_ds, tr_ds = real_ds.subset(idx[:n_ho]), real_ds.subset(idx[n_ho:])
    zmask = np.linalg.norm(tr_ds.Z, axis=1) <= 1e-8
    trc = tr_ds.subset(np.where(~zmask)[0])
    return ho_ds, trc, int(0.15 * trc.n)


def cross_fit_scores(fit_ds, score_ds):
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


def decomp(dset, head):
    head._hinv_cache = None
    m = head.margins(dset)
    dis = expit(-m) ** 2
    lev = np.einsum("ij,jk,ik->i", dset.Z, head.hinv(dset), dset.Z)
    head._hinv_cache = None
    return m, dis, lev, dis * lev


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


rows, sel_rows = [], []
t0 = time.time()
for seed in SEEDS:
    ho_ds, trc, budget = split(seed)
    head_clean = BTHead(lam=LAM).fit(trc)

    cov = trc.Z.T @ trc.Z / trc.n
    eigvals, eigvecs = np.linalg.eigh(cov)
    v_raw = eigvecs[:, -1]
    v_plus = canonical(v_raw)
    v_minus = -v_plus
    lam1, lam2 = float(eigvals[-1]), float(eigvals[-2])
    # which orientation did the stored runs actually get?
    raw_is_plus = bool(np.allclose(v_raw, v_plus))
    jmax = int(np.argmax(np.abs(v_raw)))

    arms = {"raw": v_raw, "canon_plus": v_plus, "canon_minus": v_minus}
    sels = {}
    for arm, v in arms.items():
        sel = GradientFlipAttack(target_dir=v).select_flips(trc, budget, set_seed(seed))
        sels[arm] = np.sort(sel)
        p = trc.flipped(sel)
        h = BTHead(lam=LAM).fit(p)
        fm = p.flip_mask

        comp = {
            "pre_attack": {"self_influence": decomp(trc, head_clean)[3],
                           "training_dynamics": td_det.score(trc, head_clean),
                           "cross_fit": cross_fit_scores(trc, trc)},
            "clean_head": {"self_influence": decomp(p, head_clean)[3],
                           "training_dynamics": td_det.score(p, head_clean),
                           "cross_fit": cross_fit_scores(trc, p)},
            "poisoned_refit": {"self_influence": decomp(p, h)[3],
                               "training_dynamics": td_det.score(p, h),
                               "cross_fit": cross_fit_scores(p, p)},
        }
        row = {
            "seed": seed, "arm": arm, "n_train": trc.n, "budget": budget,
            "eig1": lam1, "eig2": lam2,
            "relative_eigengap": (lam1 - lam2) / lam1,
            "raw_equals_canon_plus": raw_is_plus,
            "argmax_abs_coord": jmax,
            "v_coord_at_argmax": float(v[jmax]),
            "acc_poisoned": h.accuracy(ho_ds),
            "logloss_poisoned": float(np.mean(np.logaddexp(0.0, -h.margins(ho_ds)))),
            "acc_clean": head_clean.accuracy(ho_ds),
            "logloss_clean": float(np.mean(np.logaddexp(0.0, -head_clean.margins(ho_ds)))),
            "theta_displacement": float(np.linalg.norm(h.theta - head_clean.theta)),
            "mean_alignment_selected": float((trc.s * (trc.Z @ v))[sel].mean()),
        }
        for st in STAGES:
            c = comp[st]
            c["ensemble"] = ens([c["self_influence"], c["cross_fit"], c["training_dynamics"]])
            for dd in DETS:
                row[f"auroc_{st}__{dd}"] = auroc(c[dd], fm)
                row[f"prec_{st}__{dd}"] = precision_at_k(c[dd], fm, budget)
        for st, dset, hh in (("pre_attack", trc, head_clean), ("clean_head", p, head_clean),
                             ("poisoned_refit", p, h)):
            m, dis, lev, sip = decomp(dset, hh)
            row[f"margin_rev_{st}"] = float(m[fm].mean())
            row[f"margin_ret_{st}"] = float(m[~fm].mean())
            row[f"disagree_rev_{st}"] = float(dis[fm].mean())
            row[f"disagree_ret_{st}"] = float(dis[~fm].mean())
            row[f"leverage_rev_{st}"] = float(lev[fm].mean())
            row[f"leverage_ret_{st}"] = float(lev[~fm].mean())
            row[f"selfinf_rev_{st}"] = float(sip[fm].mean())
            row[f"selfinf_ret_{st}"] = float(sip[~fm].mean())
        rows.append(row)

    sel_rows.append({
        "seed": seed, "budget": budget,
        "jaccard_raw_vs_plus": jaccard(sels["raw"], sels["canon_plus"]),
        "jaccard_raw_vs_minus": jaccard(sels["raw"], sels["canon_minus"]),
        "jaccard_plus_vs_minus": jaccard(sels["canon_plus"], sels["canon_minus"]),
        "n_overlap_plus_minus": len(set(sels["canon_plus"].tolist())
                                    & set(sels["canon_minus"].tolist())),
        "sel_canon_plus": " ".join(map(str, sels["canon_plus"])),
        "sel_canon_minus": " ".join(map(str, sels["canon_minus"])),
        "sel_raw": " ".join(map(str, sels["raw"])),
    })
    print(f"seed {seed:2d} n={trc.n:3d} B={budget:2d} raw=canon+ {raw_is_plus} "
          f"gap={(lam1-lam2)/lam1:.3f} J(+,-)={sel_rows[-1]['jaccard_plus_vs_minus']:.3f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

per = pd.DataFrame(rows)
sel = pd.DataFrame(sel_rows)

# ===================== reproduction gate: raw arm vs stored =====================
print("\n=== reproduction gate: raw arm against the stored gradient_flip results ===")
stored_e1 = pd.read_csv(RESULTS / "repeated_split_e1.csv")
stored_e1 = stored_e1[stored_e1.attack == "gradient_flip"]
stored_stg = pd.read_csv(RESULTS / "repeated_split_stages.csv")
raw = per[per.arm == "raw"].set_index("seed")
gate, fails = [], []
SMAP = {"pre_attack": "stage0_pre_attack", "clean_head": "stage1_flipped_clean_models",
        "poisoned_refit": "stage2_poisoned_refit"}
for _, r in stored_e1.iterrows():
    got = raw.loc[r.seed, f"auroc_poisoned_refit__{r.detector}"]
    d = abs(got - r.auroc)
    gate.append({"quantity": f"E1 seed{r.seed} {r.detector} auroc", "stored": r.auroc,
                 "raw_arm": got, "abs_diff": d, "match": d <= 1e-9})
    if d > 1e-9:
        fails.append(gate[-1]["quantity"])
for _, r in stored_stg.iterrows():
    st = [k for k, v in SMAP.items() if v == r.stage][0]
    for col, want in (("auroc", r.auroc), ("prec", r.precision_at_budget)):
        got = raw.loc[r.seed, f"{col}_{st}__{r.detector}"]
        d = abs(got - want)
        gate.append({"quantity": f"stages seed{r.seed} {r.stage} {r.detector} {col}",
                     "stored": want, "raw_arm": got, "abs_diff": d, "match": d <= 1e-9})
        if d > 1e-9:
            fails.append(gate[-1]["quantity"])
g = pd.DataFrame(gate)
g.to_csv(OUT / "raw_arm_reproduction_gate.csv", index=False)
print(f"  {int(g.match.sum())}/{len(g)} quantities match the stored runs")
for q in fails[:10]:
    print("   FAIL:", q)
if fails:
    sys.exit(f"ABORT: raw arm does not reproduce ({len(fails)} mismatches); nothing interpreted")

per.to_csv(OUT / "orientation_per_split.csv", index=False)
sel.to_csv(OUT / "orientation_selected_rows.csv", index=False)

# =============================== aggregate ===============================
KEY = ([c for c in per.columns if c.startswith(("auroc_", "prec_", "margin_", "disagree_",
                                                "leverage_", "selfinf_"))]
       + ["acc_poisoned", "logloss_poisoned", "theta_displacement",
          "mean_alignment_selected", "relative_eigengap", "eig1", "eig2"])
agg = per.groupby("arm", as_index=False).agg(
    n_seeds=("seed", "size"), **{f"{c}_{s}": (c, s) for c in KEY for s in ("mean", "std")})
agg.to_csv(OUT / "orientation_summary.csv", index=False)

# direction counts: does each score sit below chance under the poisoned refit?
dc = []
for arm, gg in per.groupby("arm"):
    for dd in DETS:
        col = f"auroc_poisoned_refit__{dd}"
        dc.append({"arm": arm, "detector": dd, "mean_auroc": gg[col].mean(),
                   "sd": gg[col].std(),
                   "below_chance_splits": int((gg[col] < 0.5).sum()), "n": len(gg),
                   "mean_prec": gg[f"prec_poisoned_refit__{dd}"].mean(),
                   "clean_head_auroc": gg[f"auroc_clean_head__{dd}"].mean(),
                   "inversion_margin": gg[f"auroc_clean_head__{dd}"].mean() - gg[col].mean()})
counts = pd.DataFrame(dc)
counts.to_csv(OUT / "orientation_direction_counts.csv", index=False)

# absorption: do reversed rows end up with LARGER positive margins after refit?
ab = []
for arm, gg in per.groupby("arm"):
    d = gg.margin_rev_poisoned_refit - gg.margin_ret_poisoned_refit
    ab.append({"arm": arm, "margin_rev": gg.margin_rev_poisoned_refit.mean(),
               "margin_ret": gg.margin_ret_poisoned_refit.mean(),
               "margin_gap_mean": d.mean(), "margin_gap_sd": d.std(),
               "absorbed_splits": int((d > 0).sum()), "n": len(gg),
               "disagreement_ratio": (gg.disagree_rev_poisoned_refit
                                      / gg.disagree_ret_poisoned_refit).mean(),
               "leverage_ratio": (gg.leverage_rev_poisoned_refit
                                  / gg.leverage_ret_poisoned_refit).mean(),
               "selfinf_ratio": (gg.selfinf_rev_poisoned_refit
                                 / gg.selfinf_ret_poisoned_refit).mean()})
absb = pd.DataFrame(ab)
absb.to_csv(OUT / "orientation_absorption.csv", index=False)

meta = {
    "label": "DRAFT3_ORIENTATION_AUDIT",
    "canonical_rule": ("j = argmax_i |v_i| with numpy's smallest-index tie-break; "
                       "if v_j < 0 then v <- -v"),
    "arms": {"raw": "np.linalg.eigh(cov)[1][:, -1], the orientation the stored runs used",
             "canon_plus": "canonical(raw)", "canon_minus": "-canonical(raw)"},
    "raw_equals_canon_plus_in_splits": int(per[per.arm == "raw"].raw_equals_canon_plus.sum()),
    "n_splits": len(list(SEEDS)), "lam": LAM,
    "budget_rule": "int(0.15 * n_filtered)",
    "interval_type": ("repeated-split stability summary over one 300-comparison pool; "
                      "not a population confidence interval"),
    "reproduction_gate": {"n": len(g), "n_pass": int(g.match.sum())},
    "writes_to_results": False,
}
(OUT / "orientation_meta.json").write_text(json.dumps(meta, indent=2))

print("\n=== eigenstructure ===")
e = per[per.arm == "raw"]
print(f"  lambda1 {e.eig1.mean():.5f} (sd {e.eig1.std():.5f})")
print(f"  lambda2 {e.eig2.mean():.5f} (sd {e.eig2.std():.5f})")
print(f"  relative eigengap {e.relative_eigengap.mean():.4f} "
      f"(sd {e.relative_eigengap.std():.4f}, min {e.relative_eigengap.min():.4f}, "
      f"max {e.relative_eigengap.max():.4f})")
print(f"  raw orientation equals canonical + in {int(e.raw_equals_canon_plus.sum())} of {len(e)} splits")
print("\n=== selection overlap ===")
print(f"  Jaccard canon+ vs canon-  mean {sel.jaccard_plus_vs_minus.mean():.4f} "
      f"(max {sel.jaccard_plus_vs_minus.max():.4f})")
print(f"  Jaccard raw vs canon+     mean {sel.jaccard_raw_vs_plus.mean():.4f}")
print(f"  Jaccard raw vs canon-     mean {sel.jaccard_raw_vs_minus.mean():.4f}")
print("\n=== poisoned-refit AUROC by arm ===")
print(counts.to_string(index=False))
print("\n=== absorption by arm ===")
print(absb.to_string(index=False))
print(f"\nwrote 7 files to {OUT.relative_to(REPO)}/")
