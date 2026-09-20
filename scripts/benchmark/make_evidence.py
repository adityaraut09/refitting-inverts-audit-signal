"""Derived benchmark evidence: the budget sweep and the absorption index.

Two jobs, both scoped so that no new methodological choice is introduced.

(1) E2, the budget sweep, over the grid {0.001, 0.005, 0.01, 0.02, 0.03, 0.05}
    with `RandomFlipAttack`, `set_seed(0)`, `lam=1e-4`, the `training_dynamics`
    score, AUROC against the flip mask, and held-out accuracy on the unfiltered
    90-row holdout. Run first at seed 0 as the pre-existing protocol, then over
    split seeds 0..19 using the same seed set and split rule as
    `repeated_split_stability.py`. The second part is labelled an extension,
    not the pre-existing protocol.

(2) The absorption index. The three-stage diagnostic in
    `repeated_split_stability.py` was run for `gradient_flip` only.
    Its `decomp` quantities (signed margin, disagreement sigma(-m)^2,
    leverage z^T H^-1 z) are applied here to the poisoned refits of every E1
    attack and every E3 alpha, which are the same poisoned datasets that
    script already builds. No parameter, grid, seed, budget or detector is
    introduced. This yields one scalar per condition that can be compared
    against detector AUROC on the same condition.

Reproduction gates run first: this script rebuilds the seed-0 E1 numbers and
the 20-seed E3/stage aggregates and asserts they equal the stored values in
`results/`. If the gates fail the script stops without writing anything, so a
drifted environment cannot silently produce new numbers.

Nothing in `results/` is written or modified. Output goes to
`evidence/benchmark/`.
"""
from __future__ import annotations

import json
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
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
E2_FRACS = (0.001, 0.005, 0.01, 0.02, 0.03, 0.05)
OUT = REPO / "evidence" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results"

si_det = SelfInfluenceDetector()
td_det = TrainingDynamicsDetector()

print("loading the fixed 300-example pool")
pairs = load_preference_subset("hh-rlhf-helpful", n=300, seed=0)
real_ds = embed_pairs(pairs, embedder="sentence-transformer")
print(f"pool n={real_ds.n} d={real_ds.d}")


def cross_fit_scores(fit_ds, score_ds):
    """Identical to `cross_fit_scores` in scripts/repeated_split_stability.py."""
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
    """Identical to `decomp` in scripts/repeated_split_stability.py."""
    head._hinv_cache = None
    m = head.margins(dset)
    dis = expit(-m) ** 2
    lev = np.einsum("ij,jk,ik->i", dset.Z, head.hinv(dset), dset.Z)
    head._hinv_cache = None
    return m, dis, lev, dis * lev


def split(seed):
    """Identical split rule to scripts/repeated_split_stability.py."""
    idx = set_seed(seed).permutation(real_ds.n)
    n_ho = int(real_ds.n * 0.3)
    ho_ds, tr_ds = real_ds.subset(idx[:n_ho]), real_ds.subset(idx[n_ho:])
    zmask = np.linalg.norm(tr_ds.Z, axis=1) <= 1e-8
    trc = tr_ds.subset(np.where(~zmask)[0])
    return ho_ds, trc, int(0.15 * trc.n)


e2_rows, e2_ext_rows, absorb_rows, gate_e1_rows = [], [], [], []
t_start = time.time()

for seed in SEEDS:
    t0 = time.time()
    ho_ds, trc, budget = split(seed)
    head_clean = BTHead(lam=LAM).fit(trc)

    # ---- E2: the Chapter 4 budget grid --------------------------------------
    for frac in E2_FRACS:
        b = max(1, int(frac * trc.n))
        sel = RandomFlipAttack().select_flips(trc, b, set_seed(seed))
        pb = trc.flipped(sel)
        hb = BTHead(lam=LAM).fit(pb)
        row = {"seed": seed, "budget_frac": frac, "budget": b,
               "n_train": trc.n,
               "auroc_training_dynamics": auroc(td_det.score(pb, hb), pb.flip_mask),
               "holdout_accuracy": hb.accuracy(ho_ds),
               # the clean baseline the notebook prints separately, recorded per
               # seed so "attack success vs. budget" has a reference line
               "holdout_accuracy_clean": head_clean.accuracy(ho_ds),
               "theta_drift_from_clean": float(np.linalg.norm(hb.theta - head_clean.theta))}
        (e2_rows if seed == 0 else e2_ext_rows).append(row)
        if seed != 0:
            continue

    # ---- E1 attacks: reproduction gate plus absorption index ----------------
    conditions = {
        "random_flip": RandomFlipAttack().select_flips(trc, budget, set_seed(seed)),
        "gradient_flip": GradientFlipAttack().select_flips(trc, budget, set_seed(seed)),
        "ambiguity_flip": AmbiguityFlipAttack(alpha=1.0).select_flips(
            trc, budget, set_seed(seed), head=head_clean),
    }
    for aname, sel in conditions.items():
        p = trc.flipped(sel)
        h = BTHead(lam=LAM).fit(p)
        comp = {"self_influence": si_det.score(p, h),
                "cross_fit": cross_fit_scores(p, p),
                "training_dynamics": td_det.score(p, h)}
        comp["ensemble"] = ens([comp["self_influence"], comp["cross_fit"],
                                comp["training_dynamics"]])
        for dname, s in comp.items():
            gate_e1_rows.append({"seed": seed, "attack": aname, "detector": dname,
                                 "auroc": auroc(s, p.flip_mask),
                                 "precision_at_budget": precision_at_k(s, p.flip_mask, budget)})
        m, dis, lev, sip = decomp(p, h)
        fm = p.flip_mask
        absorb_rows.append({
            "seed": seed, "family": "E1", "condition": aname, "alpha": np.nan,
            "budget": budget, "n_train": trc.n,
            "margin_flipped": float(m[fm].mean()), "margin_clean": float(m[~fm].mean()),
            "disagree_flipped": float(dis[fm].mean()), "disagree_clean": float(dis[~fm].mean()),
            "leverage_flipped": float(lev[fm].mean()), "leverage_clean": float(lev[~fm].mean()),
            "self_influence_flipped": float(sip[fm].mean()),
            "self_influence_clean": float(sip[~fm].mean()),
            "auroc_self_influence": auroc(comp["self_influence"], fm),
            "auroc_cross_fit": auroc(comp["cross_fit"], fm),
            "auroc_training_dynamics": auroc(comp["training_dynamics"], fm),
            "auroc_ensemble": auroc(comp["ensemble"], fm),
        })

    # ---- E3 alphas: absorption index ---------------------------------------
    for a in ALPHAS:
        sel_a = AmbiguityFlipAttack(alpha=a).select_flips(
            trc, budget, set_seed(seed), head=head_clean)
        pa = trc.flipped(sel_a)
        ha = BTHead(lam=LAM).fit(pa)
        s_si, s_cf, s_td = si_det.score(pa, ha), cross_fit_scores(pa, pa), td_det.score(pa, ha)
        s_en = ens([s_si, s_cf, s_td])
        m, dis, lev, sip = decomp(pa, ha)
        fm = pa.flip_mask
        absorb_rows.append({
            "seed": seed, "family": "E3", "condition": f"alpha={a}", "alpha": a,
            "budget": budget, "n_train": trc.n,
            "margin_flipped": float(m[fm].mean()), "margin_clean": float(m[~fm].mean()),
            "disagree_flipped": float(dis[fm].mean()), "disagree_clean": float(dis[~fm].mean()),
            "leverage_flipped": float(lev[fm].mean()), "leverage_clean": float(lev[~fm].mean()),
            "self_influence_flipped": float(sip[fm].mean()),
            "self_influence_clean": float(sip[~fm].mean()),
            "auroc_self_influence": auroc(s_si, fm), "auroc_cross_fit": auroc(s_cf, fm),
            "auroc_training_dynamics": auroc(s_td, fm), "auroc_ensemble": auroc(s_en, fm),
        })
    print(f"seed {seed:2d} ok  n={trc.n:3d} budget={budget:2d}  ({time.time()-t0:.1f}s)", flush=True)

print(f"\nall seeds in {time.time()-t_start:.1f}s")

e2 = pd.DataFrame(e2_rows)
e2_ext = pd.DataFrame(e2_ext_rows + e2_rows).sort_values(["budget_frac", "seed"])
absorb = pd.DataFrame(absorb_rows)
gate_e1 = pd.DataFrame(gate_e1_rows)

# =========================== reproduction gates =============================
print("\n=== reproduction gates ===")
# The directional attack's default orientation is canonicalized
# (src/attacks/gradient_flip.py, canonical_sign). Gradient-dependent values
# therefore no longer equal the stored raw-orientation run, and are gated
# against the canonical arm of the orientation audit instead. The raw
# orientation is still reproduced exactly, by
# directional_orientation_sensitivity.py through an explicit target_dir, so the
# stored artifacts remain verifiable and nothing is silently mixed.
ORI = OUT / "directional_orientation" / "orientation_per_split.csv"
ori = pd.read_csv(ORI)
canon = ori[ori.arm == "canon_plus"].set_index("seed")
gates = []


def gate(name, got, want, tol=1e-9):
    d = abs(float(got) - float(want))
    gates.append({"gate": name, "recomputed": float(got), "reference": float(want),
                  "abs_diff": d, "pass": bool(d <= tol)})
    print(f"  [{'PASS' if d <= tol else 'FAIL'}] {name}: {got:.12f} vs {want:.12f} (d={d:.2e})")


# --- non-gradient values still gate against the frozen stored artifacts -----
stored_e1 = pd.read_csv(RESULTS / "e1_benchmark.csv")
g0 = gate_e1[gate_e1.seed == 0]
for _, r in stored_e1.iterrows():
    if r.attack == "gradient_flip":
        continue
    mine = g0[(g0.attack == r.attack) & (g0.detector == r.detector)]
    gate(f"E1 seed0 {r.attack}/{r.detector} auroc [stored]", mine.auroc.iloc[0], r.auroc)

stored_sum = pd.read_csv(RESULTS / "repeated_split_summary.csv")
e3s = stored_sum[(stored_sum.block == "E3") & (stored_sum.metric == "auroc")]
for _, r in e3s.iterrows():
    mine = absorb[(absorb.family == "E3") & (absorb.alpha == r.alpha)][f"auroc_{r.detector}"].mean()
    gate(f"E3 20-split mean alpha={r.alpha} {r.detector} [stored]", mine, r["mean"])

e1r = stored_sum[(stored_sum.block == "E1") & (stored_sum.attack == "random_flip")
                 & (stored_sum.metric == "auroc")]
for _, r in e1r.iterrows():
    mine = gate_e1[gate_e1.attack == "random_flip"].groupby("detector").auroc.mean()[r.detector]
    gate(f"E1 random_flip 20-split mean {r.detector} [stored]", mine, r["mean"])

# --- gradient-dependent values gate against the canonical orientation arm ---
gf = absorb[(absorb.family == "E1") & (absorb.condition == "gradient_flip")]
for dd in ("self_influence", "cross_fit", "training_dynamics", "ensemble"):
    gate(f"E1 gradient_flip 20-split mean {dd} [canonical]",
         gf[f"auroc_{dd}"].mean(), canon[f"auroc_poisoned_refit__{dd}"].mean())
gate("gradient_flip margin of reversed rows, poisoned refit [canonical]",
     gf.margin_flipped.mean(), canon.margin_rev_poisoned_refit.mean())
gate("gradient_flip disagreement of reversed rows, poisoned refit [canonical]",
     gf.disagree_flipped.mean(), canon.disagree_rev_poisoned_refit.mean())
gate("gradient_flip leverage of reversed rows, poisoned refit [canonical]",
     gf.leverage_flipped.mean(), canon.leverage_rev_poisoned_refit.mean())
gate("gradient_flip seed0 self-influence of reversed rows [canonical]",
     gf[gf.seed == 0].self_influence_flipped.iloc[0] if "seed" in gf.columns
     else absorb[(absorb.family == "E1") & (absorb.condition == "gradient_flip")
                 & (absorb.seed == 0)].self_influence_flipped.iloc[0],
     canon.loc[0, "selfinf_rev_poisoned_refit"])

gdf = pd.DataFrame(gates)
n_fail = int((~gdf["pass"]).sum())
print(f"\n{len(gdf) - n_fail}/{len(gdf)} gates pass")
if n_fail:
    gdf.to_csv(OUT / "reproduction_gates_FAILED.csv", index=False)
    sys.exit(f"ABORT: {n_fail} reproduction gate(s) failed; nothing else written")

# =============================== write out ==================================
gdf.to_csv(OUT / "reproduction_gates.csv", index=False)
e2.to_csv(OUT / "e2_budget_sweep_seed0.csv", index=False)
e2_ext.to_csv(OUT / "e2_budget_sweep_20seeds.csv", index=False)
absorb.to_csv(OUT / "absorption_index_per_seed.csv", index=False)
gate_e1.to_csv(OUT / "e1_per_seed_recomputed.csv", index=False)

VAL = ["margin_flipped", "margin_clean", "disagree_flipped", "disagree_clean",
       "leverage_flipped", "leverage_clean", "self_influence_flipped",
       "self_influence_clean", "auroc_self_influence", "auroc_cross_fit",
       "auroc_training_dynamics", "auroc_ensemble"]
agg = absorb.groupby(["family", "condition"], as_index=False).agg(
    n_seeds=("seed", "size"),
    **{f"{c}_{st}": (c, st) for c in VAL for st in ("mean", "std")})
agg["absorption_margin_gap"] = agg.margin_flipped_mean - agg.margin_clean_mean
agg["disagreement_ratio"] = agg.disagree_flipped_mean / agg.disagree_clean_mean
agg.to_csv(OUT / "absorption_index_summary.csv", index=False)

e2_ext["holdout_acc_delta"] = e2_ext.holdout_accuracy - e2_ext.holdout_accuracy_clean
e2sum = e2_ext.groupby("budget_frac", as_index=False).agg(
    n_seeds=("seed", "size"), budget_min=("budget", "min"), budget_max=("budget", "max"),
    auroc_mean=("auroc_training_dynamics", "mean"), auroc_sd=("auroc_training_dynamics", "std"),
    holdout_acc_mean=("holdout_accuracy", "mean"), holdout_acc_sd=("holdout_accuracy", "std"),
    holdout_acc_clean_mean=("holdout_accuracy_clean", "mean"),
    acc_delta_mean=("holdout_acc_delta", "mean"), acc_delta_sd=("holdout_acc_delta", "std"),
    acc_delta_neg=("holdout_acc_delta", lambda x: int((x < 0).sum())),
    drift_mean=("theta_drift_from_clean", "mean"), drift_sd=("theta_drift_from_clean", "std"))
e2sum.to_csv(OUT / "e2_budget_sweep_summary.csv", index=False)

# Rank agreement between the absorption index and each detector, across the
# distinct attack conditions. ambiguity_flip and alpha=1.0 are the same attack
# by construction, so the E1 duplicate is dropped to avoid a tied pair.

distinct = agg[~((agg.family == "E1") & (agg.condition == "ambiguity_flip"))]
rank_rows = []
for dname in ("training_dynamics", "self_influence", "cross_fit", "ensemble"):
    for ix, lab in (("disagreement_ratio", "disagreement ratio"),
                    ("absorption_margin_gap", "margin gap")):
        rho, pv = spearmanr(distinct[ix], distinct[f"auroc_{dname}_mean"])
        rank_rows.append({"detector": dname, "index": lab, "n_conditions": len(distinct),
                          "spearman_rho": float(rho), "p_value": float(pv)})
rank = pd.DataFrame(rank_rows)
rank.to_csv(OUT / "absorption_rank_agreement.csv", index=False)

meta = {
    "label": "DRAFT1_STAGE1_EVIDENCE",
    "pool": "hh-rlhf-helpful, n=300, seed=0, sentence-transformer all-MiniLM-L6-v2, d=384",
    "lam": LAM, "n_folds": N_FOLDS, "seeds": list(SEEDS),
    "holdout_frac": 0.3, "zero_norm_threshold": 1e-8,
    "budget_rule": "int(0.15 * n_filtered) for E1/E3; Chapter 4 grid for E2",
    "e2_protocol_source": "pre-existing budget-sweep protocol, re-run verbatim at seed 0",
    "e2_extension": "the identical grid over split seeds 0..19; labelled an extension",
    "absorption_protocol_source": "decomp() in scripts/repeated_split_stability.py, applied to E1 and E3 conditions already built there",
    "reproduction_gates": {"n": len(gdf), "n_pass": len(gdf) - n_fail},
    "e2_grid_degeneracy": "at n_train in [176, 183] the fractions 0.001, 0.005 and 0.01 all map to max(1, int(frac*n)) = 1 flipped comparison, so the three lowest grid points are the same experiment",
    "writes_to_results": False,
}
(OUT / "evidence_meta.json").write_text(json.dumps(meta, indent=2))

print("\n=== E2, Chapter 4 grid, 20-seed aggregate ===")
print(e2sum.to_string(index=False))
print("\n=== absorption index, 20-seed aggregate ===")
cols = ["family", "condition", "n_seeds", "margin_flipped_mean", "margin_clean_mean",
        "disagreement_ratio", "auroc_self_influence_mean", "auroc_cross_fit_mean",
        "auroc_training_dynamics_mean", "auroc_ensemble_mean"]
print(agg.sort_values("disagreement_ratio")[cols].to_string(index=False))
print("\n=== rank agreement, absorption index vs detector AUROC ===")
print(rank.to_string(index=False))
print(f"\nwrote 9 files to {OUT.relative_to(REPO)}/")
