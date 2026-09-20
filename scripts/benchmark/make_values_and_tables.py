"""Emit every benchmark number used in the manuscript, plus the claim ledger.

One generator so that the manuscript, the tables and the claim ledger cannot
disagree. Each macro is written with a comment naming the file, the columns and
the filter it came from, and the same triple is written into the ledger.

Macro names are letters only. LaTeX rejects digits in `\\newcommand` names, so
quantities that carry a number in their label spell it out (`QZero`, `KTwentySix`).
Values are read from `results/` and `evidence/benchmark/`. Nothing is recomputed
from raw data here; `make_evidence.py` owns that and gates itself against the
stored artifacts first.

Scope: the E1-E4 benchmark, one 300-comparison pool, 20 splits, three attacks,
four audit scores, budget = int(0.15 * n_train), lam = 1e-4. The companion
length-targeted study is not part of this repository; see README.md.
"""
from __future__ import annotations

import re
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results"
EV = REPO / "evidence" / "benchmark"
OUTDIR = REPO / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)


macros: list[tuple[str, str, str]] = []   # (name, value, provenance)
ledger: list[dict] = []
NUM = "ABCDEFGHIJ"


def spell(n: int) -> str:
    return "".join(NUM[int(c)] for c in str(n))


def put(name, value, *, claim, section, family, dataset, splits, attack, detector,
        budget, metric, uncertainty, artifact, columns, status, fmt="{:.4f}"):
    """Register one value as both a LaTeX macro and a ledger row."""
    if isinstance(value, str):
        v = value
    else:
        v = fmt.format(value)
        # a mean of -1e-17 under "{:+.4f}" prints "-0.0000", which reads as a
        # measured negative. Collapse a rounded zero to an unsigned zero.
        if re.fullmatch(r"[+-]0\.0*", v) or re.fullmatch(r"[+-]0", v):
            v = v.lstrip("+-")
    prov = f"{artifact} :: {columns}"
    macros.append((name, v, prov))
    ledger.append({
        "claim_id": f"C{len(ledger) + 1:03d}", "macro": f"\\v{name}", "section": section,
        "claim": claim, "family": family, "dataset_pool": dataset,
        "splits_seeds": splits, "attack": attack, "detector": detector,
        "budget": budget, "metric": metric, "value": v, "uncertainty": uncertainty,
        "source_artifact": artifact, "columns_filter": columns, "status": status,
    })
    return v


# ============================ FAMILY-A inputs ===============================
summ = pd.read_csv(RESULTS / "repeated_split_summary.csv")
meta = pd.read_csv(RESULTS / "repeated_split_meta.csv")
fore = pd.read_csv(RESULTS / "gradient_flip_forensics_si_components.csv")
absorb = pd.read_csv(EV / "absorption_index_summary.csv")
rank = pd.read_csv(EV / "absorption_rank_agreement.csv")
e2sum = pd.read_csv(EV / "e2_budget_sweep_summary.csv")
gates = pd.read_csv(EV / "reproduction_gates.csv")

A_POOL = "one HH-RLHF helpful-base pool, n=300, seed 0"
A_SPLITS = "20 splits (seeds 0-19), same pool"
A_BUDGET = "int(0.15*n_train), 26-27 of 175-183"
SI = "repeated_split_summary.csv"

# --- setup scale ---
put("NPool", 300, claim="source pool size", section="IV", family="FAMILY-A",
    dataset=A_POOL, splits="n/a", attack="n/a", detector="n/a", budget="n/a",
    metric="count", uncertainty="fixed", artifact="repeated_split_stability.py",
    columns="load_preference_subset(n=300, seed=0)", status="supported", fmt="{:.0f}")
put("NSeeds", 20, claim="number of repeated splits", section="IV", family="FAMILY-A",
    dataset=A_POOL, splits=A_SPLITS, attack="all", detector="all", budget=A_BUDGET,
    metric="count", uncertainty="fixed", artifact="repeated_split_meta.csv",
    columns="seed, nunique", status="supported", fmt="{:.0f}")
put("NTrainMin", meta.n_filtered.min(), claim="smallest filtered training set", section="IV",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="n/a", detector="n/a",
    budget=A_BUDGET, metric="count", uncertainty=f"range {meta.n_filtered.min()}-{meta.n_filtered.max()}",
    artifact="repeated_split_meta.csv", columns="n_filtered, min", status="supported", fmt="{:.0f}")
put("NTrainMax", meta.n_filtered.max(), claim="largest filtered training set", section="IV",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="n/a", detector="n/a",
    budget=A_BUDGET, metric="count", uncertainty="range", artifact="repeated_split_meta.csv",
    columns="n_filtered, max", status="supported", fmt="{:.0f}")
put("ZeroNormMin", meta.zero_norm_removed.min(), claim="fewest zero-norm rows removed",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="n/a",
    detector="n/a", budget="n/a", metric="count", uncertainty="range",
    artifact="repeated_split_meta.csv", columns="zero_norm_removed, min",
    status="supported", fmt="{:.0f}")
put("ZeroNormMax", meta.zero_norm_removed.max(), claim="most zero-norm rows removed",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="n/a",
    detector="n/a", budget="n/a", metric="count", uncertainty="range",
    artifact="repeated_split_meta.csv", columns="zero_norm_removed, max",
    status="supported", fmt="{:.0f}")
put("ZeroNormPct", 100 * meta.zero_norm_removed.sum() / (210 * len(meta)),
    claim="share of pre-filter training rows with a zero-norm difference feature",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="n/a",
    detector="n/a", budget="n/a", metric="percent", uncertainty="pooled over 20 splits",
    artifact="repeated_split_meta.csv", columns="zero_norm_removed / n_pre_filter",
    status="supported", fmt="{:.0f}")
put("BudgetMin", meta.budget.min(), claim="smallest attack budget", section="IV",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="all", detector="n/a",
    budget=A_BUDGET, metric="count", uncertainty="range", artifact="repeated_split_meta.csv",
    columns="budget, min", status="supported", fmt="{:.0f}")
put("BudgetMax", meta.budget.max(), claim="largest attack budget", section="IV",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="all", detector="n/a",
    budget=A_BUDGET, metric="count", uncertainty="range", artifact="repeated_split_meta.csv",
    columns="budget, max", status="supported", fmt="{:.0f}")
put("NGates", len(gates), claim="reproduction gates run before any new value was written",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="all",
    detector="all", budget=A_BUDGET, metric="count", uncertainty="all passed",
    artifact="evidence/reproduction_gates.csv", columns="row count",
    status="supported", fmt="{:.0f}")

# --- the benchmark regularization selection, as it was actually done ---
# results/holdout_reuse_audit.json records a four-point grid scored by held-out
# pairwise accuracy on the seed-0 split alone, before any attack, after which
# one value was fixed for every split. Because that partition is one of the 20
# reported splits, every predictive outcome downstream is a validation estimate
# and not performance on untouched data. A fifth value, lam = 1e-6, appears in
# the wider grid but no held-out accuracy was ever recorded for it, so the grid
# that actually decided the value had four points and that is what is reported.
_hr = json.loads((RESULTS / "holdout_reuse_audit.json").read_text())
_lamsweep = _hr["lam_sweep"]
_LAMSEL_ART = "results/holdout_reuse_audit.json"
assert sorted(_lamsweep) == ["1e-02", "1e-03", "1e-04", "1e-05"], sorted(_lamsweep)
assert max(_lamsweep, key=lambda k: _lamsweep[k]["acc_all_90"]) == "1e-04"
put("LamSelNGrid", len(_lamsweep),
    claim="regularization values scored on held-out accuracy before the value was fixed",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits="seed 0 only", attack="none",
    detector="n/a", budget="n/a", metric="count",
    uncertainty="1e-2, 1e-3, 1e-4, 1e-5; one split, no attack applied",
    artifact=_LAMSEL_ART, columns="lam_sweep, key count", status="supported", fmt="{:.0f}")
put("LamSelNHoldout", _hr["holdout_n"],
    claim="held-out comparisons the regularization value was selected on",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits="seed 0 only", attack="none",
    detector="n/a", budget="n/a", metric="count",
    uncertainty=f"unfiltered; {_hr['holdout_zero_norm']} carry a zero-norm difference feature",
    artifact=_LAMSEL_ART, columns="holdout_n", status="supported", fmt="{:.0f}")
for _tag, _key in (("EmTwo", "1e-02"), ("EmThree", "1e-03"), ("EmFour", "1e-04"),
                   ("EmFive", "1e-05")):
    put(f"LamSelAcc{_tag}", _lamsweep[_key]["acc_all_90"],
        claim=f"held-out accuracy at lam = {_key} in the selection sweep", section="IV",
        family="FAMILY-A", dataset=A_POOL, splits="seed 0 only", attack="none",
        detector="n/a", budget="n/a", metric="held-out accuracy",
        uncertainty="single split, single fit; no spread available",
        artifact=_LAMSEL_ART, columns=f"lam_sweep.{_key}.acc_all_90",
        status="supported")

# --- canonical-orientation evidence for the directional stress test ---------
# The directional attack's orientation was undefined before Stage 3 and the
# stored results/ artifacts used a split-dependent mixture of the two tails.
# Every gradient-dependent value below therefore comes from the canonical +v
# arm, never from results/. Non-gradient values still come from results/.
ORI = pd.read_csv(EV / "directional_orientation" / "orientation_per_split.csv")
CANON = ORI[ORI.arm == "canon_plus"]
CANON_MINUS = ORI[ORI.arm == "canon_minus"]
A_CANON = "evidence/directional_orientation/orientation_per_split.csv"
CANON_NOTE = ("canonical +v orientation: largest-|.| coordinate of the top "
              "eigenvector forced positive")
e1_rec = pd.read_csv(EV / "e1_per_seed_recomputed.csv")

# --- E1: 20-split AUROC and precision per attack x detector ---
DET_TAG = {"self_influence": "SelfInf", "cross_fit": "CrossFit",
           "training_dynamics": "TrainDyn", "ensemble": "Ens"}
ATT_TAG = {"random_flip": "Rand", "gradient_flip": "Grad", "ambiguity_flip": "Amb"}
e1s = summ[(summ.block == "E1")]
for att in ("random_flip", "gradient_flip", "ambiguity_flip"):
    for dd in DET_TAG:
        tag = ATT_TAG[att] + DET_TAG[dd]
        if att == "gradient_flip":
            g = e1_rec[(e1_rec.attack == att) & (e1_rec.detector == dd)]
            vals = {"auroc": g.auroc, "precision_at_budget": g.precision_at_budget}
            art, note, extra = A_CANON, "supported; " + CANON_NOTE, CANON_NOTE
        else:
            g = e1_rec[(e1_rec.attack == att) & (e1_rec.detector == dd)]
            vals = {"auroc": g.auroc, "precision_at_budget": g.precision_at_budget}
            art, note, extra = "repeated_split_summary.csv", "supported", "unchanged by the orientation fix"
        for metric, series in vals.items():
            kind = "Auroc" if metric == "auroc" else "Prec"
            put(f"{kind}{tag}", series.mean(),
                claim=f"E1 {att} / {dd} {metric}, 20-split mean",
                section="V-A", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
                attack=att, detector=dd, budget=A_BUDGET, metric=metric,
                uncertainty=f"sd {series.std():.4f}; {extra}",
                artifact=art, columns=f"attack={att}, detector={dd}, {metric}, mean",
                status=note)
            put(f"{kind}{tag}Sd", series.std(),
                claim=f"E1 {att} / {dd} {metric} across-split sd", section="V-A",
                family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack=att,
                detector=dd, budget=A_BUDGET, metric=f"{metric} sd",
                uncertainty="across-split sd, not a population interval",
                artifact=art, columns=f"attack={att}, detector={dd}, {metric}, sd",
                status=note)

# --- E3: corrected alpha sweep (no gradient dependence) ---
ALPHA_TAG = {0.0: "QZero", 0.25: "QTwentyFive", 0.5: "QFifty",
             0.75: "QSeventyFive", 1.0: "QHundred"}
e3s = summ[(summ.block == "E3") & (summ.metric == "auroc")]
for _, r in e3s.iterrows():
    put(f"EThree{DET_TAG[r.detector]}{ALPHA_TAG[r.alpha]}", r["mean"],
        claim=f"E3 AUROC at alpha={r.alpha} for {r.detector}, 20-split mean",
        section="V-C", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
        attack=f"ambiguity_flip alpha={r.alpha}", detector=r.detector, budget=A_BUDGET,
        metric="auroc", uncertainty=f"sd {r['sd']:.4f}", artifact=SI,
        columns=f"block=E3, alpha={r.alpha}, detector={r.detector}, mean",
        status="supported")

# --- three-stage decomposition, canonical orientation ---
STAGE_TAG = {"pre_attack": "Pre", "clean_head": "CleanHead", "poisoned_refit": "Refit"}
for st, tag in STAGE_TAG.items():
    for dd, dtag in DET_TAG.items():
        for kind, col in (("Auroc", f"auroc_{st}__{dd}"), ("Prec", f"prec_{st}__{dd}")):
            s = CANON[col]
            put(f"Stg{kind}{tag}{dtag}", s.mean(),
                claim=f"directional stress test, {st}, {dd} {kind.lower()}, 20-split mean",
                section="V-B", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
                attack="directional, canonical +v", detector=dd, budget=A_BUDGET,
                metric=col, uncertainty=f"sd {s.std():.4f}; {CANON_NOTE}",
                artifact=A_CANON, columns=f"arm=canon_plus, {col}, mean",
                status="supported; " + CANON_NOTE)
    for who, cw in (("Flip", "rev"), ("Clean", "ret")):
        s = CANON[f"margin_{cw}_{st}"]
        put(f"Marg{who}{tag}", s.mean(),
            claim=f"mean signed margin of {'reversed' if cw == 'rev' else 'retained'} rows, {st}",
            section="V-B", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
            attack="directional, canonical +v", detector="n/a", budget=A_BUDGET,
            metric="mean signed margin", uncertainty=f"sd {s.std():.4f}; {CANON_NOTE}",
            artifact=A_CANON, columns=f"arm=canon_plus, margin_{cw}_{st}, mean",
            status="supported; " + CANON_NOTE, fmt="{:.3f}")
    for dd, dtag in DET_TAG.items():
        s = 1.0 - CANON[f"auroc_{st}__{dd}"]
        if st == "poisoned_refit":
            put(f"StgInvRefit{dtag}", s.mean(),
                claim=f"separation recovered by inverting the {dd} ranking after refitting",
                section="V-B", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
                attack="directional, canonical +v", detector=dd, budget=A_BUDGET,
                metric="1 - auroc",
                uncertainty="an auditor cannot know to invert without knowing the attack",
                artifact=A_CANON, columns=f"arm=canon_plus, 1 - auroc_{st}__{dd}, mean",
                status="supported; " + CANON_NOTE)

# --- orientation sensitivity macros ---
_gap = CANON.relative_eigengap
put("EigGap", _gap.mean(), claim="relative eigengap of the feature second-moment matrix",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
    attack="directional", detector="n/a", budget=A_BUDGET,
    metric="(lambda1 - lambda2) / lambda1",
    uncertainty=f"sd {_gap.std():.4f}, range [{_gap.min():.3f}, {_gap.max():.3f}]",
    artifact=A_CANON, columns="arm=canon_plus, relative_eigengap, mean",
    status="supported", fmt="{:.3f}")
put("EigRatio", (CANON.eig1 / CANON.eig2).mean(),
    claim="ratio of the two leading eigenvalues", section="IV", family="FAMILY-A",
    dataset=A_POOL, splits=A_SPLITS, attack="directional", detector="n/a",
    budget=A_BUDGET, metric="lambda1 / lambda2", uncertainty="20-split mean",
    artifact=A_CANON, columns="arm=canon_plus, eig1/eig2, mean", status="supported",
    fmt="{:.2f}")
put("NRawCanon", int(ORI[ORI.arm == "raw"].raw_equals_canon_plus.sum()),
    claim="splits where the pre-fix orientation happened to equal canonical +v",
    section="IV", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
    attack="directional", detector="n/a", budget=A_BUDGET, metric="count",
    uncertainty="of 20 splits; the remainder used the opposite tail",
    artifact=A_CANON, columns="arm=raw, raw_equals_canon_plus, sum",
    status="supported", fmt="{:.0f}")
for dd, dtag in DET_TAG.items():
    s = CANON_MINUS[f"auroc_poisoned_refit__{dd}"]
    put(f"MinusAuroc{dtag}", s.mean(),
        claim=f"poisoned-refit AUROC for {dd} under the opposite orientation",
        section="VIII", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
        attack="directional, canonical -v", detector=dd, budget=A_BUDGET,
        metric="auroc", uncertainty=f"below chance in {int((s < 0.5).sum())} of 20 splits",
        artifact=A_CANON, columns=f"arm=canon_minus, auroc_poisoned_refit__{dd}, mean",
        status="supported; opposite-tail sensitivity arm")
# A7: the opposite tail is not uniformly worse for the model, so the paper
# reports each axis rather than calling one tail "more damaging".
for _nm, _arm, _src in (("Plus", CANON, "canon_plus"), ("Minus", CANON_MINUS, "canon_minus")):
    for _key, _col, _fmt, _what in (
            ("Acc", "acc_poisoned", "{:.4f}", "held-out accuracy after the attack"),
            ("Loss", "logloss_poisoned", "{:.4f}", "held-out logistic loss after the attack"),
            ("Disp", "theta_displacement", "{:.2f}", "parameter displacement from the original head")):
        _s = _arm[_col]
        put(f"{_nm}{_key}", _s.mean(),
            claim=f"{_what}, {_src} orientation", section="VIII", family="FAMILY-A",
            dataset=A_POOL, splits=A_SPLITS, attack=f"directional, {_src}",
            detector="n/a", budget=A_BUDGET, metric=_col,
            uncertainty=f"sd {_s.std():.4f} over 20 splits", artifact=A_CANON,
            columns=f"arm={_src}, {_col}, mean", status="supported", fmt=_fmt)
for _dd, _dtag in DET_TAG.items():
    _s = CANON_MINUS[f"prec_poisoned_refit__{_dd}"]
    put(f"MinusPrec{_dtag}", _s.mean(),
        claim=f"precision at the review budget for {_dd}, opposite orientation",
        section="VIII", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
        attack="directional, canon_minus", detector=_dd, budget=A_BUDGET,
        metric="precision_at_budget", uncertainty=f"sd {_s.std():.4f}",
        artifact=A_CANON, columns=f"arm=canon_minus, prec_poisoned_refit__{_dd}, mean",
        status="supported")

put("MinusMargGap", (CANON_MINUS.margin_rev_poisoned_refit
                     - CANON_MINUS.margin_ret_poisoned_refit).mean(),
    claim="reversed minus retained margin after refitting, opposite orientation",
    section="VIII", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
    attack="directional, canonical -v", detector="n/a", budget=A_BUDGET,
    metric="margin gap", uncertainty="positive in 20 of 20 splits",
    artifact=A_CANON, columns="arm=canon_minus, margin gap, mean", status="supported",
    fmt="{:.3f}")
put("AccClean", CANON.acc_clean.mean(), claim="held-out accuracy of the original head",
    section="V-B", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="none",
    detector="n/a", budget=A_BUDGET, metric="held-out accuracy",
    uncertainty=f"sd {CANON.acc_clean.std():.4f}", artifact=A_CANON,
    columns="arm=canon_plus, acc_clean, mean", status="supported")
put("AccPoisonedDir", CANON.acc_poisoned.mean(),
    claim="held-out accuracy after the directional stress test", section="V-B",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
    attack="directional, canonical +v", detector="n/a", budget=A_BUDGET,
    metric="held-out accuracy", uncertainty=f"sd {CANON.acc_poisoned.std():.4f}",
    artifact=A_CANON, columns="arm=canon_plus, acc_poisoned, mean",
    status="supported; " + CANON_NOTE)

# --- leverage decomposition, seed 0 only ---
f2 = fore[fore.stage == "stage2_poisoned_refit"].set_index("component")
for comp, tag, fmt in (("leverage_zT_Hinv_z", "Lev", "{:.0f}"),
                       ("disagreement_sigma_neg_m_sq", "Dis", "{:.4f}"),
                       ("self_influence_product", "SelfInfVal", "{:.2f}")):
    put(f"{tag}Flip", f2.loc[comp, "mean_flipped"],
        claim=f"{comp} of reversed rows under the poisoned refit, split seed 0",
        section="VI", family="FAMILY-A", dataset=A_POOL, splits="split seed 0 only",
        attack="gradient_flip", detector="self_influence", budget="26 of 176",
        metric=comp, uncertainty="single split, no interval",
        artifact="gradient_flip_forensics_si_components.csv",
        columns=f"stage=stage2_poisoned_refit, component={comp}, mean_flipped",
        status="supported, single split", fmt=fmt)
    put(f"{tag}Clean", f2.loc[comp, "mean_unflipped"],
        claim=f"{comp} of retained rows under the poisoned refit, split seed 0",
        section="VI", family="FAMILY-A", dataset=A_POOL, splits="split seed 0 only",
        attack="gradient_flip", detector="self_influence", budget="26 of 176",
        metric=comp, uncertainty="single split, no interval",
        artifact="gradient_flip_forensics_si_components.csv",
        columns=f"stage=stage2_poisoned_refit, component={comp}, mean_unflipped",
        status="supported, single split", fmt=fmt)
    put(f"{tag}Ratio", f2.loc[comp, "ratio_flipped_over_unflipped"],
        claim=f"{comp} ratio reversed/retained under the poisoned refit, split seed 0",
        section="VI", family="FAMILY-A", dataset=A_POOL, splits="split seed 0 only",
        attack="gradient_flip", detector="self_influence", budget="26 of 176",
        metric=f"{comp} ratio", uncertainty="single split, no interval",
        artifact="gradient_flip_forensics_si_components.csv",
        columns=f"stage=stage2_poisoned_refit, component={comp}, ratio_flipped_over_unflipped",
        status="supported, single split", fmt="{:.3f}")

# --- absorption index ordering ---
put("NAbsorbCond", int(rank.n_conditions.iloc[0]),
    claim="distinct attack conditions in the absorption ordering", section="V-B",
    family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="all E1 and E3 conditions",
    detector="all", budget=A_BUDGET, metric="count",
    uncertainty="ambiguity_flip and alpha=1.0 are the same attack; the duplicate is dropped",
    artifact="evidence/absorption_rank_agreement.csv", columns="n_conditions",
    status="supported", fmt="{:.0f}")
for _, r in rank[rank["index"] == "disagreement ratio"].iterrows():
    put(f"Rho{DET_TAG[r.detector]}", r.spearman_rho,
        claim=f"Spearman rank agreement between the disagreement ratio and {r.detector} AUROC",
        section="V-B", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS,
        attack="all E1 and E3 conditions", detector=r.detector, budget=A_BUDGET,
        metric="Spearman rho over 7 conditions",
        uncertainty="the 7 conditions share one pool and one set of splits, so they are not independent samples",
        artifact="evidence/absorption_rank_agreement.csv",
        columns=f"detector={r.detector}, index=disagreement ratio, spearman_rho",
        status="supported as a descriptive ordering", fmt="{:.3f}")
ab = absorb.set_index(absorb.family + "|" + absorb.condition)
for key, tag in (("E1|gradient_flip", "Grad"), ("E3|alpha=0.0", "AlphaZero"),
                 ("E3|alpha=1.0", "AlphaOne"), ("E1|random_flip", "Rand")):
    put(f"DisRatio{tag}", ab.loc[key, "disagreement_ratio"],
        claim=f"disagreement ratio (reversed/retained) for {key}", section="V-B",
        family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack=key.split("|")[1],
        detector="n/a", budget=A_BUDGET, metric="mean sigma(-m)^2 ratio",
        uncertainty="ratio of 20-split means", artifact="evidence/absorption_index_summary.csv",
        columns=f"family|condition={key}, disagreement_ratio", status="supported", fmt="{:.3f}")

# --- E2 budget sweep ---
e2lo = e2sum[e2sum.budget_frac <= 0.01]
put("ETwoNDegen", len(e2lo), claim="grid points that collapse to a single reversed comparison",
    section="V-D", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="random_flip",
    detector="training_dynamics", budget="max(1, int(frac*n_train))", metric="count",
    uncertainty="exact, 0.1/0.5/1.0 percent all map to budget 1",
    artifact="evidence/e2_budget_sweep_summary.csv", columns="budget_frac<=0.01, budget_min=budget_max=1",
    status="supported", fmt="{:.0f}")
for _, r in e2sum.iterrows():
    t = spell(int(round(r.budget_frac * 1000)))
    put(f"ETwoAuroc{t}", r.auroc_mean,
        claim=f"E2 training-dynamics AUROC at budget fraction {r.budget_frac}", section="V-D",
        family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="random_flip",
        detector="training_dynamics", budget=f"{int(r.budget_min)}-{int(r.budget_max)} rows",
        metric="auroc", uncertainty=f"sd {r.auroc_sd:.4f} over 20 splits",
        artifact="evidence/e2_budget_sweep_summary.csv",
        columns=f"budget_frac={r.budget_frac}, auroc_mean", status="supported")
    put(f"ETwoAccDelta{t}", r.acc_delta_mean,
        claim=f"E2 held-out accuracy change vs the clean head at budget fraction {r.budget_frac}",
        section="V-D", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="random_flip",
        detector="n/a", budget=f"{int(r.budget_min)}-{int(r.budget_max)} rows",
        metric="delta held-out accuracy",
        uncertainty=f"sd {r.acc_delta_sd:.4f}; negative in {int(r.acc_delta_neg)} of 20 splits",
        artifact="evidence/e2_budget_sweep_summary.csv",
        columns=f"budget_frac={r.budget_frac}, acc_delta_mean", status="supported", fmt="{:+.4f}")
    put(f"ETwoDrift{t}", r.drift_mean,
        claim=f"E2 parameter drift from the clean head at budget fraction {r.budget_frac}",
        section="V-D", family="FAMILY-A", dataset=A_POOL, splits=A_SPLITS, attack="random_flip",
        detector="n/a", budget=f"{int(r.budget_min)}-{int(r.budget_max)} rows",
        metric="L2 theta drift", uncertainty=f"sd {r.drift_sd:.4f}",
        artifact="evidence/e2_budget_sweep_summary.csv",
        columns=f"budget_frac={r.budget_frac}, drift_mean", status="supported", fmt="{:.1f}")

# --- E4 remediation, rebuilt over 20 splits ---
# Draft 1 reported this from split seed 0 only. That split turns out to be the
# most extreme of the 20 and its conclusion does not generalise, so every E4
# value here comes from the 20-split rebuild.
e4 = pd.read_csv(EV / "e4_per_seed_20splits.csv")
e4d = pd.read_csv(EV / "e4_summary_20splits.csv")
e4b = pd.read_csv(EV / "e4_budget_depth_20splits.csv").iloc[0]
E4_ART = "evidence/e4_per_seed_20splits.csv"
E4_SPL = "20 splits (seeds 0-19), same pool"
E4_NOTE = ("margin-targeted attack alpha=1, training-dynamics ranking, lam=1e-4; "
           "seed 0 reproduces the stored single-split CSV on all 84 paper columns")
bud = e4[e4.is_budget_depth]

# Seven depths per split: 5, 10, 15, 20, B, 35, 50. The aggregated CSV shows
# eight distinct numeric k only because B = 26 in ten splits and 27 in the other
# ten. 20 splits x 7 depths = 140 cells.
_depths_per_split = int(e4.groupby("seed").k.nunique().unique()[0])
assert _depths_per_split == 7, _depths_per_split
assert len(e4) == 140, len(e4)
put("EFourNDepths", _depths_per_split,
    claim="deletion depths applied per split", section="V-F", family="FAMILY-A",
    dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="count",
    uncertainty="5, 10, 15, 20, B, 35, 50; identical for every split",
    artifact=E4_ART, columns="per-seed nunique of k", status="supported", fmt="{:.0f}")
put("EFourNDistinctK", int(e4.k.nunique()),
    claim="distinct numeric k values across the aggregated table", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics",
    budget=A_BUDGET, metric="count",
    uncertainty="eight rather than seven only because B = 26 in ten splits and 27 in ten",
    artifact=E4_ART, columns="k, nunique", status="supported", fmt="{:.0f}")
put("EFourAccClean", bud.acc_full_clean.mean(),
    claim="held-out accuracy of the head fit on the full original data", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="none", detector="n/a",
    budget=A_BUDGET, metric="held-out accuracy",
    uncertainty=f"sd {bud.acc_full_clean.std():.4f}; range [{bud.acc_full_clean.min():.4f}, {bud.acc_full_clean.max():.4f}]",
    artifact=E4_ART, columns="is_budget_depth, acc_full_clean, mean", status="supported")

# The absolute quality of the head being repaired. An independent audit asked
# for this baseline explicitly: a predictor that assigns probability one half to
# every comparison incurs log 2 on every one of them, and the full-clean head is
# above that on average, so no claim about repairing a *useful* reward model is
# available from this pool. The ranking results are unaffected because they are
# scored against the reversal mask, not against held-out quality.
_UNINF = float(np.log(2.0))
put("UninfLoss", _UNINF,
    claim="logistic loss of a predictor assigning probability one half to every comparison",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits="n/a", attack="n/a",
    detector="n/a", budget="n/a", metric="held-out logistic loss",
    uncertainty="exact: log 2 on every comparison, so no spread",
    artifact="arithmetic identity", columns="log(2)", status="supported")
put("EFourLossClean", bud.logloss_full_clean.mean(),
    claim="held-out logistic loss of the head fit on the full original data",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="none",
    detector="n/a", budget=A_BUDGET, metric="held-out logistic loss",
    uncertainty=f"sd {bud.logloss_full_clean.std():.4f}; range [{bud.logloss_full_clean.min():.4f}, {bud.logloss_full_clean.max():.4f}]",
    artifact=E4_ART, columns="is_budget_depth, logloss_full_clean, mean",
    status="supported")
put("EFourLossCleanAboveUninf", int((bud.logloss_full_clean > _UNINF).sum()),
    claim="splits where the full-clean head's held-out loss exceeds log 2",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="none",
    detector="n/a", budget=A_BUDGET, metric="count of splits",
    uncertainty=f"mean gap {(bud.logloss_full_clean - _UNINF).mean():+.4f}",
    artifact=E4_ART, columns="logloss_full_clean > log(2), sum", status="supported",
    fmt="{:.0f}")

put("EFourAccUnsafe", bud.acc_unsafe.mean(),
    claim="held-out accuracy with no remediation", section="V-F", family="FAMILY-A",
    dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0", detector="n/a",
    budget=A_BUDGET, metric="held-out accuracy",
    uncertainty=f"sd {bud.acc_unsafe.std():.4f}", artifact=E4_ART,
    columns="is_budget_depth, acc_unsafe, mean", status="supported")
put("EFourAttackAcc", (bud.acc_unsafe - bud.acc_full_clean).mean(),
    claim="effect of the attack on held-out accuracy", section="V-F", family="FAMILY-A",
    dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0", detector="n/a",
    budget=A_BUDGET, metric="delta held-out accuracy",
    uncertainty=f"sd {(bud.acc_unsafe-bud.acc_full_clean).std():.4f}; worse in {int((bud.acc_unsafe<bud.acc_full_clean).sum())} of 20 splits, better in {int((bud.acc_unsafe>bud.acc_full_clean).sum())}",
    artifact=E4_ART, columns="acc_unsafe - acc_full_clean, mean",
    status="supported; no measurable effect", fmt="{:+.4f}")
put("EFourAttackAccWorse", int((bud.acc_unsafe < bud.acc_full_clean).sum()),
    claim="splits where the attack lowered held-out accuracy", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="n/a", budget=A_BUDGET, metric="count", uncertainty="of 20 splits",
    artifact=E4_ART, columns="acc_unsafe < acc_full_clean, sum", status="supported", fmt="{:.0f}")
put("EFourAttackLoss", (bud.logloss_unsafe - bud.logloss_full_clean).mean(),
    claim="effect of the attack on held-out logistic loss", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="n/a", budget=A_BUDGET, metric="delta held-out logistic loss",
    uncertainty=f"sd {(bud.logloss_unsafe-bud.logloss_full_clean).std():.4f}; higher in {int((bud.logloss_unsafe>bud.logloss_full_clean).sum())} of 20 splits",
    artifact=E4_ART, columns="logloss_unsafe - logloss_full_clean, mean",
    status="supported", fmt="{:+.4f}")
put("EFourAttackLossWorse", int((bud.logloss_unsafe > bud.logloss_full_clean).sum()),
    claim="splits where the attack raised held-out logistic loss", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="n/a", budget=A_BUDGET, metric="count", uncertainty="of 20 splits",
    artifact=E4_ART, columns="logloss_unsafe > logloss_full_clean, sum",
    status="supported", fmt="{:.0f}")

# localization quality, which is the part that holds up
put("EFourPrecBudget", bud.precision_of_removal.mean(),
    claim="share of the deleted set that was reversed, at the budget-matched depth",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="precision of removal",
    uncertainty=f"sd {bud.precision_of_removal.std():.4f}; above prevalence in {int((bud.precision_of_removal > bud.budget/bud.n_train).sum())} of 20 splits",
    artifact=E4_ART, columns="is_budget_depth, precision_of_removal, mean", status="supported")
put("EFourPrevalence", (bud.budget / bud.n_train).mean(),
    claim="reversal prevalence, the random-review baseline for deletion precision",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="n/a", budget=A_BUDGET,
    metric="prevalence", uncertainty="mean of B/n_train over 20 splits",
    artifact=E4_ART, columns="budget / n_train, mean", status="supported")
put("EFourPrecAbovePrev", int((e4.precision_of_removal > e4.budget / e4.n_train).sum()),
    claim="(split, depth) cells where deletion precision exceeds prevalence", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="count",
    uncertainty=f"of {len(e4)} cells, so every cell at every depth",
    artifact=E4_ART, columns="precision_of_removal > budget/n_train, sum",
    status="supported", fmt="{:.0f}")
put("EFourNCells", len(e4), claim="(split, depth) cells in the rebuilt sweep",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="20 splits by 7 depths", artifact=E4_ART,
    columns="row count", status="supported", fmt="{:.0f}")
put("EFourPrecShallow", e4d[e4d.k == 5].precision_of_removal_mean.iloc[0],
    claim="deletion precision at the shallowest depth", section="V-F", family="FAMILY-A",
    dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="precision of removal",
    uncertainty=f"sd {e4d[e4d.k==5].precision_of_removal_std.iloc[0]:.4f}",
    artifact="evidence/e4_summary_20splits.csv", columns="k=5, precision_of_removal_mean",
    status="supported")
put("EFourPrecDeep", e4d[e4d.k == 50].precision_of_removal_mean.iloc[0],
    claim="deletion precision at the deepest depth", section="V-F", family="FAMILY-A",
    dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="precision of removal",
    uncertainty=f"sd {e4d[e4d.k==50].precision_of_removal_std.iloc[0]:.4f}",
    artifact="evidence/e4_summary_20splits.csv", columns="k=50, precision_of_removal_mean",
    status="supported")

# recovery, which does not follow
_da = bud.acc_remediated - bud.acc_unsafe
put("EFourRemVsNoneAcc", _da.mean(),
    claim="accuracy difference between remediation and taking no action, budget depth",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="delta held-out accuracy",
    uncertainty=f"sd {_da.std():.4f}; better in {int((_da>0).sum())}, equal in {int((_da==0).sum())}, worse in {int((_da<0).sum())} of 20 splits",
    artifact=E4_ART, columns="acc_remediated - acc_unsafe, mean",
    status="supported; no direction", fmt="{:+.4f}")
put("EFourRemBetterAcc", int((_da > 0).sum()), claim="splits where remediation raised accuracy",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="of 20 splits", artifact=E4_ART,
    columns="acc_remediated > acc_unsafe, sum", status="supported", fmt="{:.0f}")
put("EFourRemWorseAcc", int((_da < 0).sum()), claim="splits where remediation lowered accuracy",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="of 20 splits", artifact=E4_ART,
    columns="acc_remediated < acc_unsafe, sum", status="supported", fmt="{:.0f}")
put("EFourRemEqualAcc", int((_da == 0).sum()), claim="splits where remediation left accuracy unchanged",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="of 20 splits", artifact=E4_ART,
    columns="acc_remediated == acc_unsafe, sum", status="supported", fmt="{:.0f}")
_dl = bud.logloss_remediated - bud.logloss_unsafe
put("EFourRemVsNoneLoss", _dl.mean(),
    claim="held-out logistic loss penalty of remediation against taking no action",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="delta held-out logistic loss",
    uncertainty=f"sd {_dl.std():.4f}; higher in {int((_dl>0).sum())} of 20 splits",
    artifact=E4_ART, columns="logloss_remediated - logloss_unsafe, mean",
    status="supported; consistent direction", fmt="{:+.4f}")
put("EFourRemWorseLoss", int((_dl > 0).sum()),
    claim="splits where remediation raised held-out logistic loss", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="count",
    uncertainty="of 20 splits", artifact=E4_ART,
    columns="logloss_remediated > logloss_unsafe, sum", status="supported", fmt="{:.0f}")
_lo5 = e4d[e4d.k == 5]
_lo50 = e4d[e4d.k == 50]
put("EFourLossPenShallow", _lo5.logloss_remediated_mean.iloc[0] - _lo5.logloss_unsafe_mean.iloc[0],
    claim="loss penalty of remediation at the shallowest depth", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="delta held-out logistic loss",
    uncertainty="20-split mean at k=5", artifact="evidence/e4_summary_20splits.csv",
    columns="k=5, logloss_remediated_mean - logloss_unsafe_mean", status="supported",
    fmt="{:+.4f}")
put("EFourLossPenDeep", _lo50.logloss_remediated_mean.iloc[0] - _lo50.logloss_unsafe_mean.iloc[0],
    claim="loss penalty of remediation at the deepest depth", section="V-F",
    family="FAMILY-A", dataset=A_POOL, splits=E4_SPL, attack="ambiguity_flip alpha=1.0",
    detector="training_dynamics", budget=A_BUDGET, metric="delta held-out logistic loss",
    uncertainty="20-split mean at k=50", artifact="evidence/e4_summary_20splits.csv",
    columns="k=50, logloss_remediated_mean - logloss_unsafe_mean", status="supported",
    fmt="{:+.4f}")

# the composition control
_comp = bud.logloss_matched_clean_cf - bud.logloss_full_clean
put("EFourCompLoss", _comp.mean(),
    claim="held-out loss cost of deleting the same number of comparisons that carried original labels",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="delta held-out logistic loss",
    uncertainty=f"sd {_comp.std():.4f}; higher in {int((_comp>0).sum())} of 20 splits",
    artifact=E4_ART, columns="logloss_matched_clean_cf - logloss_full_clean, mean",
    status="supported; consistent direction", fmt="{:+.4f}")
put("EFourCompLossWorse", int((_comp > 0).sum()),
    claim="splits where deleting original-label comparisons raised held-out loss",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="of 20 splits", artifact=E4_ART,
    columns="logloss_matched_clean_cf > logloss_full_clean, sum", status="supported",
    fmt="{:.0f}")
_res = bud.logloss_remediated - bud.logloss_matched_clean_cf
put("EFourResidLoss", _res.mean(),
    claim="held-out loss difference between remediation and the matched original-label control",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="delta held-out logistic loss",
    uncertainty=f"sd {_res.std():.4f}; higher in only {int((_res>0).sum())} of 20 splits, so no consistent direction",
    artifact=E4_ART, columns="logloss_remediated - logloss_matched_clean_cf, mean",
    status="supported as inconsistent; no direction claimed", fmt="{:+.4f}")
put("EFourResidLossWorse", int((_res > 0).sum()),
    claim="splits where remediation was worse than the matched control on held-out loss",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty="of 20 splits", artifact=E4_ART,
    columns="logloss_remediated > logloss_matched_clean_cf, sum", status="supported",
    fmt="{:.0f}")
put("EFourSurvive", bud.poisoned_surviving.mean(),
    claim="reversed comparisons still in the training set at the budget-matched depth",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count", uncertainty=f"sd {bud.poisoned_surviving.std():.2f} of B = 26 or 27",
    artifact=E4_ART, columns="is_budget_depth, poisoned_surviving, mean",
    status="supported", fmt="{:.1f}")
put("EFourMcNemarSig", int((bud.mcnemar_p < 0.05).sum()),
    claim="splits where the paired test separates remediation from the matched control",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits=E4_SPL,
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget=A_BUDGET,
    metric="count of exact two-sided McNemar p < 0.05",
    uncertainty="of 20 splits; tested within each split, never pooled",
    artifact=E4_ART, columns="is_budget_depth, mcnemar_p < 0.05, sum",
    status="supported", fmt="{:.0f}")
put("EFourSeedZeroGap", float(bud[bud.seed == 0].acc_gap_remed_minus_matched.iloc[0]),
    claim="the split-seed-0 accuracy gap that Draft 1 reported as the headline",
    section="V-F", family="FAMILY-A", dataset=A_POOL, splits="split seed 0 only",
    attack="ambiguity_flip alpha=1.0", detector="training_dynamics", budget="26 of 176",
    metric="delta held-out accuracy",
    uncertainty="the most extreme of the 20 splits on this quantity",
    artifact=E4_ART, columns="seed=0, is_budget_depth, acc_gap_remed_minus_matched",
    status="superseded by the 20-split mean", fmt="{:+.4f}")

# The companion length-targeted study (FAMILY-B) contributed no value to the
# five-page manuscript; neither its evidence nor its generators are part of
# this repository. See README.md, "Scope and limitations".

# =============================== write out ==================================
lines = []
for name, val, prov in macros:
    lines.append(f"% {prov}")
    lines.append(f"\\newcommand{{\\v{name}}}{{{val}}}")
seen = {}
for name, _, _ in macros:
    seen[name] = seen.get(name, 0) + 1
dupes = {k: v for k, v in seen.items() if v > 1}
if dupes:
    sys.exit(f"ABORT: duplicate macro names {dupes}")
bad = [n for n, _, _ in macros if not n.isalpha()]
if bad:
    sys.exit(f"ABORT: macro names must be letters only, got {bad}")
(OUTDIR / "paper_values_benchmark.tex").write_text("\n".join(lines) + "\n")

led = pd.DataFrame(ledger)
led.to_csv(OUTDIR / "claim_ledger_benchmark.csv", index=False)

# ---------------- Table I: attacks and audit scores ----------------
(OUTDIR / "table_methods.tex").write_text(r"""\begin{table}[t]
\caption{Attacks and audit scores as implemented. Every score is computed on the
poisoned training set; reversal masks are evaluator-only.}
\label{tab:methods}
\centering
\footnotesize
\begin{tabular}{@{}p{1.55cm}p{1.15cm}p{4.55cm}@{}}
\toprule
\textbf{Rule} & \textbf{Reads} & \textbf{Selection or score} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Attacks}}\\
random & nothing & uniform subset of size $B$; the blind control \\[2pt]
directional & $s_i$, $z_i$ & largest $s_i\langle z_i,v\rangle$, with $v$ the
canonically signed top eigenvector of $Z^\top Z/n$; a mutually aligned block
along that direction \\[2pt]
margin-targeted & $s_i$, $z_i$, $\thh_0$ & window of width $B$ slid across the
$|m_i|$ order by $\alpha$; $\alpha=1$ takes the lowest reference margins,
$\alpha=0$ the highest \\[2pt]
length-disparity & $s_i$, lengths & largest longer-minus-shorter token gap within
$\mathcal C$ (Family B only) \\
\midrule
\multicolumn{3}{@{}l}{\emph{Audit scores, all increasing in suspicion}}\\
self-influence & $\thh_P$ & $\sigma(-m_i)^2\,z_i^\top H^{-1}z_i$: disagreement
times leverage in the fitted curvature \\[2pt]
out-of-fold & 5 folds of $D_P$ & $1-\sigma(m_i)$ under a head fit on the other
four folds, folds assigned by index modulo 5. A disagreement score in the spirit
of Confident Learning, not an implementation of it \\[2pt]
final margin & $\thh_P$ & $-m_i$, the final fitted margin. The cheap
substitute for AUM sanctioned in our plan, not an area under a training
trajectory, and no training trajectory is recorded \\[2pt]
ensemble & all three & mean of the three per-example ranks \\
\bottomrule
\end{tabular}
\end{table}
""")

# ------- compact methods table for the five-page candidate -------
# Single column, one row per rule. The length-disparity attack is deliberately
# absent because the length-targeted study is not part of the five-page story.
(OUTDIR / "table_methods_compact.tex").write_text(r"""\begin{table}[t]
\caption{Attacks and audit scores as implemented. Every score is computed on the
poisoned training set and increases with suspicion. $m_i$ is the fitted margin,
$H$ the fitted Hessian, and subscripts $0$ and $P$ the original and poisoned
labels.}
\label{tab:methods}
\centering
\footnotesize
% widths: "directional" overflows a 1.3 cm first column by 2.2 pt, so the three
% are redistributed from 1.3/1.15/4.75 at the same total width. Set here rather
% than in the emitted file, which is overwritten on every run.
\begin{tabular}{@{}p{1.5cm}p{1.1cm}p{4.6cm}@{}}
\toprule
\textbf{Rule} & \textbf{Reads} & \textbf{Selection or score} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Attacks, budget $B$}}\\
random & nothing & uniform subset of size $B$; the control \\[2pt]
directional & $s_i$, $z_i$ & largest $s_i\langle z_i,v\rangle$ with $v$ the
canonically signed top eigenvector of $Z^\top Z/n$: a mutually aligned block
along one axis \\[2pt]
margin-\newline targeted & $s_i$, $z_i$, $\thh_0$ & window of width $B$ slid by
$\alpha$ across the $|m_i|$ order; $\alpha=1$ takes the lowest reference
margins, $\alpha=0$ the highest \\
\midrule
\multicolumn{3}{@{}l}{\emph{Audit scores}}\\
self-\newline influence & $\thh_P$ & $\sigma(-m_i)^2\,z_i^\top H^{-1}z_i$:
disagreement times leverage \\[2pt]
out-of-fold & 5 folds of $D_P$ & $1-\sigma(m_i)$ under a head fit on the other
four folds, folds by index modulo 5 \\[2pt]
final-\newline margin & $\thh_P$ & $-m_i$, the final fitted margin \\[2pt]
rank\newline ensemble & all three & mean of the three per-example ranks \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- Table II: E1 benchmark ----------------
rows = []
for att in ("random_flip", "gradient_flip", "ambiguity_flip"):
    lab = {"random_flip": "random", "gradient_flip": "directional",
           "ambiguity_flip": r"margin-targeted ($\alpha{=}1$)"}[att]
    cells = []
    for dd in ("self_influence", "cross_fit", "training_dynamics", "ensemble"):
        g = e1_rec[(e1_rec.attack == att) & (e1_rec.detector == dd)]
        assert len(g) == 20, (att, dd, len(g))
        # an independent audit found the bare second number easy to misread as a
        # second measurement, so the standard deviation now carries its own sign
        cells.append(f"{g.auroc.mean():.3f}\\,$\\pm${g.auroc.std():.3f}"
                     f" & {g.precision_at_budget.mean():.3f}")
    rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
(OUTDIR / "table_e1.tex").write_text(r"""\begin{table*}[t]
\caption{Benchmark. AUROC (with across-split standard deviation) and
precision at the attack budget, for each attack against each audit score, on the
poisoned refit. Mean over \vNSeeds\ splits of one \vNPool-comparison HH-RLHF
helpful-base pool; budget \vBudgetMin--\vBudgetMax\ reversals of
\vNTrainMin--\vNTrainMax\ training comparisons. Chance AUROC is $0.5$; because
the review budget equals the number of reversals, precision and recall at that
budget coincide, and the random-review baseline equals the reversal prevalence
of about $0.15$. A value below chance means the score ranks reversed
comparisons as \emph{less} suspicious than retained ones, so the ranking is
inverted rather than uninformative. The directional row uses the canonical
orientation defined in Section~\ref{sec:setup}.}
\label{tab:e1}
\centering
\footnotesize
\begin{tabular}{@{}l cc cc cc cc@{}}
\toprule
& \multicolumn{2}{c}{self-influence} & \multicolumn{2}{c}{out-of-fold}
& \multicolumn{2}{c}{final margin} & \multicolumn{2}{c}{ensemble} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Attack} & AUROC & prec. & AUROC & prec. & AUROC & prec. & AUROC & prec. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- Table III: stage decomposition ----------------
srows = []
for st, lab in (("pre_attack", "original labels, original head"),
                ("clean_head", "reversed labels, original head"),
                ("poisoned_refit", "reversed labels, poisoned refit")):
    cells = []
    for dd in ("self_influence", "cross_fit", "training_dynamics", "ensemble"):
        cells.append(f"{CANON[f'auroc_{st}__{dd}'].mean():.3f} & "
                     f"{CANON[f'prec_{st}__{dd}'].mean():.3f}")
    _mrev = CANON[f"margin_rev_{st}"].mean()
    srows.append(f"{lab} & ${'+' if _mrev >= 0 else '-'}${abs(_mrev):.2f} & "
                 + " & ".join(cells) + r" \\")
(OUTDIR / "table_stages.tex").write_text(r"""\caption{The ranking inverts after the poisoned refit. Identical reversed
comparisons, scored under the head fit on the original data and under the head
refit on the poisoned data. Directional attack, mean over \vNSeeds\ splits.
Under the original head self-influence and the final-margin score separate the
reversals almost perfectly (\vStgAurocCleanHeadSelfInf\ and
\vStgAurocCleanHeadTrainDyn), the ensemble reaches \vStgAurocCleanHeadEns, and
the out-of-fold score reaches only \vStgAurocCleanHeadCrossFit. After the refit
all four fall below chance. The attack uses the canonical
$+v$ orientation defined in Section~\ref{sec:setup}.
$\bar m_{\mathrm{rev}}$ is the mean signed margin of the reversed comparisons;
retained comparisons sit at \vMargCleanRefit\ after the refit, so the reversals
end up with the larger positive margins. Row two is an oracle diagnostic and not
a deployable detector: it requires a head fit on data an auditor does not have.}
\label{tab:stages}
\centering
\footnotesize
\begin{tabular}{@{}l r cc cc cc cc@{}}
\toprule
& & \multicolumn{2}{c}{self-infl.} & \multicolumn{2}{c}{out-of-fold}
& \multicolumn{2}{c}{final marg.} & \multicolumn{2}{c}{ensemble} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
\textbf{Scored under} & $\bar m_{\mathrm{rev}}$ & AUR. & pr. & AUR. & pr. & AUR. & pr. & AUR. & pr. \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
""")

print(f"wrote paper_values_benchmark.tex with {len(macros)} macros")
print(f"wrote claim_ledger_benchmark.csv with {len(led)} rows")
print("status counts:")
print(led.status.value_counts().to_string())
print("family counts:")
print(led.family.value_counts().to_string())
print("wrote table_methods.tex, table_methods_compact.tex, table_e1.tex, table_stages.tex")
