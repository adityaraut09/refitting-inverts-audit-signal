"""Forensic pass on the below-chance gradient_flip result.

Reproduces the exact filtered real-data gradient_flip experiment from E1
(n=176, budget=26, lam=1e-4, same split and seed) and audits it in five
parts:

  A  attack/index/mask/row-order alignment, with hard assertions
  B  the effective s*Z sign-reversal invariant
  C  detector-side score directions (formula level, see tests/)
  D  a three-stage score diagnostic (pre-attack / flipped-no-refit / refit)
  E  self-influence decomposed into disagreement and leverage

Row identity is tracked with an external parallel index array that is
never written into the feature matrix.

Writes results/gradient_flip_forensics_per_row.csv and
results/gradient_flip_forensics_summary.csv. Nothing here is passed
flip_mask before scoring; flip_mask is used only for post-hoc grouping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from src.attacks.gradient_flip import GradientFlipAttack
from src.data.embed import embed_pairs
from src.data.loaders import load_preference_subset
from src.detectors.cross_fit import CrossFitDetector
from src.detectors.ensemble import EnsembleDetector
from src.detectors.self_influence import SelfInfluenceDetector
from src.detectors.training_dynamics import TrainingDynamicsDetector
from src.metrics.detection import auroc, precision_at_k, recall_at_k
from src.model.bt_head import BTHead
from src.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[2]

LAM = 1e-4
N_FOLDS = 5
RESULTS = REPO / "results"
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Record rather than raise, so one failure does not hide the rest.
    Part A stops scientific interpretation if anything here fails."""
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ===========================================================================
# PART A — trace the pipeline and verify alignment
# ===========================================================================
print("=" * 78)
print("PART A — attack / index / mask / row-order alignment")
print("=" * 78)

pairs = load_preference_subset("hh-rlhf-helpful", n=300, seed=0)
real_ds = embed_pairs(pairs, embedder="sentence-transformer")

# PreferenceDataset carries no stable per-row id: `meta` is a plain dict and
# subset() passes it through unchanged, so it goes stale the moment rows are
# selected. Track identity in a standalone array instead, indexed in parallel
# with the real dataset. It is never written into Z or s, so it cannot reach
# a norm, a model, an attack, a Hessian or a detector.
src_ids_all = np.arange(real_ds.n)

idx = set_seed(0).permutation(real_ds.n)
n_ho = int(real_ds.n * 0.3)
ho_local, tr_local = idx[:n_ho], idx[n_ho:]
ho_ds, tr_ds = real_ds.subset(ho_local), real_ds.subset(tr_local)

tr_src_ids = src_ids_all[tr_local]
check(
    "row ids preserve order through subset() [train split]",
    len(tr_src_ids) == tr_ds.n
    and all(np.array_equal(tr_ds.Z[i], real_ds.Z[sid]) for i, sid in enumerate(tr_src_ids)),
)
check("train split size", tr_ds.n == 210, f"n={tr_ds.n}")

z_norms_all = np.linalg.norm(tr_ds.Z, axis=1)
valid_local = np.where(z_norms_all > 1e-8)[0]
tr_ds_clean = tr_ds.subset(valid_local)
clean_src_ids = tr_src_ids[valid_local]
check(
    "row ids preserve order through the zero-norm filter",
    len(clean_src_ids) == tr_ds_clean.n
    and np.array_equal(clean_src_ids, tr_src_ids[np.sort(valid_local)])
    and all(np.array_equal(tr_ds_clean.Z[i], real_ds.Z[sid])
            for i, sid in enumerate(clean_src_ids)),
)
check("filtered size", tr_ds_clean.n == 176, f"n={tr_ds_clean.n}")
check("source ids unique after filtering", len(np.unique(clean_src_ids)) == tr_ds_clean.n)
check(
    "filtered Z matches a direct index of the train Z",
    np.array_equal(tr_ds_clean.Z, tr_ds.Z[valid_local]),
)
check("no zero-norm row survived", bool((np.linalg.norm(tr_ds_clean.Z, axis=1) > 1e-8).all()))

head_clean = BTHead(lam=LAM).fit(tr_ds_clean)
budget = int(0.15 * tr_ds_clean.n)
check("budget", budget == 26, f"budget={budget}")

sel = GradientFlipAttack().select_flips(tr_ds_clean, budget, set_seed(0))
poisoned = tr_ds_clean.flipped(sel)
head_poisoned = BTHead(lam=LAM).fit(poisoned)

sel_mask = np.zeros(tr_ds_clean.n, dtype=bool)
sel_mask[sel] = True

check("selected indices are unique", len(np.unique(sel)) == len(sel), f"{len(np.unique(sel))}/{len(sel)}")
check("len(selected) == budget", len(sel) == budget)
check("selected indices in range", bool((sel >= 0).all() and (sel < tr_ds_clean.n).all()))
check("flip_mask.sum() == budget", int(poisoned.flip_mask.sum()) == budget,
      f"sum={int(poisoned.flip_mask.sum())}")
check(
    "flatnonzero(flip_mask) == sorted(selected local indices)",
    np.array_equal(np.flatnonzero(poisoned.flip_mask), np.sort(sel)),
)
check(
    "source ids marked by flip_mask == source ids of selected rows",
    np.array_equal(np.sort(clean_src_ids[poisoned.flip_mask]), np.sort(clean_src_ids[sel])),
)
check("baseline dataset had no flips", int(tr_ds_clean.flip_mask.sum()) == 0)
check("row count preserved by flipped()", poisoned.n == tr_ds_clean.n)

s_changed_rows = np.flatnonzero(poisoned.s != tr_ds_clean.s)
check(
    "no unintended row changed: s differs exactly on the selected rows",
    np.array_equal(s_changed_rows, np.sort(sel)),
    f"{len(s_changed_rows)} rows changed",
)
check(
    "no selected row lost or duplicated",
    len(np.unique(clean_src_ids[sel])) == budget and len(clean_src_ids[sel]) == budget,
)

# Independently recompute the attack objective and confirm the same selection.
cov = tr_ds_clean.Z.T @ tr_ds_clean.Z / tr_ds_clean.n
_, eigvecs = np.linalg.eigh(cov)
v_top = eigvecs[:, -1]
alignment = tr_ds_clean.s * (tr_ds_clean.Z @ v_top)
sel_recomputed = np.argsort(-alignment)[:budget]
check("recomputed gradient_flip objective reproduces the selection",
      np.array_equal(sel_recomputed, sel))

# ===========================================================================
# PART B — effective label-feature flip invariant
# ===========================================================================
print()
print("=" * 78)
print("PART B — effective s*Z sign reversal")
print("=" * 78)

eff_clean = tr_ds_clean.s[:, None] * tr_ds_clean.Z
eff_att = poisoned.s[:, None] * poisoned.Z

err_sel = np.abs(eff_att[sel_mask] + eff_clean[sel_mask]).max()
err_unsel = np.abs(eff_att[~sel_mask] - eff_clean[~sel_mask]).max()
row_inv_err = np.where(
    sel_mask,
    np.abs(eff_att + eff_clean).max(axis=1),
    np.abs(eff_att - eff_clean).max(axis=1),
)

scale = np.abs(eff_clean).max()
tol = 1e-12 * max(scale, 1.0)
print(f"  max |Z| = {scale:.6e}, tolerance = {tol:.3e}")
check("selected rows: effective_attacked == -effective_clean", err_sel <= tol,
      f"max abs err = {err_sel:.3e}")
check("unselected rows: effective_attacked == effective_clean", err_unsel <= tol,
      f"max abs err = {err_unsel:.3e}")

s_only = not np.array_equal(poisoned.s, tr_ds_clean.s)
z_only = not np.array_equal(poisoned.Z, tr_ds_clean.Z)
print(f"  s changed: {s_only}    Z changed: {z_only}    Z is the same object: {poisoned.Z is tr_ds_clean.Z}")
check("exactly one of (s, Z) is negated, so the flip does not cancel",
      s_only and not z_only)
check("selected rows have s exactly negated",
      np.array_equal(poisoned.s[sel_mask], -tr_ds_clean.s[sel_mask]))
check("other row data stay aligned (Z row-for-row identical)",
      np.array_equal(poisoned.Z, tr_ds_clean.Z))

row_s_changed = poisoned.s != tr_ds_clean.s
row_z_changed = np.abs(poisoned.Z - tr_ds_clean.Z).max(axis=1) > 0

# Hessian is label-independent (sigma' is even in m), so the same head gives
# the same leverage on clean and attacked labels. Verify rather than assume.
H_clean = head_clean.hessian(tr_ds_clean)
H_att = head_clean.hessian(poisoned)
check("Hessian is label-independent for a fixed head",
      np.allclose(H_clean, H_att, rtol=0, atol=1e-12),
      f"max abs diff = {np.abs(H_clean - H_att).max():.3e}")

# ===========================================================================
# PART D — three-stage score diagnostic
# ===========================================================================
print()
print("=" * 78)
print("PART D — three-stage score diagnostic")
print("=" * 78)

si_det = SelfInfluenceDetector()
td_det = TrainingDynamicsDetector()


def cross_fit_manual(fit_ds, score_ds, n_folds=N_FOLDS, lam=LAM):
    """CrossFitDetector.score, but with the fold-fitting dataset and the
    scored dataset separable. Same folds, same lam, same BTHead, same
    1 - sigma(m) score as the production detector."""
    n = score_ds.n
    fold_ids = np.arange(n) % n_folds
    scores = np.zeros(n)
    oof_margins = np.zeros(n)
    for fold in range(n_folds):
        test_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        fold_head = BTHead(lam=lam).fit(fit_ds.subset(train_idx))
        test_ds = score_ds.subset(test_idx)
        m = test_ds.s * (test_ds.Z @ fold_head.theta)
        scores[test_idx] = 1 - expit(m)
        oof_margins[test_idx] = m
    return scores, oof_margins


def ensemble_of(component_scores):
    return np.mean([rankdata(s) for s in component_scores], axis=0)


def decompose_self_influence(ds, head):
    """Exactly the production formula: sigma(-m)^2 * (z^T Hinv z), with the
    head's own Hessian and lam. No substitute approximation."""
    head._hinv_cache = None  # do not reuse a cache built on another ds
    m = head.margins(ds)
    disagreement = expit(-m) ** 2
    Hinv = head.hinv(ds)
    leverage = np.einsum("ij,jk,ik->i", ds.Z, Hinv, ds.Z)
    head._hinv_cache = None
    return m, disagreement, leverage, disagreement * leverage


# --- Stage 0: clean dataset, clean models. Where does the cohort already sit?
m0, dis0, lev0, si0 = decompose_self_influence(tr_ds_clean, head_clean)
td0 = td_det.score(tr_ds_clean, head_clean)
cf0, oof_m0 = cross_fit_manual(tr_ds_clean, tr_ds_clean)
ens0 = ensemble_of([si0, cf0, td0])

# --- Stage 1: flipped labels scored by models fit only on clean labels
m1, dis1, lev1, si1 = decompose_self_influence(poisoned, head_clean)
td1 = td_det.score(poisoned, head_clean)
cf1, oof_m1 = cross_fit_manual(tr_ds_clean, poisoned)  # clean folds, flipped labels
ens1 = ensemble_of([si1, cf1, td1])

# --- Stage 2: the normal poisoned refit, exactly as E1
m2, dis2, lev2, si2 = decompose_self_influence(poisoned, head_poisoned)
td2 = td_det.score(poisoned, head_poisoned)
cf2, oof_m2 = cross_fit_manual(poisoned, poisoned)
ens2 = ensemble_of([si2, cf2, td2])

# Cross-check the manual cross-fit and the decomposition against production.
prod_cf = CrossFitDetector(lam=LAM)
prod_ens = EnsembleDetector([si_det, prod_cf, td_det])
check("manual cross-fit reproduces production on clean data",
      np.allclose(cf0, prod_cf.score(tr_ds_clean), atol=1e-12))
check("manual cross-fit reproduces production at stage 2",
      np.allclose(cf2, prod_cf.score(poisoned), atol=1e-12))
check("self-influence decomposition reproduces production at stage 2",
      np.allclose(si2, si_det.score(poisoned, head_poisoned), atol=1e-12))
check("stage-2 scores reproduce the E1 numbers",
      np.isclose(auroc(si2, poisoned.flip_mask), 0.31666666666666665, atol=1e-12)
      and np.isclose(auroc(cf2, poisoned.flip_mask), 0.3082051282051282, atol=1e-12)
      and np.isclose(auroc(td2, poisoned.flip_mask), 0.2969230769230769, atol=1e-12))
check("ensemble decomposition reproduces production at stage 2",
      np.allclose(ens2, prod_ens.score(poisoned, head_poisoned), atol=1e-12))

cohort = sel_mask  # evaluation-only grouping, never given to a detector
stage_scores = {
    "stage0_pre_attack": {"self_influence": si0, "cross_fit": cf0, "training_dynamics": td0, "ensemble": ens0},
    "stage1_flipped_clean_models": {"self_influence": si1, "cross_fit": cf1, "training_dynamics": td1, "ensemble": ens1},
    "stage2_poisoned_refit": {"self_influence": si2, "cross_fit": cf2, "training_dynamics": td2, "ensemble": ens2},
}

QS = [0.10, 0.25, 0.50, 0.75, 0.90]
summary_rows = []
for stage, dets in stage_scores.items():
    for det_name, s in dets.items():
        f, u = s[cohort], s[~cohort]
        top = np.argsort(-s)[:budget]
        row = {
            "stage": stage,
            "detector": det_name,
            "auroc": auroc(s, cohort),
            "precision_at_budget": precision_at_k(s, cohort, budget),
            "recall_at_budget": recall_at_k(s, cohort, budget),
            "n_flipped_in_top_26": int(cohort[top].sum()),
            "mean_flipped": float(f.mean()),
            "mean_unflipped": float(u.mean()),
            "median_flipped": float(np.median(f)),
            "median_unflipped": float(np.median(u)),
        }
        for q in QS:
            row[f"q{int(q*100)}_flipped"] = float(np.quantile(f, q))
            row[f"q{int(q*100)}_unflipped"] = float(np.quantile(u, q))
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print()
print(summary_df[["stage", "detector", "auroc", "precision_at_budget", "recall_at_budget",
                  "n_flipped_in_top_26", "mean_flipped", "mean_unflipped",
                  "median_flipped", "median_unflipped"]].to_string(index=False))

# ===========================================================================
# PART E — self-influence decomposition
# ===========================================================================
print()
print("=" * 78)
print("PART E — self-influence components")
print("=" * 78)

comp_rows = []
for stage, (m, dis, lev, prod) in {
    "stage0_pre_attack": (m0, dis0, lev0, si0),
    "stage1_flipped_clean_models": (m1, dis1, lev1, si1),
    "stage2_poisoned_refit": (m2, dis2, lev2, si2),
}.items():
    for comp_name, arr in [("signed_margin_m", m), ("disagreement_sigma_neg_m_sq", dis),
                           ("leverage_zT_Hinv_z", lev), ("self_influence_product", prod)]:
        comp_rows.append({
            "stage": stage,
            "component": comp_name,
            "mean_flipped": float(arr[cohort].mean()),
            "mean_unflipped": float(arr[~cohort].mean()),
            "median_flipped": float(np.median(arr[cohort])),
            "median_unflipped": float(np.median(arr[~cohort])),
            "ratio_flipped_over_unflipped": float(arr[cohort].mean() / arr[~cohort].mean())
            if arr[~cohort].mean() != 0 else float("nan"),
        })
comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False))

# ===========================================================================
# PART F — artifacts
# ===========================================================================
per_row = pd.DataFrame({
    "source_row_id": clean_src_ids,
    "filtered_local_index": np.arange(tr_ds_clean.n),
    "selected_rank": np.where(sel_mask, np.argsort(np.argsort(-alignment)), -1),
    "flip_flag": sel_mask,
    "gradient_flip_objective": alignment,
    "z_norm": np.linalg.norm(tr_ds_clean.Z, axis=1),
    "s_clean": tr_ds_clean.s,
    "s_attacked": poisoned.s,
    "s_changed": row_s_changed,
    "z_changed": row_z_changed,
    "effective_invariant_abs_err": row_inv_err,
    "clean_margin": m0,
    "clean_head_margin_after_relabel": m1,
    "poisoned_head_margin": m2,
    "clean_oof_margin_after_relabel": oof_m1,
    "poisoned_oof_margin": oof_m2,
    "clean_oof_margin_pre_attack": oof_m0,
    "stage0_self_influence": si0,
    "stage0_cross_fit": cf0,
    "stage0_training_dynamics": td0,
    "stage0_ensemble": ens0,
    "stage1_self_influence": si1,
    "stage1_cross_fit": cf1,
    "stage1_training_dynamics": td1,
    "stage1_ensemble": ens1,
    "stage2_self_influence": si2,
    "stage2_cross_fit": cf2,
    "stage2_training_dynamics": td2,
    "stage2_ensemble": ens2,
    "stage0_si_disagreement": dis0,
    "stage0_si_leverage": lev0,
    "stage0_si_product": si0,
    "stage1_si_disagreement": dis1,
    "stage1_si_leverage": lev1,
    "stage1_si_product": si1,
    "stage2_si_disagreement": dis2,
    "stage2_si_leverage": lev2,
    "stage2_si_product": si2,
})
per_row_path = RESULTS / "gradient_flip_forensics_per_row.csv"
per_row.to_csv(per_row_path, index=False)

summary_path = RESULTS / "gradient_flip_forensics_summary.csv"
summary_df.to_csv(summary_path, index=False)
comp_path = RESULTS / "gradient_flip_forensics_si_components.csv"
comp_df.to_csv(comp_path, index=False)

meta_out = {
    "n": int(tr_ds_clean.n),
    "budget": int(budget),
    "lam": LAM,
    "n_folds": N_FOLDS,
    "split_seed": 0,
    "attack_seed": 0,
    "holdout_frac": 0.3,
    "zero_norm_threshold": 1e-8,
    "max_invariant_err_selected": float(err_sel),
    "max_invariant_err_unselected": float(err_unsel),
    "s_changed": bool(s_only),
    "z_changed": bool(z_only),
    "assertion_failures": FAILURES,
}
(RESULTS / "gradient_flip_forensics_meta.json").write_text(json.dumps(meta_out, indent=2))

print()
print("=" * 78)
print(f"per-row rows: {len(per_row)}  unique source ids: {per_row.source_row_id.nunique()}  "
      f"flip_flag true: {int(per_row.flip_flag.sum())}")
print(f"wrote {per_row_path.name}, {summary_path.name}, {comp_path.name}, "
      f"gradient_flip_forensics_meta.json")
if FAILURES:
    print(f"\n*** {len(FAILURES)} ASSERTION FAILURE(S): {FAILURES}")
    print("*** Scientific interpretation is on hold until these are resolved.")
    sys.exit(1)
print("\nAll alignment and invariant assertions passed.")
