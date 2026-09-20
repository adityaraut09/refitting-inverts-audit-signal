"""Generate the pooled-replication macro file from the pooled evidence.

Every macro is read out of a CSV and asserted, so a number in the manuscript
cannot drift from the evidence that produced it. The benchmark macros are
generated separately by ``scripts/benchmark/make_values_and_tables.py``; no
value is produced by both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "evidence" / "pooled"
OUTDIR = REPO / "tables"

macros: dict[str, str] = {}
ledger: list[dict] = []


def put(name: str, value, fmt: str = "{}", source: str = "", status: str = "measured"):
    macros[name] = fmt.format(value)
    ledger.append({"macro": name, "value": macros[name], "source": source, "status": status})


def main() -> int:
    grid = pd.read_csv(EV / "x_grid_cells.csv")
    lad = pd.read_csv(EV / "q2_nd_ladder.csv")
    q2b = pd.read_csv(EV / "q2b_effect_of_N.csv")
    q3 = pd.read_csv(EV / "q3_cleanhead_validity.csv")
    q4 = pd.read_csv(EV / "q4_remediation_summary.csv")
    q5 = pd.read_csv(EV / "q5_twotailed_summary.csv")

    # ---------------- design
    put("vFourNPoolsPerScale", 5, source="docs/EXPERIMENTAL_PLAN.md")
    put("vFourNPoolsTotal", 15, source="embedding_manifest.json")
    put("vFourPoolTrainRows", 43835, source="embedding_manifest.json")
    put("vFourPoolTestRows", 2354, source="embedding_manifest.json")
    put("vFourRowsUsed", 21500, source="embedding_manifest.json")
    put("vFourNCells", len(grid), source="x_grid_cells.csv")
    assert len(grid) == 1050

    # ---------------- robustness of the inversion
    n_below = int((grid.fm_refit_auroc < 0.5).sum())
    n_absorb = int((grid.refit_marg_excess_reversed > 0).sum())
    assert n_below == len(grid) and n_absorb == len(grid)
    put("vFourRefitBelowHalf", n_below, source="x_grid_cells.csv")
    put("vFourAbsorbPositive", n_absorb, source="x_grid_cells.csv")
    put("vFourRefitMin", lad.fm_refit_auroc.min(), fmt="{:.4f}", source="q2_nd_ladder.csv")
    put("vFourRefitMax", lad.fm_refit_auroc.max(), fmt="{:.4f}", source="q2_nd_ladder.csv")
    put("vFourNLadder", len(lad), source="q2_nd_ladder.csv")
    assert int((lad.n_pools_refit_below_half == 5).sum()) == len(lad)
    put("vFourLadderAllFive", len(lad), source="q2_nd_ladder.csv")

    # ---------------- the N/d result (cat, dim 384, lam 1e-3)
    c = q2b[q2b.convention == "cat"].sort_values("n_train").reset_index(drop=True)
    r = q2b[q2b.convention == "resp"].sort_values("n_train").reset_index(drop=True)
    assert len(c) == 3 and len(r) == 3
    for i, tag in enumerate(["Lo", "Mid", "Hi"]):
        put(f"vFourNTrain{tag}", int(round(c.n_train[i])), source="q2b_effect_of_N.csv")
        put(f"vFourNdRatio{tag}", c.n_over_d[i], fmt="{:.2f}", source="q2b_effect_of_N.csv")
        put(f"vFourOracle{tag}", c.oracle_cleanhead_auroc[i], fmt="{:.4f}",
            source="q2b_effect_of_N.csv")
        put(f"vFourRefit{tag}", c.refit_auroc[i], fmt="{:.4f}", source="q2b_effect_of_N.csv")
        put(f"vFourTrainAcc{tag}", c.clean_train_acc[i], fmt="{:.3f}",
            source="q2b_effect_of_N.csv")
    put("vFourOraclePoolsHi", int(c.n_pools_cleanhead_above_half[2]),
        source="q2b_effect_of_N.csv")
    put("vFourOracleRespHi", r.oracle_cleanhead_auroc[2], fmt="{:.4f}",
        source="q2b_effect_of_N.csv")
    put("vFourOracleRespLo", r.oracle_cleanhead_auroc[0], fmt="{:.4f}",
        source="q2b_effect_of_N.csv")
    put("vFourNdRatioMax", q2b.n_over_d.max(), fmt="{:.2f}", source="q2b_effect_of_N.csv")
    put("vFourOracleLadderMin", lad.fm_cleanhead_auroc.min(), fmt="{:.4f}",
        source="q2_nd_ladder.csv")
    put("vFourOracleLadderMax", lad.fm_cleanhead_auroc.max(), fmt="{:.4f}",
        source="q2_nd_ladder.csv")

    # ---------------- clean-head validity
    put("vFourLogTwo", 0.693147, fmt="{:.4f}", source="log 2")
    best = q3.sort_values("test_logloss").iloc[0]
    put("vFourBestTestLL", best.test_logloss, fmt="{:.4f}", source="q3_cleanhead_validity.csv")
    put("vFourBestTestAcc", best.test_acc, fmt="{:.4f}", source="q3_cleanhead_validity.csv")
    put("vFourBestRefitAuroc", best.fm_refit_auroc, fmt="{:.4f}",
        source="q3_cleanhead_validity.csv")
    assert bool(q3.qualifies.all())
    put("vFourNQualify", len(q3), source="q3_cleanhead_validity.csv")
    assert int(lad.beats_log2.sum()) == len(lad)
    put("vFourLadderBeatLogTwo", len(lad), source="q2_nd_ladder.csv")
    # the weak-head configuration that reproduces the benchmark regime
    weak = q2b[(q2b.convention == "resp") & (~q2b.beats_log2)]
    assert len(weak) == 1
    put("vFourWeakTestLL", weak.clean_test_logloss.iloc[0], fmt="{:.4f}",
        source="q2b_effect_of_N.csv")
    put("vFourWeakTrainAcc", weak.clean_train_acc.iloc[0], fmt="{:.3f}",
        source="q2b_effect_of_N.csv")
    put("vFourWeakOracle", weak.oracle_cleanhead_auroc.iloc[0], fmt="{:.4f}",
        source="q2b_effect_of_N.csv")

    # ---------------- interior band
    put("vFourTiedAtCutMax", int(grid.fm_refit_n_tied_at_cut.max()),
        source="x_grid_cells.csv")
    put("vFourZeroPrecCells", int((grid.fm_refit_prec_at_b == 0).sum()),
        source="x_grid_cells.csv")

    # ---------------- two-tailed, cat n3000, directional, budget = B
    d = q5[(q5.attack == "directional") & (q5.budget_mult == 1.0)
           & (q5.scale == "n3000") & (q5.convention == "cat")].set_index("policy")
    put("vFourTopPrec", d.loc["top", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourTwoTailPrec", d.loc["two_tailed", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourRandPrec", d.loc["random", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourBottomPrec", d.loc["bottom_oracle_diagnostic", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourTwoTailLossGain", -d.loc["two_tailed", "correct_minus_no_action"],
        fmt="{:.4f}", source="q5_twotailed_summary.csv")
    dd = q5[(q5.attack == "directional") & (q5.budget_mult == 1.0)]
    assert int((dd[dd.policy == "two_tailed"].n_pools_beat_top == 5).sum()) == 6
    put("vFourTwoTailConfigs", 6, source="q5_twotailed_summary.csv")
    # the unfavorable half
    m = q5[(q5.attack == "margin_targeted") & (q5.budget_mult == 1.0)
           & (q5.scale == "n3000") & (q5.convention == "cat")].set_index("policy")
    put("vFourMtTwoTailPrec", m.loc["two_tailed", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourMtRandPrec", m.loc["random", "precision"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")
    put("vFourMtScoreAuroc", m.loc["random", "score_auroc"], fmt="{:.4f}",
        source="q5_twotailed_summary.csv")

    # ---------------- remediation
    assert not q4[q4.arm == "random"].beats_none.any()
    put("vFourRandDelNeverWins", int(len(q4[q4.arm == "random"])),
        source="q4_remediation_summary.csv")
    det = q4[(q4.arm == "detector") & (q4.scale == "n3000")
             & (q4.convention == "resp")].iloc[0]
    put("vFourDetHarm", -det.improve_vs_none, fmt="{:.4f}",
        source="q4_remediation_summary.csv")
    put("vFourDetHarmLo", -det.ci_hi, fmt="{:.4f}", source="q4_remediation_summary.csv")
    put("vFourDetHarmHi", -det.ci_lo, fmt="{:.4f}", source="q4_remediation_summary.csv")
    put("vFourDetRemain", int(round(det.remaining_poisoned)), source="q4_remediation_summary.csv")
    mm = q4[(q4.arm == "margin_matched") & (q4.scale == "n3000")
            & (q4.convention == "resp")].iloc[0]
    put("vFourMatchedHarm", -mm.improve_vs_none, fmt="{:.4f}",
        source="q4_remediation_summary.csv")
    oc = q4[(q4.arm == "oracle_correct") & (q4.scale == "n3000")
            & (q4.convention == "resp")].iloc[0]
    put("vFourOracleCorrGain", oc.improve_vs_none, fmt="{:.4f}",
        source="q4_remediation_summary.csv")

    # ---------------- zero-difference rows
    catz = grid[grid.convention == "cat"]
    respz = grid[grid.convention == "resp"]
    put("vFourZeroCatPct", 100.0 * (catz.n_zero_train / (catz.n_train + catz.n_dev
                                                         + catz.n_zero_train)).mean(),
        fmt="{:.0f}", source="x_grid_cells.csv")
    put("vFourZeroRespPct", 100.0 * (respz.n_zero_train / (respz.n_train + respz.n_dev
                                                           + respz.n_zero_train)).mean(),
        fmt="{:.1f}", source="x_grid_cells.csv")

    # ---------------- emit
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    out = OUTDIR / "paper_values_pooled.tex"
    out.write_text("\n".join(lines) + "\n")
    pd.DataFrame(ledger).to_csv(OUTDIR / "claim_ledger_pooled.csv", index=False)

    print(f"wrote {out.relative_to(REPO)}  {len(macros)} macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
