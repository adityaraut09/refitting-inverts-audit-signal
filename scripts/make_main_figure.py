"""Build Figure 1 of the manuscript, all three panels.

Inputs, all read-only:

  panel (a)  results/repeated_split_summary.csv                  block=E3
  panel (b)  evidence/benchmark/directional_orientation/
             orientation_per_split.csv                           arm=canon_plus
  panel (c)  evidence/benchmark/e4_per_seed_20splits.csv

Several layout choices here are load-bearing rather than cosmetic and are
asserted so they cannot regress silently. ``assert_legend_inside_axes`` fails
if a legend crosses its own axes box; the saved width is asserted against the
IEEE two-column text width, because a wider figure is silently downscaled by
pdfLaTeX and every label in all three panels shrinks with it; and the on-page
height is asserted against the budget the manuscript's page count depends on.
Each is explained at the point where it is enforced.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# The backend must be selected before pyplot is imported, so this import
# cannot be hoisted with the others.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
EV = REPO / "evidence" / "benchmark"
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, colour-blind safe
OK = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
      "yellow": "#F0E442", "grey": "#7F7F7F"}
DETS = ("self_influence", "cross_fit", "training_dynamics", "ensemble")
LBL = {"self_influence": "self-infl.", "cross_fit": "out-of-fold",
       "training_dynamics": "final margin", "ensemble": "ensemble"}
COL = {"self_influence": OK["purple"], "cross_fit": OK["blue"],
       "training_dynamics": OK["vermillion"], "ensemble": OK["green"]}

RC = {"axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
      "lines.linewidth": 1.1, "figure.dpi": 150, "savefig.bbox": "tight",
      "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False}

summ = pd.read_csv(RESULTS / "repeated_split_summary.csv")
ORI = pd.read_csv(EV / "directional_orientation" / "orientation_per_split.csv")
CANON = ORI[ORI.arm == "canon_plus"]
e4 = pd.read_csv(EV / "e4_per_seed_20splits.csv")

# ordinal depth grouping: depths differ per split because k = B does
e4 = e4.assign(depth_ord=e4.groupby("seed").k.rank(method="dense").astype(int))


def by_depth(frame, series):
    """Mean, sd and x position per ordinal depth, each over all 20 splits."""
    g = frame.assign(_v=series).groupby("depth_ord")
    return g.k.mean(), g._v.mean(), g._v.std(), g._v.size()


def style(fs):
    plt.rcParams.update({**RC, "font.size": fs, "axes.labelsize": fs + 0.2,
                         "axes.titlesize": fs + 0.7, "xtick.labelsize": fs - 0.5,
                         "ytick.labelsize": fs - 0.5, "legend.fontsize": fs - 1.0})


def assert_legend_inside_axes(fig, ax, name: str, slack: float = 0.5) -> None:
    """Fail if a legend crosses its own axes box.

    A legend wider than its axes and centred on them is drawn over the
    y-axis spine. Checked in display pixels with a small slack for
    antialiasing.
    """
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None, f"{name}: no legend to check"
    lb = leg.get_window_extent(fig.canvas.get_renderer())
    ab = ax.get_window_extent()
    problems = []
    if lb.x0 < ab.x0 - slack:
        problems.append(f"overflows left by {ab.x0 - lb.x0:.1f}px (crosses the y-axis)")
    if lb.x1 > ab.x1 + slack:
        problems.append(f"overflows right by {lb.x1 - ab.x1:.1f}px")
    if lb.y1 > ab.y1 + slack:
        problems.append(f"overflows top by {lb.y1 - ab.y1:.1f}px")
    if lb.y0 < ab.y0 - slack:
        problems.append(f"overflows bottom by {ab.y0 - lb.y0:.1f}px")
    assert not problems, f"{name} legend {'; '.join(problems)}"
    print(f"    {name}: legend inside axes "
          f"(left margin {lb.x0 - ab.x0:+.1f}px, right {ab.x1 - lb.x1:+.1f}px, "
          f"top {ab.y1 - lb.y1:+.1f}px)")


# ============================== Figure 1 ====================================
def panel_e3(ax):
    e3 = summ[(summ.block == "E3") & (summ.metric == "auroc")]
    ax.axhline(0.5, color=OK["grey"], lw=0.7, ls=":")
    ax.annotate("chance", xy=(0.80, 0.515), fontsize=6.0, color=OK["grey"])
    for dd in ("self_influence", "cross_fit", "training_dynamics"):
        g = e3[e3.detector == dd].sort_values("alpha")
        ax.fill_between(g.alpha, g["mean"] - g["sd"], g["mean"] + g["sd"],
                        color=COL[dd], alpha=0.14, lw=0)
        ax.plot(g.alpha, g["mean"], marker="o", ms=2.8, color=COL[dd], label=LBL[dd])
    ax.set_xlabel(r"$\alpha$:  0 = highest reference margin $\rightarrow$ 1 = lowest")
    ax.set_ylabel("AUROC")
    ax.set_title("(a) margin-targeted sweep", loc="left")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.20, 0.92)
    # Ticks are set explicitly. The automatic locator adds a 1.0 tick, which
    # is outside ylim so its mark is not drawn, but its label stays in the
    # layout and overflows the canvas top. These are the eight actually drawn.
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.legend(loc="upper left", borderpad=0.1, labelspacing=0.18, handlelength=1.3)


def panel_stages(ax):
    """Canonical +v orientation. The stored raw-orientation run used a
    split-dependent mixture of the two tails and is not plotted."""
    order = ["pre_attack", "clean_head", "poisoned_refit"]
    short = ["orig. lab.\norig. head", "rev. lab.\norig. head", "rev. lab.\npois. refit"]
    x = np.arange(len(order))
    w = 0.2
    ax.axhline(0.5, color=OK["grey"], lw=0.7, ls=":")
    for i, dd in enumerate(DETS):
        vals = [CANON[f"auroc_{s}__{dd}"].mean() for s in order]
        errs = [CANON[f"auroc_{s}__{dd}"].std() for s in order]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, color=COL[dd], label=LBL[dd],
               error_kw={"lw": 0.6, "capsize": 1.1})
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.32)
    ax.set_title("(b) directional stress test, by stage", loc="left")
    # Two columns inside the empty upper-left of the axes: a four-column
    # legend of these labels is wider than the axes and overflows both sides.
    # The upper left is empty here because the tallest bar plus its error bar
    # reaches about 1.01 while the axis runs to 1.32.
    ax.legend(loc="upper left", ncol=2, borderpad=0.2, labelspacing=0.25,
              columnspacing=0.8, handlelength=0.9, handletextpad=0.4,
              borderaxespad=0.35)


def panel_remediation(ax):
    """Margin-targeted attack, final-margin ranking, 20 splits, 7 depths."""
    ax.axhline(0, color="black", lw=0.8)
    for col, lab, c, mk in (
            ("logloss_remediated", "delete and refit", OK["blue"], "o"),
            ("logloss_matched_clean_cf", "same rows, original labels", OK["green"], "s")):
        xs, m, s, n = by_depth(e4, e4[col] - e4.logloss_unsafe)
        assert (n == 20).all(), n.to_dict()
        ax.fill_between(xs, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.plot(xs, m, marker=mk, ms=3.0, color=c, label=lab)
    xs, m, _, _ = by_depth(e4, e4.logloss_full_clean - e4.logloss_unsafe)
    ax.plot(xs, m, color=OK["grey"], lw=0.9, ls="--", label="full original data")
    bmin, bmax = int(e4.budget.min()), int(e4.budget.max())
    ax.axvline(e4.budget.mean(), color=OK["grey"], lw=0.5, ls=":")
    ax.annotate(f"budget $B$={bmin}-{bmax}", xy=(e4.budget.mean() + 1.2, -0.049),
                fontsize=6.0, color=OK["grey"])
    ax.set_xlabel("$k$, comparisons deleted and refit")
    ax.set_ylabel(r"$\Delta$ validation loss vs. no action")
    # Wrapped at the semicolon. On one line this title runs 0.52 in past the
    # figure box, which the tight bbox absorbs by widening the saved PDF; the
    # figure is then downscaled at \textwidth and every label in all three
    # panels shrinks. Wrapping keeps the three axes aligned, because
    # tight_layout gives the row one common box.
    ax.set_title("(c) recovery does not follow;\nhigher is worse", loc="left")
    # Ticks are set explicitly. The automatic locator adds 0 and 60; both lie
    # outside xlim (2.75, 52.25) so neither mark is drawn, but the "60" label
    # stays in the layout 0.31 in beyond the axes, inflating the saved figure
    # and forcing a downscale at \textwidth. These are the five actually drawn.
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.legend(loc="upper left", borderpad=0.1, labelspacing=0.18, handlelength=1.5)


TEXTWIDTH_IN = 7.16   # IEEE two-column text width, what the figure* is placed at
BASE_FS = 7.0

# The figure must occupy no more than 7.16 x 2.210 in on the page or the
# manuscript runs to six pages. Because the figure is placed without a
# downscale, the figsize height is set so the on-page footprint stays inside
# that budget while the type renders at about 7 pt.
ONPAGE_HEIGHT_BUDGET_IN = 2.210
FIGSIZE_H = 2.20


def main() -> int:
    style(BASE_FS)
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, FIGSIZE_H))
    panel_e3(axs[0])
    panel_stages(axs[1])
    panel_remediation(axs[2])
    fig.tight_layout(pad=0.3, w_pad=1.0)

    print("  legend containment checks:")
    assert_legend_inside_axes(fig, axs[0], "panel (a)")
    assert_legend_inside_axes(fig, axs[1], "panel (b)")
    assert_legend_inside_axes(fig, axs[2], "panel (c)")

    # The manuscript includes this at \textwidth = 7.16 in. Any saved width
    # above that is silently downscaled by pdfLaTeX, shrinking every label in
    # the figure. Default tight-bbox padding alone added 0.2 in, and the two
    # out-of-range tick labels handled above add another 0.35 in. Padding is
    # reduced to a hairline and the resulting width is asserted.
    p = OUT / "fig_main.pdf"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(p.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    import pymupdf

    r = pymupdf.open(p)[0].rect
    w_in, h_in = r.width / 72.0, r.height / 72.0
    scale = TEXTWIDTH_IN / w_in
    onpage_h = h_in * scale
    print(f"  saved {w_in:.3f} x {h_in:.3f} in; placed at {TEXTWIDTH_IN} in "
          f"-> scale {scale:.4f}, on-page {TEXTWIDTH_IN:.2f} x {onpage_h:.3f} in, "
          f"effective base font {BASE_FS * scale:.2f} pt")
    assert scale >= 0.97, (
        f"figure would be downscaled to {scale:.3f} when placed at "
        f"{TEXTWIDTH_IN} in, shrinking all figure type")
    assert onpage_h <= ONPAGE_HEIGHT_BUDGET_IN, (
        f"on-page height {onpage_h:.3f} in exceeds the {ONPAGE_HEIGHT_BUDGET_IN} in "
        f"budget; the manuscript would run to six pages")
    print(f"  wrote {p} and {p.with_suffix('.png').name}")

    # plotted values, for cross-checking that nothing moved
    print("\n  panel (b) plotted means, canon_plus:")
    for dd in DETS:
        vals = [CANON[f"auroc_{s}__{dd}"].mean()
                for s in ("pre_attack", "clean_head", "poisoned_refit")]
        sds = [CANON[f"auroc_{s}__{dd}"].std()
               for s in ("pre_attack", "clean_head", "poisoned_refit")]
        print(f"    {LBL[dd]:13s} " + "  ".join(f"{v:.6f}+-{s:.6f}"
                                                for v, s in zip(vals, sds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
