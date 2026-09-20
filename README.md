# When Refitting Inverts the Audit Signal: Post-hoc Detection of Poisoned Preference Labels

Code, configuration, tests and summary artifacts for reproducing the
experiments, table and figure reported in the paper.

## What the paper asks

An attacker reverses a minority of pairwise preference annotations. The natural
defence is to audit the data with the model trained on it: score each
comparison by how much the fitted model disagrees with it, review the most
suspicious few percent, and delete what looks corrupted. This repository
measures how well that works for a linear Bradley-Terry head on frozen
sentence-embedding features of HH-RLHF helpful-base comparisons.

Two results drive the paper, and both are negative for the naive audit.

1. **The ranking inverts after the poisoned refit.** Holding the data fixed and
   changing only the scoring head, a mutually aligned block of reversals is
   separated at AUROC 0.9936 under a head fit on the original labels and 0.2620
   under the head refit on the poisoned labels. A below-chance AUROC here is an
   inverted ranking, not an absent signal: the reversals end up with *larger*
   positive fitted margins than the retained comparisons, so disagreement-based
   scores rank them low. The original head is an oracle diagnostic, not a
   deployable detector.
2. **Localization succeeds and recovery does not follow.** Deletion precision
   beats the random-review baseline in all 140 depth-by-split cells, yet
   refitting after deletion raises validation logistic loss against taking no
   action in 17 of 20 splits at the budget-matched depth.

Every score's orientation is fixed so that higher means more suspicious; an
AUROC below 0.5 is reported as an inverted ranking and never as deployable
detection.

## Repository structure

```
src/
  attacks/ data/ detectors/ metrics/ model/ remediation/ utils/
                           the shared library
  pooled/                  pooled-replication library: pools, partitions, scores
scripts/
  benchmark/               one 300-comparison pool, 20 repeated splits
  pooled/                  15 disjoint pools, three scales, untouched test split
  make_main_figure.py      Figure 1
tests/                     pytest suite
docs/EXPERIMENTAL_PLAN.md  the prespecified plan and decision rules
results/                   frozen raw benchmark outputs (reproduction-gate references)
evidence/
  benchmark/               derived benchmark evidence
  pooled/                  derived pooled-replication evidence
tables/                    generated macro files and tables (see tables/README.md)
figures/                   generated Figure 1
cache/                     embedding caches (gitignored, not redistributed)
```

There is no `configs/` directory. Every experimental constant is declared at the
top of the module that owns it rather than in an external config file, because
the values were fixed in advance and a config file would make them look
adjustable. See **Seeds and fixed constants** below for where each one lives,
and `docs/EXPERIMENTAL_PLAN.md` for the plan that fixed them.

The two pipelines are kept separate throughout and no quantity is produced by
both:

* **benchmark** — one fixed 300-comparison pool, 20 repeated splits of it,
  three attacks, four audit scores, `lam = 1e-4`. Produces Table I, Table II and
  all three panels of Figure 1. Because all splits reuse one pool, every spread
  is an across-split standard deviation over repeated splits of that pool, not a
  population confidence interval.
* **pooled** — 15 provably disjoint pools at three scales, scored on the
  official HH-RLHF `helpful-base` test file, with lambda and the projection
  chosen on a tuning partition only. Produces the N/d ladder, the
  regularization sweep, the seven-arm remediation comparison and the two-tailed
  review policy.

## Environment

```bash
conda create -n refit-audit python=3.11 -y
conda activate refit-audit
pip install -r requirements.txt
```

Nothing needs to be installed from this repository. Tests pick up the package
via the `pythonpath` setting in `pyproject.toml`, and every script is run as a
module from the repository root (`python -m scripts....`), which puts the root
on `sys.path`. No script manipulates `sys.path` itself.

Verified on macOS 15 (Apple Silicon) with Python 3.11.16, numpy 2.4.6,
scipy 1.17.1, pandas 3.0.5, matplotlib 3.11.1, sentence-transformers 6.0.1,
datasets 5.0.1, pymupdf 1.28.2, pytest 9.1.1. No version metadata was recorded
alongside the original runs, so these are the versions under which the
reproduction described below was checked, not necessarily the versions that
first produced the numbers.

The encoder runs on Apple `mps` when available and falls back to CPU.

## Data and models

`Anthropic/hh-rlhf` (subset `helpful-base`) and
`sentence-transformers/all-MiniLM-L6-v2` (frozen, `d = 384`, 256-token window).
Both are fetched from the Hugging Face Hub on first use.

**Raw HH-RLHF data are not redistributed here.** No corpus text, no encoder
weights and no embedding caches are included. See [DATA.md](DATA.md) for how to
obtain them, the pinned snapshot revision, the recorded file hashes, and the
`HH_RLHF_HELPFUL_DIR` override.

## Tests

```bash
# everything: 88 tests
pytest

# data-free subset only: 66 tests, about 1 second
pytest --ignore=tests/test_pooled.py
```

Both work from any working directory.

`tests/test_pooled.py` (22 tests) needs HH-RLHF and the encoder. It asserts
pool disjointness, that train and test come from different files, that the
projection is fit on training rows only, that corrupting the test features
cannot change the selected lambda, score orientation, tie handling, and
determinism of flip selection.

## Reproducing the reported results

Run from the repository root. Order matters within each pipeline.

### Principal benchmark — Table II and Figure 1(a)

```bash
python -m scripts.benchmark.repeated_split_stability          # ~10 s + embedding
python -m scripts.benchmark.gradient_flip_forensics           # ~2 s
python -m scripts.benchmark.directional_orientation_sensitivity  # ~19 s
python -m scripts.benchmark.rebuild_e4_20splits               # ~3 s
python -m scripts.benchmark.make_evidence                     # ~14 s
```

`directional_orientation_sensitivity.py` is what supplies every directional
("gradient flip") number in the paper: it runs three arms (`raw`, `canon_plus`,
`canon_minus`) and the paper reports `canon_plus`. `make_evidence.py` runs 35
reproduction gates against `results/` before writing anything, and
`rebuild_e4_20splits.py` runs 133 seed-0 gates; both stop without writing if a
gate fails.

### Scale and regularization — the N/d ladder and the lambda grid

```bash
python -m scripts.pooled.build_embeddings     # one-off, ~10 min, writes cache/
python -m scripts.pooled.exp_grid             # 1050 cells: 2 conventions x 3 scales
                                              #   x 5 pools x 5 dims x 7 lambdas
```

Lambda is *recorded* over the whole grid `{1e-6 … 1e0}` and *selected* later on
dev only, so the selection can be audited. Dimensions `{32, 64, 128, 256, 384}`
come from a PCA basis fit on the training partition alone.

### Remediation — seven equal-size arms

```bash
python -m scripts.pooled.exp_remediation
```

Arms: no action; detector-ranked deletion; uniformly random deletion (averaged
over 10 repetitions); margin-matched honest deletion; oracle deletion of exactly
the reversed rows; oracle correction of exactly those labels; and the detector's
surviving rows evaluated with original labels. The benchmark-side remediation
experiment behind Figure 1(c) is `rebuild_e4_20splits.py`, above.

### Two-tailed audit

```bash
python -m scripts.pooled.exp_twotailed
```

Equal-budget review policies at budget multipliers `{0.5, 1.0, 2.0}`: top-B,
bottom-B (an oracle diagnostic only, since choosing the bottom requires already
knowing the ranking is inverted), two-tailed, and random.

### Adjudication, table and figure

`scripts.pooled.analyze` evaluates the five prespecified questions Q1-Q5.
Each rule, and the verdict it produces, is stated in
[docs/EXPERIMENTAL_PLAN.md](docs/EXPERIMENTAL_PLAN.md) and printed at run time.

```bash
python -m scripts.pooled.analyze                        # prints each prespecified
                                                        #   decision rule and its verdict
python -m scripts.pooled.make_values                    # 62 macros
python -m scripts.benchmark.make_values_and_tables      # 202 macros + all tables
python -m scripts.make_main_figure                      # Figure 1
```

### Output locations

| script | writes |
|---|---|
| `scripts.benchmark.repeated_split_stability` | `results/repeated_split_{e1,e3,stages,cohort,meta,summary,e3_trend}.csv` |
| `scripts.benchmark.gradient_flip_forensics` | `results/gradient_flip_forensics_*.{csv,json}` |
| `scripts.benchmark.directional_orientation_sensitivity` | `evidence/benchmark/directional_orientation/` |
| `scripts.benchmark.rebuild_e4_20splits` | `evidence/benchmark/e4_*.{csv,json}` |
| `scripts.benchmark.make_evidence` | `evidence/benchmark/{absorption_*,e1_per_seed_recomputed,e2_*,reproduction_gates,evidence_meta}` |
| `scripts.benchmark.make_values_and_tables` | `tables/paper_values_benchmark.tex`, `tables/claim_ledger_benchmark.csv`, `tables/table_{methods,methods_compact,e1,stages}.tex` |
| `scripts.pooled.build_embeddings` | `evidence/pooled/embedding_manifest.json`, `cache/pooled/` |
| `scripts.pooled.exp_grid` | `evidence/pooled/x_grid_cells.csv` |
| `scripts.pooled.exp_remediation` | `evidence/pooled/x_remediation_arms.csv` |
| `scripts.pooled.exp_twotailed` | `evidence/pooled/x_twotailed_policies.csv` |
| `scripts.pooled.analyze` | `evidence/pooled/q*.csv`, `evidence/pooled/adjudication_values.json` |
| `scripts.pooled.make_values` | `tables/paper_values_pooled.tex`, `tables/claim_ledger_pooled.csv` |
| `scripts.make_main_figure` | `figures/fig_main.{pdf,png}` |

The artifacts under `results/`, `evidence/`, `tables/` and `figures/` are the
reported ones. Re-running overwrites them in place, so `git diff` is the check
on whether a rerun moved a number. Nothing in `tables/` or `figures/` should be
edited by hand; [tables/README.md](tables/README.md) lists which generator owns
each file.

## What reproduced, and what did not

Checked on the environment above, comparing regenerated output byte-for-byte
against the artifacts in this repository.

| stage | result |
|---|---|
| 66 data-free tests | pass |
| 22 pooled tests | pass |
| `scripts.benchmark.gradient_flip_forensics` | all 4 artifacts identical |
| `scripts.benchmark.directional_orientation_sensitivity` | all 7 artifacts identical; 560/560 raw-arm gates pass |
| `scripts.benchmark.rebuild_e4_20splits` | 4 data artifacts identical; 133/133 seed-0 gates pass |
| `scripts.benchmark.make_evidence` | 9 artifacts identical; 35/35 gates pass |
| `scripts.pooled.analyze` | all 9 pooled evidence artifacts identical |
| `scripts.pooled.exp_grid`, `exp_remediation`, `exp_twotailed` | single-cell checks reproduce the recorded values exactly; full sweeps not re-run end to end |
| `scripts.pooled.build_embeddings` | embedding hash, corpus hashes and encoder revision all match the manifest |
| both macro generators | all 87 macros the manuscript uses regenerate to the reported values |
| `scripts.make_main_figure` | identical page geometry; panel (b) plotted means match the reported AUROCs |

**One known non-reproduction.** Re-running `scripts.benchmark.repeated_split_stability` today
does *not* reproduce the `gradient_flip` rows stored in
`results/repeated_split_{e1,stages,cohort,summary}.csv`. Those rows predate the
sign-canonicalization of the directional attack. `GradientFlipAttack` now forces
the largest-magnitude coordinate of the target eigenvector positive
(`canonical_sign`), because an eigenvector is defined only up to sign and the
two signs select opposite, disjoint tails; the stored rows used whatever sign
LAPACK happened to return. The three splits that still reproduce exactly
(seeds 0, 17, 18) are precisely the three where the raw sign already equalled
the canonical one.

This affects no reported number. Every directional-attack value in the paper is
sourced from `evidence/benchmark/directional_orientation/orientation_per_split.csv`
(arm `canon_plus`) or `evidence/benchmark/absorption_index_summary.csv`, both of
which reproduce byte-for-byte. The `random_flip` and `ambiguity_flip` rows, and
the entire E3 block that Figure 1(a) plots, also reproduce byte-for-byte. The
stored rows are retained because the `raw` arm of
`directional_orientation_sensitivity.py` gates against them, 560/560.

## Seeds and fixed constants

Everything below is deterministic given these values.

**Benchmark** (`scripts/benchmark/`)

| constant | value | where |
|---|---|---|
| source pool | `load_preference_subset("hh-rlhf-helpful", n=300, seed=0)` | each script's header |
| split seeds | 0–19, via `src.utils.seed.set_seed` | `SEEDS = range(20)` |
| holdout | 30 percent per split, not zero-norm filtered | `split()` |
| zero-norm filter | `‖z‖ ≤ 1e-8`, training side only | `split()` |
| ridge lambda | `1e-4` | `LAM` |
| attack budget | `int(0.15 * n_filtered)`, 26–27 reversals | `split()` |
| cross-fit folds | 5, `lam = 1e-4` | `N_FOLDS` |
| E4 deletion depths | 5, 10, 15, 20, 35, 50, plus `k = B` | `BASE_DEPTHS` |
| E2 budget grid | 0.001, 0.005, 0.01, 0.02, 0.03, 0.05 | `E2_FRACS` |

**Pooled** (`src/pooled/data.py`, `src/pooled/core.py`)

| constant | value |
|---|---|
| `POOL_PERM_SEED` | `20260920` — one permutation of the 43,835 training rows, then consecutive disjoint blocks |
| pool spec | `n3000` ×5, `n1000` ×5, `n300` ×5 = 21,500 of 43,835 rows |
| `DEV_FRAC` | `0.30`, split seed 0 |
| `PREVALENCE` | `0.15` of the training partition |
| `LAMBDA_GRID` | `1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0` |
| `DIMS` | `32, 64, 128, 256, 384` (384 means no projection) |
| `ZERO_NORM_TOL` | `1e-8`, applied in the original 384-d space before projection |
| attack seed | the pool index |
| random-deletion seeds | `7_000_000 + 1000*pool + rep`, 10 repetitions |
| two-tailed random seeds | `9_000_000 + 1000*pool + rep`, 10 repetitions |
| paired bootstrap | 20,000 resamples, seed 0 |

## Scope and limitations

These bound what the code here can show, and are stated in the paper.

* Every outcome is a property of a **linear Bradley-Terry head on frozen
  embeddings**. Nothing downstream of the reward head is run, so nothing here
  speaks to policy optimization, to generated text, or to a deep reward model
  with a trainable encoder.
* **One dataset**, HH-RLHF `helpful-base`. Pool-level variation is estimated
  only within that subset.
* The benchmark's 20 splits **share one 300-comparison pool**, and one of those
  splits selected lambda. Outside the pooled replication no number is
  out-of-sample, and no reported spread is a population confidence interval.
* The clean head being repaired is **weak in absolute terms** in the benchmark
  setting: fit on the full original data its mean validation logistic loss is
  0.7258, above the 0.6931 of a one-half predictor. The remediation results
  therefore do not establish remediation of an already useful reward model.
* The attacks are **fixed rules, not solutions to a worst-case problem**, so one
  rule's weakness bounds nothing about all rules at the same budget. Both
  targeted rules read the observed original labels, so guarantees proved under
  independent or blind label noise apply only to the random control.
* `precision_at_k` in `src/metrics/detection.py` uses an unstable argsort and
  does not average over ties. The pooled pipeline records `n_tied_at_cut`
  alongside every precision so a tie-dependent value can be recognized;
  `tests/test_pooled.py` pins this behaviour rather than hiding it.
* The **companion length-targeted study** mentioned in the paper's limitations
  is not part of this repository. It contributes no number to the five-page
  manuscript, and neither its generators nor its evidence are included.
* The manuscript's LaTeX sources are not included. `tables/` holds the
  generated macro files and tables that a manuscript would `\input`.

## Archived evidence, and what is not regenerated

Three files under `results/` are shipped as frozen references that **no script
in this repository regenerates**. They are retained because they are the
reference values that public reproduction gates check against, or because a
reported number was read from them.

| file | role | regenerated here? |
|---|---|---|
| `e1_benchmark.csv` | seed-0 reference for the reproduction gates in `scripts.benchmark.make_evidence` | no |
| `remediation_counterfactual.csv` | seed-0 reference for the 133 reproduction gates in `scripts.benchmark.rebuild_e4_20splits` | no |
| `holdout_reuse_audit.json` | record of the benchmark lambda-selection sweep | no |

For the first two, the *quantities* are recomputed from raw data by the scripts
named above and asserted equal to the stored values, so a drifted environment
is caught. Neither file contributes a macro to the manuscript.

`holdout_reuse_audit.json` is different and is the one real gap. It records the
lambda-selection sweep (validation pairwise accuracy on the seed-0 split over a
four-point grid from `1e-2` to `1e-5`, which selected `lam = 1e-4`). It supplies
six macros, of which the manuscript uses one: `vLamSelNGrid`, the size of that
grid. **No generator for it exists**, here or in the authors' working
repository, so that value is read from the stored file and is not recomputed by
anything. It is archived evidence, not a reproduced result.

**Reproduction here is therefore not complete end to end.** The lambda
selection that fixed `lam = 1e-4` is documented but not re-executable, and the
full pooled sweeps were verified cell-by-cell rather than re-run in their
entirety. Everything else listed under *What reproduced, and what did not* was
regenerated and compared byte-for-byte.
