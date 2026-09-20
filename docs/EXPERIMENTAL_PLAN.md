# Experimental plan for the pooled replication

This is the analysis plan for the pooled replication, written and fixed
**before any of its results were observed**. Feature construction (embedding)
had been launched at the time of writing, because embedding turns text into
frozen features and reports no outcome. No model had been fit, no AUROC
computed, and no test-set quantity inspected.

It is reproduced here so that the decision rules below can be checked against
what the code actually does. `scripts/pooled/analyze.py` prints each rule and
the verdict that rule produces, whether or not it favours the paper.

The benchmark pipeline (`scripts/benchmark/`) predates this plan and is not
covered by it.

## 1. What this stage is for

The benchmark reports that a margin-based score computed on the original,
pre-attack head separates reversed comparisons well, and that the same score
computed on a head refit on the poisoned data ranks them *below* chance. The
paper calls the second an inversion rather than an absence of signal.

Six standing objections say that result may be an artifact rather than a
property of poisoning:

| # | objection | what the pooled replication does about it |
|---|---|---|
| 1 | one reused 300-comparison pool | 15 provably disjoint pools |
| 2 | no untouched final test set | the official helpful-base `test.jsonl.gz` |
| 3 | weak clean head (test log-loss 0.7258 > log 2) | a prespecified search for a configuration with test log-loss below log 2, selected without test access |
| 4 | N < d only (175–183 training rows, d = 384) | an N/d ladder crossing N < d, N ≈ d and N > d |
| 5 | insufficient remediation controls | seven equal-size arms including two oracles |
| 6 | unknown detector orientation | an equal-budget two-tailed review policy that never uses labels to choose a direction |

## 2. Data, fixed before results

All data is read offline from a local snapshot. Nothing is downloaded at run
time. See `DATA.md` for how to obtain it.

| item | value |
|---|---|
| corpus | `Anthropic/hh-rlhf`, subset `helpful-base` |
| training file | `helpful-base/train.jsonl.gz`, 43,835 rows, sha256 `518a5bf288456fc9…` |
| test file | `helpful-base/test.jsonl.gz`, 2,354 rows, sha256 `8be3fc1a13b27901…` |
| encoder | `all-MiniLM-L6-v2`, revision `1110a243fdf4706b…`, frozen, d = 384, 256-token window |
| transcript split | `rfind("\n\nAssistant:")`, identical to `src/data/loaders.py` |

**The test file is a different file from the training file.** No partition
arithmetic can leak it. It is read once per reported configuration, after the
configuration is fixed on training and tuning data alone.

### 2.1 Pools

One permutation of the 43,835 training rows under fixed seed
`POOL_PERM_SEED = 20260920`, then consecutive disjoint blocks:

| scale | pool size | pools | rows used |
|---|---|---|---|
| `n3000` | 3,000 | 5 | 15,000 |
| `n1000` | 1,000 | 5 | 5,000 |
| `n300` | 300 | 5 | 1,500 |

21,500 of 43,835 rows. Disjointness is asserted pairwise in code
(`src.pooled.data.verify_disjoint`) and tested. **`n3000` is the primary
scale**, declared here before results, because it is the only scale where the
training partition exceeds d = 384 by a wide margin.

`n300` is the scale that reproduces the benchmark's regime (about 176 usable
training rows after filtering).

### 2.2 Partitions

Within each pool: **train 70% / dev 30%**, disjoint, split seed 0. Test is
always the official file, never a slice of a pool.

Hyperparameters are selected on **dev only**. The test set is not consulted for
any choice of lambda, dimension, embedding convention, or pool.

### 2.3 Embedding conventions, both prespecified

| id | text embedded | purpose |
|---|---|---|
| `cat` | `f"{prompt}\n\n{response}"` | exactly the benchmark's convention, including its truncation artifact |
| `resp` | the response text alone | removes the long shared prompt, so the 256-token window no longer truncates away the part that differs |

`cat` is the primary convention for reproducing the benchmark. `resp` is
declared here, in advance, as the candidate for a genuinely predictive clean
head, because the benchmark's own audit attributes the zero-difference rows to
prompt truncation rather than to the data. Choosing between them for any
reported claim uses dev log-loss only.

### 2.4 Exclusions

Rows with `‖z‖ ≤ 1e-8` are excluded from train, dev and the test target set, as
in the benchmark. Counts are reported for every pool and convention, never
silently dropped. A pool whose fit fails to converge is reported as an
implementation failure and kept in the table.

## 3. Factors

| factor | prespecified levels |
|---|---|
| pool scale | 300, 1000, 3000 |
| pool | 5 independent per scale |
| embedding | `cat`, `resp` |
| dimension | 32, 64, 128, 256, 384 by PCA **fit on the training partition only**; 384 means no projection |
| lambda | fixed log grid `1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0` |
| attack | directional (`gradient_flip`) as canonical; margin-targeted (`ambiguity_flip`) for remediation |
| poison prevalence | 0.15 of the training partition, held constant across scales |

PCA is fit on train only and applied unchanged to dev and test. This is
asserted in code and tested.

## 4. Metrics

Detection, computed against the reversal mask: AUROC, AUPRC, precision@B,
recall@B, precision lift over prevalence, and the exact integer counts behind
each precision.

Model quality, on the untouched test set: log-loss, accuracy, comparison
against log 2 = 0.693147 and against the constant one-half predictor,
train-versus-test loss gap, training accuracy.

Mechanism: signed-margin absorption, Hessian condition number, parameter
distance from the clean head, and rank agreement with the clean head.

Score orientation is fixed: every detector score increases with suspicion. An
AUROC below 0.5 is reported as an inverted ranking, never as absent signal, and
never as deployable detection.

## 5. Primary questions and the criteria that decide them

Criteria are stated in advance so that no result can be reinterpreted later.
"Primary pools" means the five `n3000` pools under the `cat` convention.

**Q1. Does the inversion reproduce across genuinely independent pools, scored
on an untouched test set?**
Reproduced if the original-head AUROC exceeds 0.5 and the poisoned-refit AUROC
falls below 0.5 in at least 4 of 5 primary pools. Failed to reproduce if that
holds in 2 or fewer. Anything else is inconclusive.

**Q2. Does the inversion survive N > d?**
Survives if, at `n3000` where the training partition is far larger than
d = 384, the poisoned-refit AUROC is below 0.5 in at least 4 of 5 pools. Does
not survive if that holds in 2 or fewer.

**Q3. Is there a useful clean head, and does the inversion persist under it?**
A configuration counts as useful only if it was selected on dev alone and its
clean test log-loss is below log 2 = 0.693147. The test set is not searched for
such a configuration. If no prespecified configuration qualifies, that is
reported plainly and Q3 returns "no useful clean head available", which
*narrows* the paper rather than supporting it. If one qualifies, the
clean-versus-poisoned comparison is repeated under it, and the two statements
"inversion exists under the original weak-head setting" and "inversion persists
under a useful clean head" are reported separately and never merged.

**Q4. Does any remediation arm beat no action?**
Seven equal-size arms: no action; detector-ranked deletion; uniformly random
deletion; margin-matched honest deletion; oracle deletion of exactly the
reversed rows; oracle correction flipping exactly those labels back; and the
same selected rows kept but evaluated with original labels. An arm beats no
action if the paired mean improvement in test log-loss across the 5 primary
pools is positive and a paired bootstrap 95% interval excludes zero. A
nonsignificant result is reported as nonsignificant and never as evidence of
equivalence.

**Q5. Does an orientation-agnostic policy help at equal budget?**
At one total review budget B: top-B; bottom-B as an oracle diagnostic only;
two-tailed with top ⌊B/2⌋ and bottom B − ⌊B/2⌋; and random B. Two-tailed helps
if it recovers more reversed rows than top-B in at least 4 of 5 primary pools
at identical total budget. Bottom-B is never presented as deployable, because
choosing the bottom requires knowing the direction.

## 6. Adjudication

Every result is classified as: strengthens the central claim; narrows it;
contradicts it; inconclusive; or implementation failure. A failed replication
is evidence and is reported. Unfavourable results are retained.

Forbidden, and checked afterwards: selecting a pool, dimension, lambda,
convention or seed after seeing test performance; rerunning until a conclusion
appears; reinterpreting dev numbers as test numbers; treating overlapping
splits as independent; describing a below-chance AUROC as deployable; or
claiming anything about deep reward models, trainable encoders, or downstream
policy from a frozen-feature linear head.
