"""Independent pools, the N/d ladder, and clean-head validity.

One row per (convention, scale, pool, dim, lambda) cell. Every cell records
the clean head's predictive quality on the official untouched test split, the
directional attack, the poisoned refit, and the clean-head-versus-refit
detection contrast. Lambda is *recorded* over the whole preregistered grid and
*selected* later using dev only, so the selection can be audited.

Nothing here reads the test set to make a choice. The test block is used only
to evaluate a head that was already fit and a lambda that is recorded, never
compared, inside this script.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pooled import core as sc

OUT = sc.EVIDENCE / "x_grid_cells.csv"

SCALES = ["n300", "n1000", "n3000"]
POOLS = range(5)
CONVENTIONS = ["cat", "resp"]
ATTACK = "directional"


def grad_norm(Z: np.ndarray, s: np.ndarray, head) -> float:
    from src.data.types import PreferenceDataset

    ds = PreferenceDataset(Z=Z, s=s, flip_mask=np.zeros(Z.shape[0], bool), meta={})
    return float(np.linalg.norm(head.grad(ds)))


def run_cell(conv: str, scale: str, pool: int, dim: int, lam: float,
             prep: sc.Prepared) -> dict:
    ds_clean = sc.make_ds(prep.Ztr)
    budget = int(sc.PREVALENCE * prep.n_train)

    idx = sc.select_flips(ATTACK, ds_clean, budget, lam, seed=pool)
    ds_p = ds_clean.flipped(idx)

    h_clean = sc.fit(prep.Ztr, ds_clean.s, lam)
    h_ref = sc.fit(prep.Ztr, ds_p.s, lam)

    s_te = np.ones(prep.n_test)
    s_dev = np.ones(prep.n_dev)

    row = {
        "convention": conv, "scale": scale, "pool": pool, "dim": dim, "lam": lam,
        "n_train": prep.n_train, "n_dev": prep.n_dev, "n_test": prep.n_test,
        "n_zero_train": prep.n_zero_train, "n_zero_test": prep.n_zero_test,
        "budget": budget, "prevalence": float(ds_p.flip_mask.mean()),
        "n_over_d": prep.n_train / dim,
        # clean head quality, the only thing lambda selection may look at (dev)
        "clean_train_logloss": sc.data_logloss(prep.Ztr, ds_clean.s, h_clean.theta),
        "clean_dev_logloss": sc.data_logloss(prep.Zdev, s_dev, h_clean.theta),
        "clean_test_logloss": sc.data_logloss(prep.Zte, s_te, h_clean.theta),
        "clean_train_acc": sc.accuracy(prep.Ztr, ds_clean.s, h_clean.theta),
        "clean_dev_acc": sc.accuracy(prep.Zdev, s_dev, h_clean.theta),
        "clean_test_acc": sc.accuracy(prep.Zte, s_te, h_clean.theta),
        "clean_theta_norm": float(np.linalg.norm(h_clean.theta)),
        "clean_hessian_cond": sc.hessian_cond(prep.Ztr, ds_clean.s, h_clean.theta, lam),
        "clean_grad_norm": grad_norm(prep.Ztr, ds_clean.s, h_clean),
        # poisoned head quality
        "pois_test_logloss": sc.data_logloss(prep.Zte, s_te, h_ref.theta),
        "pois_test_acc": sc.accuracy(prep.Zte, s_te, h_ref.theta),
        "pois_theta_norm": float(np.linalg.norm(h_ref.theta)),
        "pois_grad_norm": grad_norm(prep.Ztr, ds_p.s, h_ref),
        "theta_dist": float(np.linalg.norm(h_ref.theta - h_clean.theta)),
        "theta_cos": float(
            h_ref.theta @ h_clean.theta
            / max(np.linalg.norm(h_ref.theta) * np.linalg.norm(h_clean.theta), 1e-300)
        ),
        "uninf_loss": sc.UNINF_LOSS,
        "clean_beats_uninf_test": bool(
            sc.data_logloss(prep.Zte, s_te, h_clean.theta) < sc.UNINF_LOSS),
    }

    # detection: original head vs poisoned refit, same rows, same Z
    for cond, head in (("cleanhead", h_clean), ("refit", h_ref)):
        row.update(sc.detection_metrics(
            sc.score_final_margin(ds_p, head), ds_p.flip_mask, budget,
            prefix=f"fm_{cond}_"))
        row.update(sc.detection_metrics(
            sc.score_self_influence(ds_p, head), ds_p.flip_mask, budget,
            prefix=f"si_{cond}_"))
    for k, v in sc.absorption(ds_p, h_ref).items():
        row[f"refit_{k}"] = v
    for k, v in sc.absorption(ds_p, h_clean).items():
        row[f"cleanhead_{k}"] = v
    row["inverted_separation_fm_refit"] = 1.0 - row["fm_refit_auroc"]
    return row


def main() -> int:
    t0 = time.time()
    rows = []
    for conv in CONVENTIONS:
        for scale in SCALES:
            for pool in POOLS:
                for dim in sc.DIMS:
                    prep = sc.prepare(scale, pool, conv, dim)
                    for lam in sc.LAMBDA_GRID:
                        rows.append(run_cell(conv, scale, pool, dim, lam, prep))
                print(f"  {conv}/{scale}/pool{pool}: done "
                      f"({len(rows)} cells, {time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(sc.REPO)}  rows={len(df)}  "
          f"cols={len(df.columns)}  {time.time()-t0:.1f}s")
    nc = int((df.clean_grad_norm > 1e-4).sum() + (df.pois_grad_norm > 1e-4).sum())
    print(f"non-converged fits (||grad||>1e-4): {nc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
