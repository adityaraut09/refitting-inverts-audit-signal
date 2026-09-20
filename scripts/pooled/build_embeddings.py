"""Embed every pool and the official untouched test split, once.

Writes caches under cache/pooled and a provenance manifest recording the
dataset file hashes, the encoder revision, the exact row indices per pool,
disjointness proof, and zero-difference counts.

Nothing here observes an outcome: it only turns text into frozen features.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from src.pooled import data as sd

# Committed before any result was observed. (scale, pool_n, n_pools)
POOL_SPEC = [("n3000", 3000, 5), ("n1000", 1000, 5), ("n300", 300, 5)]
CONVENTIONS = ["cat", "resp"]

OUT = sd.EVIDENCE / "embedding_manifest.json"


def main() -> int:
    t_start = time.time()
    train = sd.load_helpful_base("train")
    test = sd.load_helpful_base("test")
    print(f"helpful-base train rows: {len(train)}  sha256={train.file_sha256[:16]}")
    print(f"helpful-base test  rows: {len(test)}  sha256={test.file_sha256[:16]}")
    print(f"encoder {sd.MODEL_NAME} revision {sd.model_revision()[:16]}")

    pools = sd.allocate_pools(len(train), POOL_SPEC)
    sd.verify_disjoint(pools)
    print(f"allocated {len(pools)} disjoint pools, "
          f"{sum(len(v) for v in pools.values())} of {len(train)} train rows")

    manifest = {
        "pool_perm_seed": sd.POOL_PERM_SEED,
        "pool_spec": POOL_SPEC,
        "conventions": CONVENTIONS,
        "zero_norm_tol": sd.ZERO_NORM_TOL,
        "dataset": {
            "snapshot": sd.SNAPSHOT_REF,
            "train_rows": len(train),
            "train_file_sha256": train.file_sha256,
            "test_rows": len(test),
            "test_file_sha256": test.file_sha256,
            "test_is_official_split": True,
        },
        "encoder": {
            "model": sd.MODEL_NAME,
            "revision": sd.model_revision(),
            "d": 384,
            "max_seq_length": 256,
            "frozen": True,
        },
        "pools": {},
        "test_blocks": {},
    }

    test_idx = np.arange(len(test))
    for conv in CONVENTIONS:
        t0 = time.time()
        Zt = sd.embed_block(test, test_idx, conv)
        nz = sd.nonzero_mask(Zt)
        manifest["test_blocks"][conv] = {
            "n": int(Zt.shape[0]),
            "d": int(Zt.shape[1]),
            "n_nonzero": int(nz.sum()),
            "n_zero_norm": int((~nz).sum()),
            "zero_norm_frac": float((~nz).mean()),
            "Z_sha256_trunc": __import__("hashlib").sha256(Zt.tobytes()).hexdigest()[:16],
        }
        print(f"  test/{conv}: n={Zt.shape[0]} zero-norm={int((~nz).sum())} "
              f"({(~nz).mean():.4f})  {time.time() - t0:.1f}s")

    for name, idx in pools.items():
        entry = {
            "n": int(len(idx)),
            "index_sha256": __import__("hashlib").sha256(
                np.asarray(idx, dtype=np.int64).tobytes()).hexdigest(),
            "index_min": int(idx.min()),
            "index_max": int(idx.max()),
            "conventions": {},
        }
        for conv in CONVENTIONS:
            t0 = time.time()
            Z = sd.embed_block(train, idx, conv)
            nz = sd.nonzero_mask(Z)
            entry["conventions"][conv] = {
                "n_nonzero": int(nz.sum()),
                "n_zero_norm": int((~nz).sum()),
                "zero_norm_frac": float((~nz).mean()),
                "Z_sha256_trunc": __import__("hashlib").sha256(Z.tobytes()).hexdigest()[:16],
            }
            print(f"  {name}/{conv}: n={Z.shape[0]} zero-norm={int((~nz).sum())} "
                  f"({(~nz).mean():.4f})  {time.time() - t0:.1f}s")
        manifest["pools"][name] = entry

    manifest["wall_clock_seconds"] = round(time.time() - t_start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(sd.REPO)}  ({manifest['wall_clock_seconds']}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
