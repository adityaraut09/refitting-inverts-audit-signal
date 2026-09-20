"""Offline helpful-base loading, disjoint pool allocation, and embedding.

Everything here is deterministic and fully offline. Three properties matter:

1.  The official HH-RLHF helpful-base ``test.jsonl.gz`` split is read as a
    genuinely untouched test set. It is a different file from the training
    file, so no partition arithmetic can leak it.
2.  Pools are carved as disjoint blocks of one fixed permutation of the
    training file, so two pools provably share no comparison.
3.  Embedding caches are written under ``cache/pooled``, keyed by the source
    file hash, the row indices, the convention and the encoder revision. This
    module deliberately does NOT reuse ``src.data.embed.embed_pairs``' cache,
    which is keyed on the benchmark pipeline's own conventions.

Two embedding conventions are supported, both preregistered:

``cat``   ``f"{prompt}\\n\\n{response}"`` truncated to the encoder's 256-token
          window. This is exactly the benchmark pipeline's convention (see
          ``src/data/embed.py``), including its truncation artifact.
``resp``  the response text alone, which removes the long shared prompt and
          therefore removes the truncation artifact.

The corpus is read offline from a local Hugging Face snapshot. Set
``HH_RLHF_HELPFUL_DIR`` to point at a directory holding ``train.jsonl.gz``
and ``test.jsonl.gz`` if the snapshot lives elsewhere; see ``DATA.md``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "evidence" / "pooled"
CACHE = REPO / "cache" / "pooled"

HUB = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
) / "hub" / "datasets--Anthropic--hh-rlhf"
# The snapshot the reported results were produced from. Pinned, because the
# pool allocation indexes row positions in these exact files.
SNAPSHOT_REF = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
HELPFUL = Path(os.environ["HH_RLHF_HELPFUL_DIR"]) if os.environ.get(
    "HH_RLHF_HELPFUL_DIR") else HUB / "snapshots" / SNAPSHOT_REF / "helpful-base"

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_HUB = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
) / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"

# One fixed permutation seed, chosen before any result was observed.
POOL_PERM_SEED = 20260920
ZERO_NORM_TOL = 1e-8


# --------------------------------------------------------------------------
# raw text
# --------------------------------------------------------------------------
def _split_hh_transcript(text: str) -> tuple[str, str]:
    """Identical to ``src/data/loaders.py::_split_hh_transcript``."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx + len(marker):].strip()


@dataclass(frozen=True)
class RawSplit:
    prompts: list[str]
    chosen: list[str]
    rejected: list[str]
    source: str
    file_sha256: str

    def __len__(self) -> int:
        return len(self.prompts)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4)
def load_helpful_base(split: str) -> RawSplit:
    """Read helpful-base ``train`` or ``test`` straight from the local snapshot.

    Row order is the file's own order, never shuffled here, so any later
    permutation is reproducible from ``POOL_PERM_SEED`` alone.

    Memoized per process. ``RawSplit`` is frozen and never mutated, so sharing
    one instance across callers cannot change a result.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    path = HELPFUL / f"{split}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"helpful-base {split} not found at {path}. This module is offline only."
        )
    prompts, chosen, rejected = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p_c, c = _split_hh_transcript(row["chosen"])
            _, r = _split_hh_transcript(row["rejected"])
            prompts.append(p_c)
            chosen.append(c)
            rejected.append(r)
    return RawSplit(prompts, chosen, rejected, f"hh-rlhf/helpful-base/{split}",
                    _file_sha256(path))


def model_revision() -> str:
    ref = MODEL_HUB / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


# --------------------------------------------------------------------------
# disjoint pool allocation
# --------------------------------------------------------------------------
def allocate_pools(n_train_rows: int, spec: list[tuple[str, int, int]]) -> dict[str, np.ndarray]:
    """Carve disjoint index blocks from one fixed permutation of the train file.

    ``spec`` is a list of ``(scale_name, pool_n, n_pools)``. Blocks are taken
    consecutively from the permutation in the order given, so allocation is a
    pure function of ``POOL_PERM_SEED``, ``n_train_rows`` and ``spec``.

    Returns ``{"<scale>_pool<k>": indices}``. Every pair of returned arrays is
    disjoint by construction; ``verify_disjoint`` re-checks that as a test.
    """
    rng = np.random.default_rng(POOL_PERM_SEED)
    perm = rng.permutation(n_train_rows)
    need = sum(pool_n * n_pools for _, pool_n, n_pools in spec)
    if need > n_train_rows:
        raise ValueError(f"spec needs {need} rows, train file has {n_train_rows}")
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for scale, pool_n, n_pools in spec:
        for k in range(n_pools):
            out[f"{scale}_pool{k}"] = np.sort(perm[cursor:cursor + pool_n])
            cursor += pool_n
    return out


def verify_disjoint(pools: dict[str, np.ndarray]) -> None:
    keys = sorted(pools)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = np.intersect1d(pools[a], pools[b])
            if overlap.size:
                raise AssertionError(f"pools {a} and {b} overlap in {overlap.size} rows")


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
def _texts(raw: RawSplit, idx: np.ndarray, convention: str) -> tuple[list[str], list[str]]:
    if convention == "cat":
        a = [f"{raw.prompts[i]}\n\n{raw.chosen[i]}" for i in idx]
        b = [f"{raw.prompts[i]}\n\n{raw.rejected[i]}" for i in idx]
    elif convention == "resp":
        a = [raw.chosen[i] for i in idx]
        b = [raw.rejected[i] for i in idx]
    else:
        raise ValueError(f"unknown convention {convention!r}")
    return a, b


_MODEL = None


def _encoder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(MODEL_NAME)
        try:
            m = m.to("mps")
        except Exception:
            pass
        _MODEL = m
    return _MODEL


def embed_block(raw: RawSplit, idx: np.ndarray, convention: str,
                batch_size: int = 64) -> np.ndarray:
    """Return difference features ``Z = phi(chosen) - phi(rejected)``.

    Cached under ``cache/pooled``, keyed by the source file hash,
    the exact row indices, the convention and the model revision, so a cache
    hit can only ever correspond to identical inputs.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    h.update(f"{raw.source}|{raw.file_sha256}|{convention}|{MODEL_NAME}|{model_revision()}".encode())
    h.update(np.asarray(idx, dtype=np.int64).tobytes())
    key = f"{raw.source.replace('/', '-')}_{convention}_{len(idx)}_{h.hexdigest()[:16]}"
    if key in _MEM:
        return _MEM[key]
    path = CACHE / f"{key}.npz"
    if path.exists():
        with np.load(path) as data:
            Z = data["Z"]
        _remember(key, Z)
        return Z

    a, b = _texts(raw, idx, convention)
    enc = _encoder()
    pa = enc.encode(a, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
    pb = enc.encode(b, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
    Z = (np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64))
    np.savez_compressed(path, Z=Z)
    _remember(key, Z)
    return Z


# Small in-process memo so a sweep over dimensions does not reread the same
# .npz once per dimension. Bounded because one n3000 block is about 9 MB.
_MEM: dict[str, np.ndarray] = {}
_MEM_MAX = 6


def _remember(key: str, Z: np.ndarray) -> None:
    if len(_MEM) >= _MEM_MAX:
        _MEM.pop(next(iter(_MEM)))
    _MEM[key] = Z


def nonzero_mask(Z: np.ndarray) -> np.ndarray:
    """Zero-difference filter: keep rows with ``||z|| > 1e-8``."""
    return np.linalg.norm(Z, axis=1) > ZERO_NORM_TOL
