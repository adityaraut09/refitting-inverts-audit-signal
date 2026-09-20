"""Turn RawPairs into a PreferenceDataset via a frozen embedder.

Two embedding backends are registered below: "sentence-transformer" (default,
CPU-fast) and "hf-hidden-state" (mean-pooled hidden states of a frozen causal
LM, closer to what a real reward model sees). Both are frozen — nothing here
is trained. This module is the *only* place raw text becomes feature
vectors; everything downstream is linear algebra on the resulting Z.

To add a backend: write `_embed_<name>(texts, model_name) -> np.ndarray` and
register it with `@register_embedder("<name>")`.

`make_synthetic_dataset` (below) is the zero-download escape hatch: it
fabricates `Z` directly, bypassing text and embedders entirely, for testing
or demoing `BTHead` without a network call.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from ..utils.io import load_npz_cache, save_npz_cache
from .types import PreferenceDataset, RawPairs

_EMBEDDER_REGISTRY: dict[str, Callable[[list[str], str], np.ndarray]] = {}
_DEFAULT_MODEL = {
    "sentence-transformer": "all-MiniLM-L6-v2",
    "hf-hidden-state": "EleutherAI/pythia-160m",
}


def register_embedder(name: str):
    def deco(fn: Callable[[list[str], str], np.ndarray]) -> Callable[[list[str], str], np.ndarray]:
        _EMBEDDER_REGISTRY[name] = fn
        return fn

    return deco


def available_embedders() -> list[str]:
    return sorted(_EMBEDDER_REGISTRY)


@register_embedder("sentence-transformer")
def _embed_sentence_transformer(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float64)


@register_embedder("hf-hidden-state")
def _embed_hf_hidden_state(texts: list[str], model_name: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batch_size = 16
    pooled_batches = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled_batches.append((summed / counts).numpy())
    return np.concatenate(pooled_batches, axis=0).astype(np.float64)


def embed_texts(
    texts: list[str], embedder: str = "sentence-transformer", model_name: str | None = None
) -> np.ndarray:
    """Embed raw strings directly with a frozen embedder — the single-text
    counterpart to `embed_pairs`' difference features. Scores an arbitrary
    individual (prompt, response) with a fitted BTHead's theta, outside the
    fixed chosen/rejected pair structure. Not used by the paper experiments,
    which all operate on difference features.
    """
    if embedder not in _EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder '{embedder}'. Available: {available_embedders()}")
    resolved_model = model_name or _DEFAULT_MODEL.get(embedder, "default")
    return _EMBEDDER_REGISTRY[embedder](texts, resolved_model).astype(np.float64)


def _cache_key(pairs: RawPairs, embedder: str, model_name: str) -> str:
    h = hashlib.sha1()
    h.update(f"{pairs.source}|{embedder}|{model_name}".encode())
    for p, c, r in zip(pairs.prompts, pairs.chosen, pairs.rejected):
        h.update(p.encode())
        h.update(c.encode())
        h.update(r.encode())
    safe_source = pairs.source.replace("/", "-")
    return f"{safe_source}_{embedder}_{model_name.replace('/', '-')}_{h.hexdigest()[:10]}"


def embed_pairs(
    pairs: RawPairs,
    embedder: str = "sentence-transformer",
    model_name: str | None = None,
    use_cache: bool = True,
) -> PreferenceDataset:
    """Embed `pairs` into a PreferenceDataset of clean-orientation difference
    features: `z_i = phi(prompt_i, chosen_i) - phi(prompt_i, rejected_i)`.
    All labels start clean (s_i = +1); attacks flip afterward.
    """
    if embedder not in _EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder '{embedder}'. Available: {available_embedders()}")
    resolved_model = model_name or _DEFAULT_MODEL.get(embedder, "default")
    n = len(pairs)
    meta = {
        "source": pairs.source,
        "embedder": embedder,
        "model_name": resolved_model,
        "pair_meta": pairs.meta,
    }

    key = _cache_key(pairs, embedder, resolved_model)
    if use_cache:
        cached = load_npz_cache(key)
        if cached is not None:
            return PreferenceDataset(
                Z=cached["Z"], s=np.ones(n), flip_mask=np.zeros(n, dtype=bool), meta=meta
            )

    embed_fn = _EMBEDDER_REGISTRY[embedder]
    chosen_texts = [f"{p}\n\n{c}" for p, c in zip(pairs.prompts, pairs.chosen)]
    rejected_texts = [f"{p}\n\n{r}" for p, r in zip(pairs.prompts, pairs.rejected)]
    phi_chosen = embed_fn(chosen_texts, resolved_model)
    phi_rejected = embed_fn(rejected_texts, resolved_model)
    Z = (phi_chosen - phi_rejected).astype(np.float64)

    if use_cache:
        save_npz_cache(key, Z=Z)

    return PreferenceDataset(Z=Z, s=np.ones(n), flip_mask=np.zeros(n, dtype=bool), meta=meta)


def make_synthetic_dataset(n: int, d: int, separation: float = 1.0, seed: int = 0) -> PreferenceDataset:
    """Zero-download synthetic *features* (not text): fabricates Z directly so
    BTHead can be fit/tested/demoed without pulling any model.

    Draws a random ground-truth direction `w_true` and sets
    `z_i = separation * w_true + noise_i`, so the true preference margin
    `z_i . w_true` gets stronger (more separable, easier to learn) as
    `separation` grows; `separation=0` is pure noise. Labels start clean
    (s_i = +1); `w_true`/`separation` are stashed in `meta` for evaluation
    only (never given to detectors/attacks).
    """
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    Z = separation * w_true + rng.normal(size=(n, d))
    return PreferenceDataset(
        Z=Z,
        s=np.ones(n),
        flip_mask=np.zeros(n, dtype=bool),
        meta={"source": "synthetic-features", "w_true": w_true, "separation": separation, "seed": seed},
    )
