"""Retrieval of raw preference-pair datasets (no embedding yet).

Every loader returns a `RawPairs` object: parallel lists of (prompt, chosen,
rejected) text, plus provenance metadata. Embedding (src/data/embed.py)
turns these into difference features downstream.

To add a dataset: write a `_load_<name>(n, seed) -> RawPairs` function and
register it with `@register("<name>")`; it becomes available to
`load_preference_subset` immediately. Update the dataset table in README.md
to match the registry.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .types import RawPairs

_REGISTRY: dict[str, Callable[..., RawPairs]] = {}


def register(name: str):
    def deco(fn: Callable[..., RawPairs]) -> Callable[..., RawPairs]:
        _REGISTRY[name] = fn
        return fn

    return deco


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def load_preference_subset(name: str, n: int, seed: int = 0) -> RawPairs:
    """Load up to `n` preference pairs from dataset `name`, shuffled by `seed`."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {available_datasets()}")
    return _REGISTRY[name](n=n, seed=seed)


def _split_hh_transcript(text: str) -> tuple[str, str]:
    """hh-rlhf stores the full multi-turn transcript ending in the response;
    split at the final 'Assistant:' turn to recover (prompt, response)."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx + len(marker):].strip()


def _load_hh_rlhf(subset: str, n: int, seed: int) -> RawPairs:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    prompts, chosen, rejected, meta = [], [], [], []
    for row in ds:
        prompt, chosen_resp = _split_hh_transcript(row["chosen"])
        _, rejected_resp = _split_hh_transcript(row["rejected"])
        prompts.append(prompt)
        chosen.append(chosen_resp)
        rejected.append(rejected_resp)
        meta.append({"subset": subset})
    return RawPairs(prompts, chosen, rejected, meta, source=f"hh-rlhf/{subset}")


@register("hh-rlhf-helpful")
def _load_hh_rlhf_helpful(n: int, seed: int) -> RawPairs:
    return _load_hh_rlhf(subset="helpful-base", n=n, seed=seed)


@register("hh-rlhf-harmless")
def _load_hh_rlhf_harmless(n: int, seed: int) -> RawPairs:
    return _load_hh_rlhf(subset="harmless-base", n=n, seed=seed)


@register("shp")
def _load_shp(n: int, seed: int) -> RawPairs:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/SHP", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    prompts, chosen, rejected, meta = [], [], [], []
    for row in ds:
        a, b = row["human_ref_A"], row["human_ref_B"]
        c, r = (a, b) if row["labels"] == 1 else (b, a)
        prompts.append(row["history"])
        chosen.append(c)
        rejected.append(r)
        meta.append({"domain": row.get("domain"), "post_id": row.get("post_id")})
    return RawPairs(prompts, chosen, rejected, meta, source="shp")


@register("synthetic-toy")
def _load_synthetic_toy(n: int, seed: int) -> RawPairs:
    """Fully offline toy dataset: no downloads, deterministic given `seed`.

    Chosen is always the longer, more detailed reply, so every downstream
    step (embedding, fitting, attacking, detecting) can be sanity-checked
    without network access.
    """
    rng = np.random.default_rng(seed)
    topics = [
        "How do I bake sourdough bread?",
        "Explain how a binary search works.",
        "What causes ocean tides?",
        "Give me tips for learning a new language.",
        "How does a car engine convert fuel into motion?",
        "What is the difference between weather and climate?",
        "How do vaccines train the immune system?",
        "Explain what a hash table is.",
    ]
    prompts, chosen, rejected, meta = [], [], [], []
    for i in range(n):
        topic = topics[rng.integers(len(topics))]
        n_words = int(rng.integers(20, 60))
        chosen_resp = f"[detailed, {n_words} words] " + " ".join(["step"] * n_words)
        rejected_resp = "[terse] not sure."
        prompts.append(topic)
        chosen.append(chosen_resp)
        rejected.append(rejected_resp)
        meta.append({"idx": i})
    return RawPairs(prompts, chosen, rejected, meta, source="synthetic-toy")
