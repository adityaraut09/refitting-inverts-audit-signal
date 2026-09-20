import numpy as np

from src.data.embed import embed_pairs, register_embedder
from src.data.loaders import load_preference_subset


@register_embedder("_test-hash")
def _embed_hash(texts, model_name):
    """Deterministic, offline, dependency-free stand-in embedder for tests:
    hashes each text into a fixed-size vector so embed_pairs' orchestration
    (caching, difference features, dataset shape) can be tested without
    downloading a real model."""
    dim = 8
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        out[i] = rng.normal(size=dim)
    return out


def test_embed_pairs_shape_and_clean_labels():
    pairs = load_preference_subset("synthetic-toy", n=10, seed=0)
    ds = embed_pairs(pairs, embedder="_test-hash", use_cache=False)
    assert ds.Z.shape == (10, 8)
    assert np.all(ds.s == 1)
    assert not np.any(ds.flip_mask)
    assert ds.n == 10 and ds.d == 8


def test_embed_pairs_cache_roundtrip(tmp_path, monkeypatch):
    import src.utils.io as io

    monkeypatch.setattr(io, "CACHE_DIR", tmp_path)
    pairs = load_preference_subset("synthetic-toy", n=6, seed=1)

    ds1 = embed_pairs(pairs, embedder="_test-hash", use_cache=True)
    cached_files = list(tmp_path.glob("*.npz"))
    assert len(cached_files) == 1

    ds2 = embed_pairs(pairs, embedder="_test-hash", use_cache=True)
    np.testing.assert_array_equal(ds1.Z, ds2.Z)


def test_flipped_negates_only_selected_indices():
    pairs = load_preference_subset("synthetic-toy", n=5, seed=0)
    ds = embed_pairs(pairs, embedder="_test-hash", use_cache=False)
    flipped = ds.flipped([1, 3])
    assert flipped.s[1] == -1 and flipped.s[3] == -1
    assert flipped.s[0] == 1 and flipped.s[2] == 1 and flipped.s[4] == 1
    assert flipped.flip_mask.sum() == 2
    assert ds.s[1] == 1, "original dataset must be untouched"
