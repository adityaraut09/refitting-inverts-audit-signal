# Data

No corpus is redistributed in this repository. Raw HH-RLHF comparisons, the
encoder weights, and the embedding caches derived from them are all absent and
must be obtained from their original sources. Everything under `results/`,
`evidence/`, `tables/` and `figures/` is a derived summary artifact.

## What you need

| item | source | size |
|---|---|---|
| `Anthropic/hh-rlhf`, subset `helpful-base` | Hugging Face Hub | ~50 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face Hub | ~90 MB |

Both are downloaded automatically on first use into the standard Hugging Face
cache (`$HF_HOME`, default `~/.cache/huggingface`). Nothing else is fetched.

```bash
python -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
load_dataset('Anthropic/hh-rlhf', data_dir='helpful-base', split='train')
snapshot_download('Anthropic/hh-rlhf')
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
"
```

Please read and accept the licence and intended-use terms attached to
`Anthropic/hh-rlhf` on the Hub before using it. This repository states no
licence for that data and grants no rights to it.

## The two pipelines read the corpus differently

`scripts/benchmark/` goes through `datasets.load_dataset` (`src/data/loaders.py`),
which resolves the dataset through the `datasets` library cache.

`scripts/pooled/` reads `helpful-base/train.jsonl.gz` and `test.jsonl.gz`
directly off a local Hub snapshot, fully offline, because the pool allocation
indexes row positions in those exact files. The snapshot revision is pinned in
`src/pooled/data.py`:

```
SNAPSHOT_REF = 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa
train.jsonl.gz   43,835 rows   sha256 518a5bf288456fc9...
test.jsonl.gz     2,354 rows   sha256 8be3fc1a13b27901...
```

If your snapshot lives elsewhere, point at the directory holding the two files:

```bash
export HH_RLHF_HELPFUL_DIR=/path/to/hh-rlhf/helpful-base
```

The file hashes and the encoder revision (`1110a243fdf4706b...`) are recorded in
`evidence/pooled/embedding_manifest.json` and are re-derived on every run, so a
substituted corpus shows up as a hash mismatch rather than as silently
different numbers.

## Embedding caches

Difference features are cached under `cache/`, which is gitignored. They are
regenerated deterministically from the corpus and the frozen encoder; the
pooled cache key includes the source file hash, the row indices, the embedding
convention and the encoder revision, so a cache hit can only correspond to
identical inputs. Expect roughly 20 MB of cache for the full pooled sweep.

## Dataset registry

`src/data/loaders.py` registers four datasets. Only two are exercised here.

| name | source | used by |
|---|---|---|
| `hh-rlhf-helpful` | `Anthropic/hh-rlhf`, `helpful-base` | every reported experiment |
| `synthetic-toy` | generated offline, no download | tests |
| `hh-rlhf-harmless` | `Anthropic/hh-rlhf`, `harmless-base` | not used by the paper |
| `shp` | `stanfordnlp/SHP` | not used by the paper |
