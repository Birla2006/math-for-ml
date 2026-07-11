# Dot Product, Cosine Similarity & Nearest-Vector Search

A from-scratch look at the operation behind embeddings, RAG retrieval, and
attention: measuring how *aligned* two vectors are. Every primitive is written by
hand (explicit loops, no `np.dot` / `np.linalg`) and verified against NumPy.

## What it does

`similarity.py`:

1. **Primitives by hand** — `dot(a, b)`, `norm(a)`, and
   `cosine_similarity(a, b) = (a·b) / (|a||b|)`. Cosine normalizes out magnitude,
   so it compares *direction only* (the angle between vectors), which is what you
   want when comparing embeddings.
2. **Verification** — checks the hand-written cosine against NumPy
   (`np.dot` / `np.linalg.norm`) with `np.isclose` on same-direction, orthogonal,
   and opposite cases.
3. **Nearest-vector demo** — a tiny hand-made "embedding space" of 5 words. For a
   query word it ranks all vectors by cosine similarity and returns the top-k —
   exactly what a vector database does.

## Run

From the repo root (see the [root README](../README.md) for one-time venv setup):

```bash
.venv/bin/python similarity/similarity.py
```

## Sample output

```
STEP 1 — Correctness vs NumPy
same direction  [1,2,3] vs [2,4,6] | mine=+1.0000  numpy=+1.0000  match=True
orthogonal      [1,0]   vs [0,1]   | mine=+0.0000  numpy=+0.0000  match=True
opposite        [1,1]   vs [-1,-1] | mine=-1.0000  numpy=-1.0000  match=True

STEP 2 — Nearest-vector search (what a vector DB does)
nearest to 'king':  king (1.000), queen (0.992), man (0.880)
nearest to 'apple':  apple (1.000), banana (0.997), queen (0.268)
```

## Takeaway

The hand-written cosine matches NumPy exactly. In the demo space, royalty words
(`king ~ queen ~ man`) cluster together and fruit words (`apple ~ banana`) cluster
together, with almost no cross-similarity between the two groups.

That ranking — "which stored vectors point most like my query?" — **is** RAG
retrieval. Softened (via softmax) over every token in a sequence, the same dot-
product-similarity is **attention**. One small operation, scaled up, underpins
modern ML.