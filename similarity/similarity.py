"""Dot product, cosine similarity, and a nearest-vector demo — by hand vs NumPy.

The same idea that powers embeddings, RAG retrieval, and attention:
  a . b   measures how aligned two vectors are.
  cosine  normalizes out length, leaving pure direction (the angle between them).
  nearest ranks a set of vectors by how similar they are to a query.

Verifies the hand-written versions against NumPy with np.allclose.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Primitives (by hand — explicit loops, no np.dot / np.linalg)
# ---------------------------------------------------------------------------
def dot(a, b):
    """Dot product of two equal-length vectors."""
    assert len(a) == len(b), f"length mismatch: {len(a)} != {len(b)}"
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def norm(a):
    """Euclidean length (L2 norm) of a vector: sqrt(a . a)."""
    return dot(a, a) ** 0.5


def cosine_similarity(a, b):
    """cos(theta) = (a . b) / (|a| |b|).  Range: -1 .. 0 .. 1.

    Cosine over raw dot product normalizes out magnitude, so it compares
    *direction* only — a long vector and a short one pointing the same way
    score 1.0. That is exactly what we want when comparing embeddings.
    """
    denom = norm(a) * norm(b)
    if denom == 0.0:
        return 0.0  # a zero vector has no direction; define similarity as 0
    return dot(a, b) / denom


# ---------------------------------------------------------------------------
# 2. Verify against NumPy
# ---------------------------------------------------------------------------
def verify():
    print("=" * 56)
    print("STEP 1 — Correctness vs NumPy")
    print("=" * 56)

    cases = {
        "same direction  [1,2,3] vs [2,4,6]": ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        "orthogonal      [1,0]   vs [0,1]  ": ([1.0, 0.0], [0.0, 1.0]),
        "opposite        [1,1]   vs [-1,-1]": ([1.0, 1.0], [-1.0, -1.0]),
    }
    for label, (a, b) in cases.items():
        mine = cosine_similarity(a, b)
        theirs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ok = np.isclose(mine, theirs)
        print(f"{label} | mine={mine:+.4f}  numpy={theirs:+.4f}  match={ok}")
    print()


# ---------------------------------------------------------------------------
# 3. Nearest-vector demo — a tiny hand-made "embedding space"
# ---------------------------------------------------------------------------
VECTORS = {
    "king":   [0.90, 0.80, 0.10],
    "queen":  [0.80, 0.90, 0.15],
    "man":    [0.85, 0.20, 0.10],
    "apple":  [0.10, 0.10, 0.95],
    "banana": [0.15, 0.05, 0.90],
}


def nearest(query, k=3):
    """Rank all named vectors by cosine similarity to `query`. Returns top-k."""
    scored = [(name, cosine_similarity(query, vec)) for name, vec in VECTORS.items()]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def demo():
    print("=" * 56)
    print("STEP 2 — Nearest-vector search (what a vector DB does)")
    print("=" * 56)
    for word in ("king", "apple"):
        ranked = nearest(VECTORS[word], k=3)
        pretty = ", ".join(f"{name} ({score:.3f})" for name, score in ranked)
        print(f"nearest to '{word}':  {pretty}")
    print()
    print("royalty clusters (king~queen~man), fruit clusters (apple~banana).")
    print("This ranking IS RAG retrieval and, softened over all tokens, attention.")


if __name__ == "__main__":
    verify()
    demo()
