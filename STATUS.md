# Status — where I left off

> 10-minute reopen ritual: skim this file, then do 5 min of Anki. Update the
> **Next up** and **Last touched** lines when you stop for the day.

**Last touched:** 2026-07-22
**Focus:** math-for-ml — build core ML math from scratch, verify each piece vs NumPy.

## Done ✅

| Experiment | Status | One-line takeaway |
| ---------- | ------ | ----------------- |
| [`matmul/`](matmul/README.md) | ✅ complete | Hand-loop matmul matches `np.matmul` (~2e-14 error); BLAS is ~9,000× faster. |
| [`similarity/`](similarity/README.md) | ✅ complete | Hand cosine matches NumPy; nearest-vector search *is* RAG retrieval / attention. |
| [`eigen/`](eigen/README.md) | ✅ complete | Hand 2×2 eigenvalues match NumPy; verified `A v = λ v` — eigenvector only scales, others rotate. |

## Next up 🔜

- [ ] Pick the next experiment (candidates below).
- [ ] Anki: review today's due cards before coding.

### Candidate next experiments
- **softmax + attention** — natural sequel to `similarity/`: turn dot-product
  scores into weights, build a tiny single-head attention by hand vs NumPy.
- **gradient descent** — fit a line by hand-derived gradients vs a closed-form /
  autograd check.
- **PCA / SVD** — dimensionality reduction from scratch, verify against
  `np.linalg.svd` (builds directly on `eigen/`).

## Quick commands

```bash
# one-time setup
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

# run what exists
.venv/bin/python matmul/matmul_compare.py
.venv/bin/python similarity/similarity.py
.venv/bin/python eigen/eigen.py
```

## Repo shape

```
math-for-ml/
├── README.md         # index of experiments
├── STATUS.md         # this file — reopen snapshot
├── requirements.txt  # numpy
├── matmul/           # matmul by hand vs np.matmul
├── similarity/       # dot / cosine / nearest-vector search
└── eigen/            # eigenvalues/vectors — verify A v = λ v
```
