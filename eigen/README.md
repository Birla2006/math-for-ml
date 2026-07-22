# Eigenvalues & Eigenvectors — `A v = λ v`

The 3Blue1Brown Ch.13–14 idea, verified in NumPy. An **eigenvector** `v` of a
matrix `A` is a vector whose *direction* survives the transformation — applying
`A` only stretches (or flips) it by a scalar, the **eigenvalue** `λ`:

```
A v = λ v
```

Every other vector gets knocked off its span (its direction rotates); an
eigenvector stays on its own line. Eigenvalues by hand are checked against NumPy,
then the defining equation itself is verified.

## What it does

`eigen.py`:

1. **Eigenvalues by hand** — `eigvals_2x2_by_hand(A)` solves the characteristic
   polynomial `det(A − λI) = 0`, which for a 2×2 is the quadratic
   `λ² − trace·λ + det = 0`, and checks it against `np.linalg.eigvals`.
2. **Verify `A v = λ v`** — takes NumPy's eigenvectors (`np.linalg.eig`, returned
   as *columns*) and confirms applying `A` equals scaling by `λ` with
   `np.allclose`.
3. **Geometry** — measures the angle a vector turns under `A`: an eigenvector
   turns `0°` (only scaled), while a generic vector rotates.

Test matrix is symmetric with clean eigenvalues `3` and `1` (eigenvectors `[1,1]`
and `[1,−1]`).

## Run

From the repo root (see the [root README](../README.md) for one-time venv setup):

```bash
.venv/bin/python eigen/eigen.py
```

## Sample output

```
STEP 1 — Eigenvalues: by hand vs np.linalg
by hand : [1. 3.]
numpy   : [1.+0.j 3.+0.j]
match   : True

STEP 2 — The defining equation  A v = lambda v
lambda = +3.000   v = [0.707 0.707]
    A v      = [2.121 2.121]
    lambda v = [2.121 2.121]   ->  A v = lambda v ? True
lambda = +1.000   v = [-0.707  0.707]
    A v      = [-0.707  0.707]
    lambda v = [-0.707  0.707]   ->  A v = lambda v ? True

STEP 3 — Eigenvector stays on its span; other vectors rotate
eigenvector      turns by   0.00 deg   (unchanged direction)
generic [1,0]    turns by  26.57 deg   (direction rotated)
```

## Takeaway

The hand-derived eigenvalues match NumPy, and both eigenvectors satisfy
`A v = λ v` exactly. Geometrically, an eigenvector is an **axis the transformation
acts along** — `A` merely stretches it by `λ` while every other direction rotates.

That is why eigen-decomposition reveals a matrix's "natural axes." The same idea
powers **PCA** (principal axes = eigenvectors of the covariance matrix),
**PageRank** (the ranking is the dominant eigenvector), and **stability analysis**
(eigenvalue signs decide whether a system grows or decays).