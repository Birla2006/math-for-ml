"""Eigenvalues & eigenvectors — the 3Blue1Brown Ch.13–14 idea, verified in NumPy.

An eigenvector v of a matrix A is a special vector whose *direction* is unchanged
by the transformation A — applying A only stretches (or flips) it by a scalar, the
eigenvalue lambda:

        A v = lambda v

Every other vector gets knocked off its span (its direction rotates); an
eigenvector stays on its own line. This script:

  1. Finds the eigenvalues of a 2x2 by hand (characteristic polynomial) and checks
     them against np.linalg.eigvals.
  2. Takes NumPy's eigenvectors and verifies the defining equation  A v = lambda v
     with np.allclose  (the whole point of the exercise).
  3. Shows the geometry: an eigenvector only scales, a generic vector rotates.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Eigenvalues of a 2x2 by hand — the characteristic polynomial
# ---------------------------------------------------------------------------
def eigvals_2x2_by_hand(A):
    """Eigenvalues of a 2x2 via  det(A - lambda I) = 0.

    For a 2x2 this expands to the quadratic
        lambda^2 - trace*lambda + det = 0
    which we solve with the quadratic formula. (Real eigenvalues only here.)
    """
    (a, b), (c, d) = A
    trace = a + d          # sum of the diagonal
    det = a * d - b * c    # determinant
    disc = trace**2 - 4 * det
    assert disc >= 0, f"complex eigenvalues (discriminant {disc:.3f} < 0)"
    root = disc**0.5
    return np.array([(trace + root) / 2, (trace - root) / 2])


# ---------------------------------------------------------------------------
# 2. Verify  A v = lambda v  against NumPy
# ---------------------------------------------------------------------------
def verify(A):
    print("=" * 60)
    print("STEP 1 — Eigenvalues: by hand vs np.linalg")
    print("=" * 60)
    print(f"A =\n{A}\n")

    mine = np.sort(eigvals_2x2_by_hand(A))
    theirs = np.sort(np.linalg.eigvals(A))
    print(f"by hand : {mine}")
    print(f"numpy   : {theirs}")
    print(f"match   : {np.allclose(mine, theirs)}\n")

    print("=" * 60)
    print("STEP 2 — The defining equation  A v = lambda v")
    print("=" * 60)
    # np.linalg.eig returns eigenvalues and the eigenvectors as *columns* of V.
    # It uses complex dtype in general; our eigenvalues are real, so drop the
    # (zero) imaginary parts to keep the output and arithmetic clean.
    vals, V = np.linalg.eig(A)
    vals, V = vals.real, V.real
    for i in range(len(vals)):
        lam = vals[i]
        v = V[:, i]                 # i-th eigenvector (a column)
        Av = A @ v                  # apply the transformation
        lam_v = lam * v             # just scale the vector
        ok = np.allclose(Av, lam_v)
        print(f"lambda = {lam:+.3f}   v = {np.round(v, 3)}")
        print(f"    A v      = {np.round(Av, 3)}")
        print(f"    lambda v = {np.round(lam_v, 3)}   ->  A v = lambda v ? {ok}")
    print()


# ---------------------------------------------------------------------------
# 3. Geometry — eigenvector only scales, a generic vector rotates
# ---------------------------------------------------------------------------
def angle_between(u, w):
    """Angle in degrees between two vectors (0 = same direction)."""
    cos = (u @ w) / (np.linalg.norm(u) * np.linalg.norm(w))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def geometry(A):
    print("=" * 60)
    print("STEP 3 — Eigenvector stays on its span; other vectors rotate")
    print("=" * 60)
    _, V = np.linalg.eig(A)
    eigvec = V[:, 0].real
    generic = np.array([1.0, 0.0])   # an arbitrary, non-eigen direction

    for name, v in (("eigenvector", eigvec), ("generic [1,0]", generic)):
        turned = angle_between(v, A @ v)
        note = "unchanged direction" if turned < 1e-6 else "direction rotated"
        print(f"{name:<16} turns by {turned:6.2f} deg   ({note})")
    print()
    print("An eigenvector is the axis the transformation acts along — it is only")
    print("stretched by lambda. This is why eigen-decomposition reveals a matrix's")
    print("'natural axes': PCA, PageRank, and stability analysis all live here.")


if __name__ == "__main__":
    # Symmetric matrix with clean integer eigenvalues (3 and 1),
    # eigenvectors [1,1] and [1,-1].
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
    verify(A)
    geometry(A)