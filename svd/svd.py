"""PCA via SVD directly — X = U S Vt — and low-rank reconstruction on Iris.

`pca/` took the covariance route: center → covariance → eigen-decompose. SVD gets
the SAME principal axes in one numerically-stable step, without ever forming the
covariance matrix:

        X_centered = U @ diag(s) @ Vt

  - rows of Vt        ARE the principal axes (right-singular vectors)
  - singular values s relate to the covariance eigenvalues by  lambda_i = s_i^2 / (n-1)
  - U @ diag(s)       ARE the projected scores (same as X_centered @ Vt.T)

This script shows the SVD route agrees with both the covariance route and
sklearn.PCA, then uses SVD for the thing it is really famous for: the best rank-k
approximation of a matrix (Eckart–Young).
"""

import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")  # headless backend — save to file, no display needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. PCA via SVD, and agreement with the covariance route + sklearn
# ---------------------------------------------------------------------------
def pca_via_svd(X, n_components=2):
    """Return (scores, components, explained_variance) using the SVD route."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]

    # Economy SVD: U is (n x r), s is (r,), Vt is (r x d).
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    components = Vt[:n_components]                       # (k x d) principal axes
    scores = (U[:, :n_components] * s[:n_components])    # (n x k) == Xc @ components.T
    explained_variance = (s**2 / (n - 1))[:n_components]  # lambda_i = s_i^2 / (n-1)
    return scores, components, explained_variance


def covariance_eigenvalues(X):
    """The covariance route from `pca/`, for a head-to-head comparison."""
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigh(cov)[0]
    return np.sort(eigvals)[::-1]  # largest first


def align_signs(mine, ref):
    """Flip each column of `mine` to match `ref` (a singular vector is only ±-defined)."""
    flips = np.sign(np.sum(mine * ref, axis=0))
    flips[flips == 0] = 1.0
    return mine * flips


def verify(X):
    print("=" * 62)
    print("STEP 1 — PCA via SVD  ==  covariance route  ==  sklearn.PCA")
    print("=" * 62)

    scores, _, ev_svd = pca_via_svd(X, n_components=2)
    ev_cov = covariance_eigenvalues(X)[:2]
    sk = PCA(n_components=2).fit(X)
    sk_scores = sk.transform(X)

    print(f"eigenvalues  s^2/(n-1)  [SVD]     : {np.round(ev_svd, 4)}")
    print(f"eigenvalues  of covariance [eigh] : {np.round(ev_cov, 4)}")
    print(f"explained_variance_       [sklearn]: {np.round(sk.explained_variance_, 4)}")
    print(f"    SVD == covariance route : {np.allclose(ev_svd, ev_cov)}")
    print(f"    SVD == sklearn          : {np.allclose(ev_svd, sk.explained_variance_)}")
    print(f"    projected scores match sklearn (sign-aligned): "
          f"{np.allclose(align_signs(scores, sk_scores), sk_scores)}")
    print()


# ---------------------------------------------------------------------------
# 2. Low-rank reconstruction — what SVD is really famous for (Eckart–Young)
# ---------------------------------------------------------------------------
def reconstruction(X, out_path):
    print("=" * 62)
    print("STEP 2 — Best rank-k reconstruction of the data (Eckart–Young)")
    print("=" * 62)

    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    total = (s**2).sum()

    errors = []
    ranks = range(1, len(s) + 1)
    for k in ranks:
        # Keep the top-k singular triplets and rebuild the matrix.
        Xk = (U[:, :k] * s[:k]) @ Vt[:k]
        rel_err = np.linalg.norm(Xc - Xk) / np.linalg.norm(Xc)
        captured = (s[:k] ** 2).sum() / total
        errors.append(rel_err)
        print(f"rank {k}: variance captured {captured * 100:5.1f}%   "
              f"relative reconstruction error {rel_err:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(list(ranks), errors, "o-")
    plt.xlabel("rank k (singular values kept)")
    plt.ylabel("relative reconstruction error")
    plt.title("Iris: low-rank reconstruction error vs rank")
    plt.xticks(list(ranks))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nsaved plot -> {out_path}")
    print()
    print("SVD gives the provably-best rank-k approximation of any matrix. Rank 2")
    print("already rebuilds Iris almost perfectly — the same fact that makes PCA,")
    print("image compression, and latent-factor / recommender models work.")


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data

    verify(X)
    out = os.path.join(os.path.dirname(__file__), "iris_svd_reconstruction.png")
    reconstruction(X, out)
