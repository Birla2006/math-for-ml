"""PCA from scratch on Iris — covariance → eigen → project, checked vs sklearn.

The intuition (the SVD / eigen story from `eigen/`):
  A dataset is a cloud of points. Its covariance matrix describes the shape of
  that cloud — which directions it spreads along and by how much. The
  *eigenvectors* of the covariance matrix are those directions (the principal
  axes); the *eigenvalues* are the variance along each. PCA just rotates the data
  onto those axes and keeps the few with the most variance.

  SVD is the same thing done in one numerically-stable step: X = U S Vt, and the
  right-singular vectors (rows of Vt) ARE the principal axes, with singular values
  s_i related to eigenvalues by  lambda_i = s_i^2 / (n - 1).

This script builds PCA by hand (covariance → np.linalg.eigh → project), verifies
it against sklearn.decomposition.PCA, and plots the 2-D projection of Iris.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")  # headless backend — save to file, no display needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. PCA by hand — covariance, eigen-decomposition, projection
# ---------------------------------------------------------------------------
def pca_by_hand(X, n_components=2):
    """Return (scores, components, explained_variance) via the covariance route.

    scores              : X projected onto the top principal axes  (n x k)
    components          : the principal axes themselves            (k x d)
    explained_variance  : variance captured by each axis (the eigenvalues)
    """
    # 1. Center the data — PCA is about variance *around the mean*.
    Xc = X - X.mean(axis=0)

    # 2. Covariance matrix (d x d). rowvar=False: columns are the features.
    cov = np.cov(Xc, rowvar=False)

    # 3. Eigen-decomposition. cov is symmetric -> use eigh (real, orthonormal).
    #    eigh returns eigenvalues ascending, so reverse to get largest first.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 4. Keep the top-k axes and project the centered data onto them.
    components = eigvecs[:, :n_components].T          # (k x d)
    scores = Xc @ components.T                        # (n x k)

    # Fraction of TOTAL variance each kept axis explains (uses the full spectrum:
    # the eigenvalues sum to the total variance of the data).
    explained_ratio = eigvals[:n_components] / eigvals.sum()
    return scores, components, eigvals[:n_components], explained_ratio


# ---------------------------------------------------------------------------
# 2. Verify against sklearn (fixing the arbitrary sign of each axis)
# ---------------------------------------------------------------------------
def align_signs(mine, ref):
    """Flip each column of `mine` to match `ref`'s sign convention.

    An eigenvector is only defined up to a sign (+v and -v are both valid axes),
    so hand-PCA and sklearn can differ by a per-axis flip. Align before comparing.
    """
    flips = np.sign(np.sum(mine * ref, axis=0))
    flips[flips == 0] = 1.0
    return mine * flips


def verify(X):
    print("=" * 60)
    print("STEP 1 — PCA by hand vs sklearn.decomposition.PCA")
    print("=" * 60)

    scores, components, ev, ratio = pca_by_hand(X, n_components=2)

    sk = PCA(n_components=2).fit(X)
    sk_scores = sk.transform(X)

    scores_aligned = align_signs(scores, sk_scores)
    scores_ok = np.allclose(scores_aligned, sk_scores)
    var_ok = np.allclose(ev, sk.explained_variance_)
    ratio_ok = np.allclose(ratio, sk.explained_variance_ratio_)

    print(f"explained variance (by hand): {np.round(ev, 4)}")
    print(f"explained variance (sklearn): {np.round(sk.explained_variance_, 4)}")
    print(f"    match: {var_ok}")
    print(f"explained-variance ratio (by hand): {np.round(ratio, 4)}")
    print(f"explained-variance ratio (sklearn): {np.round(sk.explained_variance_ratio_, 4)}")
    print(f"    match: {ratio_ok}")
    print(f"projected scores match sklearn (after sign-align): {scores_ok}")
    print()
    print(f"PC1 captures {ratio[0] * 100:.1f}% and PC2 {ratio[1] * 100:.1f}% "
          f"of the TOTAL variance ({ratio.sum() * 100:.1f}% in just 2 of 4 dims).")
    print()
    return scores_aligned


# ---------------------------------------------------------------------------
# 3. Plot the 2-D projection
# ---------------------------------------------------------------------------
def plot(scores, y, target_names, out_path):
    print("=" * 60)
    print("STEP 2 — Project 4-D Iris down to 2-D and plot")
    print("=" * 60)

    plt.figure(figsize=(7, 5))
    for i, name in enumerate(target_names):
        pts = scores[y == i]
        plt.scatter(pts[:, 0], pts[:, 1], label=name, alpha=0.8, edgecolor="k",
                    linewidth=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Iris projected onto its first two principal components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved plot -> {out_path}")
    print()
    print("Four measurements collapse to two axes and the three species still")
    print("separate cleanly — PC1 alone carries most of what distinguishes them.")


if __name__ == "__main__":
    import os

    iris = load_iris()
    X, y = iris.data, iris.target

    scores = verify(X)

    out = os.path.join(os.path.dirname(__file__), "iris_pca.png")
    plot(scores, y, iris.target_names, out)
