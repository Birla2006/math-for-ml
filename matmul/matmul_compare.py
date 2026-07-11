
"""Compare a hand-written matrix multiply against np.matmul.

Verifies numerical agreement with np.allclose and times both implementations.
"""

import time

import numpy as np


def matmul_by_hand(A, B):
    """Naive triple-loop matrix multiply. A is (m x n), B is (n x p) -> (m x p)."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"inner dimensions must match: {n} != {n2}"

    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def time_it(fn, *args, repeats=1):
    """Return (result, best_seconds) over `repeats` runs."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - start)
    return result, best


def main():
    rng = np.random.default_rng(0)
    m, n, p = 128, 96, 112
    A = rng.standard_normal((m, n))
    B = rng.standard_normal((n, p))

    c_hand, t_hand = time_it(matmul_by_hand, A, B)
    c_np, t_np = time_it(np.matmul, A, B, repeats=100)

    match = np.allclose(c_hand, c_np)
    max_abs_err = np.max(np.abs(c_hand - c_np))

    print(f"Shapes: A={A.shape}  B={B.shape}  ->  C={c_hand.shape}")
    print(f"np.allclose(by_hand, np.matmul): {match}")
    print(f"max absolute error:              {max_abs_err:.3e}")
    print()
    print(f"by hand (triple loop): {t_hand * 1e3:9.3f} ms")
    print(f"np.matmul (best of 100): {t_np * 1e3:9.3f} ms")
    print(f"speedup:                {t_hand / t_np:9.1f}x")


if __name__ == "__main__":
    main()