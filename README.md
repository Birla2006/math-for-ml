# Matmul by Hand vs `np.matmul`

A small experiment comparing a hand-written matrix multiplication against NumPy's
`np.matmul`. It verifies the two produce the same result (`np.allclose`) and times
both to show how much faster a compiled BLAS backend is than pure-Python loops.

## What it does

`matmul_compare.py`:

1. **`matmul_by_hand(A, B)`** — a naive triple-loop matrix multiply
   (`A` is `m×n`, `B` is `n×p`, result is `m×p`).
2. **Verification** — checks the hand-written result against `np.matmul` with
   `np.allclose`, and reports the maximum absolute error.
3. **Timing** — times both implementations (`np.matmul` is timed as best-of-100)
   and prints the speedup.

Test matrices are `128×96` and `96×112`, seeded (`np.random.default_rng(0)`) so
runs are reproducible.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python matmul_compare.py
```

### Running in IntelliJ IDEA

1. **Settings** (`Cmd + ,`) → **Project → Python Interpreter** → **Add Local
   Interpreter → Existing** → point it at `.venv/bin/python`.
2. Open `matmul_compare.py` and click the green **▶** in the gutter, or use the
   built-in terminal with the run command above.

> Requires the Python plugin. If the interpreter still can't find numpy, make sure
> the selected interpreter is the project `.venv`, not the system Python.

## Sample output

```
Shapes: A=(128, 96)  B=(96, 112)  ->  C=(128, 112)
np.allclose(by_hand, np.matmul): True
max absolute error:              2.132e-14

by hand (triple loop):   437.308 ms
np.matmul (best of 100):     0.047 ms
speedup:                   9376.8x
```

## Takeaway

Both implementations compute the **same math** — the tiny `~2e-14` difference is
just floating-point rounding from a different summation order.

The `~9,000×` speed gap comes from `np.matmul` dispatching to a compiled **BLAS**
kernel (vectorized SIMD, cache-blocked, multi-threaded), while the by-hand version
pays Python interpreter overhead on every scalar multiply-add. Same algorithm,
vastly different execution — this is why numerical code leans on NumPy/BLAS rather
than hand-rolled loops.

## Files

| File               | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `matmul_compare.py`| The experiment (implementation + benchmark). |
| `requirements.txt` | Pinned dependency (`numpy`).                 |
| `.gitignore`       | Excludes `.venv/`, caches, and `.idea/`.     |
