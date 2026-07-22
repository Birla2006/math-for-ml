# Math for ML

Small, self-contained experiments that build core ML math from scratch and verify
each hand-written version against NumPy. Each experiment lives in its own folder
with its own README.

## Experiments

| Folder                                   | Experiment                                                    |
| ---------------------------------------- | ------------------------------------------------------------- |
| [`matmul/`](matmul/README.md)            | Hand-written matrix multiply vs `np.matmul` (correctness + BLAS speedup). |
| [`similarity/`](similarity/README.md)    | Dot product, cosine similarity, and nearest-vector search (the math behind embeddings, RAG, and attention). |
| [`eigen/`](eigen/README.md)              | Eigenvalues & eigenvectors — verify `A v = λ v` in NumPy (3B1B Ch.13–14). |
| [`pca/`](pca/README.md)                  | PCA from scratch on Iris (covariance → eigen → project), compared to `sklearn.PCA`. |

## Setup (one time)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

> `matmul/`, `similarity/`, and `eigen/` need only numpy; `pca/` also uses
> scikit-learn (Iris + the comparison) and matplotlib (the plot).

## Run

```bash
.venv/bin/python matmul/matmul_compare.py
.venv/bin/python similarity/similarity.py
.venv/bin/python eigen/eigen.py
.venv/bin/python pca/pca.py
```

### Running in IntelliJ IDEA

1. **Settings** (`Cmd + ,`) → **Project → Python Interpreter** → **Add Local
   Interpreter → Existing** → point it at `.venv/bin/python`.
2. Open a script and click the green **▶** in the gutter, or use the built-in
   terminal with a run command above.

> Requires the Python plugin. If the interpreter still can't find numpy, make sure
> the selected interpreter is the project `.venv`, not the system Python.

## Layout

```
math-for-ml/
├── README.md            # this index
├── requirements.txt     # pinned deps (numpy, scikit-learn, matplotlib)
├── matmul/
│   ├── matmul_compare.py
│   └── README.md
├── similarity/
│   ├── similarity.py
│   └── README.md
├── eigen/
│   ├── eigen.py
│   └── README.md
└── pca/
    ├── pca.py
    ├── iris_pca.png
    └── README.md
```