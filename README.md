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

## Setup (one time)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python matmul/matmul_compare.py
.venv/bin/python similarity/similarity.py
.venv/bin/python eigen/eigen.py
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
├── requirements.txt     # shared pinned dependency (numpy)
├── matmul/
│   ├── matmul_compare.py
│   └── README.md
├── similarity/
│   ├── similarity.py
│   └── README.md
└── eigen/
    ├── eigen.py
    └── README.md
```