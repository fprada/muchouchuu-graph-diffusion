#!/usr/bin/env python3

from pathlib import Path
import os
import numpy as np
from scipy import sparse


N = 108_000_000
K = 12

WORK = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/"
    "muchouchuu_diffusion/muchouchuu_graph_work"
)

OUT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/"
    "muchouchuu_diffusion/muchouchuu_graphs"
)

IP = WORK / "knn_indices.i32"

# Write to a temporary filename first.
TMP = OUT / "knn_periodic_recovery_tmp.npz"
FINAL = OUT / "knn_periodic.npz"


print("MuchoUchuu kNN graph recovery", flush=True)
print("N =", N, flush=True)
print("K =", K, flush=True)
print("indices =", IP, flush=True)
print("temporary output =", TMP, flush=True)


expected = N * K * np.dtype(np.int32).itemsize

if IP.stat().st_size != expected:
    raise RuntimeError(
        "Unexpected knn_indices.i32 size: {} vs expected {}".format(
            IP.stat().st_size,
            expected,
        )
    )


# ------------------------------------------------------------
# Neighbor indices remain disk-backed.
# ------------------------------------------------------------

idx = np.memmap(
    str(IP),
    mode="r",
    dtype=np.int32,
    shape=(N, K),
)

print("Mapped neighbor table.", flush=True)


# ------------------------------------------------------------
# CSR row offsets.
#
# 108,000,001 int64 values ~864 MB.
# ------------------------------------------------------------

print("Creating CSR indptr...", flush=True)

indptr = np.arange(
    0,
    N * K + 1,
    K,
    dtype=np.int64,
)

print("Created CSR indptr.", flush=True)


# ------------------------------------------------------------
# Use uint8 rather than float32 for the binary adjacency.
#
# This preserves exactly the same graph topology while reducing
# the temporary data array from ~5.18 GB to ~1.30 GB.
# ------------------------------------------------------------

print("Allocating binary directed-edge data...", flush=True)

data = np.ones(
    N * K,
    dtype=np.uint8,
)

print("Building directed CSR view...", flush=True)

directed = sparse.csr_matrix(
    (
        data,
        idx.reshape(-1),
        indptr,
    ),
    shape=(N, N),
    copy=False,
)

print(
    "Directed graph:",
    "shape =", directed.shape,
    "nnz =", directed.nnz,
    "dtype =", directed.dtype,
    flush=True,
)


# ------------------------------------------------------------
# Symmetrize exactly as in the original graph builder:
#
#       G = maximum(A, A.T)
#
# No self-edges were present in the original kNN table.
# ------------------------------------------------------------

print("Building transpose...", flush=True)

directed_T = directed.transpose().tocsr()

print(
    "Transpose built:",
    "nnz =", directed_T.nnz,
    flush=True,
)


print("Symmetrizing with maximum(A, A.T)...", flush=True)

g = directed.maximum(directed_T)

print(
    "Symmetrization complete:",
    "nnz =", g.nnz,
    flush=True,
)


# Free structures no longer needed as early as possible.
del directed_T
del directed
del data
del indptr


print("Canonicalizing CSR...", flush=True)

g = g.tocsr(copy=False)
g.sum_duplicates()
g.eliminate_zeros()
g.sort_indices()

print(
    "Final graph:",
    "shape =", g.shape,
    "nnz =", g.nnz,
    "dtype =", g.dtype,
    "undirected edges =", g.nnz // 2,
    flush=True,
)


# ------------------------------------------------------------
# Save uint8 graph.
#
# The diffusion script later converts it to float32:
#
# adjacency.astype(np.float32, copy=False)
#
# so topology and transition operator are unchanged.
# ------------------------------------------------------------

if TMP.exists():
    TMP.unlink()

print("Saving temporary NPZ...", flush=True)

sparse.save_npz(
    str(TMP),
    g,
    compressed=False,
)

print("Temporary graph written:", TMP, flush=True)


# ------------------------------------------------------------
# Validate from disk before replacing final path.
# ------------------------------------------------------------

print("Reloading temporary graph for validation...", flush=True)

del g

test = sparse.load_npz(str(TMP))

if test.shape != (N, N):
    raise RuntimeError(
        "Recovered graph has wrong shape: {}".format(test.shape)
    )

if test.nnz <= N * K:
    print(
        "WARNING: symmetrized nnz is unexpectedly <= directed nnz",
        flush=True,
    )

print(
    "Validation:",
    "shape =", test.shape,
    "nnz =", test.nnz,
    "dtype =", test.dtype,
    flush=True,
)

del test


# Atomic replacement after successful validation.
os.replace(str(TMP), str(FINAL))

print()
print("SUCCESS", flush=True)
print("Recovered graph:", FINAL, flush=True)
