#!/usr/bin/env python3

from pathlib import Path
import os
import numpy as np
from scipy import sparse


# ============================================================
# MuchoUchuu kNN robustness graph: k = 6
#
# Source:
#   Existing ordered 12-neighbour table:
#       knn_indices.i32
#   shape = (108,000,000, 12)
#
# We retain only the first six nearest neighbours of every node,
# then apply exactly the same symmetrisation rule as the
# fiducial k=12 graph:
#
#       A_sym = maximum(A, A.T)
#
# The fiducial k=12 graph is NEVER modified.
# ============================================================


N = 108_000_000

K_SOURCE = 12
K_TARGET = 6

BASE = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/"
    "muchouchuu_diffusion"
)

WORK = BASE / "muchouchuu_graph_work"

OUT = BASE / "k_robustness" / "k06"

IP12 = WORK / "knn_indices.i32"

# Disk-backed contiguous k=6 neighbour table.
IP6 = OUT / "knn_indices_k06.i32"

# Graph output.
TMP = OUT / "knn_periodic_k06_tmp.npz"
FINAL = OUT / "knn_periodic.npz"


# Number of rows copied at a time from the k=12 table.
# 1,000,000 rows means:
# source block ~48 MB
# output block ~24 MB
CHUNK_ROWS = 1_000_000


print("=" * 72, flush=True)
print("MuchoUchuu kNN robustness graph construction", flush=True)
print("=" * 72, flush=True)
print("N              =", N, flush=True)
print("source K       =", K_SOURCE, flush=True)
print("target K       =", K_TARGET, flush=True)
print("source indices =", IP12, flush=True)
print("k=6 indices    =", IP6, flush=True)
print("temporary NPZ  =", TMP, flush=True)
print("final graph    =", FINAL, flush=True)
print("=" * 72, flush=True)


# ============================================================
# Prepare output directory.
# ============================================================

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Validate original k=12 neighbour table.
# ============================================================

expected12 = (
    N
    * K_SOURCE
    * np.dtype(np.int32).itemsize
)

actual12 = IP12.stat().st_size

print(
    "Source neighbour file size:",
    f"{actual12:,}",
    "bytes",
    flush=True,
)

if actual12 != expected12:
    raise RuntimeError(
        "Unexpected source knn_indices.i32 size: "
        f"{actual12:,} vs expected {expected12:,}"
    )

print("Source k=12 neighbour table size is correct.", flush=True)


# ============================================================
# Memory-map original ordered k=12 neighbour table.
# ============================================================

idx12 = np.memmap(
    str(IP12),
    mode="r",
    dtype=np.int32,
    shape=(N, K_SOURCE),
)

print(
    "Mapped source neighbour table:",
    idx12.shape,
    flush=True,
)


# ============================================================
# Create contiguous disk-backed k=6 neighbour table.
#
# IMPORTANT:
# Do not use
#
#     idx12[:, :6].reshape(-1)
#
# on the complete 108M-node catalogue because the slice is
# strided and NumPy may create a very large RAM copy.
#
# Instead copy block-by-block to a contiguous disk-backed file.
# ============================================================

expected6 = (
    N
    * K_TARGET
    * np.dtype(np.int32).itemsize
)

reuse_existing = False

if IP6.exists():

    if IP6.stat().st_size == expected6:

        print(
            "Existing contiguous k=6 neighbour table found.",
            flush=True,
        )
        print(
            "Reusing:",
            IP6,
            flush=True,
        )

        reuse_existing = True

    else:

        print(
            "Removing incomplete/wrong-size k=6 neighbour file:",
            IP6,
            flush=True,
        )

        IP6.unlink()


if not reuse_existing:

    print(
        "Creating contiguous disk-backed k=6 neighbour table...",
        flush=True,
    )

    idx6_write = np.memmap(
        str(IP6),
        mode="w+",
        dtype=np.int32,
        shape=(N, K_TARGET),
    )

    for start in range(0, N, CHUNK_ROWS):

        stop = min(start + CHUNK_ROWS, N)

        idx6_write[start:stop, :] = idx12[
            start:stop,
            :K_TARGET,
        ]

        idx6_write.flush()

        print(
            "Copied rows "
            f"{start:,} -- {stop:,} "
            f"({100.0 * stop / N:6.2f}%)",
            flush=True,
        )

    idx6_write.flush()

    del idx6_write

    print(
        "Finished contiguous k=6 neighbour table.",
        flush=True,
    )


# Validate k=6 intermediate file size.
actual6 = IP6.stat().st_size

if actual6 != expected6:
    raise RuntimeError(
        "Unexpected k=6 neighbour file size: "
        f"{actual6:,} vs expected {expected6:,}"
    )

print(
    "Validated k=6 neighbour file size:",
    f"{actual6:,}",
    "bytes",
    flush=True,
)


# Source table no longer needed.
del idx12


# ============================================================
# Map contiguous k=6 neighbour table.
# ============================================================

idx6 = np.memmap(
    str(IP6),
    mode="r",
    dtype=np.int32,
    shape=(N, K_TARGET),
)

print(
    "Mapped contiguous k=6 neighbour table.",
    flush=True,
)


# ============================================================
# Basic neighbour-index validation.
#
# Check in blocks so we never require a full-array temporary.
# ============================================================

print("Validating neighbour-index range...", flush=True)

global_min = N
global_max = -1
n_self = 0

for start in range(0, N, CHUNK_ROWS):

    stop = min(start + CHUNK_ROWS, N)

    block = idx6[start:stop]

    bmin = int(block.min())
    bmax = int(block.max())

    global_min = min(global_min, bmin)
    global_max = max(global_max, bmax)

    rows = np.arange(
        start,
        stop,
        dtype=np.int32,
    )[:, None]

    n_self += int(
        np.count_nonzero(block == rows)
    )

    if bmin < 0 or bmax >= N:
        raise RuntimeError(
            f"Invalid neighbour IDs in rows "
            f"{start:,}:{stop:,}: "
            f"min={bmin}, max={bmax}"
        )

    if stop % 10_000_000 == 0 or stop == N:
        print(
            "Validated rows through",
            f"{stop:,}",
            flush=True,
        )


print(
    "Neighbour ID range:",
    global_min,
    "...",
    global_max,
    flush=True,
)

print(
    "Self-neighbour entries:",
    f"{n_self:,}",
    flush=True,
)

if n_self != 0:
    print(
        "WARNING: self-neighbour entries were found.",
        flush=True,
    )


# ============================================================
# Construct directed k=6 CSR graph.
# ============================================================

print("Creating CSR indptr...", flush=True)

indptr = np.arange(
    0,
    N * K_TARGET + 1,
    K_TARGET,
    dtype=np.int64,
)

print(
    "CSR indptr created:",
    f"{indptr.nbytes / 1024**3:.3f} GiB",
    flush=True,
)


print(
    "Allocating uint8 directed-edge data...",
    flush=True,
)

data = np.ones(
    N * K_TARGET,
    dtype=np.uint8,
)

print(
    "Directed data size:",
    f"{data.nbytes / 1024**3:.3f} GiB",
    flush=True,
)


print("Building directed CSR graph...", flush=True)

directed = sparse.csr_matrix(
    (
        data,
        idx6.reshape(-1),
        indptr,
    ),
    shape=(N, N),
    dtype=np.uint8,
    copy=False,
)

print(
    "Directed graph:",
    "shape =", directed.shape,
    "nnz =", directed.nnz,
    "dtype =", directed.dtype,
    flush=True,
)


# idx6 is now owned indirectly through the CSR indices.
# Keep the mapped file itself on disk.
del idx6


# ============================================================
# Transpose and symmetrise.
# ============================================================

print("Building transpose...", flush=True)

directed_T = directed.transpose().tocsr()

print(
    "Transpose built:",
    "nnz =", directed_T.nnz,
    flush=True,
)


print(
    "Symmetrizing with maximum(A, A.T)...",
    flush=True,
)

g = directed.maximum(directed_T)

print(
    "Symmetrization complete:",
    "nnz =", g.nnz,
    flush=True,
)


# Free structures no longer needed.
del directed_T
del directed
del data
del indptr


# ============================================================
# Canonical CSR representation.
# ============================================================

print("Canonicalizing CSR...", flush=True)

g = g.tocsr(copy=False)
g.sum_duplicates()
g.eliminate_zeros()
g.sort_indices()

degree = np.diff(g.indptr)

print()
print("Final k=6 graph:", flush=True)
print(
    "shape              =",
    g.shape,
    flush=True,
)
print(
    "nnz                =",
    f"{g.nnz:,}",
    flush=True,
)
print(
    "dtype              =",
    g.dtype,
    flush=True,
)
print(
    "undirected edges   =",
    f"{g.nnz // 2:,}",
    flush=True,
)
print(
    "mean degree        =",
    f"{degree.mean():.6f}",
    flush=True,
)
print(
    "minimum degree     =",
    int(degree.min()),
    flush=True,
)
print(
    "maximum degree     =",
    int(degree.max()),
    flush=True,
)
print(
    "zero-degree nodes  =",
    f"{np.count_nonzero(degree == 0):,}",
    flush=True,
)


# ============================================================
# Save to temporary NPZ first.
# ============================================================

if TMP.exists():
    TMP.unlink()


print(
    "Saving temporary graph NPZ...",
    flush=True,
)

sparse.save_npz(
    str(TMP),
    g,
    compressed=False,
)

print(
    "Temporary graph written:",
    TMP,
    flush=True,
)


# ============================================================
# Validate from disk before final atomic rename.
# ============================================================

print(
    "Reloading temporary graph for validation...",
    flush=True,
)

expected_nnz = g.nnz

del degree
del g


test = sparse.load_npz(
    str(TMP)
).tocsr()


if test.shape != (N, N):

    raise RuntimeError(
        "Recovered k=6 graph has wrong shape: "
        f"{test.shape}"
    )


if test.nnz <= N * K_TARGET:

    print(
        "WARNING: symmetrized nnz is unexpectedly "
        "<= directed nnz.",
        flush=True,
    )


print()
print("Disk validation:", flush=True)
print(
    "shape =",
    test.shape,
    flush=True,
)
print(
    "nnz =",
    f"{test.nnz:,}",
    flush=True,
)
print(
    "dtype =",
    test.dtype,
    flush=True,
)


if test.nnz != expected_nnz:

    raise RuntimeError(
        "NNZ changed after saving/reloading: "
        f"{test.nnz:,} vs {expected_nnz:,}"
    )


# Check exact symmetry.
print(
    "Checking exact symmetry...",
    flush=True,
)

asym = (test != test.T).nnz

print(
    "asymmetric entries =",
    f"{asym:,}",
    flush=True,
)

if asym != 0:
    raise RuntimeError(
        "Final k=6 graph is not exactly symmetric."
    )


degree_test = np.diff(test.indptr)

if degree_test.min() == 0:

    raise RuntimeError(
        "Final k=6 graph contains zero-degree nodes."
    )


print(
    "minimum degree after reload =",
    int(degree_test.min()),
    flush=True,
)

print(
    "maximum degree after reload =",
    int(degree_test.max()),
    flush=True,
)

print(
    "mean degree after reload =",
    f"{degree_test.mean():.6f}",
    flush=True,
)


del degree_test
del test


# ============================================================
# Atomic replacement after successful validation.
# ============================================================

if FINAL.exists():
    raise RuntimeError(
        "Refusing to overwrite an existing final k=6 graph: "
        f"{FINAL}"
    )


os.replace(
    str(TMP),
    str(FINAL),
)


print()
print("=" * 72, flush=True)
print("SUCCESS", flush=True)
print(
    "MuchoUchuu k=6 graph:",
    FINAL,
    flush=True,
)
print("=" * 72, flush=True)
