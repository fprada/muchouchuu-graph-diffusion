#!/usr/bin/env python3
"""
build_periodic_delaunay_graph.py

Build a periodic 3-D Delaunay graph from an XYZ float32 catalogue.

Designed for controlled graph-construction comparisons with the kNN
diffusion pipeline.

IMPORTANT
---------
True periodic Delaunay is implemented here by a 3x3x3 tiling of the
point catalogue followed by scipy.spatial.Delaunay.

This is computationally expensive in 3-D. Do NOT use this directly
on the full MuchoUchuu catalogue. Use a reduced matched point sample
for the construction-systematics experiment.

Outputs
-------
selected_xyz_f32.bin
selected_indices.npy             [if --sample-n is used]
delaunay_periodic.npz            sparse symmetric adjacency matrix
delaunay_metadata.json
"""

from __future__ import print_function

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


def load_xyz(filename):
    x = np.fromfile(str(filename), dtype=np.float32)
    if x.size % 3 != 0:
        raise ValueError("XYZ file does not contain a multiple of 3 floats")
    return x.reshape((-1, 3))


def select_points(pts, sample_n=None, seed=12345):
    n = len(pts)

    if sample_n is None or sample_n >= n:
        idx = np.arange(n, dtype=np.int64)
        return pts.copy(), idx

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=sample_n, replace=False)
    idx.sort()

    return pts[idx].copy(), idx


def tile_periodic(pts, L):
    """
    Construct 27 periodic copies.

    Returns
    -------
    tiled : (27*N,3)
    owner : original vertex index for every tiled point
    central : bool mask selecting the zero-shift image
    """
    n = len(pts)

    shifts = np.array(
        list(itertools.product((-1, 0, 1), repeat=3)),
        dtype=np.int8
    )

    tiled = np.empty((27 * n, 3), dtype=np.float64)
    owner = np.empty(27 * n, dtype=np.int64)
    central = np.zeros(27 * n, dtype=bool)

    base_owner = np.arange(n, dtype=np.int64)

    offset = 0

    for shift in shifts:
        sl = slice(offset, offset + n)

        tiled[sl] = pts.astype(np.float64) + shift[None, :] * float(L)
        owner[sl] = base_owner

        if np.all(shift == 0):
            central[sl] = True

        offset += n

    return tiled, owner, central


def delaunay_edges_from_simplices(tri, owner, central):
    """
    Extract unique graph edges.

    Only Delaunay edges with at least one endpoint belonging to the
    central periodic image are retained. Periodic-image indices are
    mapped back onto the original N vertices.
    """

    simplices = tri.simplices
    n_tetra = len(simplices)

    print("Delaunay tetrahedra: {:,}".format(n_tetra), flush=True)

    # Six edges of a tetrahedron.
    pairs = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )

    edge_chunks = []

    for ia, ib in pairs:
        a = simplices[:, ia]
        b = simplices[:, ib]

        # An edge is relevant if one of its image endpoints lies
        # in the central box.
        keep = central[a] | central[b]

        a = owner[a[keep]]
        b = owner[b[keep]]

        # Remove periodic self-edges.
        good = a != b
        a = a[good]
        b = b[good]

        # Canonical undirected ordering.
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        edge_chunks.append(np.column_stack((lo, hi)))

    edges = np.concatenate(edge_chunks, axis=0)

    print("Raw mapped Delaunay edges: {:,}".format(len(edges)), flush=True)

    # Unique undirected pairs.
    edges = np.unique(edges, axis=0)

    print("Unique periodic edges: {:,}".format(len(edges)), flush=True)

    return edges


def edges_to_csr(edges, n):
    i = edges[:, 0]
    j = edges[:, 1]

    rows = np.concatenate((i, j))
    cols = np.concatenate((j, i))

    data = np.ones(len(rows), dtype=np.uint8)

    g = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n, n),
        dtype=np.uint8
    ).tocsr()

    g.sum_duplicates()
    g.eliminate_zeros()
    g.sort_indices()

    return g


def main():
    p = argparse.ArgumentParser()

    p.add_argument("xyz_binary",
                   help="Input raw float32 XYZ catalogue")

    p.add_argument("--box-size",
                   type=float,
                   required=True,
                   help="Periodic box size")

    p.add_argument("--output-dir",
                   required=True)

    p.add_argument("--sample-n",
                   type=int,
                   default=None,
                   help="Randomly select this many vertices before "
                        "constructing either graph")

    p.add_argument("--seed",
                   type=int,
                   default=12345,
                   help="Seed for the matched subsample")

    p.add_argument(
        "--qhull-options",
        default="Qbb Qc Q12 QJ",
        help="Options passed to scipy.spatial.Delaunay"
    )

    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("Loading:", args.xyz_binary, flush=True)
    pts0 = load_xyz(args.xyz_binary)

    print("Input N = {:,}".format(len(pts0)), flush=True)

    pts, selected = select_points(
        pts0,
        sample_n=args.sample_n,
        seed=args.seed
    )

    n = len(pts)

    print("Working N = {:,}".format(n), flush=True)

    # Ensure coordinates live inside [0,L).
    pts %= np.float32(args.box_size)

    selected_xyz = out / "selected_xyz_f32.bin"
    pts.astype(np.float32).tofile(str(selected_xyz))

    np.save(str(out / "selected_indices.npy"), selected)

    print("Creating 27 periodic images...", flush=True)

    t0 = time.time()

    tiled, owner, central = tile_periodic(
        pts,
        args.box_size
    )

    print(
        "Tiled point count = {:,}".format(len(tiled)),
        flush=True
    )

    print("Computing 3-D Delaunay tessellation...", flush=True)

    tri = Delaunay(
        tiled,
        qhull_options=args.qhull_options
    )

    print(
        "Delaunay completed in {:.1f} s".format(time.time() - t0),
        flush=True
    )

    edges = delaunay_edges_from_simplices(
        tri,
        owner,
        central
    )

    # Free the very large Delaunay objects before CSR construction.
    del tri
    del tiled
    del owner
    del central

    print("Constructing sparse adjacency...", flush=True)

    g = edges_to_csr(edges, n)

    graphfile = out / "delaunay_periodic.npz"

    sparse.save_npz(
        str(graphfile),
        g,
        compressed=False
    )

    deg = np.diff(g.indptr)

    metadata = {
        "input_xyz": str(Path(args.xyz_binary).resolve()),
        "selected_xyz": str(selected_xyz),
        "graph_file": str(graphfile),
        "box_size": args.box_size,
        "input_n": int(len(pts0)),
        "n": int(n),
        "sample_n": args.sample_n,
        "seed": args.seed,
        "construction": "periodic_3d_delaunay_27_tile",
        "qhull_options": args.qhull_options,
        "undirected_edges": int(g.nnz // 2),
        "nnz": int(g.nnz),
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "degree_mean": float(deg.mean()),
        "degree_median": float(np.median(deg)),
        "isolated_vertices": int(np.sum(deg == 0)),
    }

    with open(str(out / "delaunay_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Finished")
    print("Graph:", graphfile)
    print("N:", n)
    print("Edges:", g.nnz // 2)
    print(
        "Degree: min={} median={:.1f} mean={:.3f} max={}".format(
            deg.min(),
            np.median(deg),
            deg.mean(),
            deg.max()
        )
    )
    print("Isolated vertices:", np.sum(deg == 0))


if __name__ == "__main__":
    main()
