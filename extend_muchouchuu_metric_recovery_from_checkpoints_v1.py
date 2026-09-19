#!/usr/bin/env python3
"""
Checkpoint-only extended MuchoUchuu metric-recovery diagnostics.

This script does NOT rerun graph diffusion. It:
  1. samples periodic halo pairs out to a larger physical separation,
  2. loads existing state_tXXXXXXX.npy diffusion checkpoints,
  3. computes diffusion-distance curves and linearity diagnostics.

Designed for extending the existing 0--1800 Mpc/h test to 3000 Mpc/h.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


def periodic_distances(xyz, i_idx, j_idx, box_size):
    d = np.abs(
        np.asarray(xyz[i_idx], dtype=np.float64)
        - np.asarray(xyz[j_idx], dtype=np.float64)
    )
    d = np.minimum(d, box_size - d)
    return np.sqrt(np.sum(d * d, axis=1))


def sample_pairs_stratified(
    xyz,
    box_size,
    rng,
    n_bins,
    pairs_per_bin,
    dmin,
    dmax,
    max_rounds,
    outdir,
):
    pair_file = outdir / "sampled_pairs_extended.npz"
    edge_file = outdir / "sampled_pair_bin_edges.npy"

    edges = np.linspace(dmin, dmax, n_bins + 1)

    if pair_file.exists() and edge_file.exists():
        saved_edges = np.load(edge_file)
        z = np.load(pair_file)
        if (
            np.array_equal(saved_edges, edges)
            and int(z["pairs_per_bin"]) == pairs_per_bin
        ):
            print("Reusing saved extended pair sample:", pair_file, flush=True)
            return z["i"], z["j"], z["distance"], edges
        raise RuntimeError(
            "Existing pair sample has incompatible bins/settings. "
            "Use a new output directory or delete sampled_pairs_extended.npz."
        )

    n = len(xyz)
    si = [[] for _ in range(n_bins)]
    sj = [[] for _ in range(n_bins)]
    sd = [[] for _ in range(n_bins)]
    counts = np.zeros(n_bins, dtype=np.int64)

    # Keep memory moderate while giving sparse low-r bins enough candidates.
    batch = max(1_000_000, n_bins * pairs_per_bin * 10)

    for rnd in range(max_rounds):
        if np.all(counts >= pairs_per_bin):
            break

        i = rng.randint(0, n, size=batch).astype(np.int64)
        j = rng.randint(0, n, size=batch).astype(np.int64)
        mask = i != j
        i, j = i[mask], j[mask]

        dist = periodic_distances(xyz, i, j, box_size)
        bid = np.searchsorted(edges, dist, side="right") - 1

        for b in range(n_bins):
            need = pairs_per_bin - counts[b]
            if need <= 0:
                continue
            found = np.flatnonzero(bid == b)[:need]
            if found.size:
                si[b].append(i[found])
                sj[b].append(j[found])
                sd[b].append(dist[found])
                counts[b] += found.size

        if rnd % 10 == 0 or np.all(counts >= pairs_per_bin):
            print(
                "pair round {} min/median/max bin count = {}/{}/{}".format(
                    rnd,
                    int(counts.min()),
                    int(np.median(counts)),
                    int(counts.max()),
                ),
                flush=True,
            )

    if np.any(counts < pairs_per_bin):
        deficient = np.flatnonzero(counts < pairs_per_bin)
        raise RuntimeError(
            "Pair sampling incomplete after {} rounds. Deficient bins: {} "
            "with counts {}".format(
                max_rounds,
                deficient.tolist(),
                counts[deficient].tolist(),
            )
        )

    ia = np.concatenate([np.concatenate(v) for v in si])
    ja = np.concatenate([np.concatenate(v) for v in sj])
    da = np.concatenate([np.concatenate(v) for v in sd])

    np.savez(
        pair_file,
        i=ia,
        j=ja,
        distance=da,
        pairs_per_bin=np.int64(pairs_per_bin),
    )
    np.save(edge_file, edges)
    return ia, ja, da, edges


def fit_linearity(distance, delta, lo, hi, bins):
    edges = np.linspace(lo, hi, bins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (distance >= a) & (distance < b)
        if mask.sum() >= 50:
            xs.append(np.median(distance[mask]))
            ys.append(np.median(delta[mask]))

    x = np.asarray(xs)
    y = np.asarray(ys)
    if len(x) < 5:
        return (np.nan,) * 5

    scale = np.dot(x, y) / max(np.dot(x, x), 1e-30)
    pred = scale * x
    err = np.sqrt(np.mean((y - pred) ** 2)) / max(
        np.sqrt(np.mean(y * y)), 1e-30
    )
    r2 = linregress(x, y).rvalue ** 2
    return (
        float(x.min()),
        float(x.max()),
        float(scale),
        float(err),
        float(r2),
    )


def diagnostics(
    graph_name,
    step,
    state,
    i,
    j,
    physical,
    windows,
    fit_bins,
    curve_edges,
):
    delta = np.linalg.norm(
        np.asarray(state[i], dtype=np.float64)
        - np.asarray(state[j], dtype=np.float64),
        axis=1,
    )
    # Same normalization as the original metric-recovery run.
    delta /= max(np.sqrt(np.mean(delta * delta)), 1e-30)

    diag_rows = []
    for k in range(0, len(windows), 2):
        lo, hi = windows[k], windows[k + 1]
        xmin, xmax, scale, err, r2 = fit_linearity(
            physical, delta, lo, hi, fit_bins
        )
        diag_rows.append(
            dict(
                graph=graph_name,
                diffusion_time=step,
                requested_window_min=lo,
                requested_window_max=hi,
                window_min_mpc_h=xmin,
                window_max_mpc_h=xmax,
                linear_scale=scale,
                relative_rmse=err,
                linear_r2=r2,
            )
        )

    curve_rows = []
    for lo, hi in zip(curve_edges[:-1], curve_edges[1:]):
        mask = (physical >= lo) & (physical < hi)
        if np.any(mask):
            curve_rows.append(
                dict(
                    graph=graph_name,
                    diffusion_time=step,
                    distance_bin_center_mpc_h=0.5 * (lo + hi),
                    pair_count=int(mask.sum()),
                    median_diffusion_distance=float(np.median(delta[mask])),
                    q16=float(np.quantile(delta[mask], 0.16)),
                    q84=float(np.quantile(delta[mask], 0.84)),
                )
            )
    return diag_rows, curve_rows


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz_binary")
    ap.add_argument("state_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--box-size", type=float, default=6000.0)
    ap.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384],
    )
    ap.add_argument("--sketch-dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12346)
    ap.add_argument("--pair-distance-bins", type=int, default=96)
    ap.add_argument("--pairs-per-bin", type=int, default=1000)
    ap.add_argument("--curve-distance-min", type=float, default=50.0)
    ap.add_argument("--curve-distance-max", type=float, default=3000.0)
    ap.add_argument("--max-pair-rounds", type=int, default=5000)
    ap.add_argument(
        "--fit-windows",
        nargs="+",
        type=float,
        default=[
            75, 150,
            150, 300,
            300, 600,
            600, 1200,
            1200, 1800,
            1800, 2400,
            2400, 3000,
        ],
    )
    ap.add_argument("--fit-bins", type=int, default=16)
    ap.add_argument("--graph-name", default="knn_periodic")
    args = ap.parse_args()

    if len(args.fit_windows) % 2:
        raise ValueError("--fit-windows must contain lo hi pairs")

    xyz_path = Path(args.xyz_binary).expanduser().resolve()
    state_dir = Path(args.state_dir).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bytes_per_row = 3 * np.dtype(np.float32).itemsize
    nbytes = xyz_path.stat().st_size
    if nbytes % bytes_per_row:
        raise ValueError("XYZ binary size is not divisible by 12 bytes")
    n = nbytes // bytes_per_row

    xyz = np.memmap(
        xyz_path, mode="r", dtype=np.float32, shape=(n, 3)
    )

    print(f"N={n:,}", flush=True)
    print(f"Box={args.box_size} h^-1 Mpc", flush=True)
    print(
        f"Extended range={args.curve_distance_min}--"
        f"{args.curve_distance_max} h^-1 Mpc",
        flush=True,
    )

    rng = np.random.RandomState(args.seed)
    i, j, physical, curve_edges = sample_pairs_stratified(
        xyz,
        args.box_size,
        rng,
        args.pair_distance_bins,
        args.pairs_per_bin,
        args.curve_distance_min,
        args.curve_distance_max,
        args.max_pair_rounds,
        outdir,
    )

    all_diag, all_curve = [], []
    for step in sorted(set(args.times)):
        state_path = state_dir / f"state_t{step:07d}.npy"
        if not state_path.exists():
            raise FileNotFoundError(state_path)

        print("Loading checkpoint:", state_path, flush=True)
        state = np.load(state_path, mmap_mode="r")
        if state.shape != (n, args.sketch_dim):
            raise ValueError(
                f"{state_path.name} has shape {state.shape}, "
                f"expected {(n, args.sketch_dim)}"
            )

        dr, cr = diagnostics(
            args.graph_name,
            step,
            state,
            i,
            j,
            physical,
            args.fit_windows,
            args.fit_bins,
            curve_edges,
        )
        all_diag.extend(dr)
        all_curve.extend(cr)
        print(f"Computed extended diagnostics for t={step}", flush=True)

    diag_fields = [
        "graph",
        "diffusion_time",
        "requested_window_min",
        "requested_window_max",
        "window_min_mpc_h",
        "window_max_mpc_h",
        "linear_scale",
        "relative_rmse",
        "linear_r2",
    ]
    curve_fields = [
        "graph",
        "diffusion_time",
        "distance_bin_center_mpc_h",
        "pair_count",
        "median_diffusion_distance",
        "q16",
        "q84",
    ]

    write_csv(outdir / "euclidean_diagnostics_extended.csv", diag_fields, all_diag)
    write_csv(
        outdir / "diffusion_distance_vs_physical_separation_extended.csv",
        curve_fields,
        all_curve,
    )

    metadata = vars(args).copy()
    metadata.update(
        xyz_binary=str(xyz_path),
        state_dir=str(state_dir),
        inferred_n=int(n),
        operation="checkpoint-only metric-recovery extension",
        reran_diffusion=False,
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Finished:", outdir, flush=True)


if __name__ == "__main__":
    main()
