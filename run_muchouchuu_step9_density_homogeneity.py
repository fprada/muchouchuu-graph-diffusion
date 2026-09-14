#!/usr/bin/env python3
"""
MuchoUchuu Step 9: periodic counts-in-spheres and density homogeneity.

Method
------
1. Build a periodic linked-cell index for the 108M-point float32 XYZ catalogue.
2. Select halo centers uniformly without replacement.
3. Count neighbours within a set of radii around each center.
4. Compute:
     Nbar(<r) = <N(<r)> / [nbar * 4*pi*r^3/3]
     D2(r)    = d ln <N(<r)> / d ln r
5. Bootstrap centers and estimate persistent homogeneity crossings.

The center halo itself is removed from every count.

This is an exact center-sampled calculation within r_max; it does not rerun
the graph-diffusion pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit, prange, set_num_threads
except Exception as exc:
    raise SystemExit(
        "This script requires numba. Install it in the active environment."
    ) from exc


@njit(cache=True)
def build_linked_cells(xyz, box_size, ncell):
    n = xyz.shape[0]
    ncells = ncell * ncell * ncell
    counts = np.zeros(ncells, dtype=np.int64)

    inv = ncell / box_size
    for p in range(n):
        ix = int(xyz[p, 0] * inv)
        iy = int(xyz[p, 1] * inv)
        iz = int(xyz[p, 2] * inv)
        if ix >= ncell:
            ix = ncell - 1
        if iy >= ncell:
            iy = ncell - 1
        if iz >= ncell:
            iz = ncell - 1
        cid = (ix * ncell + iy) * ncell + iz
        counts[cid] += 1

    offsets = np.empty(ncells + 1, dtype=np.int64)
    offsets[0] = 0
    for c in range(ncells):
        offsets[c + 1] = offsets[c] + counts[c]

    cursor = offsets[:-1].copy()
    order = np.empty(n, dtype=np.int32)
    for p in range(n):
        ix = int(xyz[p, 0] * inv)
        iy = int(xyz[p, 1] * inv)
        iz = int(xyz[p, 2] * inv)
        if ix >= ncell:
            ix = ncell - 1
        if iy >= ncell:
            iy = ncell - 1
        if iz >= ncell:
            iz = ncell - 1
        cid = (ix * ncell + iy) * ncell + iz
        pos = cursor[cid]
        order[pos] = p
        cursor[cid] += 1

    return offsets, order


@njit(cache=True, parallel=True)
def count_spheres(
    xyz,
    centers,
    radii2,
    box_size,
    ncell,
    offsets,
    order,
):
    ncent = centers.shape[0]
    nr = radii2.shape[0]
    out = np.zeros((ncent, nr), dtype=np.int64)

    cell_size = box_size / ncell
    rmax = math.sqrt(radii2[-1])
    span = int(math.ceil(rmax / cell_size))
    half_box = 0.5 * box_size
    inv = ncell / box_size

    for ci in prange(ncent):
        p0 = centers[ci]
        x0 = float(xyz[p0, 0])
        y0 = float(xyz[p0, 1])
        z0 = float(xyz[p0, 2])

        ix0 = int(x0 * inv)
        iy0 = int(y0 * inv)
        iz0 = int(z0 * inv)
        if ix0 >= ncell:
            ix0 = ncell - 1
        if iy0 >= ncell:
            iy0 = ncell - 1
        if iz0 >= ncell:
            iz0 = ncell - 1

        hist = np.zeros(nr, dtype=np.int64)

        for ox in range(-span, span + 1):
            ix = (ix0 + ox) % ncell
            for oy in range(-span, span + 1):
                iy = (iy0 + oy) % ncell
                for oz in range(-span, span + 1):
                    iz = (iz0 + oz) % ncell
                    cid = (ix * ncell + iy) * ncell + iz
                    lo = offsets[cid]
                    hi = offsets[cid + 1]

                    for qpos in range(lo, hi):
                        p = order[qpos]
                        if p == p0:
                            continue

                        dx = abs(float(xyz[p, 0]) - x0)
                        dy = abs(float(xyz[p, 1]) - y0)
                        dz = abs(float(xyz[p, 2]) - z0)
                        if dx > half_box:
                            dx = box_size - dx
                        if dy > half_box:
                            dy = box_size - dy
                        if dz > half_box:
                            dz = box_size - dz

                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= radii2[-1]:
                            # First radius whose squared edge includes d2.
                            j = np.searchsorted(radii2, d2)
                            if j < nr:
                                hist[j] += 1

        running = 0
        for j in range(nr):
            running += hist[j]
            out[ci, j] = running

    return out


def local_dimension(r, mean_counts, window_points):
    n = len(r)
    out = np.full(n, np.nan)
    half = window_points // 2
    xall = np.log(r)
    yall = np.log(mean_counts)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < window_points:
            if lo == 0:
                hi = min(n, window_points)
            else:
                lo = max(0, n - window_points)
        x = xall[lo:hi]
        y = yall[lo:hi]
        if np.all(np.isfinite(y)) and len(x) >= 3:
            out[i] = np.polyfit(x, y, 1)[0]
    return out


def first_persistent(mask, persistence):
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if bool(ok) else 0
        if run >= persistence:
            return i - persistence + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz_binary")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--box-size", type=float, default=6000.0)
    ap.add_argument("--number-density", type=float, default=5e-4)
    ap.add_argument("--centers", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=24680)
    ap.add_argument("--r-min", type=float, default=20.0)
    ap.add_argument("--r-max", type=float, default=600.0)
    ap.add_argument("--radius-bins", type=int, default=80)
    ap.add_argument("--radius-spacing", choices=["log", "linear"], default="log")
    ap.add_argument("--cell-size", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=13579)
    ap.add_argument("--dimension-window-points", type=int, default=5)
    ap.add_argument("--scaled-count-tolerance", type=float, default=0.01)
    ap.add_argument("--dimension-tolerance", type=float, default=0.03)
    ap.add_argument("--persistence", type=int, default=3)
    args = ap.parse_args()

    if args.dimension_window_points < 3 or args.dimension_window_points % 2 == 0:
        raise ValueError("--dimension-window-points must be odd and >=3")
    if args.r_max >= 0.5 * args.box_size:
        raise ValueError("--r-max must be below L/2 for this linked-cell search")
    if args.centers < 10:
        raise ValueError("Use at least 10 centers")

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(args.xyz_binary).resolve()
    nbytes = xyz_path.stat().st_size
    if nbytes % 12:
        raise ValueError("XYZ binary size must be divisible by 12")
    n = nbytes // 12
    xyz = np.memmap(xyz_path, mode="r", dtype=np.float32, shape=(n, 3))

    ncell = int(math.floor(args.box_size / args.cell_size))
    if ncell < 3:
        raise ValueError("Too few linked cells")
    actual_cell_size = args.box_size / ncell
    span = int(math.ceil(args.r_max / actual_cell_size))
    if 2 * span + 1 >= ncell:
        raise ValueError("Search span wraps onto duplicate cells; reduce r-max or cell size")

    set_num_threads(args.threads)
    print(f"N={n:,}", flush=True)
    print(f"Centers={args.centers:,}", flush=True)
    print(f"Numba threads={args.threads}", flush=True)
    print(
        f"Linked grid: {ncell}^3 cells, cell side={actual_cell_size:.6f}, "
        f"search span={span}",
        flush=True,
    )

    if args.radius_spacing == "log":
        radii = np.geomspace(args.r_min, args.r_max, args.radius_bins)
    else:
        radii = np.linspace(args.r_min, args.r_max, args.radius_bins)

    rng = np.random.default_rng(args.seed)
    centers = rng.choice(n, size=args.centers, replace=False).astype(np.int64)
    np.save(outdir / "center_indices.npy", centers)
    np.save(outdir / "radii_mpc_h.npy", radii)

    print("Building linked-cell index...", flush=True)
    offsets, order = build_linked_cells(xyz, args.box_size, ncell)
    print(
        f"Linked-cell index built: offsets={offsets.nbytes/2**20:.1f} MiB, "
        f"order={order.nbytes/2**30:.2f} GiB",
        flush=True,
    )

    print("Counting periodic neighbours around sampled centers...", flush=True)
    counts = count_spheres(
        xyz,
        centers,
        radii * radii,
        args.box_size,
        ncell,
        offsets,
        order,
    )
    np.save(outdir / "counts_per_center.npy", counts)
    print("Counts complete.", flush=True)

    mean_counts = counts.mean(axis=0)
    volume = 4.0 * np.pi * radii**3 / 3.0
    poisson_counts = args.number_density * volume
    scaled = mean_counts / poisson_counts
    d2 = local_dimension(radii, mean_counts, args.dimension_window_points)

    count_ok = np.abs(scaled - 1.0) <= args.scaled_count_tolerance
    d2_ok = np.abs(d2 - 3.0) <= args.dimension_tolerance
    joint_ok = count_ok & d2_ok
    central_idx = first_persistent(joint_ok, args.persistence)

    # Center bootstrap.
    brng = np.random.default_rng(args.bootstrap_seed)
    cross = np.full(args.bootstrap, np.nan)
    scaled_boot = np.empty((args.bootstrap, len(radii)), dtype=np.float64)
    d2_boot = np.empty_like(scaled_boot)

    for b in range(args.bootstrap):
        idx = brng.integers(0, args.centers, size=args.centers)
        mc = counts[idx].mean(axis=0)
        sb = mc / poisson_counts
        db = local_dimension(radii, mc, args.dimension_window_points)
        scaled_boot[b] = sb
        d2_boot[b] = db
        ok = (
            (np.abs(sb - 1.0) <= args.scaled_count_tolerance)
            & (np.abs(db - 3.0) <= args.dimension_tolerance)
        )
        j = first_persistent(ok, args.persistence)
        if j is not None:
            cross[b] = radii[j]

    q_scaled = np.quantile(scaled_boot, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    q_d2 = np.quantile(d2_boot, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)

    result = pd.DataFrame({
        "radius_mpc_h": radii,
        "mean_count": mean_counts,
        "poisson_expected_count": poisson_counts,
        "scaled_count": scaled,
        "scaled_count_q025": q_scaled[0],
        "scaled_count_q16": q_scaled[1],
        "scaled_count_bootstrap_median": q_scaled[2],
        "scaled_count_q84": q_scaled[3],
        "scaled_count_q975": q_scaled[4],
        "correlation_dimension_D2": d2,
        "D2_q025": q_d2[0],
        "D2_q16": q_d2[1],
        "D2_bootstrap_median": q_d2[2],
        "D2_q84": q_d2[3],
        "D2_q975": q_d2[4],
        "scaled_count_within_tolerance": count_ok,
        "D2_within_tolerance": d2_ok,
        "joint_criterion": joint_ok,
    })
    result.to_csv(outdir / "density_homogeneity_curve.csv", index=False)

    valid = cross[np.isfinite(cross)]
    summary = {
        "simulation": "MuchoUchuu",
        "n_objects": int(n),
        "box_size_mpc_h": args.box_size,
        "number_density_h3_mpc3": args.number_density,
        "sampled_centers": args.centers,
        "radius_range_mpc_h": [args.r_min, args.r_max],
        "radius_bins": args.radius_bins,
        "radius_spacing": args.radius_spacing,
        "dimension_window_points": args.dimension_window_points,
        "scaled_count_tolerance": args.scaled_count_tolerance,
        "dimension_tolerance": args.dimension_tolerance,
        "persistence": args.persistence,
        "central_first_persistent_radius_mpc_h": (
            None if central_idx is None else float(radii[central_idx])
        ),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_crossing_success_fraction": float(len(valid) / args.bootstrap),
        "bootstrap_crossing_quantiles_mpc_h": (
            None if len(valid) == 0 else {
                "q025": float(np.quantile(valid, 0.025)),
                "q16": float(np.quantile(valid, 0.16)),
                "median": float(np.quantile(valid, 0.50)),
                "q84": float(np.quantile(valid, 0.84)),
                "q975": float(np.quantile(valid, 0.975)),
            }
        ),
        "transport_scale_fiducial_mpc_h": 273.0071462496935,
        "note": (
            "Density and transport scales are operational quantities with "
            "different estimators; compare their magnitudes and uncertainties, "
            "not as identical definitions."
        ),
    }
    (outdir / "density_homogeneity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(radii, q_scaled[0], q_scaled[4], alpha=0.18, label="95% bootstrap")
    ax.fill_between(radii, q_scaled[1], q_scaled[3], alpha=0.30, label="68% bootstrap")
    ax.plot(radii, scaled, label=r"$\mathcal{N}(<r)$")
    ax.axhline(1.0, linestyle="--")
    ax.axhspan(
        1.0 - args.scaled_count_tolerance,
        1.0 + args.scaled_count_tolerance,
        alpha=0.10,
        label="Scaled-count tolerance",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Radius [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"Scaled counts $\mathcal{N}(<r)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "scaled_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(radii, q_d2[0], q_d2[4], alpha=0.18, label="95% bootstrap")
    ax.fill_between(radii, q_d2[1], q_d2[3], alpha=0.30, label="68% bootstrap")
    ax.plot(radii, d2, label=r"$D_2(r)$")
    ax.axhline(3.0, linestyle="--")
    ax.axhspan(
        3.0 - args.dimension_tolerance,
        3.0 + args.dimension_tolerance,
        alpha=0.10,
        label="Dimension tolerance",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Radius [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"Correlation dimension $D_2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "correlation_dimension.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote:", outdir / "counts_per_center.npy", flush=True)
    print("Wrote:", outdir / "density_homogeneity_curve.csv", flush=True)
    print("Wrote:", outdir / "density_homogeneity_summary.json", flush=True)
    print("Wrote:", outdir / "scaled_counts.png", flush=True)
    print("Wrote:", outdir / "correlation_dimension.png", flush=True)


if __name__ == "__main__":
    main()
