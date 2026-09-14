#!/usr/bin/env python3
"""
Fourier-mode diffusion calibration for periodic halo graphs.

For low integer wavevectors n=(nx,ny,nz), initialize cos(k.x) and sin(k.x),
propagate them with P = D^{-1} A, and measure

    C_k(t) = [<cos, P^t cos> + <sin, P^t sin>] /
             [<cos, cos> + <sin, sin>].

For continuum diffusion, C_k(t) ~= exp(-D k^2 t). The script fits D and
reports ell_RMS(t) = sqrt(6 D t) = A_RMS sqrt(t).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def parse_nvecs(values):
    if len(values) % 3:
        raise ValueError("--nvecs must contain triples nx ny nz")
    out = []
    for i in range(0, len(values), 3):
        v = tuple(int(x) for x in values[i:i+3])
        if v == (0, 0, 0):
            raise ValueError("Zero wavevector is not allowed")
        out.append(v)
    return out


def spmm(A, X, backend):
    if backend == "mkl":
        try:
            from sparse_dot_mkl import dot_product_mkl
        except Exception as exc:
            raise RuntimeError(
                "backend=mkl requested but sparse_dot_mkl is unavailable"
            ) from exc
        return np.asarray(dot_product_mkl(A, X), dtype=np.float32)
    return np.asarray(A @ X, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz_binary")
    ap.add_argument("graph_npz")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--box-size", type=float, required=True)
    ap.add_argument("--times", nargs="+", type=int,
                    default=[0, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--nvecs", nargs="+", type=int,
                    default=[
                        1,0,0, 0,1,0, 0,0,1,
                        2,0,0, 0,2,0, 0,0,2,
                    ])
    ap.add_argument("--backend", choices=["mkl", "scipy"], default="mkl")
    ap.add_argument("--progress-every", type=int, default=32)
    ap.add_argument("--fit-tmin", type=int, default=256)
    ap.add_argument("--fit-tmax", type=int, default=4096)
    ap.add_argument("--fit-max-n2", type=int, default=4)
    args = ap.parse_args()

    times = sorted(set(args.times))
    if not times or times[0] != 0:
        times = [0] + times
    nvecs = parse_nvecs(args.nvecs)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(args.xyz_binary).resolve()
    nbytes = xyz_path.stat().st_size
    if nbytes % 12:
        raise ValueError("XYZ binary size is not divisible by 12 bytes")
    n = nbytes // 12
    xyz = np.memmap(xyz_path, mode="r", dtype=np.float32, shape=(n, 3))

    print(f"N={n:,}", flush=True)
    print(f"Loading graph: {Path(args.graph_npz).resolve()}", flush=True)
    A = sparse.load_npz(args.graph_npz).tocsr()

    if A.dtype != np.float32:
        print(
            f"Casting graph from {A.dtype} to float32 for MKL...",
            flush=True,
        )
        A = A.astype(np.float32, copy=False)

    if A.shape != (n, n):
        raise ValueError(f"Graph shape {A.shape} does not match N={n}")
    A.sort_indices()

    degree = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    if np.any(degree <= 0):
        raise ValueError("Graph contains zero-degree nodes")
    inv_degree = (1.0 / degree).astype(np.float32)

    m = 2 * len(nvecs)
    X0 = np.empty((n, m), dtype=np.float32)
    norms = np.empty(len(nvecs), dtype=np.float64)
    k2 = np.empty(len(nvecs), dtype=np.float64)

    twopi_over_L = 2.0 * np.pi / args.box_size
    print(f"Initializing {len(nvecs)} Fourier vectors ({m} columns)...", flush=True)
    for q, nv in enumerate(nvecs):
        phase = twopi_over_L * (
            nv[0] * np.asarray(xyz[:, 0], dtype=np.float64)
            + nv[1] * np.asarray(xyz[:, 1], dtype=np.float64)
            + nv[2] * np.asarray(xyz[:, 2], dtype=np.float64)
        )
        c = np.cos(phase).astype(np.float32)
        s = np.sin(phase).astype(np.float32)
        X0[:, 2*q] = c
        X0[:, 2*q+1] = s
        norms[q] = float(np.dot(c.astype(np.float64), c.astype(np.float64))
                         + np.dot(s.astype(np.float64), s.astype(np.float64)))
        k2[q] = (twopi_over_L ** 2) * sum(v*v for v in nv)

    X = X0.copy()
    rows = []

    def record(t):
        for q, nv in enumerate(nvecs):
            c0 = X0[:, 2*q]
            s0 = X0[:, 2*q+1]
            ct = X[:, 2*q]
            st = X[:, 2*q+1]
            corr = (
                np.dot(c0.astype(np.float64), ct.astype(np.float64))
                + np.dot(s0.astype(np.float64), st.astype(np.float64))
            ) / norms[q]
            rows.append({
                "diffusion_time": int(t),
                "nx": nv[0], "ny": nv[1], "nz": nv[2],
                "n2": int(sum(v*v for v in nv)),
                "k2_mpc_h2": float(k2[q]),
                "correlation": float(corr),
                "minus_log_correlation": float(-math.log(corr)) if corr > 0 else math.nan,
                "D_estimate": (
                    float(-math.log(corr) / (k2[q] * t))
                    if corr > 0 and t > 0 else math.nan
                ),
            })
        print(f"Recorded t={t}", flush=True)

    record(0)
    target = set(times[1:])
    tmax = max(times)
    for t in range(1, tmax + 1):
        Y = spmm(A, X, args.backend)
        Y *= inv_degree[:, None]
        X = Y
        if t % args.progress_every == 0:
            print(f"t={t}", flush=True)
        if t in target:
            record(t)

    df = pd.DataFrame(rows)
    df.to_csv(out / "fourier_mode_decay.csv", index=False)

    fit = df[
        (df["diffusion_time"] >= args.fit_tmin)
        & (df["diffusion_time"] <= args.fit_tmax)
        & (df["n2"] <= args.fit_max_n2)
        & np.isfinite(df["minus_log_correlation"])
        & (df["correlation"] > 0)
    ].copy()
    if len(fit) < 3:
        raise RuntimeError("Too few valid Fourier decay points for fit")

    x = fit["k2_mpc_h2"].to_numpy(float) * fit["diffusion_time"].to_numpy(float)
    y = fit["minus_log_correlation"].to_numpy(float)
    D = float(np.dot(x, y) / np.dot(x, x))
    pred = D * x
    resid = y - pred
    dof = max(len(y) - 1, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    Dse = float(np.sqrt(sigma2 / np.dot(x, x)))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else math.nan

    A_rms = float(math.sqrt(6.0 * D))
    A_rms_se = float(3.0 * Dse / A_rms) if A_rms > 0 else math.nan

    mode_fits = []
    for nv, g in fit.groupby(["nx", "ny", "nz"]):
        xx = g["k2_mpc_h2"].to_numpy(float) * g["diffusion_time"].to_numpy(float)
        yy = g["minus_log_correlation"].to_numpy(float)
        dm = float(np.dot(xx, yy) / np.dot(xx, xx))
        mode_fits.append({
            "nx": int(nv[0]), "ny": int(nv[1]), "nz": int(nv[2]),
            "n2": int(g["n2"].iloc[0]),
            "D": dm,
            "A_RMS": float(math.sqrt(6.0 * dm)),
            "n_points": int(len(g)),
        })

    summary = {
        "xyz_binary": str(xyz_path),
        "graph_npz": str(Path(args.graph_npz).resolve()),
        "box_size_mpc_h": args.box_size,
        "n_objects": int(n),
        "operator": "P = D^{-1} A",
        "fit_tmin": args.fit_tmin,
        "fit_tmax": args.fit_tmax,
        "fit_max_n2": args.fit_max_n2,
        "n_fit_points": int(len(fit)),
        "D_mpc_h2_per_step": D,
        "D_standard_error_regression": Dse,
        "fit_r2": r2,
        "A_RMS_mpc_h_per_sqrt_step": A_rms,
        "A_RMS_standard_error_regression": A_rms_se,
        "length_definition": "ell_RMS(t) = sqrt(6 D t)",
        "mode_fits": mode_fits,
        "warning": (
            "Regression errors quantify fit scatter only. Inspect mode-to-mode "
            "variation and repeat fit ranges to assess systematic uncertainty."
        ),
    }
    (out / "fourier_diffusion_calibration.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for nv, g in df[df["diffusion_time"] > 0].groupby(["nx","ny","nz"]):
        ax.plot(
            g["diffusion_time"], g["minus_log_correlation"],
            marker="o", label=str(tuple(int(v) for v in nv))
        )
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time")
    ax.set_ylabel(r"$-\ln C_k(t)$")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fourier_mode_decay.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote:", out / "fourier_mode_decay.csv", flush=True)
    print("Wrote:", out / "fourier_diffusion_calibration.json", flush=True)
    print("Wrote:", out / "fourier_mode_decay.png", flush=True)


if __name__ == "__main__":
    main()
