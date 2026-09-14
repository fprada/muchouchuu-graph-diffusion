#!/usr/bin/env python3
"""
Upgrade Step 6 for diffusion return-trace analysis.

Features
--------
1. Fits log P_return versus log t in sliding local windows.
2. Computes d_s = -2 * slope.
3. Uses a probe-level bootstrap when per-probe files are available.
4. Falls back to a parametric bootstrap from the summary standard error.
5. Requires persistence across consecutive windows for the operational crossing.
6. Writes a CSV, JSON summary, and diagnostic plot.

The probe bootstrap resamples the same probe index across all diffusion times,
preserving each probe's trajectory.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_CANDIDATES = (
    "diffusion_time", "time", "t", "step", "diffusion_steps"
)
MEAN_CANDIDATES = (
    "mean_return_probability", "return_probability",
    "mean_return_prob", "p_return", "mean_trace_per_node"
)
SE_CANDIDATES = (
    "mean_return_probability_se", "return_probability_se",
    "trace_se", "standard_error", "se"
)


def pick_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(
        f"Could not identify {label} column. Available columns: {list(df.columns)}"
    )


def local_window_indices(n: int, center: int, width: int) -> np.ndarray:
    """Return a strictly centred local window.

    A window is valid only when exactly width//2 sampled times exist
    on each side of the nominal centre. Edge-shifted/repeated windows
    are not allowed.
    """
    if width < 3 or width % 2 != 1:
        raise ValueError("--window-points must be an odd integer >= 3")
    if width > n:
        raise ValueError(
            f"--window-points={width} exceeds number of times={n}"
        )

    half = width // 2
    lo = center - half
    hi = center + half + 1

    if lo < 0 or hi > n:
        return np.empty(0, dtype=np.int64)

    return np.arange(lo, hi, dtype=np.int64)


def fit_ds(times: np.ndarray, probs: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return d_s, R^2, and relative RMS residual in log-probability."""
    n = len(times)
    ds = np.full(n, np.nan, dtype=float)
    r2 = np.full(n, np.nan, dtype=float)
    rel_rmse = np.full(n, np.nan, dtype=float)

    xall = np.log(times.astype(float))
    yall = np.log(probs.astype(float))

    for i in range(n):
        idx = local_window_indices(n, i, width)

        # Strict-centred estimator: edge locations without the full
        # symmetric window remain NaN and are not used downstream.
        if idx.size != width:
            continue

        x = xall[idx]
        y = yall[idx]
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        resid = y - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        ds[i] = -2.0 * slope
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # RMS multiplicative/log residual; small values approximate fractional error.
        rel_rmse[i] = float(np.sqrt(np.mean(resid**2)))
    return ds, r2, rel_rmse


def extract_probe_values(path: str) -> tuple[int, np.ndarray]:
    m = re.search(r"[tT](\d{3,})", Path(path).name)
    if not m:
        raise ValueError(f"Cannot extract diffusion time from probe filename: {path}")
    t = int(m.group(1))

    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric probe values found in {path}")

    # Drop obvious metadata columns.
    drop = []
    for c in numeric.columns:
        lc = str(c).lower()
        if lc in TIME_CANDIDATES or any(k in lc for k in ("time", "step", "probe_id", "index")):
            drop.append(c)
    numeric = numeric.drop(columns=drop, errors="ignore")
    if numeric.empty:
        raise ValueError(f"No usable numeric probe values found in {path}")

    # Prefer columns whose names suggest return probability or trace estimate.
    preferred = [
        c for c in numeric.columns
        if any(k in str(c).lower() for k in ("return", "trace", "estimate", "value"))
        and not any(k in str(c).lower() for k in ("se", "error", "std"))
    ]
    if preferred:
        numeric = numeric[preferred]

    arr = numeric.to_numpy(dtype=float)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        raise ValueError(f"No finite probe values found in {path}")

    # Trace values may be totals; normalize only if explicitly supplied upstream.
    return t, vals.astype(float)


def load_probe_matrix(pattern: str, expected_times: np.ndarray) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched --probe-glob={pattern!r}")

    by_time: dict[int, np.ndarray] = {}
    for f in files:
        t, vals = extract_probe_values(f)
        by_time[t] = vals

    missing = [int(t) for t in expected_times if int(t) not in by_time]
    if missing:
        raise ValueError(f"Probe files are missing times: {missing}")

    counts = [len(by_time[int(t)]) for t in expected_times]
    if len(set(counts)) != 1:
        raise ValueError(
            "Probe counts differ by time; cannot preserve probe trajectories. "
            f"Counts: {dict(zip(map(int, expected_times), counts))}"
        )

    # Shape: n_times x n_probes
    matrix = np.vstack([by_time[int(t)] for t in expected_times])
    if np.any(matrix <= 0):
        raise ValueError("All probe return-probability estimates must be positive")
    return matrix


def first_persistent_time(mask: np.ndarray, times: np.ndarray, persistence: int) -> float:
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if bool(ok) else 0
        if run >= persistence:
            return float(times[i - persistence + 1])
    return math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--probe-glob", default=None,
                    help="Glob for per-time probe CSV files, e.g. 'output/*probe*t*.csv'")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window-points", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tolerance", type=float, default=0.03,
                    help="Half-width around d_s=3; 0.03 is a 1%% band")
    ap.add_argument("--persistence", type=int, default=3,
                    help="Required number of consecutive qualifying windows")
    ap.add_argument("--min-r2", type=float, default=0.98)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    tcol = pick_column(summary, TIME_CANDIDATES, "diffusion-time")
    mcol = pick_column(summary, MEAN_CANDIDATES, "mean return-probability")
    try:
        secol = pick_column(summary, SE_CANDIDATES, "return-probability SE")
    except ValueError:
        secol = None

    summary = summary.sort_values(tcol).reset_index(drop=True)
    times = summary[tcol].to_numpy(dtype=float)
    mean_p = summary[mcol].to_numpy(dtype=float)

    if np.any(times <= 0) or np.any(mean_p <= 0):
        raise ValueError("Diffusion times and mean return probabilities must be positive")

    central_ds, central_r2, central_log_rmse = fit_ds(
        times, mean_p, args.window_points
    )

    rng = np.random.default_rng(args.seed)
    boot_ds = np.empty((args.bootstrap, len(times)), dtype=float)
    bootstrap_mode: str
    n_probes: int | None = None

    if args.probe_glob:
        probes = load_probe_matrix(args.probe_glob, times.astype(int))
        n_probes = probes.shape[1]
        bootstrap_mode = "probe_resampling"
        for b in range(args.bootstrap):
            idx = rng.integers(0, n_probes, size=n_probes)
            p = probes[:, idx].mean(axis=1)
            boot_ds[b], _, _ = fit_ds(times, p, args.window_points)
    else:
        if secol is None:
            raise ValueError(
                "No --probe-glob was supplied and no standard-error column was found."
            )
        se = summary[secol].to_numpy(dtype=float)
        if np.any(se < 0):
            raise ValueError("Standard errors must be nonnegative")
        bootstrap_mode = "parametric_from_summary_se"
        # Lognormal approximation keeps every realization positive.
        rel = np.divide(se, mean_p, out=np.zeros_like(se), where=mean_p > 0)
        sigma_log = np.sqrt(np.log1p(rel**2))
        mu_log = np.log(mean_p) - 0.5 * sigma_log**2
        for b in range(args.bootstrap):
            p = np.exp(rng.normal(mu_log, sigma_log))
            boot_ds[b], _, _ = fit_ds(times, p, args.window_points)

    q025, q16, q50, q84, q975 = np.quantile(
        boot_ds, [0.025, 0.16, 0.50, 0.84, 0.975], axis=0
    )
    p_within = np.mean(np.abs(boot_ds - 3.0) <= args.tolerance, axis=0)

    central_in_band = np.abs(central_ds - 3.0) <= args.tolerance
    ci95_contains_3 = (q025 <= 3.0) & (q975 >= 3.0)
    fit_quality_ok = central_r2 >= args.min_r2
    qualifies = central_in_band & ci95_contains_3 & fit_quality_ok

    persistent = np.zeros(len(times), dtype=bool)
    for i in range(len(times)):
        if i + 1 >= args.persistence:
            persistent[i] = bool(
                np.all(qualifies[i - args.persistence + 1:i + 1])
            )

    # Crossing distribution: first persistent run inside the tolerance band
    # with adequate fit quality for each bootstrap realization.
    boot_cross = np.full(args.bootstrap, np.nan, dtype=float)
    for b in range(args.bootstrap):
        mask = (np.abs(boot_ds[b] - 3.0) <= args.tolerance) & fit_quality_ok
        boot_cross[b] = first_persistent_time(mask, times, args.persistence)

    valid_cross = boot_cross[np.isfinite(boot_cross)]
    crossing_fraction = float(valid_cross.size / args.bootstrap)
    crossing_quantiles = None
    if valid_cross.size:
        crossing_quantiles = {
            "q16": float(np.quantile(valid_cross, 0.16)),
            "median": float(np.quantile(valid_cross, 0.50)),
            "q84": float(np.quantile(valid_cross, 0.84)),
            "q025": float(np.quantile(valid_cross, 0.025)),
            "q975": float(np.quantile(valid_cross, 0.975)),
        }

    result = pd.DataFrame({
        "diffusion_time": times.astype(int),
        "window_points": args.window_points,
        "central_ds": central_ds,
        "bootstrap_ds_median": q50,
        "bootstrap_ds_q025": q025,
        "bootstrap_ds_q16": q16,
        "bootstrap_ds_q84": q84,
        "bootstrap_ds_q975": q975,
        "bootstrap_probability_within_tolerance": p_within,
        "local_loglog_r2": central_r2,
        "local_log_rmse": central_log_rmse,
        "central_within_tolerance": central_in_band,
        "ci95_contains_3": ci95_contains_3,
        "fit_quality_ok": fit_quality_ok,
        "qualifies": qualifies,
        "persistent_criterion_met_here": persistent,
    })
    result.to_csv(outdir / "spectral_dimension_sliding_bootstrap.csv", index=False)

    first_central_persistent = first_persistent_time(
        qualifies, times, args.persistence
    )
    metadata = {
        "summary_csv": str(Path(args.summary_csv).resolve()),
        "probe_glob": args.probe_glob,
        "bootstrap_mode": bootstrap_mode,
        "number_of_probes": n_probes,
        "window_points": args.window_points,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "target_dimension": 3.0,
        "tolerance": args.tolerance,
        "persistence": args.persistence,
        "minimum_local_r2": args.min_r2,
        "first_central_persistent_time": (
            None if not np.isfinite(first_central_persistent)
            else int(first_central_persistent)
        ),
        "bootstrap_crossing_success_fraction": crossing_fraction,
        "bootstrap_crossing_time_quantiles": crossing_quantiles,
        "warning": (
            "A low bootstrap crossing success fraction means the available "
            "probe count does not constrain a persistent crossing robustly."
        ),
    }
    (outdir / "crossing_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    # Plot without requiring seaborn.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(times, q025, q975, alpha=0.18, label="95% bootstrap interval")
    ax.fill_between(times, q16, q84, alpha=0.30, label="68% bootstrap interval")
    ax.plot(times, central_ds, marker="o", label="Sliding-window central estimate")
    ax.axhline(3.0, linestyle="--", label=r"$d_s=3$")
    ax.axhspan(3.0 - args.tolerance, 3.0 + args.tolerance,
               alpha=0.10, label="Tolerance band")
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time")
    ax.set_ylabel(r"Effective spectral dimension $d_s$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "spectral_dimension_sliding_bootstrap.png", dpi=180)
    plt.close(fig)

    print(result.to_string(index=False))
    print("\nWrote:", outdir / "spectral_dimension_sliding_bootstrap.csv")
    print("Wrote:", outdir / "crossing_summary.json")
    print("Wrote:", outdir / "spectral_dimension_sliding_bootstrap.png")
    print("\nBootstrap mode:", bootstrap_mode)
    print("First central persistent time:", metadata["first_central_persistent_time"])
    print("Bootstrap crossing success fraction:", crossing_fraction)
    print("Bootstrap crossing quantiles:", crossing_quantiles)


if __name__ == "__main__":
    main()
