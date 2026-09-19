#!/usr/bin/env python3
"""
Bootstrap spectral-dimension analysis for matched-phase Gaussian-growth
graph-diffusion experiments.

Python >= 3.7 compatible.

IMPORTANT:
    This script does NOT compute a spectral dimension separately for each
    Hutchinson probe and then average those dimensions.

    Instead, for every bootstrap realization it:

        1. resamples probes with replacement;
        2. forms the mean heat-trace estimate H(t);
        3. fits ln H versus ln t in strict-centered windows;
        4. computes d_s = -2 * slope;
        5. applies the local R^2 criterion;
        6. tests the persistent |d_s - 3| <= tolerance criterion.

This reproduces the statistical ordering used in the MuchoUchuu
probe-resampling analysis.

Input directory
---------------
Must contain files named like

    trace_probe_estimates_t0000064.csv
    trace_probe_estimates_t0000096.csv
    ...

with columns

    graph
    diffusion_time
    probe_index
    individual_trace_estimate

Example
-------
python analyze_gaussian_growth_bootstrap.py \
    /work/fprada/DIFFUSION/GROWTH/gaussian_growth_pilot/D1p000/return_trace \
    --output-dir \
    /work/fprada/DIFFUSION/GROWTH/gaussian_growth_pilot/D1p000/bootstrap_ds \
    --window 7 \
    --bootstrap 10000 \
    --seed 12345 \
    --r2-min 0.98 \
    --tolerance 0.03 \
    --persistence 3
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------

TRACE_PATTERN = re.compile(r"trace_probe_estimates_t(\d+)\.csv$")


def discover_trace_files(input_dir):
    """Find and sort trace_probe_estimates_t*.csv by diffusion time."""
    files = glob.glob(
        os.path.join(str(input_dir), "trace_probe_estimates_t*.csv")
    )

    found = []
    for f in files:
        m = TRACE_PATTERN.search(os.path.basename(f))
        if m is None:
            continue
        t_from_name = int(m.group(1))
        found.append((t_from_name, f))

    found.sort(key=lambda x: x[0])

    if not found:
        raise RuntimeError(
            "No trace_probe_estimates_t*.csv files found in {}".format(
                input_dir
            )
        )

    return found


def read_probe_file(path):
    """
    Read one probe-level trace CSV.

    Returns
    -------
    diffusion_time : int
    probe_indices  : ndarray, shape (nprobe,)
    values         : ndarray, shape (nprobe,)
    graph_name     : str
    """
    indices = []
    values = []
    times = []
    graph_names = []

    with open(str(path), "r") as f:
        reader = csv.DictReader(f)

        required = {
            "graph",
            "diffusion_time",
            "probe_index",
            "individual_trace_estimate",
        }

        if reader.fieldnames is None:
            raise RuntimeError("Missing CSV header in {}".format(path))

        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "{} missing columns: {}".format(path, sorted(missing))
            )

        for row in reader:
            graph_names.append(row["graph"])
            times.append(int(float(row["diffusion_time"])))
            indices.append(int(row["probe_index"]))
            values.append(float(row["individual_trace_estimate"]))

    if not values:
        raise RuntimeError("No probe rows in {}".format(path))

    unique_times = sorted(set(times))
    if len(unique_times) != 1:
        raise RuntimeError(
            "{} contains multiple diffusion times: {}".format(
                path, unique_times
            )
        )

    unique_graphs = sorted(set(graph_names))
    if len(unique_graphs) != 1:
        raise RuntimeError(
            "{} contains multiple graph names: {}".format(
                path, unique_graphs
            )
        )

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(indices)
    indices = indices[order]
    values = values[order]

    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(
            "Duplicate probe indices found in {}".format(path)
        )

    return unique_times[0], indices, values, unique_graphs[0]


def load_probe_matrix(input_dir):
    """
    Construct trace matrix H_probe(t).

    Returns
    -------
    times : ndarray, shape (nt,)
    traces : ndarray, shape (nt, nprobe)
    probe_indices : ndarray, shape (nprobe,)
    graph_name : str
    """
    files = discover_trace_files(input_dir)

    all_times = []
    rows = []
    reference_indices = None
    graph_name = None

    for t_name, path in files:
        t_csv, indices, values, graph = read_probe_file(path)

        if t_csv != t_name:
            raise RuntimeError(
                "Time mismatch: filename says {}, CSV says {} in {}".format(
                    t_name, t_csv, path
                )
            )

        if reference_indices is None:
            reference_indices = indices.copy()
        elif not np.array_equal(indices, reference_indices):
            raise RuntimeError(
                "Probe indices do not match across files; problem at {}".format(
                    path
                )
            )

        if graph_name is None:
            graph_name = graph
        elif graph != graph_name:
            raise RuntimeError(
                "Graph name changed from {} to {} in {}".format(
                    graph_name, graph, path
                )
            )

        all_times.append(t_csv)
        rows.append(values)

    times = np.asarray(all_times, dtype=np.float64)
    traces = np.vstack(rows).astype(np.float64)

    if np.any(np.diff(times) <= 0):
        raise RuntimeError("Diffusion times are not strictly increasing")

    if not np.all(np.isfinite(traces)):
        raise RuntimeError("Non-finite individual trace estimates found")

    return times, traces, reference_indices, graph_name


# ----------------------------------------------------------------------
# Spectral dimension
# ----------------------------------------------------------------------

def linear_fit_loglog(times, trace):
    """
    Fit ln(trace) = intercept + slope * ln(t).

    Returns
    -------
    ds : float
        Spectral dimension = -2*slope
    slope : float
    intercept : float
    r2 : float
    """
    times = np.asarray(times, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)

    if len(times) < 2:
        return np.nan, np.nan, np.nan, np.nan

    if (
        np.any(~np.isfinite(times))
        or np.any(~np.isfinite(trace))
        or np.any(times <= 0.0)
        or np.any(trace <= 0.0)
    ):
        return np.nan, np.nan, np.nan, np.nan

    x = np.log(times)
    y = np.log(trace)

    xm = np.mean(x)
    ym = np.mean(y)

    dx = x - xm
    dy = y - ym

    denom = np.sum(dx * dx)
    if denom <= 0.0:
        return np.nan, np.nan, np.nan, np.nan

    slope = np.sum(dx * dy) / denom
    intercept = ym - slope * xm

    yfit = intercept + slope * x

    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - ym) ** 2)

    if ss_tot <= 0.0:
        r2 = 1.0 if ss_res <= 1.0e-30 else np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    ds = -2.0 * slope

    return ds, slope, intercept, r2


def centered_spectral_dimension(times, mean_trace, window, r2_min):
    """
    Strict-centered sliding-window spectral dimension.

    No shifted edge windows are allowed.

    Returns dictionary containing one entry per valid geometrical center.
    d_s is set to NaN when local R^2 < r2_min.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    half = window // 2
    nt = len(times)

    centers = []
    raw_ds = []
    eligible_ds = []
    slopes = []
    intercepts = []
    r2s = []
    start_times = []
    end_times = []

    for i in range(half, nt - half):
        lo = i - half
        hi = i + half + 1

        tw = times[lo:hi]
        hw = mean_trace[lo:hi]

        ds, slope, intercept, r2 = linear_fit_loglog(tw, hw)

        centers.append(times[i])
        raw_ds.append(ds)
        slopes.append(slope)
        intercepts.append(intercept)
        r2s.append(r2)
        start_times.append(tw[0])
        end_times.append(tw[-1])

        if np.isfinite(ds) and np.isfinite(r2) and r2 >= r2_min:
            eligible_ds.append(ds)
        else:
            eligible_ds.append(np.nan)

    return {
        "center_time": np.asarray(centers, dtype=np.float64),
        "window_time_min": np.asarray(start_times, dtype=np.float64),
        "window_time_max": np.asarray(end_times, dtype=np.float64),
        "ds_raw": np.asarray(raw_ds, dtype=np.float64),
        "ds": np.asarray(eligible_ds, dtype=np.float64),
        "slope": np.asarray(slopes, dtype=np.float64),
        "intercept": np.asarray(intercepts, dtype=np.float64),
        "r2": np.asarray(r2s, dtype=np.float64),
    }


# ----------------------------------------------------------------------
# Persistent Euclidean crossing
# ----------------------------------------------------------------------

def persistent_crossing(
    center_times,
    ds,
    r2,
    tolerance=0.03,
    persistence=3,
    r2_min=0.98,
):
    """
    Find first persistent crossing into |d_s - 3| <= tolerance.

    A point is eligible only when:
        finite d_s
        finite R^2
        R^2 >= r2_min

    A crossing is accepted when `persistence` consecutive eligible
    windows lie inside the Euclidean band.

    If possible, linearly interpolate the first entry into the band
    using the margin

        margin = tolerance - |d_s - 3|

    versus diffusion time.

    Returns
    -------
    crossing_time : float
        NaN if no persistent crossing.
    start_index : int
        -1 if no crossing.
    """
    center_times = np.asarray(center_times, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    eligible = (
        np.isfinite(ds)
        & np.isfinite(r2)
        & (r2 >= r2_min)
    )

    margin = tolerance - np.abs(ds - 3.0)
    inside = eligible & (margin >= 0.0)

    n = len(center_times)

    if persistence < 1:
        raise ValueError("persistence must be >= 1")

    for j in range(0, n - persistence + 1):
        if np.all(inside[j:j + persistence]):

            # First qualifying persistent run begins at j.
            if j == 0:
                return float(center_times[j]), int(j)

            # Interpolate only if previous point is eligible,
            # finite, and outside the band.
            if (
                eligible[j - 1]
                and np.isfinite(margin[j - 1])
                and np.isfinite(margin[j])
                and margin[j - 1] < 0.0
                and margin[j] >= 0.0
            ):
                t0 = center_times[j - 1]
                t1 = center_times[j]
                m0 = margin[j - 1]
                m1 = margin[j]

                denom = m1 - m0

                if denom != 0.0:
                    frac = -m0 / denom
                    frac = min(1.0, max(0.0, frac))
                    tcross = t0 + frac * (t1 - t0)
                    return float(tcross), int(j)

            return float(center_times[j]), int(j)

    return np.nan, -1


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------

def bootstrap_analysis(
    times,
    traces,
    window,
    n_bootstrap,
    seed,
    r2_min,
    tolerance,
    persistence,
    progress_every=1000,
):
    """
    Probe-resampling bootstrap.

    traces has shape:
        (n_times, n_probes)

    Each realization resamples n_probes columns with replacement.
    """
    nt, nprobe = traces.shape

    # Central estimator: mean across all actual probes.
    central_mean_trace = np.mean(traces, axis=1)

    central = centered_spectral_dimension(
        times,
        central_mean_trace,
        window=window,
        r2_min=r2_min,
    )

    ncenter = len(central["center_time"])

    ds_boot = np.full(
        (n_bootstrap, ncenter),
        np.nan,
        dtype=np.float64,
    )

    r2_boot = np.full(
        (n_bootstrap, ncenter),
        np.nan,
        dtype=np.float64,
    )

    crossing_boot = np.full(
        n_bootstrap,
        np.nan,
        dtype=np.float64,
    )

    rng = np.random.RandomState(seed)

    print(
        "Bootstrap: {} realizations, {} probes, {} time samples, {} centers".format(
            n_bootstrap, nprobe, nt, ncenter
        ),
        flush=True,
    )

    for b in range(n_bootstrap):

        idx = rng.randint(0, nprobe, size=nprobe)

        # Correct statistical ordering:
        # resample probes -> mean trace -> ln -> slope -> d_s
        mean_trace_b = np.mean(traces[:, idx], axis=1)

        result = centered_spectral_dimension(
            times,
            mean_trace_b,
            window=window,
            r2_min=r2_min,
        )

        ds_boot[b, :] = result["ds"]
        r2_boot[b, :] = result["r2"]

        tcross, _ = persistent_crossing(
            result["center_time"],
            result["ds_raw"],
            result["r2"],
            tolerance=tolerance,
            persistence=persistence,
            r2_min=r2_min,
        )

        crossing_boot[b] = tcross

        if (
            progress_every > 0
            and (
                (b + 1) % progress_every == 0
                or b + 1 == n_bootstrap
            )
        ):
            valid_cross = np.sum(np.isfinite(crossing_boot[:b + 1]))
            print(
                " bootstrap {}/{} crossing_success={:.4f}".format(
                    b + 1,
                    n_bootstrap,
                    valid_cross / float(b + 1),
                ),
                flush=True,
            )

    return (
        central_mean_trace,
        central,
        ds_boot,
        r2_boot,
        crossing_boot,
    )


# ----------------------------------------------------------------------
# Quantiles/output helpers
# ----------------------------------------------------------------------

def finite_quantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.percentile(x, q))


def finite_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def write_mean_trace(path, times, traces):
    means = np.mean(traces, axis=1)

    if traces.shape[1] > 1:
        std = np.std(traces, axis=1, ddof=1)
        se = std / math.sqrt(float(traces.shape[1]))
    else:
        std = np.full(len(times), np.nan)
        se = np.full(len(times), np.nan)

    with open(str(path), "w") as f:
        w = csv.writer(f)

        w.writerow([
            "diffusion_time",
            "mean_trace",
            "std_across_probes",
            "standard_error_mean_trace",
            "relative_standard_error",
            "n_probes",
        ])

        for i in range(len(times)):
            rel = (
                se[i] / means[i]
                if means[i] != 0.0
                else np.nan
            )

            w.writerow([
                "{:.16g}".format(times[i]),
                "{:.16g}".format(means[i]),
                "{:.16g}".format(std[i]),
                "{:.16g}".format(se[i]),
                "{:.16g}".format(rel),
                traces.shape[1],
            ])


def write_central_curve(path, central):
    with open(str(path), "w") as f:
        w = csv.writer(f)

        w.writerow([
            "center_time",
            "window_time_min",
            "window_time_max",
            "spectral_dimension_raw",
            "spectral_dimension_eligible",
            "slope",
            "intercept",
            "r2",
        ])

        n = len(central["center_time"])

        for i in range(n):
            w.writerow([
                "{:.16g}".format(central["center_time"][i]),
                "{:.16g}".format(central["window_time_min"][i]),
                "{:.16g}".format(central["window_time_max"][i]),
                "{:.16g}".format(central["ds_raw"][i]),
                "{:.16g}".format(central["ds"][i]),
                "{:.16g}".format(central["slope"][i]),
                "{:.16g}".format(central["intercept"][i]),
                "{:.16g}".format(central["r2"][i]),
            ])


def write_bootstrap_curve(
    path,
    central,
    ds_boot,
    r2_boot,
):
    with open(str(path), "w") as f:
        w = csv.writer(f)

        w.writerow([
            "center_time",
            "window_time_min",
            "window_time_max",
            "central_ds",
            "central_r2",
            "bootstrap_ds_p2p5",
            "bootstrap_ds_p16",
            "bootstrap_ds_p50",
            "bootstrap_ds_p84",
            "bootstrap_ds_p97p5",
            "bootstrap_valid_fraction",
            "bootstrap_mean_r2",
        ])

        nboot = ds_boot.shape[0]

        for j in range(ds_boot.shape[1]):
            x = ds_boot[:, j]
            valid = np.isfinite(x)

            w.writerow([
                "{:.16g}".format(central["center_time"][j]),
                "{:.16g}".format(central["window_time_min"][j]),
                "{:.16g}".format(central["window_time_max"][j]),
                "{:.16g}".format(central["ds_raw"][j]),
                "{:.16g}".format(central["r2"][j]),
                "{:.16g}".format(finite_quantile(x, 2.5)),
                "{:.16g}".format(finite_quantile(x, 16.0)),
                "{:.16g}".format(finite_quantile(x, 50.0)),
                "{:.16g}".format(finite_quantile(x, 84.0)),
                "{:.16g}".format(finite_quantile(x, 97.5)),
                "{:.16g}".format(
                    np.sum(valid) / float(nboot)
                ),
                "{:.16g}".format(
                    finite_mean(r2_boot[:, j])
                ),
            ])


def write_crossings(path, crossing_boot):
    with open(str(path), "w") as f:
        w = csv.writer(f)

        w.writerow([
            "bootstrap_index",
            "crossing_time",
            "crossing_success",
        ])

        for i, x in enumerate(crossing_boot):
            w.writerow([
                i,
                "{:.16g}".format(x),
                int(np.isfinite(x)),
            ])


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def make_plot(
    path,
    central,
    ds_boot,
    crossing_central,
    crossing_summary,
    tolerance,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "WARNING: unable to import matplotlib; skipping plot: {}".format(
                exc
            ),
            file=sys.stderr,
        )
        return

    t = central["center_time"]

    p16 = np.asarray([
        finite_quantile(ds_boot[:, j], 16.0)
        for j in range(ds_boot.shape[1])
    ])

    p50 = np.asarray([
        finite_quantile(ds_boot[:, j], 50.0)
        for j in range(ds_boot.shape[1])
    ])

    p84 = np.asarray([
        finite_quantile(ds_boot[:, j], 84.0)
        for j in range(ds_boot.shape[1])
    ])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.axhspan(
        3.0 - tolerance,
        3.0 + tolerance,
        alpha=0.15,
        label="Euclidean tolerance band",
    )

    ax.axhline(
        3.0,
        linestyle="--",
        linewidth=1.2,
        label=r"$d_s=3$",
    )

    ax.fill_between(
        t,
        p16,
        p84,
        alpha=0.25,
        label="probe-bootstrap 68%",
    )

    ax.plot(
        t,
        p50,
        marker="o",
        linewidth=1.5,
        label="bootstrap median",
    )

    ax.plot(
        t,
        central["ds_raw"],
        marker="s",
        linewidth=1.3,
        label="central mean-trace curve",
    )

    if np.isfinite(crossing_central):
        ax.axvline(
            crossing_central,
            linestyle=":",
            linewidth=1.2,
            label="central persistent crossing",
        )

    c16 = crossing_summary.get("p16", np.nan)
    c84 = crossing_summary.get("p84", np.nan)

    if np.isfinite(c16) and np.isfinite(c84):
        ax.axvspan(
            c16,
            c84,
            alpha=0.12,
            label="crossing 68%",
        )

    ax.set_xscale("log")
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel(r"spectral dimension $d_s$")
    ax.set_title("Probe-resampling spectral dimension")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Probe-resampling bootstrap spectral-dimension analysis "
            "from trace_probe_estimates_t*.csv files."
        )
    )

    p.add_argument(
        "input_dir",
        help="Directory containing trace_probe_estimates_t*.csv",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: INPUT_DIR/bootstrap_spectral_dimension"
        ),
    )

    p.add_argument(
        "--window",
        type=int,
        default=7,
        help="Strict-centered sliding-window width. Default: 7",
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of probe-resampling bootstrap realizations.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Bootstrap random seed.",
    )

    p.add_argument(
        "--r2-min",
        type=float,
        default=0.98,
        help="Minimum local log-log fit R^2. Default: 0.98",
    )

    p.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help=(
            "Absolute tolerance around d_s=3. "
            "Default 0.03 corresponds to one percent."
        ),
    )

    p.add_argument(
        "--persistence",
        type=int,
        default=3,
        help="Required consecutive eligible Euclidean windows.",
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print bootstrap progress every N realizations.",
    )

    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate PNG diagnostic plot.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()

    if args.output_dir is None:
        output_dir = input_dir / "bootstrap_spectral_dimension"
    else:
        output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.window < 3 or args.window % 2 == 0:
        raise ValueError("--window must be an odd integer >= 3")

    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be >= 1")

    if args.persistence < 1:
        raise ValueError("--persistence must be >= 1")

    print("Loading probe-level traces from:", input_dir, flush=True)

    times, traces, probe_indices, graph_name = load_probe_matrix(
        input_dir
    )

    nt, nprobe = traces.shape

    print("Graph:", graph_name, flush=True)
    print("Times:", [int(x) for x in times], flush=True)
    print("Number of time samples:", nt, flush=True)
    print("Number of probes:", nprobe, flush=True)
    print("Window:", args.window, flush=True)

    ncenter = nt - args.window + 1

    if ncenter <= 0:
        raise RuntimeError(
            "Not enough time samples ({}) for window {}".format(
                nt, args.window
            )
        )

    print(
        "Strict-centered spectral-dimension centers expected:",
        ncenter,
        flush=True,
    )

    (
        central_mean_trace,
        central,
        ds_boot,
        r2_boot,
        crossing_boot,
    ) = bootstrap_analysis(
        times=times,
        traces=traces,
        window=args.window,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        r2_min=args.r2_min,
        tolerance=args.tolerance,
        persistence=args.persistence,
        progress_every=args.progress_every,
    )

    crossing_central, crossing_start_index = persistent_crossing(
        central["center_time"],
        central["ds_raw"],
        central["r2"],
        tolerance=args.tolerance,
        persistence=args.persistence,
        r2_min=args.r2_min,
    )

    valid_crossings = crossing_boot[np.isfinite(crossing_boot)]

    crossing_success_fraction = (
        len(valid_crossings) / float(args.bootstrap)
    )

    crossing_summary = {
        "success_fraction": crossing_success_fraction,
        "n_success": int(len(valid_crossings)),
        "n_bootstrap": int(args.bootstrap),
        "p2p5": finite_quantile(valid_crossings, 2.5),
        "p16": finite_quantile(valid_crossings, 16.0),
        "p50": finite_quantile(valid_crossings, 50.0),
        "p84": finite_quantile(valid_crossings, 84.0),
        "p97p5": finite_quantile(valid_crossings, 97.5),
    }

    # --------------------------------------------------------------
    # Write products
    # --------------------------------------------------------------

    write_mean_trace(
        output_dir / "mean_trace.csv",
        times,
        traces,
    )

    write_central_curve(
        output_dir / "central_spectral_dimension.csv",
        central,
    )

    write_bootstrap_curve(
        output_dir / "bootstrap_spectral_dimension.csv",
        central,
        ds_boot,
        r2_boot,
    )

    write_crossings(
        output_dir / "bootstrap_crossings.csv",
        crossing_boot,
    )

    # Save full numerical bootstrap arrays as compressed NPZ.
    np.savez_compressed(
        str(output_dir / "bootstrap_arrays.npz"),
        center_time=central["center_time"],
        ds_bootstrap=ds_boot,
        r2_bootstrap=r2_boot,
        crossing_time_bootstrap=crossing_boot,
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "graph": graph_name,
        "n_times": int(nt),
        "times": [int(x) for x in times],
        "n_probes": int(nprobe),
        "probe_indices": [int(x) for x in probe_indices],
        "window": int(args.window),
        "n_centered_windows": int(len(central["center_time"])),
        "center_times": [
            float(x) for x in central["center_time"]
        ],
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "r2_min": float(args.r2_min),
        "euclidean_value": 3.0,
        "tolerance": float(args.tolerance),
        "persistence": int(args.persistence),
        "central_persistent_crossing_time": (
            float(crossing_central)
            if np.isfinite(crossing_central)
            else None
        ),
        "central_crossing_start_index": int(
            crossing_start_index
        ),
        "bootstrap_crossing": crossing_summary,
    }

    with open(str(output_dir / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if not args.no_plot:
        make_plot(
            output_dir / "spectral_dimension_bootstrap.png",
            central,
            ds_boot,
            crossing_central,
            crossing_summary,
            args.tolerance,
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print("")
    print("=" * 72)
    print("CENTRAL STRICT-CENTERED SPECTRAL DIMENSION")
    print("=" * 72)

    for i in range(len(central["center_time"])):
        print(
            "t={:8.0f}  ds={:8.4f}  R2={:.5f}  "
            "bootstrap=[{:8.4f}, {:8.4f}, {:8.4f}] "
            "valid={:.3f}".format(
                central["center_time"][i],
                central["ds_raw"][i],
                central["r2"][i],
                finite_quantile(ds_boot[:, i], 16.0),
                finite_quantile(ds_boot[:, i], 50.0),
                finite_quantile(ds_boot[:, i], 84.0),
                np.mean(np.isfinite(ds_boot[:, i])),
            )
        )

    print("")
    print("=" * 72)
    print("PERSISTENT EUCLIDEAN CROSSING")
    print("=" * 72)

    if np.isfinite(crossing_central):
        print(
            "Central crossing time: {:.6f}".format(
                crossing_central
            )
        )
    else:
        print("Central curve: no qualifying persistent crossing.")

    print(
        "Bootstrap crossing success fraction: {:.4f} ({}/{})".format(
            crossing_success_fraction,
            len(valid_crossings),
            args.bootstrap,
        )
    )

    if len(valid_crossings):
        print(
            "Crossing t median: {:.6f}".format(
                crossing_summary["p50"]
            )
        )
        print(
            "68% interval: [{:.6f}, {:.6f}]".format(
                crossing_summary["p16"],
                crossing_summary["p84"],
            )
        )
        print(
            "95% interval: [{:.6f}, {:.6f}]".format(
                crossing_summary["p2p5"],
                crossing_summary["p97p5"],
            )
        )

    print("")
    print("Outputs written to:", output_dir)
    print("  mean_trace.csv")
    print("  central_spectral_dimension.csv")
    print("  bootstrap_spectral_dimension.csv")
    print("  bootstrap_crossings.csv")
    print("  bootstrap_arrays.npz")
    print("  summary.json")

    if not args.no_plot:
        print("  spectral_dimension_bootstrap.png")


if __name__ == "__main__":
    main()
