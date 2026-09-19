#!/usr/bin/env python3
"""
Bootstrap the diffusion-based homogeneity scale from probe-level heat traces.

This script resamples stochastic probes with replacement. For each bootstrap
realization it:

1. Computes the ensemble-mean trace as a function of diffusion time.
2. Fits the spectral dimension in sliding windows:
       ln K(t) = c - (d_s/2) ln t.
3. Maps each fitted window to the measured physical diffusion length ell(t).
4. Finds the first persistent crossing into:
       |d_s - 3| <= tolerance.
5. Stores the crossing scale.

Unlike individual-probe crossing statistics, this procedure uses all probes
in every bootstrap realization and therefore avoids conditioning on whether
a noisy single probe crosses the Euclidean band.

Matplotlib is optional.
"""

from __future__ import print_function

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    plt = None
    HAVE_MATPLOTLIB = False


TIME_RE = re.compile(r"trace_probe_estimates_t(\d+)\.csv$")


def load_probe_tables(input_dir):
    files = sorted(
        glob.glob(str(Path(input_dir) / "trace_probe_estimates_t*.csv"))
    )
    if not files:
        raise FileNotFoundError(
            "No trace_probe_estimates_t*.csv files found in {}".format(
                input_dir
            )
        )

    frames = []

    for path in files:
        frame = pd.read_csv(path)

        required = {"probe_index", "individual_trace_estimate"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                "{} is missing columns {}".format(path, sorted(missing))
            )

        if "diffusion_time" not in frame.columns:
            match = TIME_RE.search(Path(path).name)
            if not match:
                raise ValueError(
                    "Cannot infer diffusion time from {}".format(path)
                )
            frame["diffusion_time"] = int(match.group(1))

        frames.append(
            frame[
                [
                    "diffusion_time",
                    "probe_index",
                    "individual_trace_estimate",
                ]
            ].copy()
        )

    data = pd.concat(frames, ignore_index=True)

    data = (
        data.sort_values(["diffusion_time", "probe_index"])
        .drop_duplicates(
            ["diffusion_time", "probe_index"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if np.any(
        data["individual_trace_estimate"].to_numpy(float) <= 0
    ):
        raise ValueError("All individual trace estimates must be positive.")

    return data


def load_lengths(path):
    frame = pd.read_csv(path).sort_values("diffusion_time")

    required = {
        "diffusion_time",
        "physical_diffusion_length_mpc_h",
    }
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "{} is missing columns {}".format(path, sorted(missing))
        )

    return frame.drop_duplicates("diffusion_time", keep="last")


def make_trace_matrix(data):
    pivot = data.pivot(
        index="probe_index",
        columns="diffusion_time",
        values="individual_trace_estimate",
    ).sort_index(axis=0).sort_index(axis=1)

    if pivot.isna().any().any():
        missing = int(pivot.isna().sum().sum())
        raise ValueError(
            "Probe-by-time matrix has {} missing entries. "
            "Use only times available for every probe.".format(missing)
        )

    probe_ids = pivot.index.to_numpy(int)
    times = pivot.columns.to_numpy(int)
    matrix = pivot.to_numpy(float)

    return probe_ids, times, matrix


def interpolate_length(lengths, time_value):
    t = lengths["diffusion_time"].to_numpy(float)
    ell = lengths[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(float)

    if time_value < t.min() or time_value > t.max():
        return np.nan

    return float(
        np.exp(
            np.interp(
                np.log(time_value),
                np.log(t),
                np.log(ell),
            )
        )
    )


def make_windows(times, window_size, step_size, lengths):
    times = np.asarray(times, dtype=int)
    windows = []

    for start in range(0, len(times) - window_size + 1, step_size):
        indices = np.arange(start, start + window_size)
        window_times = times[indices].astype(float)

        effective_time = float(
            np.exp(np.mean(np.log(window_times)))
        )
        ell = interpolate_length(lengths, effective_time)

        if not np.isfinite(ell):
            continue

        x = np.log(window_times)
        x_centered = x - np.mean(x)
        denominator = float(np.dot(x_centered, x_centered))

        windows.append(
            {
                "window_index": len(windows),
                "indices": indices,
                "window_time_min": int(window_times.min()),
                "window_time_max": int(window_times.max()),
                "effective_diffusion_time": effective_time,
                "physical_diffusion_length_mpc_h": ell,
                "x_centered": x_centered,
                "denominator": denominator,
            }
        )

    if not windows:
        raise RuntimeError(
            "No sliding windows overlap the available physical-length range."
        )

    return windows


def fit_ds_from_trace(mean_trace, windows):
    rows = []

    for window in windows:
        y = np.log(mean_trace[window["indices"]])
        y_centered = y - np.mean(y)

        slope = float(
            np.dot(window["x_centered"], y_centered)
            / window["denominator"]
        )
        ds = -2.0 * slope

        prediction = np.mean(y) + slope * window["x_centered"]
        residual = y - prediction

        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum(y_centered ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append(
            {
                "window_index": window["window_index"],
                "window_time_min": window["window_time_min"],
                "window_time_max": window["window_time_max"],
                "effective_diffusion_time": (
                    window["effective_diffusion_time"]
                ),
                "physical_diffusion_length_mpc_h": (
                    window["physical_diffusion_length_mpc_h"]
                ),
                "spectral_dimension": ds,
                "r2": r2,
            }
        )

    return pd.DataFrame(rows)


def first_persistent_crossing(summary, tolerance, persistence):
    summary = summary.sort_values(
        "physical_diffusion_length_mpc_h"
    ).reset_index(drop=True)

    ell = summary[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(float)
    ds = summary["spectral_dimension"].to_numpy(float)

    inside = np.abs(ds - 3.0) <= tolerance

    for i in range(0, len(inside) - persistence + 1):
        if np.all(inside[i : i + persistence]):
            return float(ell[i]), int(summary.loc[i, "window_index"])

    return np.nan, -1


def run_bootstrap(
    trace_matrix,
    windows,
    n_bootstrap,
    tolerance,
    persistence,
    rng,
    progress_every,
):
    n_probes = trace_matrix.shape[0]
    rows = []

    for iteration in range(n_bootstrap):
        sample_indices = rng.integers(
            0,
            n_probes,
            size=n_probes,
        )

        mean_trace = np.mean(
            trace_matrix[sample_indices, :],
            axis=0,
        )

        fit_table = fit_ds_from_trace(mean_trace, windows)

        crossing, window_index = first_persistent_crossing(
            fit_table,
            tolerance,
            persistence,
        )

        rows.append(
            {
                "bootstrap_iteration": iteration,
                "crossing_scale_mpc_h": crossing,
                "crossing_window_index": window_index,
                "has_crossing": bool(np.isfinite(crossing)),
            }
        )

        if (
            (iteration + 1) % progress_every == 0
            or iteration + 1 == n_bootstrap
        ):
            valid = sum(row["has_crossing"] for row in rows)
            print(
                "bootstrap {}/{} valid_crossings={}".format(
                    iteration + 1,
                    n_bootstrap,
                    valid,
                ),
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize_bootstrap(
    central_fit,
    bootstrap,
    tolerance,
    persistence,
    n_bootstrap,
    seed,
):
    central_crossing, central_window = first_persistent_crossing(
        central_fit,
        tolerance,
        persistence,
    )

    valid = bootstrap[
        np.isfinite(bootstrap["crossing_scale_mpc_h"])
    ]["crossing_scale_mpc_h"].to_numpy(float)

    row = {
        "central_crossing_scale_mpc_h": central_crossing,
        "central_crossing_window_index": central_window,
        "n_bootstrap": int(n_bootstrap),
        "n_bootstrap_with_crossing": int(len(valid)),
        "bootstrap_crossing_fraction": float(
            len(valid) / n_bootstrap
        ),
        "tolerance": float(tolerance),
        "persistence_windows": int(persistence),
        "random_seed": int(seed),
    }

    if len(valid):
        row.update(
            {
                "bootstrap_mean_mpc_h": float(np.mean(valid)),
                "bootstrap_median_mpc_h": float(np.median(valid)),
                "bootstrap_std_mpc_h": float(
                    np.std(valid, ddof=1)
                )
                if len(valid) > 1
                else np.nan,
                "bootstrap_q025_mpc_h": float(
                    np.quantile(valid, 0.025)
                ),
                "bootstrap_q16_mpc_h": float(
                    np.quantile(valid, 0.16)
                ),
                "bootstrap_q84_mpc_h": float(
                    np.quantile(valid, 0.84)
                ),
                "bootstrap_q975_mpc_h": float(
                    np.quantile(valid, 0.975)
                ),
            }
        )
    else:
        row.update(
            {
                "bootstrap_mean_mpc_h": np.nan,
                "bootstrap_median_mpc_h": np.nan,
                "bootstrap_std_mpc_h": np.nan,
                "bootstrap_q025_mpc_h": np.nan,
                "bootstrap_q16_mpc_h": np.nan,
                "bootstrap_q84_mpc_h": np.nan,
                "bootstrap_q975_mpc_h": np.nan,
            }
        )

    return pd.DataFrame([row])


def plot_results(
    central_fit,
    bootstrap,
    summary,
    output_path,
):
    if not HAVE_MATPLOTLIB:
        print(
            "matplotlib not installed: skipping {}".format(output_path),
            flush=True,
        )
        return

    valid = bootstrap[
        np.isfinite(bootstrap["crossing_scale_mpc_h"])
    ]["crossing_scale_mpc_h"].to_numpy(float)

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(111)

    if len(valid):
        ax.hist(valid, bins=30, alpha=0.7)

    central = summary.iloc[0]["central_crossing_scale_mpc_h"]
    if np.isfinite(central):
        ax.axvline(
            central,
            linestyle="--",
            linewidth=1.5,
            label="Central crossing",
        )

    q16 = summary.iloc[0]["bootstrap_q16_mpc_h"]
    q84 = summary.iloc[0]["bootstrap_q84_mpc_h"]
    if np.isfinite(q16) and np.isfinite(q84):
        ax.axvspan(
            q16,
            q84,
            alpha=0.15,
            label="Bootstrap 16–84% interval",
        )

    ax.set_xlabel(
        r"Diffusion homogeneity scale "
        r"$R_{\rm H}^{\rm diff}\,[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel("Bootstrap realizations")
    ax.set_title("Ensemble-bootstrap homogeneity scale")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_probe_tables(args.input_dir)
    lengths = load_lengths(args.physical_length_csv)

    probe_ids, times, trace_matrix = make_trace_matrix(data)

    windows = make_windows(
        times,
        args.window_size,
        args.step_size,
        lengths,
    )

    central_mean_trace = np.mean(trace_matrix, axis=0)
    central_fit = fit_ds_from_trace(
        central_mean_trace,
        windows,
    )

    rng = np.random.default_rng(args.seed)

    bootstrap = run_bootstrap(
        trace_matrix=trace_matrix,
        windows=windows,
        n_bootstrap=args.n_bootstrap,
        tolerance=args.tolerance,
        persistence=args.persistence,
        rng=rng,
        progress_every=args.progress_every,
    )

    summary = summarize_bootstrap(
        central_fit=central_fit,
        bootstrap=bootstrap,
        tolerance=args.tolerance,
        persistence=args.persistence,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    central_fit.to_csv(
        output_dir / "central_sliding_window_fit.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "bootstrap_crossing_scales.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "bootstrap_homogeneity_scale_summary.csv",
        index=False,
    )

    metadata = {
        "n_probes": int(len(probe_ids)),
        "probe_ids": probe_ids.tolist(),
        "times": times.tolist(),
        "window_size": args.window_size,
        "step_size": args.step_size,
        "tolerance": args.tolerance,
        "persistence": args.persistence,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "matplotlib_available": HAVE_MATPLOTLIB,
    }

    (
        output_dir / "bootstrap_homogeneity_scale_metadata.json"
    ).write_text(json.dumps(metadata, indent=2))

    plot_results(
        central_fit,
        bootstrap,
        summary,
        output_dir / "bootstrap_homogeneity_scale.png",
    )

    print("\nBootstrap homogeneity-scale summary:")
    print(summary.to_string(index=False))

    print("\nWrote:")
    for path in sorted(output_dir.iterdir()):
        print(" ", path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing trace_probe_estimates_t*.csv files."
        ),
    )
    parser.add_argument(
        "physical_length_csv",
        help="physical_diffusion_length.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="bootstrap_diffusion_homogeneity",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
