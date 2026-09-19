#!/usr/bin/env python3
"""
Estimate a diffusion-based homogeneity scale with probe-level uncertainties.

Inputs
------
1. A directory containing per-probe trace files:
     trace_probe_estimates_t*.csv
   with columns:
     diffusion_time
     probe_index
     individual_trace_estimate

2. physical_diffusion_length.csv
   with columns:
     diffusion_time
     physical_diffusion_length_mpc_h

Method
------
For each sliding time window [t_i, ..., t_{i+w-1}] and each probe a, fit

    ln K_a(t) = c_a - (d_{s,a}/2) ln t.

Assign that fitted dimension to the geometric-mean diffusion time

    t_eff = exp(mean(ln t)),

and interpolate the measured physical diffusion length ell(t_eff).

Across probes, compute:
  * mean and median d_s
  * standard deviation and SEM
  * 16th-84th percentile interval

Define the central-value diffusion homogeneity scale as the first ell where

    |mean(d_s) - 3| <= tolerance

and require persistence for a configurable number of consecutive windows.

Estimate an uncertainty on the crossing from the probe-level crossings:
for each probe, linearly interpolate the first persistent crossing into the
band [3-tolerance, 3+tolerance].

Outputs
-------
sliding_window_probe_fits.csv
sliding_window_spectral_dimension_summary.csv
probe_crossing_scales.csv
diffusion_homogeneity_scale_summary.csv
diffusion_homogeneity_scale.png   (if matplotlib is installed)

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
        glob.glob(
            str(Path(input_dir) / "trace_probe_estimates_t*.csv")
        )
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
        required = {
            "probe_index",
            "individual_trace_estimate",
        }
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

        frame = frame[
            [
                "diffusion_time",
                "probe_index",
                "individual_trace_estimate",
            ]
        ].copy()
        frames.append(frame)

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
        raise ValueError(
            "All individual trace estimates must be positive."
        )

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


def interpolate_length(lengths, time_value):
    t = lengths["diffusion_time"].to_numpy(float)
    ell = lengths[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(float)

    if time_value < t.min() or time_value > t.max():
        return np.nan

    # Interpolate log ell against log t because ell approximately follows sqrt(t).
    return float(
        np.exp(
            np.interp(
                np.log(time_value),
                np.log(t),
                np.log(ell),
            )
        )
    )


def fit_probe_window(frame):
    t = frame["diffusion_time"].to_numpy(float)
    trace = frame["individual_trace_estimate"].to_numpy(float)

    if len(t) < 3:
        return None

    x = np.log(t)
    y = np.log(trace)

    design = np.column_stack([np.ones_like(x), x])
    coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = coeff

    prediction = intercept + slope * x
    residual = y - prediction

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "spectral_dimension": float(-2.0 * slope),
        "r2": float(r2),
        "rms_log_residual": float(
            np.sqrt(np.mean(residual ** 2))
        ),
    }


def make_sliding_windows(times, window_size, step_size):
    times = sorted(set(int(x) for x in times))
    windows = []

    for start in range(0, len(times) - window_size + 1, step_size):
        window = times[start : start + window_size]
        windows.append(window)

    return windows


def run_sliding_fits(data, lengths, window_size, step_size):
    times = sorted(
        data["diffusion_time"].astype(int).unique().tolist()
    )
    probes = sorted(
        data["probe_index"].astype(int).unique().tolist()
    )

    windows = make_sliding_windows(times, window_size, step_size)
    rows = []

    for window_index, window_times in enumerate(windows):
        t_eff = float(
            np.exp(
                np.mean(np.log(np.asarray(window_times, dtype=float)))
            )
        )
        ell_eff = interpolate_length(lengths, t_eff)

        selected = data[
            data["diffusion_time"].astype(int).isin(window_times)
        ]

        for probe in probes:
            part = selected[
                selected["probe_index"].astype(int) == probe
            ].sort_values("diffusion_time")

            if len(part) != len(window_times):
                continue

            fit = fit_probe_window(part)
            if fit is None:
                continue

            rows.append(
                {
                    "window_index": window_index,
                    "window_time_min": int(min(window_times)),
                    "window_time_max": int(max(window_times)),
                    "window_n_times": int(len(window_times)),
                    "effective_diffusion_time": t_eff,
                    "physical_diffusion_length_mpc_h": ell_eff,
                    "probe_index": int(probe),
                    **fit,
                }
            )

    if not rows:
        raise RuntimeError("No sliding-window fits were produced.")

    return pd.DataFrame(rows)


def summarize_windows(fits):
    rows = []

    group_cols = [
        "window_index",
        "window_time_min",
        "window_time_max",
        "window_n_times",
        "effective_diffusion_time",
        "physical_diffusion_length_mpc_h",
    ]

    for keys, part in fits.groupby(group_cols):
        ds = part["spectral_dimension"].to_numpy(float)
        r2 = part["r2"].to_numpy(float)

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_probes": int(len(part)),
                "mean_spectral_dimension": float(np.mean(ds)),
                "median_spectral_dimension": float(np.median(ds)),
                "std_spectral_dimension": float(np.std(ds, ddof=1)),
                "sem_spectral_dimension": float(
                    np.std(ds, ddof=1) / np.sqrt(len(ds))
                ),
                "q16_spectral_dimension": float(np.quantile(ds, 0.16)),
                "q84_spectral_dimension": float(np.quantile(ds, 0.84)),
                "mean_r2": float(np.mean(r2)),
                "minimum_r2": float(np.min(r2)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("window_index")


def first_persistent_crossing(x, y, tolerance, persistence):
    """
    Return first x where |y-3| <= tolerance for persistence consecutive points.
    Uses the first point of the persistent run as the crossing location.
    """
    inside = np.abs(y - 3.0) <= tolerance

    for i in range(0, len(inside) - persistence + 1):
        if np.all(inside[i : i + persistence]):
            return float(x[i])

    return np.nan


def probe_crossings(fits, tolerance, persistence):
    rows = []

    for probe, part in fits.groupby("probe_index"):
        part = part.sort_values(
            "physical_diffusion_length_mpc_h"
        )

        x = part[
            "physical_diffusion_length_mpc_h"
        ].to_numpy(float)
        y = part["spectral_dimension"].to_numpy(float)

        crossing = first_persistent_crossing(
            x,
            y,
            tolerance,
            persistence,
        )

        rows.append(
            {
                "probe_index": int(probe),
                "crossing_scale_mpc_h": crossing,
                "has_persistent_crossing": bool(
                    np.isfinite(crossing)
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_homogeneity_scale(
    summary,
    crossings,
    tolerance,
    persistence,
):
    x = summary[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(float)
    y = summary[
        "mean_spectral_dimension"
    ].to_numpy(float)

    central_crossing = first_persistent_crossing(
        x,
        y,
        tolerance,
        persistence,
    )

    valid = crossings[
        np.isfinite(crossings["crossing_scale_mpc_h"])
    ]["crossing_scale_mpc_h"].to_numpy(float)

    if len(valid):
        result = {
            "central_crossing_scale_mpc_h": central_crossing,
            "n_probes_with_crossing": int(len(valid)),
            "n_total_probes": int(len(crossings)),
            "probe_crossing_mean_mpc_h": float(np.mean(valid)),
            "probe_crossing_median_mpc_h": float(np.median(valid)),
            "probe_crossing_std_mpc_h": float(np.std(valid, ddof=1))
            if len(valid) > 1
            else np.nan,
            "probe_crossing_sem_mpc_h": float(
                np.std(valid, ddof=1) / np.sqrt(len(valid))
            )
            if len(valid) > 1
            else np.nan,
            "probe_crossing_q16_mpc_h": float(np.quantile(valid, 0.16)),
            "probe_crossing_q84_mpc_h": float(np.quantile(valid, 0.84)),
            "tolerance": tolerance,
            "persistence_windows": persistence,
        }
    else:
        result = {
            "central_crossing_scale_mpc_h": central_crossing,
            "n_probes_with_crossing": 0,
            "n_total_probes": int(len(crossings)),
            "probe_crossing_mean_mpc_h": np.nan,
            "probe_crossing_median_mpc_h": np.nan,
            "probe_crossing_std_mpc_h": np.nan,
            "probe_crossing_sem_mpc_h": np.nan,
            "probe_crossing_q16_mpc_h": np.nan,
            "probe_crossing_q84_mpc_h": np.nan,
            "tolerance": tolerance,
            "persistence_windows": persistence,
        }

    return pd.DataFrame([result])


def make_plot(summary, homogeneity_summary, output_path):
    if not HAVE_MATPLOTLIB:
        print(
            "matplotlib not installed: skipping {}".format(output_path),
            flush=True,
        )
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.9))

    x = summary[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(float)
    y = summary[
        "mean_spectral_dimension"
    ].to_numpy(float)
    yerr = summary[
        "sem_spectral_dimension"
    ].to_numpy(float)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="Sliding-window probe mean ± SEM",
    )

    ax.fill_between(
        x,
        summary["q16_spectral_dimension"].to_numpy(float),
        summary["q84_spectral_dimension"].to_numpy(float),
        alpha=0.15,
        label="Probe 16–84% interval",
    )

    ax.axhline(
        3.0,
        linestyle="--",
        linewidth=1.2,
        label=r"Euclidean expectation $d_s=3$",
    )
    ax.axhline(
        3.03,
        linestyle=":",
        linewidth=1.0,
        label="1% band",
    )
    ax.axhline(
        2.97,
        linestyle=":",
        linewidth=1.0,
    )

    crossing = homogeneity_summary.iloc[0][
        "central_crossing_scale_mpc_h"
    ]
    if np.isfinite(crossing):
        ax.axvline(
            crossing,
            linestyle="-.",
            linewidth=1.2,
            label=(
                r"$R_{\rm H}^{\rm diff}\approx"
                + "{:.0f}".format(crossing)
                + r"\,h^{-1}\mathrm{Mpc}$"
            ),
        )

    q16 = homogeneity_summary.iloc[0][
        "probe_crossing_q16_mpc_h"
    ]
    q84 = homogeneity_summary.iloc[0][
        "probe_crossing_q84_mpc_h"
    ]
    if np.isfinite(q16) and np.isfinite(q84):
        ax.axvspan(
            q16,
            q84,
            alpha=0.12,
            label="Probe crossing 16–84% interval",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Physical diffusion length $\ell\,[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(r"Effective spectral dimension $d_s$")
    ax.set_title(
        "Probe-level diffusion homogeneity scale"
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_probe_tables(input_dir)
    lengths = load_lengths(args.physical_length_csv)

    fits = run_sliding_fits(
        data,
        lengths,
        args.window_size,
        args.step_size,
    )
    summary = summarize_windows(fits)
    crossings = probe_crossings(
        fits,
        args.tolerance,
        args.persistence,
    )
    homogeneity = summarize_homogeneity_scale(
        summary,
        crossings,
        args.tolerance,
        args.persistence,
    )

    fits.to_csv(
        output_dir / "sliding_window_probe_fits.csv",
        index=False,
    )
    summary.to_csv(
        output_dir
        / "sliding_window_spectral_dimension_summary.csv",
        index=False,
    )
    crossings.to_csv(
        output_dir / "probe_crossing_scales.csv",
        index=False,
    )
    homogeneity.to_csv(
        output_dir
        / "diffusion_homogeneity_scale_summary.csv",
        index=False,
    )

    make_plot(
        summary,
        homogeneity,
        output_dir / "diffusion_homogeneity_scale.png",
    )

    metadata = {
        "window_size": args.window_size,
        "step_size": args.step_size,
        "tolerance": args.tolerance,
        "persistence": args.persistence,
        "matplotlib_available": HAVE_MATPLOTLIB,
    }
    (
        output_dir / "diffusion_homogeneity_scale_metadata.json"
    ).write_text(json.dumps(metadata, indent=2))

    print("\nHomogeneity-scale summary:")
    print(homogeneity.to_string(index=False))

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
        default="diffusion_homogeneity_scale_results",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Number of consecutive times in each spectral fit.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Sliding-window step in number of time samples.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Euclidean band: |d_s - 3| <= tolerance.",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=2,
        help=(
            "Number of consecutive sliding windows required inside "
            "the Euclidean band."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
