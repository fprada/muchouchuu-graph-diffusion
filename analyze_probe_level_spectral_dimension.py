#!/usr/bin/env python3
"""
Probe-level spectral-dimension analysis for BigUchuu diffusion traces.

This script reads the per-probe files produced by:

    extend_diffusion_return_trace.py --save-per-probe

Expected filenames:
    trace_probe_estimates_t0000512.csv
    trace_probe_estimates_t0000640.csv
    ...

Expected columns:
    graph
    diffusion_time
    probe_index
    individual_trace_estimate

For every probe, it fits

    ln K_a(t) = c_a - (d_{s,a}/2) ln t

over one or more requested time ranges.

Because the same probe is followed across all times, this preserves the
strong temporal correlation induced by shared Hutchinson vectors.

Outputs:
    probe_level_spectral_dimension.csv
    probe_level_spectral_dimension_summary.csv
    probe_level_spectral_dimension_distributions.png
    probe_level_spectral_dimension_vs_tmin.png
    probe_level_fit_quality.png
"""

from __future__ import print_function

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TIME_RE = re.compile(r"trace_probe_estimates_t(\d+)\.csv$")


def parse_time_from_filename(path):
    match = TIME_RE.search(Path(path).name)
    if not match:
        raise ValueError(
            "Cannot parse diffusion time from filename: {}".format(path)
        )
    return int(match.group(1))


def load_probe_tables(input_dir):
    pattern = str(
        Path(input_dir) / "trace_probe_estimates_t*.csv"
    )
    files = sorted(glob.glob(pattern))

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
                "{} is missing columns: {}".format(
                    path, sorted(missing)
                )
            )

        if "diffusion_time" not in frame.columns:
            frame["diffusion_time"] = parse_time_from_filename(path)

        frame["source_file"] = Path(path).name
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
        data["individual_trace_estimate"].to_numpy(dtype=float) <= 0
    ):
        bad = data[
            data["individual_trace_estimate"] <= 0
        ]
        raise ValueError(
            "Non-positive individual trace estimates found:\n{}".format(
                bad.head().to_string(index=False)
            )
        )

    return data


def fit_one_probe(frame):
    t = frame["diffusion_time"].to_numpy(dtype=float)
    trace = frame["individual_trace_estimate"].to_numpy(
        dtype=float
    )

    if len(t) < 3:
        return None

    x = np.log(t)
    y = np.log(trace)

    design = np.column_stack(
        [np.ones_like(x), x]
    )
    coefficients, residuals, rank, singular_values = (
        np.linalg.lstsq(design, y, rcond=None)
    )
    intercept, slope = coefficients

    prediction = intercept + slope * x
    residual_vector = y - prediction

    ss_res = np.sum(residual_vector ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0.0
        else np.nan
    )

    dof = len(x) - 2
    if dof > 0:
        residual_variance = ss_res / dof
        covariance = residual_variance * np.linalg.inv(
            design.T @ design
        )
        slope_se = np.sqrt(covariance[1, 1])
    else:
        slope_se = np.nan

    ds = -2.0 * slope
    ds_internal_se = 2.0 * slope_se

    return {
        "n_times": len(t),
        "time_min": float(t.min()),
        "time_max": float(t.max()),
        "intercept": float(intercept),
        "slope": float(slope),
        "spectral_dimension": float(ds),
        "internal_fit_se": float(ds_internal_se),
        "r2": float(r2),
        "rms_log_residual": float(
            np.sqrt(np.mean(residual_vector ** 2))
        ),
    }


def run_probe_fits(data, fit_ranges):
    results = []

    probes = sorted(
        data["probe_index"].astype(int).unique().tolist()
    )

    for tmin, tmax in fit_ranges:
        selected = data[
            (data["diffusion_time"] >= tmin)
            & (data["diffusion_time"] <= tmax)
        ].copy()

        available_times = sorted(
            selected["diffusion_time"].astype(int).unique().tolist()
        )

        if len(available_times) < 3:
            print(
                "Skipping range {}-{}: fewer than three times.".format(
                    tmin, tmax
                )
            )
            continue

        for probe in probes:
            part = selected[
                selected["probe_index"].astype(int) == probe
            ].sort_values("diffusion_time")

            if len(part) != len(available_times):
                continue

            fit = fit_one_probe(part)
            if fit is None:
                continue

            fit.update(
                {
                    "probe_index": int(probe),
                    "requested_time_min": int(tmin),
                    "requested_time_max": int(tmax),
                }
            )
            results.append(fit)

    if not results:
        raise RuntimeError(
            "No complete probe fits could be constructed."
        )

    return pd.DataFrame(results)


def summarize_probe_fits(fits):
    rows = []

    for (tmin, tmax), part in fits.groupby(
        ["requested_time_min", "requested_time_max"]
    ):
        ds = part["spectral_dimension"].to_numpy(dtype=float)
        r2 = part["r2"].to_numpy(dtype=float)

        rows.append(
            {
                "requested_time_min": int(tmin),
                "requested_time_max": int(tmax),
                "n_probes": int(len(part)),
                "mean_spectral_dimension": float(np.mean(ds)),
                "std_spectral_dimension": float(
                    np.std(ds, ddof=1)
                ),
                "standard_error_of_mean": float(
                    np.std(ds, ddof=1) / np.sqrt(len(ds))
                ),
                "median_spectral_dimension": float(
                    np.median(ds)
                ),
                "q16_spectral_dimension": float(
                    np.quantile(ds, 0.16)
                ),
                "q84_spectral_dimension": float(
                    np.quantile(ds, 0.84)
                ),
                "minimum_spectral_dimension": float(
                    np.min(ds)
                ),
                "maximum_spectral_dimension": float(
                    np.max(ds)
                ),
                "mean_r2": float(np.mean(r2)),
                "minimum_r2": float(np.min(r2)),
                "mean_minus_three": float(
                    np.mean(ds) - 3.0
                ),
                "consistent_with_three_1sigma": bool(
                    abs(np.mean(ds) - 3.0)
                    <= np.std(ds, ddof=1) / np.sqrt(len(ds))
                ),
                "within_one_percent_central_value": bool(
                    abs(np.mean(ds) - 3.0) <= 0.03
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["requested_time_min", "requested_time_max"]
    )


def plot_distributions(fits, output_path):
    groups = list(
        fits.groupby(
            ["requested_time_min", "requested_time_max"]
        )
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    positions = np.arange(1, len(groups) + 1)
    values = [
        part["spectral_dimension"].to_numpy(dtype=float)
        for _, part in groups
    ]

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        showmeans=True,
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
        label=r"1% bounds",
    )
    ax.axhline(
        2.97,
        linestyle=":",
        linewidth=1.0,
    )

    labels = [
        "{}–{}".format(int(tmin), int(tmax))
        for (tmin, tmax), _ in groups
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Diffusion-time fit range")
    ax.set_ylabel(r"Probe-level spectral dimension $d_{s,a}$")
    ax.set_title(
        "Distribution of probe-level spectral-dimension fits"
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_summary(summary, output_path):
    summary = summary.sort_values(
        "requested_time_min"
    ).copy()

    x = summary["requested_time_min"].to_numpy(dtype=float)
    y = summary[
        "mean_spectral_dimension"
    ].to_numpy(dtype=float)
    yerr = summary[
        "standard_error_of_mean"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.7))

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        capsize=4,
        label="Probe-level mean ± SEM",
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
        label=r"1% bounds",
    )
    ax.axhline(
        2.97,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Minimum diffusion time in fitted interval")
    ax.set_ylabel(r"Mean spectral dimension")
    ax.set_title(
        "Late-time convergence of the spectral dimension"
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_fit_quality(fits, output_path):
    groups = list(
        fits.groupby(
            ["requested_time_min", "requested_time_max"]
        )
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    positions = np.arange(1, len(groups) + 1)
    values = [
        part["r2"].to_numpy(dtype=float)
        for _, part in groups
    ]

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        showmeans=True,
    )
    ax.axhline(
        0.99,
        linestyle="--",
        linewidth=1.1,
        label=r"$R^2=0.99$",
    )

    labels = [
        "{}–{}".format(int(tmin), int(tmax))
        for (tmin, tmax), _ in groups
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Diffusion-time fit range")
    ax.set_ylabel(r"Per-probe log-log fit $R^2$")
    ax.set_ylim(0.95, 1.001)
    ax.set_title("Quality of probe-level power-law fits")
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_fit_ranges(values, available_min, available_max):
    if not values:
        return [
            (512, available_max),
            (1024, available_max),
            (2048, available_max),
        ]

    if len(values) % 2 != 0:
        raise ValueError(
            "--fit-ranges requires min/max pairs."
        )

    ranges = []
    for index in range(0, len(values), 2):
        ranges.append(
            (int(values[index]), int(values[index + 1]))
        )
    return ranges


def main(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_probe_tables(input_dir)

    available_min = int(data["diffusion_time"].min())
    available_max = int(data["diffusion_time"].max())

    fit_ranges = parse_fit_ranges(
        args.fit_ranges,
        available_min,
        available_max,
    )

    fits = run_probe_fits(data, fit_ranges)
    summary = summarize_probe_fits(fits)

    fits_path = (
        output_dir / "probe_level_spectral_dimension.csv"
    )
    summary_path = (
        output_dir
        / "probe_level_spectral_dimension_summary.csv"
    )

    fits.to_csv(fits_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_distributions(
        fits,
        output_dir
        / "probe_level_spectral_dimension_distributions.png",
    )
    plot_summary(
        summary,
        output_dir
        / "probe_level_spectral_dimension_vs_tmin.png",
    )
    plot_fit_quality(
        fits,
        output_dir / "probe_level_fit_quality.png",
    )

    print("\nProbe-level summary:")
    print(summary.to_string(index=False))

    print("\nWrote:")
    for path in sorted(output_dir.iterdir()):
        print(" ", path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing "
            "trace_probe_estimates_t*.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="probe_level_spectral_dimension_results",
    )
    parser.add_argument(
        "--fit-ranges",
        nargs="*",
        type=int,
        help=(
            "Optional min/max pairs, e.g. "
            "--fit-ranges 512 4096 1024 4096 2048 4096"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
