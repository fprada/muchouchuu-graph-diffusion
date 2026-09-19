#!/usr/bin/env python3
"""
Measure the physical diffusion length of a periodic graph random walk.

For the row-stochastic random-walk operator

    P = D^{-1} A,

define the periodic Fourier correlation

    C(k,t) = (1/N) sum_i Re[
        exp(-i k.x_i) (P^t exp(i k.x))_i
    ].

For isotropic Euclidean diffusion,

    C(k,t) = exp[-k^2 <r^2>(t) / 6],

so the physical diffusion length is

    ell(t) = sqrt(<r^2>(t)).

The script propagates cosine/sine Fourier modes along x, y, and z with the
same MKL sparse-dense backend used by the diffusion analysis. It then fits

    -ln C(k,t) = (ell^2 / 6) k^2

through the origin across the requested low-wavenumber modes.

Outputs
-------
fourier_mode_correlations.csv
physical_diffusion_length.csv
physical_diffusion_length.png

Optional merge with an existing effective-spectral-dimension CSV:
spectral_dimension_vs_physical_scale.csv
spectral_dimension_vs_physical_scale.png

The coordinate binary must contain float32 x,y,z rows, and the graph must be
the same periodic graph used for the diffusion analysis.
"""

from __future__ import print_function

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

try:
    from sparse_dot_mkl import dot_product_mkl
    HAVE_MKL = True
except Exception:
    dot_product_mkl = None
    HAVE_MKL = False


AXES = ("x", "y", "z")


class Multiplier(object):
    def __init__(self, adjacency, backend):
        self.adjacency = adjacency
        self.backend = backend
        if backend == "mkl" and not HAVE_MKL:
            raise RuntimeError(
                "MKL requested but sparse_dot_mkl is unavailable."
            )

    def multiply(self, source, destination):
        if self.backend == "mkl":
            destination.fill(0.0)
            result = dot_product_mkl(
                self.adjacency,
                source,
                cast=False,
                out=destination,
                out_scalar=0.0,
            )
            if result is not destination:
                destination[:] = result
        else:
            destination[:] = self.adjacency @ source


def build_fourier_modes(
    xyz,
    box_size,
    harmonics,
    output_path,
    chunk_rows,
):
    n = xyz.shape[0]
    ncols = 2 * 3 * len(harmonics)

    modes = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(n, ncols),
    )

    two_pi_over_l = 2.0 * np.pi / float(box_size)

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        coords = np.asarray(xyz[start:stop], dtype=np.float64)

        col = 0
        for harmonic in harmonics:
            k = two_pi_over_l * float(harmonic)
            for axis_index in range(3):
                phase = k * coords[:, axis_index]
                modes[start:stop, col] = np.cos(phase).astype(np.float32)
                modes[start:stop, col + 1] = np.sin(phase).astype(np.float32)
                col += 2

        if stop == n or stop % (10 * chunk_rows) == 0:
            print(
                "Fourier mode rows {}/{}".format(stop, n),
                flush=True,
            )

    modes.flush()
    return modes


def evaluate_correlations(
    initial_modes,
    state,
    harmonics,
    box_size,
    chunk_rows,
):
    n = state.shape[0]
    accum = np.zeros((len(harmonics), 3), dtype=np.float64)

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        initial = np.asarray(
            initial_modes[start:stop],
            dtype=np.float64,
        )
        current = np.asarray(
            state[start:stop],
            dtype=np.float64,
        )

        col = 0
        for h_index, harmonic in enumerate(harmonics):
            for axis_index in range(3):
                accum[h_index, axis_index] += np.sum(
                    initial[:, col] * current[:, col]
                    + initial[:, col + 1] * current[:, col + 1],
                    dtype=np.float64,
                )
                col += 2

    accum /= float(n)

    rows = []
    two_pi_over_l = 2.0 * np.pi / float(box_size)

    for h_index, harmonic in enumerate(harmonics):
        k = two_pi_over_l * float(harmonic)
        axis_values = accum[h_index]
        mean_c = float(np.mean(axis_values))
        std_c = float(np.std(axis_values, ddof=1))
        sem_c = std_c / np.sqrt(3.0)

        for axis_index, axis_name in enumerate(AXES):
            rows.append(
                {
                    "harmonic": int(harmonic),
                    "axis": axis_name,
                    "wavenumber_h_mpc": k,
                    "k_squared": k * k,
                    "correlation": float(axis_values[axis_index]),
                    "axis_mean_correlation": mean_c,
                    "axis_std_correlation": std_c,
                    "axis_sem_correlation": sem_c,
                }
            )

    return pd.DataFrame(rows)


def fit_diffusion_length(correlation_frame, min_correlation):
    shell = (
        correlation_frame[
            [
                "harmonic",
                "wavenumber_h_mpc",
                "k_squared",
                "axis_mean_correlation",
                "axis_sem_correlation",
            ]
        ]
        .drop_duplicates("harmonic")
        .sort_values("harmonic")
        .copy()
    )

    shell = shell[
        np.isfinite(shell["axis_mean_correlation"])
        & (shell["axis_mean_correlation"] > min_correlation)
        & (shell["axis_mean_correlation"] < 1.0)
    ].copy()

    if len(shell) < 2:
        return None, shell

    x = shell["k_squared"].to_numpy(dtype=np.float64)
    c = shell["axis_mean_correlation"].to_numpy(dtype=np.float64)
    c_sem = shell["axis_sem_correlation"].to_numpy(dtype=np.float64)

    y = -np.log(c)

    sigma_y = np.divide(
        c_sem,
        c,
        out=np.full_like(c, np.nan),
        where=c > 0,
    )

    valid_weight = np.isfinite(sigma_y) & (sigma_y > 0)
    if np.count_nonzero(valid_weight) >= 2:
        weights = np.zeros_like(sigma_y)
        weights[valid_weight] = 1.0 / sigma_y[valid_weight] ** 2
    else:
        weights = np.ones_like(y)

    denominator = np.sum(weights * x * x)
    if denominator <= 0:
        return None, shell

    slope = np.sum(weights * x * y) / denominator
    prediction = slope * x

    residual = y - prediction
    weighted_sse = np.sum(weights * residual ** 2)
    weighted_y2 = np.sum(weights * y ** 2)
    relative_rmse = (
        np.sqrt(weighted_sse / np.sum(weights))
        / max(np.sqrt(weighted_y2 / np.sum(weights)), 1e-30)
    )

    ell2 = 6.0 * slope
    ell = np.sqrt(max(ell2, 0.0))

    # Axis-based scatter estimate:
    axis_ell_values = []
    for axis_name in AXES:
        axis_part = correlation_frame[
            correlation_frame["axis"] == axis_name
        ].sort_values("harmonic")

        axis_part = axis_part[
            axis_part["harmonic"].isin(shell["harmonic"])
        ]

        axis_c = axis_part["correlation"].to_numpy(dtype=np.float64)
        axis_x = axis_part["k_squared"].to_numpy(dtype=np.float64)

        valid = (
            np.isfinite(axis_c)
            & (axis_c > min_correlation)
            & (axis_c < 1.0)
        )
        if np.count_nonzero(valid) >= 2:
            axis_y = -np.log(axis_c[valid])
            axis_x_valid = axis_x[valid]
            axis_slope = (
                np.dot(axis_x_valid, axis_y)
                / np.dot(axis_x_valid, axis_x_valid)
            )
            axis_ell_values.append(
                np.sqrt(max(6.0 * axis_slope, 0.0))
            )

    if len(axis_ell_values) >= 2:
        ell_axis_std = float(np.std(axis_ell_values, ddof=1))
        ell_axis_sem = ell_axis_std / np.sqrt(len(axis_ell_values))
    else:
        ell_axis_std = np.nan
        ell_axis_sem = np.nan

    result = {
        "physical_diffusion_length_mpc_h": float(ell),
        "mean_squared_displacement_mpc_h2": float(ell2),
        "fit_slope_mpc_h2": float(slope),
        "n_fourier_shells_used": int(len(shell)),
        "minimum_correlation_used": float(c.min()),
        "maximum_correlation_used": float(c.max()),
        "relative_log_fit_rmse": float(relative_rmse),
        "axis_scatter_length_std_mpc_h": ell_axis_std,
        "axis_scatter_length_sem_mpc_h": ell_axis_sem,
    }

    return result, shell


def plot_lengths(length_frame, output_path):
    fig, ax = plt.subplots(figsize=(8.3, 5.7))

    t = length_frame["diffusion_time"].to_numpy(dtype=float)
    ell = length_frame[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(dtype=float)
    err = length_frame[
        "axis_scatter_length_sem_mpc_h"
    ].to_numpy(dtype=float)

    ax.errorbar(
        t,
        ell,
        yerr=err,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label=r"Measured $\ell(t)$",
    )

    # Euclidean expectation ell proportional to sqrt(t), anchored to first point.
    if len(t):
        reference = ell[0] * np.sqrt(t / t[0])
        ax.plot(
            t,
            reference,
            linestyle="--",
            linewidth=1.3,
            label=r"Euclidean expectation $\ell\propto\sqrt{t}$",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Diffusion time")
    ax.set_ylabel(
        r"Physical diffusion length $\ell(t)\,[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_title("Physical calibration of graph diffusion time")
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=220)
    plt.close(fig)


def merge_spectral_dimension(
    length_frame,
    spectral_csv,
    output_dir,
):
    spectral = pd.read_csv(str(spectral_csv))
    required = {
        "diffusion_time",
        "effective_spectral_dimension",
    }
    missing = required - set(spectral.columns)
    if missing:
        raise ValueError(
            "Spectral CSV is missing columns: {}".format(
                sorted(missing)
            )
        )

    merged = pd.merge(
        spectral,
        length_frame,
        on="diffusion_time",
        how="inner",
    ).sort_values("physical_diffusion_length_mpc_h")

    merged["within_one_percent_of_three"] = (
        np.abs(
            merged["effective_spectral_dimension"] - 3.0
        )
        <= 0.03
    )

    merged_path = (
        output_dir / "spectral_dimension_vs_physical_scale.csv"
    )
    merged.to_csv(str(merged_path), index=False)

    fig, ax = plt.subplots(figsize=(8.3, 5.7))

    x = merged[
        "physical_diffusion_length_mpc_h"
    ].to_numpy(dtype=float)
    y = merged[
        "effective_spectral_dimension"
    ].to_numpy(dtype=float)

    if "effective_spectral_dimension_se" in merged.columns:
        yerr = merged[
            "effective_spectral_dimension_se"
        ].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=r"$d_s(\ell)$",
        )
    else:
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label=r"$d_s(\ell)$",
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

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Physical diffusion length $\ell\,[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(r"Effective spectral dimension $d_s$")
    ax.set_title(
        "Diffusion-based analogue of the cosmological homogeneity test"
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False)

    fig.tight_layout()
    figure_path = (
        output_dir / "spectral_dimension_vs_physical_scale.png"
    )
    fig.savefig(str(figure_path), dpi=220)
    plt.close(fig)

    passing = merged[
        merged["within_one_percent_of_three"]
    ]
    if not passing.empty:
        first = passing.iloc[0]
        print(
            "First central-value 1% crossing: "
            "ell = {:.3f} h^-1 Mpc at t = {}".format(
                first["physical_diffusion_length_mpc_h"],
                int(first["diffusion_time"]),
            ),
            flush=True,
        )
    else:
        print(
            "No central-value 1% crossing is present in the merged times.",
            flush=True,
        )


def main(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = np.memmap(
        args.xyz_binary,
        mode="r",
        dtype=np.float32,
        shape=(args.n, 3),
    )

    graph = sparse.load_npz(args.graph).tocsr()
    graph = graph.astype(np.float32, copy=False)
    graph.sort_indices()

    if graph.shape != (args.n, args.n):
        raise ValueError(
            "Graph shape {} does not match N={}.".format(
                graph.shape, args.n
            )
        )

    degree = np.asarray(
        graph.sum(axis=1)
    ).ravel().astype(np.float32)
    degree = np.maximum(degree, np.float32(1e-20))

    backend = args.backend
    if backend == "auto":
        backend = "mkl" if HAVE_MKL else "scipy"

    modes_path = output_dir / "fourier_modes_t0.npy"
    initial_modes = build_fourier_modes(
        xyz,
        args.box_size,
        args.harmonics,
        modes_path,
        args.io_chunk_rows,
    )

    ncols = initial_modes.shape[1]
    current_path = output_dir / "fourier_work_current.npy"
    next_path = output_dir / "fourier_work_next.npy"

    current = np.lib.format.open_memmap(
        str(current_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, ncols),
    )
    next_state = np.lib.format.open_memmap(
        str(next_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, ncols),
    )

    for start in range(0, args.n, args.io_chunk_rows):
        stop = min(start + args.io_chunk_rows, args.n)
        current[start:stop] = initial_modes[start:stop]
    current.flush()

    multiplier = Multiplier(graph, backend)
    requested_times = sorted(set(args.times))
    max_time = max(requested_times)

    correlation_rows = []
    length_rows = []

    propagation_start = time.time()

    for step in range(1, max_time + 1):
        step_start = time.time()

        multiplier.multiply(current, next_state)
        next_state /= degree[:, None]
        current, next_state = next_state, current

        if step % args.progress_every == 0 or step in requested_times:
            mean_seconds = (
                time.time() - propagation_start
            ) / step
            print(
                "t={} step_seconds={:.3f} mean_seconds={:.3f} "
                "estimated_remaining_seconds={:.1f}".format(
                    step,
                    time.time() - step_start,
                    mean_seconds,
                    mean_seconds * (max_time - step),
                ),
                flush=True,
            )

        if step not in requested_times:
            continue

        frame = evaluate_correlations(
            initial_modes,
            current,
            args.harmonics,
            args.box_size,
            args.io_chunk_rows,
        )
        frame.insert(0, "diffusion_time", step)
        correlation_rows.append(frame)

        fit, used_shells = fit_diffusion_length(
            frame,
            args.min_correlation,
        )
        if fit is None:
            print(
                "Unable to fit diffusion length at t={}".format(step),
                flush=True,
            )
            continue

        fit["diffusion_time"] = step
        fit["backend"] = backend
        fit["harmonics_requested"] = ",".join(
            str(value) for value in args.harmonics
        )
        length_rows.append(fit)

        print(
            "length t={} ell={:.4f} h^-1 Mpc "
            "MSD={:.4f} shells={} relative_rmse={:.4e}".format(
                step,
                fit["physical_diffusion_length_mpc_h"],
                fit["mean_squared_displacement_mpc_h2"],
                fit["n_fourier_shells_used"],
                fit["relative_log_fit_rmse"],
            ),
            flush=True,
        )

    correlation_frame = pd.concat(
        correlation_rows,
        ignore_index=True,
    )
    length_frame = pd.DataFrame(length_rows).sort_values(
        "diffusion_time"
    )

    correlation_frame.to_csv(
        output_dir / "fourier_mode_correlations.csv",
        index=False,
    )
    length_frame.to_csv(
        output_dir / "physical_diffusion_length.csv",
        index=False,
    )

    plot_lengths(
        length_frame,
        output_dir / "physical_diffusion_length.png",
    )

    metadata = vars(args).copy()
    metadata["selected_backend"] = backend
    metadata["n_fourier_columns"] = int(ncols)
    (output_dir / "physical_diffusion_length_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    if args.spectral_dimension_csv:
        merge_spectral_dimension(
            length_frame,
            Path(args.spectral_dimension_csv),
            output_dir,
        )

    print("Finished:", output_dir, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("xyz_binary")
    parser.add_argument("n", type=int)
    parser.add_argument("graph")

    parser.add_argument(
        "--box-size",
        type=float,
        default=4000.0,
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--harmonics",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help=(
            "Low periodic Fourier harmonics. Start with 1 2 3 4."
        ),
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.05,
        help=(
            "Ignore Fourier shells whose mean correlation has decayed "
            "below this value."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mkl", "scipy"],
        default="auto",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--io-chunk-rows",
        type=int,
        default=500000,
    )
    parser.add_argument(
        "--spectral-dimension-csv",
        default=None,
        help=(
            "Optional effective_spectral_dimension.csv to merge with "
            "the measured physical diffusion lengths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="biguchuu_physical_diffusion_length",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
