#!/usr/bin/env python3
"""
Generate Figure 3: return probability and heat-trace scaling.

The figure contains two panels:

Top:
    Mean return probability P_return(t) versus diffusion time t
    on logarithmic axes, with:
      - stochastic error bars;
      - a fitted Euclidean reference P_return ∝ t^(-3/2);
      - the finite-graph mixing floor 1/N.

Bottom:
    Compensated return probability:
        P_return(t) * t^(3/2)
    normalized to its median over a selected reference interval.

A homogeneous three-dimensional diffusion regime appears approximately flat
in the compensated panel.

Expected CSV columns
--------------------
Required:
    diffusion_time
    mean_return_probability

Uncertainty may be provided as either:
    mean_return_probability_se
or:
    relative_trace_se

Optional:
    stochastic_heat_trace

The script automatically detects common column-name variants.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TIME_CANDIDATES = [
    "diffusion_time",
    "time",
    "t",
]

RETURN_CANDIDATES = [
    "mean_return_probability",
    "return_probability",
    "p_return",
    "mean_return",
]

ABSOLUTE_ERROR_CANDIDATES = [
    "mean_return_probability_se",
    "return_probability_se",
    "p_return_se",
]

RELATIVE_ERROR_CANDIDATES = [
    "relative_trace_se",
    "relative_return_probability_se",
    "relative_se",
]


def find_column(frame, explicit, candidates, label, required=True):
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(
                "{} column '{}' not found. Available columns: {}".format(
                    label,
                    explicit,
                    list(frame.columns),
                )
            )
        return explicit

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            "Could not identify {} column. Available columns: {}".format(
                label,
                list(frame.columns),
            )
        )

    return None


def fit_fixed_slope_amplitude(t, p, slope=-1.5):
    """
    Fit ln p = ln A + slope ln t with fixed slope.
    """
    x = np.log(t)
    y = np.log(p)

    log_amplitude = float(np.mean(y - slope * x))
    amplitude = float(np.exp(log_amplitude))

    prediction = amplitude * t ** slope
    residual = np.log(p) - np.log(prediction)

    return amplitude, prediction, residual


def fit_free_power_law(t, p):
    """
    Fit ln p = intercept + slope ln t.
    """
    x = np.log(t)
    y = np.log(p)

    design = np.column_stack([np.ones_like(x), x])
    coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

    intercept, slope = coeff
    prediction = np.exp(intercept + slope * x)

    residual = y - np.log(prediction)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    spectral_dimension = -2.0 * slope

    return {
        "amplitude": float(np.exp(intercept)),
        "slope": float(slope),
        "spectral_dimension": float(spectral_dimension),
        "r2_log_space": float(r2),
        "prediction": prediction,
    }


def main(args):
    frame = pd.read_csv(args.input_csv)

    time_col = find_column(
        frame,
        args.time_column,
        TIME_CANDIDATES,
        "diffusion time",
    )
    return_col = find_column(
        frame,
        args.return_column,
        RETURN_CANDIDATES,
        "mean return probability",
    )
    absolute_error_col = find_column(
        frame,
        args.error_column,
        ABSOLUTE_ERROR_CANDIDATES,
        "absolute uncertainty",
        required=False,
    )
    relative_error_col = find_column(
        frame,
        args.relative_error_column,
        RELATIVE_ERROR_CANDIDATES,
        "relative uncertainty",
        required=False,
    )

    data = frame[
        np.isfinite(frame[time_col].to_numpy(float))
        & np.isfinite(frame[return_col].to_numpy(float))
        & (frame[time_col].to_numpy(float) > 0)
        & (frame[return_col].to_numpy(float) > 0)
    ].copy()

    data = (
        data.sort_values(time_col)
        .drop_duplicates(time_col, keep="last")
        .reset_index(drop=True)
    )

    if args.min_time is not None:
        data = data[data[time_col] >= args.min_time]

    if args.max_time is not None:
        data = data[data[time_col] <= args.max_time]

    if data.empty:
        raise RuntimeError("No valid rows remain after filtering.")

    t = data[time_col].to_numpy(float)
    p = data[return_col].to_numpy(float)

    if absolute_error_col is not None:
        p_error = data[absolute_error_col].to_numpy(float)
    elif relative_error_col is not None:
        p_error = (
            data[relative_error_col].to_numpy(float)
            * p
        )
    else:
        p_error = np.full_like(p, np.nan)

    reference_mask = np.ones_like(t, dtype=bool)

    if args.reference_min_time is not None:
        reference_mask &= t >= args.reference_min_time

    if args.reference_max_time is not None:
        reference_mask &= t <= args.reference_max_time

    if np.count_nonzero(reference_mask) < 2:
        raise ValueError(
            "The selected reference interval contains fewer than two points."
        )

    t_reference = t[reference_mask]
    p_reference = p[reference_mask]

    fixed_amplitude, _, _ = fit_fixed_slope_amplitude(
        t_reference,
        p_reference,
        slope=-1.5,
    )

    fixed_reference = fixed_amplitude * t ** (-1.5)

    free_fit = fit_free_power_law(
        t_reference,
        p_reference,
    )
    free_reference = (
        free_fit["amplitude"]
        * t ** free_fit["slope"]
    )

    compensated = p * t ** 1.5
    compensated_reference = np.median(
        compensated[reference_mask]
    )
    compensated_normalized = (
        compensated / compensated_reference
    )

    compensated_error = np.divide(
        p_error,
        p,
        out=np.full_like(p_error, np.nan),
        where=p > 0,
    ) * compensated_normalized

    floor = 1.0 / float(args.n_nodes)

    figure, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.1, 1.25],
            "hspace": 0.06,
        },
    )

    ax_top.errorbar(
        t,
        p,
        yerr=p_error,
        marker="o",
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        label="Stochastic heat-trace estimate",
    )

    ax_top.plot(
        t,
        fixed_reference,
        linestyle="--",
        linewidth=1.5,
        label=r"Euclidean reference $P_{\rm return}\propto t^{-3/2}$",
    )

    if args.show_free_fit:
        ax_top.plot(
            t,
            free_reference,
            linestyle="-.",
            linewidth=1.3,
            label=(
                r"Free fit: "
                + rf"$d_s={free_fit['spectral_dimension']:.3f}$"
            ),
        )

    ax_top.axhline(
        floor,
        linestyle=":",
        linewidth=1.4,
        label=rf"Finite-graph floor $1/N={floor:.3g}$",
    )

    if args.reference_min_time is not None or args.reference_max_time is not None:
        lower = (
            args.reference_min_time
            if args.reference_min_time is not None
            else t.min()
        )
        upper = (
            args.reference_max_time
            if args.reference_max_time is not None
            else t.max()
        )

        ax_top.axvspan(
            lower,
            upper,
            alpha=0.09,
            label="Reference fitting interval",
        )
        ax_bottom.axvspan(
            lower,
            upper,
            alpha=0.09,
        )

    if args.finite_box_start is not None:
        ax_top.axvspan(
            args.finite_box_start,
            t.max(),
            alpha=0.08,
            hatch="//",
            label="Finite-volume affected regime",
        )
        ax_bottom.axvspan(
            args.finite_box_start,
            t.max(),
            alpha=0.08,
            hatch="//",
        )

    ax_bottom.errorbar(
        t,
        compensated_normalized,
        yerr=compensated_error,
        marker="o",
        markersize=4.0,
        linewidth=1.3,
        capsize=2.5,
    )

    ax_bottom.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label=r"$t^{-3/2}$ scaling",
    )

    ax_top.set_xscale("log", base=2)
    ax_top.set_yscale("log")

    ax_bottom.set_xscale("log", base=2)

    ax_top.set_ylabel(
        r"Mean return probability "
        r"$P_{\rm return}(t)$"
    )
    ax_bottom.set_ylabel(
        r"$P_{\rm return}(t)t^{3/2}$"
        "\nnormalized"
    )
    ax_bottom.set_xlabel("Diffusion time $t$")

    ax_top.set_title(
        "Heat-trace scaling and the emergence of 3D Euclidean diffusion"
    )

    ax_top.grid(True, alpha=0.25)
    ax_bottom.grid(True, alpha=0.25)

    ax_top.tick_params(
        direction="in",
        top=True,
        right=True,
    )
    ax_bottom.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    ax_top.legend(
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    diagnostics = pd.DataFrame(
        [
            {
                "reference_min_time": float(t_reference.min()),
                "reference_max_time": float(t_reference.max()),
                "n_reference_points": int(len(t_reference)),
                "fixed_slope": -1.5,
                "fixed_slope_amplitude": fixed_amplitude,
                "free_fit_slope": free_fit["slope"],
                "free_fit_spectral_dimension": (
                    free_fit["spectral_dimension"]
                ),
                "free_fit_r2_log_space": (
                    free_fit["r2_log_space"]
                ),
                "finite_graph_floor": floor,
            }
        ]
    )

    diagnostics_path = output.with_name(
        output.stem + "_fit_diagnostics.csv"
    )
    diagnostics.to_csv(diagnostics_path, index=False)

    plotted = data.copy()
    plotted["return_probability_error"] = p_error
    plotted["euclidean_reference_t_minus_3_over_2"] = (
        fixed_reference
    )
    plotted["compensated_return_probability"] = compensated
    plotted["compensated_normalized"] = compensated_normalized

    plotted_path = output.with_name(
        output.stem + "_plotted_data.csv"
    )
    plotted.to_csv(plotted_path, index=False)

    print("Columns used:")
    print("  diffusion time:", time_col)
    print("  return probability:", return_col)
    print("  absolute uncertainty:", absolute_error_col)
    print("  relative uncertainty:", relative_error_col)

    print("\nReference-interval fit:")
    print(diagnostics.to_string(index=False))

    print("\nWrote:", output)
    print("Wrote:", diagnostics_path)
    print("Wrote:", plotted_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_csv",
        help=(
            "CSV containing diffusion_time and mean_return_probability."
        ),
    )

    parser.add_argument(
        "--n-nodes",
        type=int,
        default=32_000_000,
        help="Number of graph nodes used for the finite mixing floor 1/N.",
    )

    parser.add_argument("--time-column", default=None)
    parser.add_argument("--return-column", default=None)
    parser.add_argument("--error-column", default=None)
    parser.add_argument("--relative-error-column", default=None)

    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reference-min-time",
        type=float,
        default=512,
        help="Beginning of interval used to normalize the t^-3/2 reference.",
    )
    parser.add_argument(
        "--reference-max-time",
        type=float,
        default=4096,
        help="End of interval used to normalize the t^-3/2 reference.",
    )
    parser.add_argument(
        "--finite-box-start",
        type=float,
        default=8192,
        help=(
            "Optional diffusion time at which to shade the "
            "finite-volume affected regime."
        ),
    )
    parser.add_argument(
        "--show-free-fit",
        action="store_true",
        help="Also plot a free power-law fit over the reference interval.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--output",
        default="figure3_heat_trace_scaling.png",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
