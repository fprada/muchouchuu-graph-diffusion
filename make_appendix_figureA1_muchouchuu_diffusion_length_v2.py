#!/usr/bin/env python3
"""
Appendix Figure A.1: MuchoUchuu calibration of diffusion time to physical length.

The script reads a MuchoUchuu calibration CSV containing diffusion time and a measured
physical length scale, and compares it with the fiducial relation

    ell(t) = A_RMS sqrt(t),

using the final MuchoUchuu calibration

    A_RMS = 10.409557615 h^-1 Mpc step^-1/2.

and produces a two-panel publication figure:

  (a) measured physical diffusion length and the best-fit sqrt(t) relation;
  (b) compensated ratio ell / (A sqrt(t)).

The script is designed for non-interactive HPC use and prints all detected
columns and output paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Final MuchoUchuu calibration used in the fiducial analysis.
MUCHOUCHUU_A_RMS = 10.409557615168367
MUCHOUCHUU_A_RMS_ERR = 0.0008030807269589261


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def find_column(frame: pd.DataFrame, candidates: list[str], role: str, required: bool = True):
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(
            f"Could not identify the {role} column. Tried {candidates}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def load_data(path: Path, requested_length_col: str | None = None):
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    time_col = find_column(
        frame,
        ["diffusion_time", "time", "t", "step", "diffusion_step"],
        "diffusion-time",
    )

    if requested_length_col is not None:
        if requested_length_col not in frame.columns:
            raise ValueError(
                f"Requested length column {requested_length_col!r} was not found. "
                f"Available columns: {list(frame.columns)}"
            )
        length_col = requested_length_col
    else:
        length_col = find_column(
            frame,
            [
                "physical_diffusion_length_mpc_h",
                "diffusion_length_mpc_h",
                "length_mpc_h",
                "ell_mpc_h",
                "ell",
                "physical_length_mpc_h",
                # MuchoUchuu Step-7 calibration products.  The f=0.500
                # definition is the central/default calibration branch.
                "ell_f0.500_mpc_h",
                "ell_f0.400_mpc_h",
                "ell_f0.600_mpc_h",
            ],
            "physical diffusion length",
        )

    error_col = find_column(
        frame,
        [
            "axis_scatter_length_sem_mpc_h",
            "physical_diffusion_length_se_mpc_h",
            "diffusion_length_se_mpc_h",
            "length_sem_mpc_h",
            "length_se_mpc_h",
            "ell_se_mpc_h",
            "ell_error_mpc_h",
        ],
        "length uncertainty",
        required=False,
    )

    q16_col = find_column(
        frame,
        ["physical_diffusion_length_q16_mpc_h", "length_q16_mpc_h", "ell_q16_mpc_h", "q16"],
        "lower quantile",
        required=False,
    )

    q84_col = find_column(
        frame,
        ["physical_diffusion_length_q84_mpc_h", "length_q84_mpc_h", "ell_q84_mpc_h", "q84"],
        "upper quantile",
        required=False,
    )

    data = frame[[time_col, length_col] + ([error_col] if error_col else []) +
                 ([q16_col] if q16_col else []) + ([q84_col] if q84_col else [])].copy()

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[time_col, length_col])
    data = data[(data[time_col] > 0) & (data[length_col] > 0)]
    data = data.sort_values(time_col).reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("Need at least two valid positive data points.")

    return data, time_col, length_col, error_col, q16_col, q84_col


def fit_prefactor(t: np.ndarray, ell: np.ndarray, sigma: np.ndarray | None) -> float:
    x = np.sqrt(t)

    if sigma is not None:
        valid = np.isfinite(sigma) & (sigma > 0)
        if np.count_nonzero(valid) >= 2:
            w = 1.0 / sigma[valid] ** 2
            return float(np.sum(w * x[valid] * ell[valid]) / np.sum(w * x[valid] ** 2))

    return float(np.sum(x * ell) / np.sum(x * x))


def main(args) -> None:
    paper_style()

    input_path = Path(args.input_csv)
    data, time_col, length_col, error_col, q16_col, q84_col = load_data(input_path, args.length_column)

    print(f"Read input: {input_path.resolve()}")
    print(f"Using time column: {time_col}")
    print(f"Using length column: {length_col}")
    print(f"Using uncertainty column: {error_col or 'none'}")
    if q16_col and q84_col:
        print(f"Using quantile columns: {q16_col}, {q84_col}")

    t = data[time_col].to_numpy(float)
    ell = data[length_col].to_numpy(float)
    sigma = data[error_col].to_numpy(float) if error_col else None

    fit_mask = np.ones_like(t, dtype=bool)
    if args.fit_min_time is not None:
        fit_mask &= t >= args.fit_min_time

    # By default, exclude the finite-box caution regime from the fit.
    # An explicit --fit-max-time overrides this automatic choice.
    if args.fit_max_time is not None:
        fit_mask &= t <= args.fit_max_time
    elif args.finite_box_start is not None:
        fit_mask &= t < args.finite_box_start

    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("The requested fit interval contains fewer than two points.")

    fit_sigma = sigma[fit_mask] if sigma is not None else None
    fitted_prefactor = fit_prefactor(t[fit_mask], ell[fit_mask], fit_sigma)
    prefactor = fitted_prefactor if args.refit_prefactor else MUCHOUCHUU_A_RMS

    model = prefactor * np.sqrt(t)
    ratio = ell / model
    ratio_err = sigma / model if sigma is not None else None

    q16_ratio = data[q16_col].to_numpy(float) / model if q16_col else None
    q84_ratio = data[q84_col].to_numpy(float) / model if q84_col else None

    residual = ell - model
    fractional_residual = ratio - 1.0
    rms_fractional = float(np.sqrt(np.mean(fractional_residual[fit_mask] ** 2)))
    max_abs_fractional = float(np.max(np.abs(fractional_residual[fit_mask])))

    fig = plt.figure(figsize=(10.4, 8.4))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[2.7, 1.2],
        left=0.11,
        right=0.96,
        bottom=0.13,
        top=0.86,
        hspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    if q16_col and q84_col:
        q16 = data[q16_col].to_numpy(float)
        q84 = data[q84_col].to_numpy(float)
        ax1.fill_between(t, q16, q84, alpha=0.18, label="16–84% interval")
        ax1.plot(t, ell, marker="o", ms=4.5, lw=1.5, label=r"Measured $\ell(t)$")
    elif sigma is not None:
        ax1.errorbar(
            t, ell, yerr=sigma,
            marker="o", ms=4.5, lw=1.5,
            elinewidth=0.8, capsize=2.5, alpha=0.95,
            label=r"Measured $\ell(t)$",
        )
    else:
        ax1.plot(t, ell, marker="o", ms=4.5, lw=1.5, label=r"Measured $\ell(t)$")

    ax1.plot(
        t, model, "--", lw=1.4,
        label=rf"Fiducial: $\ell(t)=A_{{\rm RMS}}\sqrt{{t}}$, $A_{{\rm RMS}}={prefactor:.4f}$",
    )

    fit_lo = float(t[fit_mask].min())
    fit_hi = float(t[fit_mask].max())
    ax1.axvspan(
        fit_lo, fit_hi,
        facecolor="0.90", edgecolor="none", alpha=0.65,
        zorder=0, label="Normal-diffusion fit interval",
    )

    if args.finite_box_start is not None and args.finite_box_start <= float(t.max()):
        ax1.axvspan(
            args.finite_box_start, float(t.max()),
            facecolor="none", edgecolor="0.20", hatch="//",
            linewidth=0.0, zorder=1,
            label="Increasing finite-box influence",
        )

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"Physical diffusion length $\ell(t)\,[h^{-1}\,\mathrm{Mpc}]$")
    ax1.set_title("(a) MuchoUchuu physical diffusion-length calibration", loc="left", fontweight="bold")
    ax1.grid(True, alpha=0.22)
    ax1.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        columnspacing=1.3,
        handlelength=2.1,
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    if q16_ratio is not None and q84_ratio is not None:
        ax2.fill_between(t, q16_ratio, q84_ratio, alpha=0.18)
        ax2.plot(t, ratio, marker="o", ms=4.2, lw=1.4)
    elif ratio_err is not None:
        ax2.errorbar(
            t, ratio, yerr=ratio_err,
            marker="o", ms=4.2, lw=1.4,
            elinewidth=0.8, capsize=2.3,
        )
    else:
        ax2.plot(t, ratio, marker="o", ms=4.2, lw=1.4)

    ax2.axhline(1.0, ls="--", lw=1.1, color="black")
    ax2.axvspan(
        fit_lo, fit_hi,
        facecolor="0.90", edgecolor="none", alpha=0.65,
        zorder=0,
    )
    if args.finite_box_start is not None and args.finite_box_start <= float(t.max()):
        ax2.axvspan(
            args.finite_box_start, float(t.max()),
            facecolor="none", edgecolor="0.20", hatch="//",
            linewidth=0.0, zorder=1,
        )

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Diffusion time $t$")
    ax2.set_ylabel(r"$\ell(t)/(A\sqrt{t})$")
    ax2.set_title(r"(b) Compensated $\sqrt{t}$ scaling", loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.22)

    fig.suptitle(
        "MuchoUchuu calibration of diffusion time to physical scale",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.91,
        r"The measured diffusion length follows $\ell(t)=A_{\rm RMS}\sqrt{t}$ over the calibrated interval.",
        ha="center",
        va="center",
        fontsize=10.8,
    )

    note = (
        rf"Fiducial MuchoUchuu calibration: "
        rf"$A_{{\rm RMS}}={MUCHOUCHUU_A_RMS:.6f}\pm{MUCHOUCHUU_A_RMS_ERR:.6f}\ "
        rf"h^{{-1}}\mathrm{{Mpc}}\,\mathrm{{step}}^{{-1/2}}$; "
        rf"fractional RMS over the displayed fit interval = {rms_fractional:.3%}."
    )
    fig.text(
        0.11, 0.045, note,
        ha="left", va="center", fontsize=9.8,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )

    output_pdf = Path(args.output)
    output_png = Path(args.png)
    diagnostics_path = Path(args.diagnostics_csv)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    diagnostics = pd.DataFrame([{
        "input_csv": str(input_path),
        "time_column": time_col,
        "length_column": length_col,
        "uncertainty_column": error_col or "",
        "fit_min_time": fit_lo,
        "fit_max_time": fit_hi,
        "fiducial_length_prefactor_mpc_h_step_minus_half": prefactor,
        "independently_fitted_prefactor_mpc_h_step_minus_half": fitted_prefactor,
        "fiducial_prefactor_uncertainty": MUCHOUCHUU_A_RMS_ERR,
        "fit_fractional_rms": rms_fractional,
        "fit_max_abs_fractional_residual": max_abs_fractional,
        "number_of_points": len(t),
        "number_of_fit_points": int(np.count_nonzero(fit_mask)),
    }])
    diagnostics.to_csv(diagnostics_path, index=False)

    derived = data.copy()
    derived["best_fit_length_mpc_h"] = model
    derived["compensated_length_ratio"] = ratio
    derived["fractional_residual"] = fractional_residual
    derived["absolute_residual_mpc_h"] = residual
    derived_path = diagnostics_path.with_name(diagnostics_path.stem + "_derived.csv")
    derived.to_csv(derived_path, index=False)

    print(f"Fiducial MuchoUchuu A_RMS = {prefactor:.12f} h^-1 Mpc step^-1/2")
    print(f"Independent fit to input data = {fitted_prefactor:.12f} h^-1 Mpc step^-1/2")
    print(f"Fit fractional RMS = {rms_fractional:.6f}")
    print(f"Wrote PDF: {output_pdf.resolve()}")
    print(f"Wrote PNG: {output_png.resolve()}")
    print(f"Wrote diagnostics: {diagnostics_path.resolve()}")
    print(f"Wrote derived table: {derived_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument(
        "--length-column",
        default=None,
        help=(
            "Physical-length column to plot. For the MuchoUchuu Step-7 table, "
            "use ell_f0.500_mpc_h (default when auto-detected), or explicitly "
            "select ell_f0.400_mpc_h or ell_f0.600_mpc_h for sensitivity checks."
        ),
    )
    parser.add_argument("--fit-min-time", type=float, default=None)
    parser.add_argument("--fit-max-time", type=float, default=None)
    parser.add_argument(
        "--finite-box-start",
        type=float,
        default=None,
        help="Optional diffusion time at and above which finite-box influence is highlighted and, unless --fit-max-time is supplied, excluded from the fit. By default no finite-box hatch is drawn and the full calibrated range is fitted.",
    )
    parser.add_argument("--output", default="appendix_figureA1_muchouchuu_diffusion_length.pdf")
    parser.add_argument("--png", default="appendix_figureA1_muchouchuu_diffusion_length.png")
    parser.add_argument("--diagnostics-csv", default="appendix_figureA1_muchouchuu_diffusion_length_diagnostics.csv")
    parser.add_argument(
        "--refit-prefactor",
        action="store_true",
        help="Use a new fit to the supplied CSV instead of the fiducial MuchoUchuu A_RMS value.",
    )
    main(parser.parse_args())
