#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the MuchoUchuu Fourier-calibration appendix figure.

Panel (a) groups the nearly degenerate Cartesian Fourier modes by n^2 and
shows the normal-diffusion relation

    -ln C_k(t) = D k^2 t.

Panel (b) shows the sensitivity of A_RMS = sqrt(6D) to the tested fitting
intervals and mode selections.  The reference line is the final fiducial
MuchoUchuu calibration used in the paper, not one individual sensitivity test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FIDUCIAL_A_RMS = 10.409557615168367
FIDUCIAL_A_RMS_ERROR = 0.0008030807269589261


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


def validate_inputs(decay: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    required_decay = {
        "diffusion_time", "nx", "ny", "nz", "n2", "k2_mpc_h2",
        "minus_log_correlation",
    }
    missing = required_decay - set(decay.columns)
    if missing:
        raise ValueError(f"Fourier-decay CSV is missing columns: {sorted(missing)}")

    required_sensitivity = {"test", "A_RMS_mpc_h_per_sqrt_step", "r2"}
    missing = required_sensitivity - set(sensitivity.columns)
    if missing:
        raise ValueError(f"Sensitivity CSV is missing columns: {sorted(missing)}")


def make_figure(
    mode_decay_csv: Path,
    sensitivity_csv: Path,
    output_png: Path,
    output_pdf: Path,
    fiducial_a: float,
    fiducial_a_error: float,
    dpi: int,
) -> None:
    paper_style()

    decay = pd.read_csv(mode_decay_csv)
    sensitivity = pd.read_csv(sensitivity_csv)
    validate_inputs(decay, sensitivity)

    decay = decay.loc[
        (decay["diffusion_time"] > 0)
        & np.isfinite(decay["minus_log_correlation"])
        & np.isfinite(decay["k2_mpc_h2"])
    ].copy()
    if decay.empty:
        raise ValueError("No valid positive-time Fourier-mode measurements were found.")

    decay["k2t"] = decay["k2_mpc_h2"] * decay["diffusion_time"]
    fiducial_d = fiducial_a**2 / 6.0

    y = sensitivity["A_RMS_mpc_h_per_sqrt_step"].to_numpy(float)
    x = np.arange(len(sensitivity), dtype=float)
    spread = float(np.ptp(y))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.6), constrained_layout=True
    )

    # Panel (a): combine Cartesian directions into the two n^2 families.
    markers = {1: "o", 4: "s"}
    for n2, group in decay.groupby("n2", sort=True):
        group = group.sort_values("k2t")
        ax1.scatter(
            group["k2t"],
            group["minus_log_correlation"],
            s=22,
            marker=markers.get(int(n2), "o"),
            alpha=0.58,
            linewidths=0,
            label=rf"Fourier modes with $n^2={int(n2)}$",
            zorder=3,
        )

    xfit = np.linspace(0.0, float(decay["k2t"].max()) * 1.04, 300)
    ax1.plot(
        xfit,
        fiducial_d * xfit,
        "--",
        linewidth=2.0,
        label=(
            r"Fiducial normal-diffusion fit: "
            r"$-\ln C_{\boldsymbol{k}}=Dk^2t$"
        ),
        zorder=4,
    )

    ax1.set_xlabel(r"$k^2t$")
    ax1.set_ylabel(r"$-\ln C_{\boldsymbol{k}}(t)$")
    ax1.set_title("(a) Fourier-mode decay", loc="left", fontweight="bold")
    ax1.grid(True, alpha=0.22)
    ax1.legend(frameon=False, loc="upper left")

    ax1.text(
        0.98, 0.035,
        rf"$D=A_{{\rm RMS}}^2/6={fiducial_d:.6f}$ "
        r"$(h^{-1}\mathrm{Mpc})^2\,\mathrm{step}^{-1}$"
        "\n"
        rf"$A_{{\rm RMS}}={fiducial_a:.6f}\pm{fiducial_a_error:.6f}$ "
        r"$h^{-1}\mathrm{Mpc}\,\mathrm{step}^{-1/2}$",
        transform=ax1.transAxes,
        ha="right", va="bottom", fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
              "edgecolor": "0.55", "alpha": 0.92},
    )

    # Panel (b): sensitivity tests relative to the final paper calibration.
    ax2.plot(x, y, marker="o", linewidth=1.5, markersize=6)
    ax2.axhspan(
        fiducial_a - fiducial_a_error,
        fiducial_a + fiducial_a_error,
        alpha=0.14,
        label=r"Fiducial $1\sigma$ interval",
    )
    ax2.axhline(
        fiducial_a,
        linestyle="--",
        linewidth=1.8,
        label=rf"Fiducial $A_{{\rm RMS}}={fiducial_a:.6f}$",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(sensitivity["test"].astype(str), rotation=28, ha="right")
    ax2.set_ylabel(
        r"$A_{\rm RMS}\,[h^{-1}\mathrm{Mpc}\,\mathrm{step}^{-1/2}]$"
    )
    ax2.set_title("(b) Calibration sensitivity", loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.22)
    ax2.legend(frameon=False, loc="upper right")

    # Alternate annotation offsets prevent labels from touching borders/markers.
    offsets = [(-8, 10), (0, 10), (0, -19), (8, 10)]
    for i, (xi, yi, r2_value) in enumerate(zip(x, y, sensitivity["r2"])):
        dx, dy = offsets[i % len(offsets)]
        ax2.annotate(
            rf"$R^2={float(r2_value):.6f}$",
            xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
            ha="center", va="bottom" if dy >= 0 else "top", fontsize=7.8,
        )

    ax2.text(
        0.97, 0.055,
        rf"Full tested spread: $\Delta A_{{\rm RMS}}={spread:.6f}$ "
        r"$h^{-1}\mathrm{Mpc}\,\mathrm{step}^{-1/2}$",
        transform=ax2.transAxes,
        ha="right", va="bottom", fontsize=9.0,
    )

    # Leave a little headroom for the upper annotation.
    ymin = min(float(y.min()), fiducial_a - fiducial_a_error)
    ymax = max(float(y.max()), fiducial_a + fiducial_a_error)
    yrange = ymax - ymin
    ax2.set_ylim(ymin - 0.12 * yrange, ymax + 0.16 * yrange)

    fig.suptitle(
        "Fourier calibration of the MuchoUchuu diffusion length",
        fontsize=15.5,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Read mode-decay data: {mode_decay_csv.resolve()}")
    print(f"Read sensitivity data: {sensitivity_csv.resolve()}")
    print(f"Fiducial A_RMS = {fiducial_a:.12f} +/- {fiducial_a_error:.12f}")
    print(f"Fiducial D = {fiducial_d:.12f}")
    print(f"Sensitivity spread = {spread:.12f}")
    print(f"Wrote PNG: {output_png.resolve()}")
    print(f"Wrote PDF: {output_pdf.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the MuchoUchuu Fourier-calibration appendix figure."
    )
    parser.add_argument("mode_decay_csv", type=Path)
    parser.add_argument("sensitivity_csv", type=Path)
    parser.add_argument(
        "--fiducial-a", type=float, default=FIDUCIAL_A_RMS,
        help="Final A_RMS value adopted in the paper.",
    )
    parser.add_argument(
        "--fiducial-a-error", type=float, default=FIDUCIAL_A_RMS_ERROR,
        help="One-sigma uncertainty of the final A_RMS value.",
    )
    parser.add_argument(
        "--output-png", type=Path,
        default=Path("appendix_fourier_calibration_muchouchuu_v2.png"),
    )
    parser.add_argument(
        "--output-pdf", type=Path,
        default=Path("appendix_fourier_calibration_muchouchuu_v2.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(
        mode_decay_csv=args.mode_decay_csv,
        sensitivity_csv=args.sensitivity_csv,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        fiducial_a=args.fiducial_a,
        fiducial_a_error=args.fiducial_a_error,
        dpi=args.dpi,
    )
