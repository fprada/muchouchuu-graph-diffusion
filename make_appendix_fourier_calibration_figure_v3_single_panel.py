#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the single-panel MuchoUchuu Fourier-calibration appendix figure.

The nearly degenerate Cartesian Fourier modes are grouped by n^2 and compared
with the normal-diffusion relation

    -ln C_k(t) = D k^2 t,

where D = A_RMS^2 / 6 and A_RMS is the final calibration adopted in the paper.

The sensitivity CSV is still read so that the script can report the tested
spread and R^2 range in the terminal; those robustness results are intended for
the caption or appendix text rather than a separate figure panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    sensitivity_a = sensitivity["A_RMS_mpc_h_per_sqrt_step"].to_numpy(float)
    spread = float(np.ptp(sensitivity_a))
    r2_min = float(np.nanmin(sensitivity["r2"].to_numpy(float)))
    r2_max = float(np.nanmax(sensitivity["r2"].to_numpy(float)))

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)

    markers = {1: "o", 4: "s"}
    for n2, group in decay.groupby("n2", sort=True):
        group = group.sort_values("k2t")
        ax.scatter(
            group["k2t"],
            group["minus_log_correlation"],
            s=24,
            marker=markers.get(int(n2), "o"),
            alpha=0.60,
            linewidths=0,
            label=rf"Fourier modes with $n^2={int(n2)}$",
            zorder=3,
        )

    xfit = np.linspace(0.0, float(decay["k2t"].max()) * 1.04, 300)
    ax.plot(
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

    ax.set_xlabel(r"$k^2t$")
    ax.set_ylabel(r"$-\ln C_{\boldsymbol{k}}(t)$")
    ax.set_title(
        "Fourier calibration of the MuchoUchuu diffusion length",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper left")

    ax.text(
        0.98,
        0.035,
        rf"$D=A_{{\rm RMS}}^2/6={fiducial_d:.6f}$ "
        r"$(h^{-1}\mathrm{Mpc})^2\,\mathrm{step}^{-1}$"
        "\n"
        rf"$A_{{\rm RMS}}={fiducial_a:.6f}\pm{fiducial_a_error:.6f}$ "
        r"$h^{-1}\mathrm{Mpc}\,\mathrm{step}^{-1/2}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.92,
        },
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
    print(f"Sensitivity spread in A_RMS = {spread:.12f}")
    print(f"Sensitivity R^2 range = [{r2_min:.9f}, {r2_max:.9f}]")
    print(f"Wrote PNG: {output_png.resolve()}")
    print(f"Wrote PDF: {output_pdf.resolve()}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the single-panel MuchoUchuu Fourier-calibration appendix figure."
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
        default=Path("appendix_fourier_calibration_muchouchuu_v3.png"),
    )
    parser.add_argument(
        "--output-pdf", type=Path,
        default=Path("appendix_fourier_calibration_muchouchuu_v3.pdf"),
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
