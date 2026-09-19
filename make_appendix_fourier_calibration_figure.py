#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Appendix figure for the MuchoUchuu Fourier diffusion calibration.

Panel (a):
    Fourier-mode decay:
        -ln C_k(t) versus k^2 t
    together with the adopted linear fit:
        -ln C_k(t) = D (k^2 t)

Panel (b):
    Calibration sensitivity:
        A_RMS = sqrt(6 D)
    for the alternative fitting choices listed in
    fourier_calibration_sensitivity.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_adopted_calibration(cal_json, sensitivity_csv):
    """Load adopted D and A_RMS from JSON or a preferred sensitivity row."""
    if cal_json is not None and Path(cal_json).is_file():
        with open(cal_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        D = float(data["D_mpc_h2_per_step"])
        A = float(data["A_RMS_mpc_h_per_sqrt_step"])
        sigma_A = float(data.get("A_RMS_standard_error_regression", np.nan))
        r2 = float(data.get("fit_r2", np.nan))
        label = "adopted calibration (JSON)"
        return D, A, sigma_A, r2, label

    sensitivity = pd.read_csv(sensitivity_csv)

    if "test" in sensitivity.columns:
        preferred = sensitivity.loc[
            sensitivity["test"].astype(str) == "all_512_4096"
        ]
        if len(preferred) == 1:
            row = preferred.iloc[0]
            D = float(row["D_mpc_h2_per_step"])
            A = float(row["A_RMS_mpc_h_per_sqrt_step"])
            sigma_A = np.nan
            r2 = float(row["r2"])
            label = "all_512_4096"
            return D, A, sigma_A, r2, label

    D = float(np.median(sensitivity["D_mpc_h2_per_step"]))
    A = float(np.median(sensitivity["A_RMS_mpc_h_per_sqrt_step"]))
    sigma_A = np.nan
    r2 = (
        float(np.median(sensitivity["r2"]))
        if "r2" in sensitivity.columns
        else np.nan
    )
    label = "median of sensitivity table"
    return D, A, sigma_A, r2, label


def format_mode_label(nx, ny, nz):
    return rf"$({int(nx)},{int(ny)},{int(nz)})$"


def make_figure(
    mode_decay_csv,
    sensitivity_csv,
    calibration_json=None,
    output_png="appendix_fourier_calibration_muchouchuu.png",
    output_pdf="appendix_fourier_calibration_muchouchuu.pdf",
):
    decay = pd.read_csv(mode_decay_csv)
    sensitivity = pd.read_csv(sensitivity_csv)

    D, A, sigma_A, fit_r2, adopted_source = load_adopted_calibration(
        calibration_json,
        sensitivity_csv,
    )

    required_decay = {
        "diffusion_time",
        "nx",
        "ny",
        "nz",
        "k2_mpc_h2",
        "correlation",
        "minus_log_correlation",
    }
    missing = required_decay - set(decay.columns)
    if missing:
        raise ValueError(
            f"{mode_decay_csv} is missing required columns: {sorted(missing)}"
        )

    if "test" not in sensitivity.columns:
        raise ValueError(f"{sensitivity_csv} must contain a 'test' column.")

    decay = decay.loc[
        (decay["diffusion_time"] > 0)
        & np.isfinite(decay["minus_log_correlation"])
        & np.isfinite(decay["k2_mpc_h2"])
    ].copy()

    if decay.empty:
        raise ValueError("No valid positive-time Fourier-decay rows were found.")

    decay["k2t"] = decay["k2_mpc_h2"] * decay["diffusion_time"]
    sensitivity = sensitivity.copy()
    sensitivity["index_plot"] = np.arange(len(sensitivity))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    for (nx, ny, nz), group in decay.groupby(
        ["nx", "ny", "nz"],
        sort=True,
    ):
        group = group.sort_values("k2t")
        ax1.plot(
            group["k2t"],
            group["minus_log_correlation"],
            marker="o",
            linewidth=1.3,
            markersize=3.8,
            label=format_mode_label(nx, ny, nz),
        )

    xfit = np.linspace(0.0, decay["k2t"].max() * 1.05, 300)
    ax1.plot(
        xfit,
        D * xfit,
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Adopted fit: $-\ln C_{\mathbf{k}}=Dk^2t$"
            "\n"
            rf"$D={D:.5f}\,(h^{{-1}}\mathrm{{Mpc}})^2\,"
            r"\mathrm{step}^{-1}$"
        ),
    )

    ax1.set_xlabel(r"$k^2t$")
    ax1.set_ylabel(r"$-\ln C_{\mathbf{k}}(t)$")
    ax1.set_title("(a) Fourier-mode decay")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")

    calibration_text = (
        rf"$A_{{\rm RMS}}=\sqrt{{6D}}={A:.6f}\,"
        r"h^{-1}\mathrm{Mpc}\,\mathrm{step}^{-1/2}$"
    )
    if np.isfinite(sigma_A):
        calibration_text += "\n" + rf"$\sigma(A_{{\rm RMS}})={sigma_A:.6f}$"
    if np.isfinite(fit_r2):
        calibration_text += "\n" + rf"$R^2={fit_r2:.6f}$"

    ax1.text(
        0.98,
        0.02,
        calibration_text,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
        },
    )

    x = sensitivity["index_plot"].to_numpy()
    y = sensitivity["A_RMS_mpc_h_per_sqrt_step"].to_numpy(float)

    ax2.plot(x, y, marker="o", linewidth=1.5)
    ax2.axhline(
        A,
        linestyle="--",
        linewidth=2.0,
        label=rf"Adopted $A_{{\rm RMS}}={A:.6f}$",
    )

    if np.isfinite(sigma_A) and sigma_A > 0:
        ax2.axhspan(A - sigma_A, A + sigma_A, alpha=0.15)

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        sensitivity["test"].astype(str),
        rotation=30,
        ha="right",
    )
    ax2.set_ylabel(
        r"$A_{\rm RMS}\,[h^{-1}\mathrm{Mpc}\,"
        r"\mathrm{step}^{-1/2}]$"
    )
    ax2.set_title("(b) Calibration sensitivity")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    if "r2" in sensitivity.columns:
        for xi, yi, r2_value in zip(x, y, sensitivity["r2"]):
            ax2.annotate(
                rf"$R^2={float(r2_value):.6f}$",
                (xi, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    spread = float(y.max() - y.min())
    summary = (
        f"Adopted source: {adopted_source}\n"
        f"minimum = {y.min():.6f}\n"
        f"maximum = {y.max():.6f}\n"
        f"full spread = {spread:.6f}"
    )
    ax2.text(
        0.98,
        0.02,
        summary,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
        },
    )

    fig.suptitle(
        "Fourier calibration of the MuchoUchuu diffusion length",
        fontsize=16,
    )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {output_png}")
    print(f"Wrote: {output_pdf}")
    print(f"D = {D:.10f} (h^-1 Mpc)^2 per step")
    print(f"A_RMS = {A:.10f} h^-1 Mpc per sqrt(step)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the MuchoUchuu Fourier-calibration appendix figure."
    )
    parser.add_argument("mode_decay_csv")
    parser.add_argument("sensitivity_csv")
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument(
        "--output-png",
        default="appendix_fourier_calibration_muchouchuu.png",
    )
    parser.add_argument(
        "--output-pdf",
        default="appendix_fourier_calibration_muchouchuu.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(
        mode_decay_csv=args.mode_decay_csv,
        sensitivity_csv=args.sensitivity_csv,
        calibration_json=args.calibration_json,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
    )
