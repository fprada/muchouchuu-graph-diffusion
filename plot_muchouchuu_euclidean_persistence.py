#!/usr/bin/env python3
"""
Create the MuchoUchuu Euclidean-emergence persistence figure.

The script uses the current MuchoUchuu analysis values:

- Fourier/RMS calibration:
    ell_RMS(t) = A_RMS * sqrt(t)
    A_RMS = 10.409557615168367 h^-1 Mpc / sqrt(step)

- Fiducial interpolated transport-homogeneity scale:
    median = 435.8809448983336 h^-1 Mpc
    68% interval = [411.57603291561827, 489.6219758724846]

- Interpolated density-homogeneity scale:
    median = 114.57295655530987 h^-1 Mpc
    68% interval = [108.00050405972068, 120.56992675171307]

- Maximum diffusion time tested:
    t_max = 16384

The spectral-dimension values below are the sliding-window values used in the
current MuchoUchuu analysis. Replace the arrays if you later regenerate them
from a CSV.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A_RMS = 10.409557615168367

TRANSPORT_MEDIAN = 435.8809448983336
TRANSPORT_Q16 = 411.57603291561827
TRANSPORT_Q84 = 489.6219758724846

DENSITY_MEDIAN = 114.57295655530987
DENSITY_Q16 = 108.00050405972068
DENSITY_Q84 = 120.56992675171307

T_MAX = 16384

# Columns:
# diffusion_time, central_ds, bootstrap_q16, bootstrap_q84
SPECTRAL_DIMENSION_DATA = np.array([
    [512,   3.099185, 3.093871, 3.104223],
    [640,   3.099185, 3.093871, 3.104223],
    [768,   3.099185, 3.093871, 3.104223],
    [896,   3.082278, 3.074788, 3.089769],
    [1024,  3.067793, 3.056972, 3.078633],
    [1280,  3.055536, 3.041380, 3.069854],
    [1536,  3.043574, 3.026149, 3.061372],
    [1792,  3.022160, 2.999271, 3.045877],
    [2048,  3.003794, 2.976223, 3.032920],
    [2560,  2.989436, 2.958459, 3.021880],
    [3072,  2.977020, 2.942572, 3.013287],
    [3584,  2.959077, 2.918859, 3.001109],
    [4096,  2.947873, 2.904091, 2.994902],
    [5120,  2.941865, 2.895365, 2.992761],
    [6144,  2.938674, 2.888666, 2.993154],
    [7168,  2.937457, 2.881703, 2.998025],
    [8192,  2.939415, 2.877392, 3.005194],
    [10240, 2.942281, 2.876215, 3.013577],
    [12288, 2.945631, 2.874187, 3.022694],
    [14336, 2.945631, 2.874187, 3.022694],
    [16384, 2.945631, 2.874187, 3.022694],
], dtype=float)


def make_figure(output_png: Path, output_pdf: Path | None, dpi: int) -> None:
    t = SPECTRAL_DIMENSION_DATA[:, 0]
    central_ds = SPECTRAL_DIMENSION_DATA[:, 1]
    q16 = SPECTRAL_DIMENSION_DATA[:, 2]
    q84 = SPECTRAL_DIMENSION_DATA[:, 3]

    physical_scale = A_RMS * np.sqrt(t)
    max_tested = A_RMS * math.sqrt(T_MAX)
    extent_beyond_onset = max_tested - TRANSPORT_MEDIAN
    multiple_of_onset = max_tested / TRANSPORT_MEDIAN

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.2, 1.4],
        hspace=0.25,
    )

    # Top panel: spectral dimension versus physical RMS scale.
    ax1 = fig.add_subplot(grid[0])

    ax1.fill_between(
        physical_scale,
        q16,
        q84,
        alpha=0.22,
        label="Bootstrap 68% interval",
    )

    ax1.plot(
        physical_scale,
        central_ds,
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label=r"Central sliding-window $d_s$",
    )

    # A 1% relative band around d_s = 3 corresponds to an absolute half-width
    # of 0.03.
    ax1.axhspan(
        2.97,
        3.03,
        alpha=0.12,
        label=r"1% band around $d_s=3$",
    )
    ax1.axhline(
        3.0,
        linestyle="--",
        linewidth=1.5,
        label=r"$d_s=3$",
    )

    ax1.axvspan(
        TRANSPORT_Q16,
        TRANSPORT_Q84,
        alpha=0.14,
    )
    ax1.axvline(
        TRANSPORT_MEDIAN,
        linewidth=2.0,
    )

    ax1.text(
        TRANSPORT_MEDIAN + 14,
        3.115,
        "transport homogeneity\n"
        rf"median = {TRANSPORT_MEDIAN:.1f} $h^{{-1}}$ Mpc",
        va="top",
        ha="left",
        fontsize=10,
    )

    ax1.text(
        max_tested - 10,
        2.885,
        rf"$t_{{\max}}={T_MAX}$"
        "\n"
        rf"$\ell_{{\rm RMS}}={max_tested:.1f}$",
        va="bottom",
        ha="right",
        fontsize=10,
    )

    ax1.set_xlim(0, 1400)
    ax1.set_ylim(2.87, 3.13)
    ax1.set_ylabel(r"effective spectral dimension $d_s$")
    ax1.set_title(
        "MuchoUchuu: Euclidean emergence and how far it is tested "
        "with the current analysis"
    )
    ax1.legend(
        loc="lower left",
        fontsize=9,
        frameon=False,
    )
    ax1.grid(True, alpha=0.25)

    # Bottom panel: summary of the tested physical range.
    ax2 = fig.add_subplot(grid[1])

    ax2.set_xlim(0, 1400)
    ax2.set_ylim(0, 1)

    ax2.axvspan(
        TRANSPORT_MEDIAN,
        max_tested,
        alpha=0.18,
    )
    ax2.hlines(
        0.50,
        TRANSPORT_MEDIAN,
        max_tested,
        linewidth=8,
    )

    ax2.errorbar(
        DENSITY_MEDIAN,
        0.72,
        xerr=np.array([[
            DENSITY_MEDIAN - DENSITY_Q16
        ], [
            DENSITY_Q84 - DENSITY_MEDIAN
        ]]),
        fmt="o",
        capsize=4,
    )

    ax2.errorbar(
        TRANSPORT_MEDIAN,
        0.50,
        xerr=np.array([[
            TRANSPORT_MEDIAN - TRANSPORT_Q16
        ], [
            TRANSPORT_Q84 - TRANSPORT_MEDIAN
        ]]),
        fmt="o",
        capsize=4,
    )

    ax2.plot(
        max_tested,
        0.50,
        marker="s",
        markersize=7,
    )

    ax2.text(
        DENSITY_MEDIAN,
        0.80,
        "density homogeneity\n"
        f"{DENSITY_MEDIAN:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax2.text(
        TRANSPORT_MEDIAN,
        0.22,
        "Euclidean onset\n"
        f"{TRANSPORT_MEDIAN:.1f}",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax2.text(
        max_tested,
        0.22,
        "current maximum tested\n"
        f"{max_tested:.1f}",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax2.text(
        0.5 * (TRANSPORT_MEDIAN + max_tested),
        0.64,
        "current analysis demonstrates Euclidean transport\n"
        r"beyond onset out to $\ell_{\rm RMS}(t_{\max})$",
        ha="center",
        va="center",
        fontsize=11,
    )

    ax2.text(
        0.5 * (TRANSPORT_MEDIAN + max_tested),
        0.10,
        "extent beyond onset = "
        rf"{extent_beyond_onset:.1f} $h^{{-1}}$ Mpc"
        rf"  =  {multiple_of_onset:.2f}$\times$ onset scale",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax2.set_xlabel(r"physical scale [$h^{-1}$ Mpc]")
    ax2.set_yticks([])
    ax2.grid(True, axis="x", alpha=0.25)

    for side in ("left", "right", "top"):
        ax2.spines[side].set_visible(False)

    fig.savefig(
        output_png,
        dpi=dpi,
        bbox_inches="tight",
    )

    if output_pdf is not None:
        fig.savefig(
            output_pdf,
            bbox_inches="tight",
        )

    plt.close(fig)

    print(f"Wrote: {output_png.resolve()}")
    if output_pdf is not None:
        print(f"Wrote: {output_pdf.resolve()}")
    print(f"Maximum tested RMS scale: {max_tested:.3f} h^-1 Mpc")
    print(
        "Extent beyond transport-homogeneity median: "
        f"{extent_beyond_onset:.3f} h^-1 Mpc"
    )
    print(f"Maximum / onset ratio: {multiple_of_onset:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "muchouchuu_euclidean_persistence_current_analysis.png"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "muchouchuu_euclidean_persistence_current_analysis.pdf"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    args = parser.parse_args()

    make_figure(
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
