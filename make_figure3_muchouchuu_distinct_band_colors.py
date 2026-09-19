#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated MuchoUchuu Figure 3 with distinct colors for each shaded region.

Shaded regions:
- 1% band around d_s=3: light gray
- probe-bootstrap 68% interval: light blue
- R_hom 68% interval: light orange

Lines:
- d_s=3 reference: blue dashed
- centered MuchoUchuu d_s: blue with circular markers
- R_hom vertical line: blue solid

Inputs:
1) spectral_dimension_sliding_bootstrap.csv
2) crossing_summary.json
3) late_time_plateau_summary.json (optional)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def parse_args():
    p = argparse.ArgumentParser(
        description="Create MuchoUchuu Figure 3 with distinct band colors."
    )
    p.add_argument(
        "centered_csv",
        help="Path to spectral_dimension_sliding_bootstrap.csv",
    )
    p.add_argument(
        "crossing_summary_json",
        help="Path to crossing_summary.json",
    )
    p.add_argument(
        "--plateau-summary-json",
        default=None,
        help=(
            "Optional path to late_time_plateau_summary.json. "
            "If omitted, the path stored in crossing_summary.json is used."
        ),
    )
    p.add_argument(
        "--output",
        default="figure3_muchouchuu_centered_distinct_bands",
        help="Output basename without extension.",
    )
    p.add_argument(
        "--a-step",
        type=float,
        default=10.4095576,
        help="MuchoUchuu calibration in h^-1 Mpc step^-1/2.",
    )
    p.add_argument(
        "--title",
        default="Euclidean transport in MuchoUchuu",
        help="Plot title.",
    )
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--y-min", type=float, default=2.82)
    p.add_argument("--y-max", type=float, default=3.11)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def t_to_R(t, a_step):
    if t is None:
        return None
    try:
        if np.isnan(t):
            return None
    except TypeError:
        pass
    return a_step * math.sqrt(float(t))


def fmt_pm(median, q16, q84, ndigits=1):
    """
    Return MathText content WITHOUT outer dollar delimiters.
    Example:
        446.3^{+41.7}_{-27.7}
    """
    if median is None or q16 is None or q84 is None:
        return r"\mathrm{unavailable}"

    plus = q84 - median
    minus = median - q16

    return (
        f"{median:.{ndigits}f}"
        + rf"^{{+{plus:.{ndigits}f}}}"
        + rf"_{{-{minus:.{ndigits}f}}}"
    )


def main():
    args = parse_args()

    centered_csv = Path(args.centered_csv)
    crossing_json = Path(args.crossing_summary_json)

    # ============================================================
    # Load strict-centered spectral-dimension results
    # ============================================================

    df = pd.read_csv(centered_csv)

    required = [
        "diffusion_time",
        "central_ds",
        "bootstrap_ds_q16",
        "bootstrap_ds_q84",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                "Missing required column {!r} in {}".format(col, centered_csv)
            )

    clean = df[np.isfinite(df["central_ds"].values)].copy()

    clean["R_hMpc"] = (
        args.a_step
        * np.sqrt(clean["diffusion_time"].astype(float).values)
    )

    # ============================================================
    # Load R_hom summary
    # ============================================================

    crossing = load_json(crossing_json)

    q = crossing.get("bootstrap_crossing_time_quantiles", {})

    t_hom_med = q.get("median")
    t_hom_q16 = q.get("q16")
    t_hom_q84 = q.get("q84")

    t_hom_first_central = crossing.get(
        "first_central_persistent_time"
    )

    R_hom_med = t_to_R(t_hom_med, args.a_step)
    R_hom_q16 = t_to_R(t_hom_q16, args.a_step)
    R_hom_q84 = t_to_R(t_hom_q84, args.a_step)
    R_hom_first_central = t_to_R(
        t_hom_first_central,
        args.a_step,
    )

    # ============================================================
    # Load late-time plateau
    # ============================================================

    plateau_path = args.plateau_summary_json

    if plateau_path is None:
        plateau_path = crossing.get(
            "late_time_plateau_summary"
        )

    ds_late_med = None
    ds_late_q16 = None
    ds_late_q84 = None

    if plateau_path is not None and Path(plateau_path).exists():
        plateau = load_json(plateau_path)

        pq = plateau.get(
            "bootstrap_plateau_quantiles",
            {},
        )

        ds_late_med = pq.get("median")
        ds_late_q16 = pq.get("q16")
        ds_late_q84 = pq.get("q84")

        if ds_late_med is None:
            ds_late_med = plateau.get(
                "central_plateau_mean"
            )

    # ============================================================
    # Colors
    # ============================================================

    curve_color = "tab:blue"

    euclidean_band_color = "0.75"   # light neutral gray
    bootstrap_band_color = "#8ec5f4"  # light blue
    rhom_band_color = "#f4b35e"       # light orange / amber

    # ============================================================
    # Figure
    # ============================================================

    fig, ax = plt.subplots(figsize=(12.0, 6.2))

    # ------------------------------------------------------------
    # 1% Euclidean band
    # ------------------------------------------------------------

    ax.axhspan(
        2.97,
        3.03,
        color=euclidean_band_color,
        alpha=0.28,
        zorder=0,
    )

    # ------------------------------------------------------------
    # R_hom 68% interval
    # ------------------------------------------------------------

    if (
        R_hom_q16 is not None
        and R_hom_q84 is not None
    ):
        ax.axvspan(
            R_hom_q16,
            R_hom_q84,
            color=rhom_band_color,
            alpha=0.35,
            zorder=0.5,
        )

    # ------------------------------------------------------------
    # d_s 68% bootstrap interval
    # ------------------------------------------------------------

    ax.fill_between(
        clean["R_hMpc"].values,
        clean["bootstrap_ds_q16"].values,
        clean["bootstrap_ds_q84"].values,
        color=bootstrap_band_color,
        alpha=0.42,
        zorder=1,
    )

    # ------------------------------------------------------------
    # Euclidean reference line
    # ------------------------------------------------------------

    ax.axhline(
        3.0,
        color=curve_color,
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )

    # ------------------------------------------------------------
    # Central strict-centered d_s curve
    # ------------------------------------------------------------

    ax.plot(
        clean["R_hMpc"].values,
        clean["central_ds"].values,
        color=curve_color,
        marker="o",
        markersize=5.0,
        linewidth=2.0,
        zorder=3,
    )

    # ------------------------------------------------------------
    # Bootstrap-median R_hom line
    # ------------------------------------------------------------

    if R_hom_med is not None:
        ax.axvline(
            R_hom_med,
            color=curve_color,
            linewidth=2.0,
            zorder=4,
        )

    # ============================================================
    # Annotation box
    # ============================================================

    text_lines = ["MuchoUchuu:"]

    if (
        R_hom_med is not None
        and R_hom_q16 is not None
        and R_hom_q84 is not None
    ):
        text_lines.append(
            r"$R_{\rm hom}="
            + fmt_pm(
                R_hom_med,
                R_hom_q16,
                R_hom_q84,
                ndigits=1,
            )
            + r"\ h^{-1}\,\mathrm{Mpc}$"
        )

    if (
        ds_late_med is not None
        and ds_late_q16 is not None
        and ds_late_q84 is not None
    ):
        text_lines.append(
            r"$d_s^{\rm late}="
            + fmt_pm(
                ds_late_med,
                ds_late_q16,
                ds_late_q84,
                ndigits=3,
            )
            + r"$"
        )

    ax.text(
        0.985,
        0.955,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.65",
            alpha=0.94,
        ),
        zorder=10,
    )

    # ============================================================
    # Legend
    # ============================================================

    legend_handles = [
        Patch(
            facecolor=euclidean_band_color,
            alpha=0.28,
            edgecolor="none",
            label=r"1\% band around $d_s=3$",
        ),
        Line2D(
            [0], [0],
            color=curve_color,
            linestyle="--",
            linewidth=1.8,
            label=r"$d_s=3$",
        ),
        Line2D(
            [0], [0],
            color=curve_color,
            marker="o",
            markersize=5.0,
            linewidth=2.0,
            label=r"MuchoUchuu: centered 64-probe $d_s$",
        ),
        Patch(
            facecolor=bootstrap_band_color,
            alpha=0.42,
            edgecolor="none",
            label=r"MuchoUchuu: probe-bootstrap 68\% interval",
        ),
    ]

    if R_hom_med is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=curve_color,
                linewidth=2.0,
                label=r"MuchoUchuu: $R_{\rm hom}$",
            )
        )

    if (
        R_hom_q16 is not None
        and R_hom_q84 is not None
    ):
        legend_handles.append(
            Patch(
                facecolor=rhom_band_color,
                alpha=0.35,
                edgecolor="none",
                label=r"MuchoUchuu: $R_{\rm hom}$ 68\% interval",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=11.2,
    )

    # ============================================================
    # Axes and style
    # ============================================================

    ax.set_title(
        args.title,
        fontsize=17,
    )

    ax.set_xlabel(
        r"physical diffusion scale $R\ [h^{-1}\,\mathrm{Mpc}]$",
        fontsize=14,
    )

    ax.set_ylabel(
        r"effective spectral dimension $d_s$",
        fontsize=14,
    )

    x_max = args.x_max

    if x_max is None:
        x_max = max(
            1350.0,
            clean["R_hMpc"].max() * 1.05,
        )

    ax.set_xlim(
        args.x_min,
        x_max,
    )

    ax.set_ylim(
        args.y_min,
        args.y_max,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        labelsize=12,
    )

    fig.tight_layout()

    # ============================================================
    # Save
    # ============================================================

    outbase = Path(args.output)

    pdf_path = str(outbase) + ".pdf"
    png_path = str(outbase) + ".png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    print("Wrote:")
    print(pdf_path)
    print(png_path)

    print("\nR_hom:")
    print("  first central persistent t =", t_hom_first_central)
    print("  first central R            =", R_hom_first_central)
    print("  bootstrap median t         =", t_hom_med)
    print("  bootstrap q16/q84 t        =", t_hom_q16, t_hom_q84)
    print("  bootstrap median R         =", R_hom_med)
    print("  bootstrap q16/q84 R        =", R_hom_q16, R_hom_q84)

    print("\nLate-time plateau:")
    print("  median =", ds_late_med)
    print("  q16/q84 =", ds_late_q16, ds_late_q84)


if __name__ == "__main__":
    main()
