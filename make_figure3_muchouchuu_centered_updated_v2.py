#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make an updated Figure 3 for MuchoUchuu only, using the strict centered Step-6 results.

Features:
- centered d_s(R) curve
- 68% bootstrap interval on d_s
- 1% band around d_s = 3
- horizontal line at d_s = 3
- vertical line at R_hom (bootstrap-median crossing scale)
- shaded 68% interval for R_hom
- legend entries for all of the above
- annotation box with R_hom and late-time d_s summary

Expected inputs:
1) spectral_dimension_sliding_bootstrap.csv
2) crossing_summary.json
3) optionally late_time_plateau_summary.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser(
        description="Create updated MuchoUchuu-only Figure 3 from centered Step-6 outputs."
    )
    p.add_argument(
        "centered_csv",
        help="Path to spectral_dimension_sliding_bootstrap.csv"
    )
    p.add_argument(
        "crossing_summary_json",
        help="Path to crossing_summary.json"
    )
    p.add_argument(
        "--plateau-summary-json",
        default=None,
        help="Optional path to late_time_plateau_summary.json. "
             "If omitted, the script tries to read the path from crossing_summary.json."
    )
    p.add_argument(
        "--output",
        default="figure3_muchouchuu_centered_updated",
        help="Output basename (without extension). Default: figure3_muchouchuu_centered_updated"
    )
    p.add_argument(
        "--a-step",
        type=float,
        default=10.409557280900008,
        help="Calibration factor converting sqrt(t) to physical scale R = A_step * sqrt(t). "
             "Default is set for the MuchoUchuu calibration used previously."
    )
    p.add_argument(
        "--title",
        default="Euclidean transport in MuchoUchuu",
        help="Plot title"
    )
    p.add_argument(
        "--x-min",
        type=float,
        default=0.0,
        help="Minimum x-axis value"
    )
    p.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum x-axis value. Default: auto from data"
    )
    p.add_argument(
        "--y-min",
        type=float,
        default=2.82,
        help="Minimum y-axis value"
    )
    p.add_argument(
        "--y-max",
        type=float,
        default=3.11,
        help="Maximum y-axis value"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG dpi"
    )
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
    if median is None or q16 is None or q84 is None:
        return "unavailable"
    plus = q84 - median
    minus = median - q16
    return (
        f"{median:.{ndigits}f}"
        + r"$^{+"
        + f"{plus:.{ndigits}f}"
        + r"}_{-"
        + f"{minus:.{ndigits}f}"
        + r"}$"
    )


def fmt_pm3(median, q16, q84):
    if median is None or q16 is None or q84 is None:
        return "unavailable"
    plus = q84 - median
    minus = median - q16
    return (
        f"{median:.3f}"
        + r"$^{+"
        + f"{plus:.3f}"
        + r"}_{-"
        + f"{minus:.3f}"
        + r"}$"
    )


def main():
    args = parse_args()

    centered_csv = Path(args.centered_csv)
    crossing_json = Path(args.crossing_summary_json)

    df = pd.read_csv(centered_csv)

    required = [
        "diffusion_time",
        "central_ds",
        "bootstrap_ds_q16",
        "bootstrap_ds_q84",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError("Missing required column '{}' in {}".format(col, centered_csv))

    # keep only valid centered rows
    clean = df.copy()
    clean = clean[np.isfinite(clean["central_ds"].values)].copy()

    # physical scale
    clean["R_hMpc"] = args.a_step * np.sqrt(clean["diffusion_time"].astype(float).values)

    # optional q025/q975 if available
    has_q025 = "bootstrap_ds_q025" in clean.columns
    has_q975 = "bootstrap_ds_q975" in clean.columns

    crossing = load_json(crossing_json)

    # Correct keys from your JSON
    # We use the bootstrap median as the displayed R_hom central value
    q = crossing.get("bootstrap_crossing_time_quantiles", {})
    t_hom_med = q.get("median", None)
    t_hom_q16 = q.get("q16", None)
    t_hom_q84 = q.get("q84", None)
    t_hom_q025 = q.get("q025", None)
    t_hom_q975 = q.get("q975", None)

    # this is also available but not used as the main displayed R_hom
    t_hom_first_central = crossing.get("first_central_persistent_time", None)

    R_hom_med = t_to_R(t_hom_med, args.a_step)
    R_hom_q16 = t_to_R(t_hom_q16, args.a_step)
    R_hom_q84 = t_to_R(t_hom_q84, args.a_step)
    R_hom_q025 = t_to_R(t_hom_q025, args.a_step)
    R_hom_q975 = t_to_R(t_hom_q975, args.a_step)
    R_hom_first_central = t_to_R(t_hom_first_central, args.a_step)

    plateau_path = args.plateau_summary_json
    if plateau_path is None:
        plateau_path = crossing.get("late_time_plateau_summary", None)

    plateau = None
    if plateau_path is not None and Path(plateau_path).exists():
        plateau = load_json(plateau_path)

    ds_late_med = None
    ds_late_q16 = None
    ds_late_q84 = None

    if plateau is not None:
        bpq = plateau.get("bootstrap_plateau_quantiles", {})
        ds_late_med = bpq.get("median", None)
        ds_late_q16 = bpq.get("q16", None)
        ds_late_q84 = bpq.get("q84", None)

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1% band around ds=3
    onepct_lo = 2.97
    onepct_hi = 3.03
    band1 = ax.axhspan(
        onepct_lo,
        onepct_hi,
        color="tab:blue",
        alpha=0.12,
        zorder=0
    )

    # R_hom 68% interval
    rhom_band = None
    if R_hom_q16 is not None and R_hom_q84 is not None:
        rhom_band = ax.axvspan(
            R_hom_q16,
            R_hom_q84,
            color="tab:blue",
            alpha=0.08,
            zorder=0
        )

    # ds bootstrap 68% band
    ds_band = ax.fill_between(
        clean["R_hMpc"].values,
        clean["bootstrap_ds_q16"].values,
        clean["bootstrap_ds_q84"].values,
        color="tab:blue",
        alpha=0.18,
        zorder=1
    )

    # optional 95% band, very light if available
    if has_q025 and has_q975:
        ax.fill_between(
            clean["R_hMpc"].values,
            clean["bootstrap_ds_q025"].values,
            clean["bootstrap_ds_q975"].values,
            color="tab:blue",
            alpha=0.06,
            zorder=1
        )

    # ds=3 line
    ds3_line = ax.axhline(
        3.0,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        zorder=2
    )

    # MuchoUchuu curve
    curve_line, = ax.plot(
        clean["R_hMpc"].values,
        clean["central_ds"].values,
        color="tab:blue",
        marker="o",
        markersize=5,
        linewidth=2.0,
        zorder=3
    )

    # R_hom vertical line at bootstrap median
    rhom_line = None
    if R_hom_med is not None:
        rhom_line = ax.axvline(
            R_hom_med,
            color="tab:blue",
            linewidth=2.0,
            alpha=0.95,
            zorder=4
        )

    # Optional thin dotted line for first central persistent time
    # Uncomment if you want to show it explicitly
    # if R_hom_first_central is not None:
    #     ax.axvline(
    #         R_hom_first_central,
    #         color="tab:blue",
    #         linestyle=":",
    #         linewidth=1.5,
    #         alpha=0.9,
    #         zorder=4
    #     )

    # axis labels and title
    ax.set_title(args.title, fontsize=18)
    ax.set_xlabel(r"physical diffusion scale $R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=15)
    ax.set_ylabel(r"effective spectral dimension $d_s$", fontsize=15)

    ax.set_xlim(args.x_min, args.x_max if args.x_max is not None else clean["R_hMpc"].max() * 1.06)
    ax.set_ylim(args.y_min, args.y_max)

    ax.grid(True, alpha=0.25)

    # ---------- text box ----------
    rhom_text = r"MuchoUchuu: $R_{\rm hom}=$ unavailable"
    if R_hom_med is not None and R_hom_q16 is not None and R_hom_q84 is not None:
        rhom_text = (
            r"MuchoUchuu: $R_{\rm hom} = "
            + fmt_pm(R_hom_med, R_hom_q16, R_hom_q84, ndigits=1)
            + r"\ h^{-1}\,\mathrm{Mpc}$"
        )

    late_text = r"$d_s^{\rm late}=$ unavailable"
    if ds_late_med is not None and ds_late_q16 is not None and ds_late_q84 is not None:
        late_text = (
            r"$d_s^{\rm late} = "
            + fmt_pm3(ds_late_med, ds_late_q16, ds_late_q84)
            + r"$"
        )

    textbox = rhom_text + "\n" + late_text

    ax.text(
        0.98, 0.95,
        textbox,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.7",
            alpha=0.92
        )
    )

    # ---------- legend ----------
    handles = [
        Patch(facecolor="tab:blue", alpha=0.12, edgecolor="none",
              label=r"1\% band around $d_s = 3$"),
        Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.8,
               label=r"$d_s = 3$"),
        Line2D([0], [0], color="tab:blue", marker="o", linewidth=2.0, markersize=5,
               label=r"MuchoUchuu: centered $d_s$"),
        Patch(facecolor="tab:blue", alpha=0.18, edgecolor="none",
              label=r"MuchoUchuu: probe-bootstrap 68\% interval"),
    ]

    if rhom_line is not None:
        handles.append(
            Line2D([0], [0], color="tab:blue", linewidth=2.0,
                   label=r"MuchoUchuu: $R_{\rm hom}$")
        )
    if rhom_band is not None:
        handles.append(
            Patch(facecolor="tab:blue", alpha=0.08, edgecolor="none",
                  label=r"MuchoUchuu: $R_{\rm hom}$ 68\% interval")
        )

    ax.legend(
        handles=handles,
        loc="lower center",
        fontsize=12,
        frameon=False,
        ncol=2
    )

    fig.tight_layout()

    outbase = Path(args.output)
    png_path = str(outbase) + ".png"
    pdf_path = str(outbase) + ".pdf"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("Wrote:", png_path)
    print("Wrote:", pdf_path)

    print("\nSummary")
    print("-------")
    print("R_hom first central persistent time =", t_hom_first_central)
    print("R_hom first central physical scale  =", R_hom_first_central)
    print("R_hom bootstrap median time         =", t_hom_med)
    print("R_hom bootstrap q16 time            =", t_hom_q16)
    print("R_hom bootstrap q84 time            =", t_hom_q84)
    print("R_hom bootstrap median scale        =", R_hom_med)
    print("R_hom bootstrap q16 scale           =", R_hom_q16)
    print("R_hom bootstrap q84 scale           =", R_hom_q84)
    print("Late-time d_s median                =", ds_late_med)
    print("Late-time d_s q16                   =", ds_late_q16)
    print("Late-time d_s q84                   =", ds_late_q84)


if __name__ == "__main__":
    main()
