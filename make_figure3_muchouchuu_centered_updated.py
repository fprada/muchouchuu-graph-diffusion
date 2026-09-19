#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# Paths and constants
# ============================================================

ROOT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace/step6_centered_w7_s64_b10000"
)

CSV_PATH = ROOT / "spectral_dimension_sliding_bootstrap.csv"
CROSSING_JSON = ROOT / "crossing_summary.json"
PLATEAU_JSON = ROOT / "late_time_plateau_summary.json"

OUTBASE = ROOT / "figure3_muchouchuu_centered_updated"

# Euclidean RMS calibration
A_RMS = 10.4095576  # h^-1 Mpc step^-1/2

# Axis limits
XMIN = 0.0
YMIN = 2.82
YMAX = 3.11

# ============================================================
# Helpers
# ============================================================

def r_of_t(t):
    return A_RMS * np.sqrt(np.asarray(t, dtype=float))


def get_first(mapping, keys, default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


# ============================================================
# Load strict-centered spectral-dimension data
# ============================================================

df = pd.read_csv(CSV_PATH)

# Keep only valid strict-centered points.
df = df[np.isfinite(df["central_ds"])].copy()

# Convert diffusion time to physical diffusion scale.
df["R_hinvMpc"] = r_of_t(df["diffusion_time"].values)

# ============================================================
# Load crossing / homogeneity-scale summary
# ============================================================

with open(CROSSING_JSON, "r") as f:
    crossing = json.load(f)

cross_q = get_first(
    crossing,
    [
        "bootstrap_crossing_quantiles",
        "crossing_quantiles",
        "bootstrap_quantiles",
    ],
    {},
)

t_cross_med = get_first(cross_q, ["median", "q50"])
t_cross_q16 = get_first(cross_q, ["q16"])
t_cross_q84 = get_first(cross_q, ["q84"])
t_cross_q025 = get_first(cross_q, ["q025", "q2.5"])
t_cross_q975 = get_first(cross_q, ["q975", "q97.5"])

if t_cross_med is None:
    t_cross_med = get_first(
        crossing,
        [
            "first_central_persistent_time",
            "central_persistent_time",
            "crossing_time",
        ],
    )

R_cross_med = r_of_t(t_cross_med) if t_cross_med is not None else None
R_cross_q16 = r_of_t(t_cross_q16) if t_cross_q16 is not None else None
R_cross_q84 = r_of_t(t_cross_q84) if t_cross_q84 is not None else None
R_cross_q025 = r_of_t(t_cross_q025) if t_cross_q025 is not None else None
R_cross_q975 = r_of_t(t_cross_q975) if t_cross_q975 is not None else None

# ============================================================
# Load late-time plateau summary
# ============================================================

with open(PLATEAU_JSON, "r") as f:
    plateau = json.load(f)

plateau_mean = get_first(
    plateau,
    ["central_plateau_mean", "plateau_mean", "central_mean"],
    np.nan,
)

plateau_q = get_first(
    plateau,
    [
        "bootstrap_plateau_quantiles",
        "plateau_quantiles",
        "bootstrap_quantiles",
    ],
    {},
)

plateau_q16 = get_first(plateau_q, ["q16"], np.nan)
plateau_q84 = get_first(plateau_q, ["q84"], np.nan)

# ============================================================
# Build figure
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 5.2))

# ------------------------------------------------------------
# 1% Euclidean band around d_s = 3
# ------------------------------------------------------------

ax.axhspan(
    2.97,
    3.03,
    color="tab:blue",
    alpha=0.10,
    zorder=0,
)

# Euclidean reference line
ax.axhline(
    3.0,
    color="tab:blue",
    linestyle="--",
    linewidth=1.8,
    zorder=1,
)

# ------------------------------------------------------------
# Homogeneity-scale uncertainty and central value
# ------------------------------------------------------------

# 68% interval
if R_cross_q16 is not None and R_cross_q84 is not None:
    ax.axvspan(
        R_cross_q16,
        R_cross_q84,
        color="tab:blue",
        alpha=0.14,
        zorder=0.45,
    )

# Median homogeneity scale
if R_cross_med is not None:
    ax.axvline(
        R_cross_med,
        color="tab:blue",
        linewidth=2.0,
        zorder=2,
    )

# ------------------------------------------------------------
# Spectral-dimension bootstrap band and central curve
# ------------------------------------------------------------

ax.fill_between(
    df["R_hinvMpc"].values,
    df["bootstrap_ds_q16"].values,
    df["bootstrap_ds_q84"].values,
    color="tab:blue",
    alpha=0.18,
    zorder=2,
)

ax.plot(
    df["R_hinvMpc"].values,
    df["central_ds"].values,
    color="tab:blue",
    marker="o",
    markersize=5.0,
    linewidth=2.0,
    zorder=3,
)

# ============================================================
# Annotation box
# ============================================================

textbox_lines = []

if (
    R_cross_med is not None
    and R_cross_q16 is not None
    and R_cross_q84 is not None
):
    rhom_up = R_cross_q84 - R_cross_med
    rhom_down = R_cross_med - R_cross_q16

    textbox_lines.append(
        r"$R_{\rm hom}="
        + rf"{R_cross_med:.1f}"
        + rf"^{{+{rhom_up:.1f}}}"
        + rf"_{{-{rhom_down:.1f}}}"
        + r"\ h^{-1}\,\mathrm{Mpc}$"
    )

if (
    np.isfinite(plateau_mean)
    and np.isfinite(plateau_q16)
    and np.isfinite(plateau_q84)
):
    ds_up = plateau_q84 - plateau_mean
    ds_down = plateau_mean - plateau_q16

    textbox_lines.append(
        r"$d_s^{\rm late}="
        + rf"{plateau_mean:.3f}"
        + rf"^{{+{ds_up:.3f}}}"
        + rf"_{{-{ds_down:.3f}}}"
        + r"$"
    )

if textbox_lines:
    ax.text(
        0.985,
        0.955,
        "MuchoUchuu:  " + r",\quad ".join(textbox_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
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
        facecolor="tab:blue",
        alpha=0.10,
        edgecolor="none",
        label=r"1\% band around $d_s=3$",
    ),
    Line2D(
        [0], [0],
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label=r"$d_s=3$",
    ),
    Line2D(
        [0], [0],
        color="tab:blue",
        marker="o",
        linewidth=2.0,
        markersize=5.0,
        label=r"MuchoUchuu: 64-probe $d_s$",
    ),
    Patch(
        facecolor="tab:blue",
        alpha=0.18,
        edgecolor="none",
        label=r"MuchoUchuu: probe-bootstrap 68\% interval",
    ),
]

if R_cross_med is not None:
    legend_handles.append(
        Line2D(
            [0], [0],
            color="tab:blue",
            linewidth=2.0,
            label=r"MuchoUchuu: $R_{\rm hom}$",
        )
    )

if R_cross_q16 is not None and R_cross_q84 is not None:
    legend_handles.append(
        Patch(
            facecolor="tab:blue",
            alpha=0.14,
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
    fontsize=11.5,
)

# ============================================================
# Axes and styling
# ============================================================

ax.set_title("Euclidean transport in MuchoUchuu", fontsize=17)

ax.set_xlabel(
    r"physical diffusion scale $R\ [h^{-1}\,\mathrm{Mpc}]$",
    fontsize=14,
)

ax.set_ylabel(
    r"effective spectral dimension $d_s$",
    fontsize=14,
)

ax.set_xlim(
    XMIN,
    max(df["R_hinvMpc"].max() * 1.05, 1350.0),
)

ax.set_ylim(YMIN, YMAX)

ax.grid(True, alpha=0.25)

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

pdf_path = str(OUTBASE) + ".pdf"
png_path = str(OUTBASE) + ".png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print("Wrote:")
print(pdf_path)
print(png_path)

print("\nHomogeneity-scale summary:")
if R_cross_med is not None:
    print("R_hom median =", R_cross_med)
if R_cross_q16 is not None and R_cross_q84 is not None:
    print("R_hom q16/q84 =", R_cross_q16, R_cross_q84)

print("\nLate-time plateau:")
print("central =", plateau_mean)
print("q16/q84 =", plateau_q16, plateau_q84)

print("\nPlotted strict-centered points:")
print(
    df[
        [
            "diffusion_time",
            "R_hinvMpc",
            "central_ds",
            "bootstrap_ds_q16",
            "bootstrap_ds_q84",
        ]
    ].to_string(index=False)
)
