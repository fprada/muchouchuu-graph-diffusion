from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# Inputs
# ============================================================

ROOT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace/step6_centered_w7_s64_b10000"
)

Z0_DIR = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)

CSV_PATH = Z0_DIR / "spectral_dimension_sliding_bootstrap.csv"
CROSSING_JSON = Z0_DIR / "crossing_summary.json"
PLATEAU_JSON = Z0_DIR / "late_time_plateau_summary.json"

OUTBASE = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "growth_test/z203/figure_z0_z203_ds_physical"
)

# Step-length calibration
A_RMS = 10.4095576  # z=0, h^-1 Mpc per sqrt(diffusion step)

A_RMS_Z203 = 10.598993479525339

Z203_DIR = (
    Path("/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion")
    / "growth_test/z203/step6_window7_s64_strictcentered_tmin3584_fixed"
)

Z203_CSV = Z203_DIR / "spectral_dimension_sliding_bootstrap.csv"
Z203_CROSSING_JSON = Z203_DIR / "crossing_summary.json"
Z203_PLATEAU_JSON = Z203_DIR / "late_time_plateau_summary.json"


# Plot limits
XMIN = 0.0
YMIN = 2.82
YMAX = 3.11

# ============================================================
# Helpers
# ============================================================

def r_of_t(t):
    return A_RMS * np.sqrt(np.asarray(t, dtype=float))

def fmt_pm(center, lo, hi, digits=1):
    """
    Format center^{+up}_{-down} with independent upper/lower errors.
    """
    up = hi - center
    down = center - lo
    return (
        rf"{center:.{digits}f}"
        + rf"^{{+{up:.{digits}f}}}"
        + rf"_{{-{down:.{digits}f}}}"
    )

def fmt_pm_symmetric(center, q16, q84, digits=3):
    """
    Format central value with asymmetric q16/q84 error bar.
    """
    up = q84 - center
    down = center - q16
    return (
        rf"{center:.{digits}f}"
        + rf"^{{+{up:.{digits}f}}}"
        + rf"_{{-{down:.{digits}f}}}"
    )

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

# Keep only valid strict-centered points
df = df[np.isfinite(df["central_ds"])].copy()

# Physical diffusion scale
df["R_hinvMpc"] = r_of_t(df["diffusion_time"].values)

# ============================================================
# z = 2.03 comparison sample
# ============================================================

df_z203 = pd.read_csv(Z203_CSV)

# Ensure numerical columns are actually numeric.
for col in [
    "diffusion_time",
    "central_ds",
    "bootstrap_ds_median",
    "bootstrap_ds_q16",
    "bootstrap_ds_q84",
]:
    df_z203[col] = pd.to_numeric(df_z203[col], errors="coerce")

# Retain only valid strict-centred estimates.
# t=12288,14336,16384 are therefore automatically excluded.
df_z203 = df_z203[np.isfinite(df_z203["central_ds"])].copy()

df_z203["R_hinvMpc"] = (
    A_RMS_Z203
    * np.sqrt(df_z203["diffusion_time"].to_numpy(float))
)

with open(Z203_CROSSING_JSON, "r") as f:
    crossing_z203 = json.load(f)

with open(Z203_PLATEAU_JSON, "r") as f:
    plateau_z203 = json.load(f)

cross_q_z203 = crossing_z203.get("bootstrap_crossing_quantiles", {})

t_cross_z203_med = cross_q_z203.get("median", None)
t_cross_z203_q16 = cross_q_z203.get("q16", None)
t_cross_z203_q84 = cross_q_z203.get("q84", None)

R_cross_z203_med = (
    A_RMS_Z203 * np.sqrt(t_cross_z203_med)
    if t_cross_z203_med is not None else None
)

R_cross_z203_q16 = (
    A_RMS_Z203 * np.sqrt(t_cross_z203_q16)
    if t_cross_z203_q16 is not None else None
)

R_cross_z203_q84 = (
    A_RMS_Z203 * np.sqrt(t_cross_z203_q84)
    if t_cross_z203_q84 is not None else None
)


with open(CROSSING_JSON, "r") as f:
    crossing = json.load(f)

with open(PLATEAU_JSON, "r") as f:
    plateau = json.load(f)

# ============================================================
# Homogeneity scale from crossing summary
# ============================================================

# Prefer interpolated bootstrap crossing quantiles if available
cross_q = crossing.get("bootstrap_crossing_quantiles", {})
t_cross_med = cross_q.get("median", None)
t_cross_q16 = cross_q.get("q16", None)
t_cross_q84 = cross_q.get("q84", None)
t_cross_q025 = cross_q.get("q025", None)
t_cross_q975 = cross_q.get("q975", None)

# Fallback to first central persistent time if needed
if t_cross_med is None:
    t_cross_med = crossing.get("first_central_persistent_time", None)
    t_cross_q16 = None
    t_cross_q84 = None
    t_cross_q025 = None
    t_cross_q975 = None

R_cross_med = r_of_t(t_cross_med) if t_cross_med is not None else None
R_cross_q16 = r_of_t(t_cross_q16) if t_cross_q16 is not None else None
R_cross_q84 = r_of_t(t_cross_q84) if t_cross_q84 is not None else None
R_cross_q025 = r_of_t(t_cross_q025) if t_cross_q025 is not None else None
R_cross_q975 = r_of_t(t_cross_q975) if t_cross_q975 is not None else None

# ============================================================
# Late-time plateau summary
# ============================================================

plateau_mean = plateau.get("central_plateau_mean", np.nan)
plateau_q = plateau.get("bootstrap_plateau_quantiles", {})
plateau_q16 = plateau_q.get("q16", np.nan)
plateau_q84 = plateau_q.get("q84", np.nan)

# ============================================================
# Build figure
# ============================================================

fig, ax = plt.subplots(figsize=(11.5, 5.2))

# 1% band around d_s=3
band_patch = ax.axhspan(
    2.97, 3.03,
    color="tab:blue",
    alpha=0.10,
    zorder=0
)

# d_s = 3 reference
ref_line = ax.axhline(
    3.0,
    color="tab:blue",
    linestyle="--",
    linewidth=1.8,
    zorder=1
)

# Homogeneity-scale uncertainty region (68%)
if R_cross_q16 is not None and R_cross_q84 is not None:
    ax.axvspan(
        R_cross_q16, R_cross_q84,
        color="tab:blue",
        alpha=0.08,
        zorder=0.5
    )

# Optional wider 95% region, very faint
if R_cross_q025 is not None and R_cross_q975 is not None:
    ax.axvspan(
        R_cross_q025, R_cross_q975,
        color="tab:blue",
        alpha=0.03,
        zorder=0.4
    )

# Vertical line at median homogeneity scale
if R_cross_med is not None:
    ax.axvline(
        R_cross_med,
        color="tab:blue",
        linewidth=2.0,
        zorder=2
    )

# 68% bootstrap band on d_s
fill_handle = ax.fill_between(
    df["R_hinvMpc"].values,
    df["bootstrap_ds_q16"].values,
    df["bootstrap_ds_q84"].values,
    color="tab:blue",
    alpha=0.18,
    zorder=2,
)

# Central curve
line_handle, = ax.plot(
    df["R_hinvMpc"].values,
    df["central_ds"].values,
    color="tab:blue",
    marker="o",
    markersize=5.0,
    linewidth=2.0,
    zorder=3,
)

# ============================================================
# Labels and style
# ============================================================

ax.set_title(r"Redshift evolution of Euclidean graph transport at fixed $\bar n$")
ax.set_xlabel(r"physical diffusion scale $R\ [h^{-1}\,\mathrm{Mpc}]$", fontsize=14)
ax.set_ylabel(r"effective spectral dimension $d_s$", fontsize=14)

ax.set_xlim(200, 1100)
ax.set_ylim(2.85, 3.10)

ax.grid(True, alpha=0.25)
ax.tick_params(direction="in", top=True, right=True, labelsize=12)

# ============================================================
# Annotation box
# ============================================================

textbox_lines = []

if (
    R_cross_med is not None and
    R_cross_q16 is not None and
    R_cross_q84 is not None
):
    rhom_text = (
        r"$R_{\rm hom} = "
        + fmt_pm(R_cross_med, R_cross_q16, R_cross_q84, digits=1)
        + r"\ h^{-1}\,\mathrm{Mpc}$"
    )
    textbox_lines.append(rhom_text)

if np.isfinite(plateau_mean) and np.isfinite(plateau_q16) and np.isfinite(plateau_q84):
    ds_text = (
        r"$d_s^{\rm late} = "
        + fmt_pm_symmetric(plateau_mean, plateau_q16, plateau_q84, digits=3)
        + r"$"
    )
    textbox_lines.append(ds_text)

if textbox_lines:
    ax.text(
        0.98, 0.95,
        "MuchoUchuu: " + ", ".join(textbox_lines),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="0.7")
    )


# ============================================================
# z = 2.03 trajectory
# ============================================================

ax.fill_between(
    df_z203["R_hinvMpc"],
    df_z203["bootstrap_ds_q16"],
    df_z203["bootstrap_ds_q84"],
    color="tab:orange",
    alpha=0.18,
    linewidth=0,
    zorder=2,
)

ax.plot(
    df_z203["R_hinvMpc"],
    df_z203["central_ds"],
    color="tab:orange",
    marker="s",
    markersize=4.5,
    linewidth=2.0,
    zorder=4,
)

# z=2.03 bootstrap homogeneity-scale interval
if R_cross_z203_q16 is not None and R_cross_z203_q84 is not None:
    ax.axvspan(
        R_cross_z203_q16,
        R_cross_z203_q84,
        color="tab:orange",
        alpha=0.08,
        linewidth=0,
        zorder=1,
    )

if R_cross_z203_med is not None:
    ax.axvline(
        R_cross_z203_med,
        color="tab:orange",
        linewidth=2.0,
        zorder=3,
    )

# Numerical comparison
comparison_text = (
    r"$z=0:\ R_{\rm hom}=446.1^{+40.8}_{-27.6}\ h^{-1}\,\mathrm{Mpc}$"
    "\n"
    r"$z=2.03:\ R_{\rm hom}=453.7^{+34.8}_{-25.3}\ h^{-1}\,\mathrm{Mpc}$"
)

ax.text(
    0.98,
    0.82,
    comparison_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11.0,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        alpha=0.92,
        edgecolor="0.7",
    ),
)

# ============================================================
# Legend
# ============================================================

legend_handles = [
    Patch(facecolor="tab:blue", alpha=0.10, edgecolor="none",
          label=r"1\% band around $d_s=3$"),
    Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.8,
           label=r"$d_s=3$"),
    Line2D([0], [0], color="tab:blue", marker="o", linewidth=2.0,
           markersize=5.0, label=r"MuchoUchuu: 64-probe $d_s$"),
    Patch(facecolor="tab:blue", alpha=0.18, edgecolor="tab:blue",
          label=r"MuchoUchuu: probe-bootstrap 68\% interval"),
]

if R_cross_med is not None:
    legend_handles.append(
        Line2D([0], [0], color="tab:blue", linewidth=2.0,
               label=r"MuchoUchuu: $R_{\rm hom}$")
    )

if R_cross_q16 is not None and R_cross_q84 is not None:
    legend_handles.append(
        Patch(facecolor="tab:blue", alpha=0.08, edgecolor="tab:blue",
              label=r"MuchoUchuu: $R_{\rm hom}$ 68\% interval")
    )


legend_handles.extend([
    Line2D(
        [0], [0],
        color="tab:orange",
        marker="s",
        linewidth=2.0,
        markersize=5.0,
        label=r"$z=2.03$: 64-probe $d_s$",
    ),
    Patch(
        facecolor="tab:orange",
        alpha=0.18,
        edgecolor="tab:orange",
        label=r"$z=2.03$: probe-bootstrap 68\% interval",
    ),
])

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.06),
    ncol=2,
    frameon=False,
    fontsize=12
)

fig.tight_layout()

# ============================================================
# Save
# ============================================================

fig.savefig(str(OUTBASE) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUTBASE) + ".png", dpi=300, bbox_inches="tight")

print("Wrote:")
print(str(OUTBASE) + ".pdf")
print(str(OUTBASE) + ".png")

print("\nFinal plotted table:")
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
