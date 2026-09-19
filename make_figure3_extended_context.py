from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

F = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace_t131072/combined_64probe/"
    "muchouchuu_extended_64probe_bootstrap_ds.csv"
)

OUT_PNG = Path("figure3_euclidean_transport_extended_context.png")
OUT_PDF = Path("figure3_euclidean_transport_extended_context.pdf")

df = pd.read_csv(F)

# Keep valid strict-centred estimates
df = df[
    np.isfinite(df["R_mpc_h"]) &
    np.isfinite(df["central_ds"]) &
    np.isfinite(df["ds_q16"]) &
    np.isfinite(df["ds_q84"])
].copy()

df = df.sort_values("R_mpc_h")

# ------------------------------------------------------------
# Fiducial quantities
# ------------------------------------------------------------

R_FID_MAX = 1332.4
R_CENTRAL_EXIT = 1884.33

R_HOM = 446.3
R_HOM_LO = 446.3 - 27.7
R_HOM_HI = 446.3 + 41.7

# Split into primary and extended regions
primary = df[df["R_mpc_h"] <= R_FID_MAX].copy()
extended = df[
    (df["R_mpc_h"] > R_FID_MAX) &
    (df["R_mpc_h"] <= 2000.0)
].copy()

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10.0, 4.8))

# Euclidean 1% band
ax.axhspan(
    2.97, 3.03,
    alpha=0.12,
    label=r"1% band around $d_s=3$"
)

ax.axhline(
    3.0,
    linestyle="--",
    linewidth=1.2,
    label=r"$d_s=3$"
)

# ------------------------------------------------------------
# Primary curve
# ------------------------------------------------------------

line, = ax.plot(
    primary["R_mpc_h"],
    primary["central_ds"],
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"MuchoUchuu: fiducial 64-probe $d_s$"
)

c = line.get_color()

ax.fill_between(
    primary["R_mpc_h"].to_numpy(),
    primary["ds_q16"].to_numpy(),
    primary["ds_q84"].to_numpy(),
    alpha=0.25,
    color=c,
    label="MuchoUchuu: fiducial 68% interval"
)

# ------------------------------------------------------------
# Extended points, excluded from fiducial inference
# ------------------------------------------------------------

ax.plot(
    extended["R_mpc_h"],
    extended["central_ds"],
    linestyle="--",
    marker="o",
    markerfacecolor="none",
    markersize=5,
    linewidth=1.5,
    color=c,
    label="Extended trajectory (not used in fiducial inference)"
)

ax.fill_between(
    extended["R_mpc_h"].to_numpy(),
    extended["ds_q16"].to_numpy(),
    extended["ds_q84"].to_numpy(),
    alpha=0.10,
    color=c
)

# ------------------------------------------------------------
# Homogeneity scale
# ------------------------------------------------------------

ax.axvspan(
    R_HOM_LO,
    R_HOM_HI,
    alpha=0.22,
    label=r"MuchoUchuu: $R_{\rm hom}$ 68% interval"
)

ax.axvline(
    R_HOM,
    linewidth=1.5,
    label=r"MuchoUchuu: $R_{\rm hom}$"
)

# ------------------------------------------------------------
# Fiducial analysis limit
# ------------------------------------------------------------

ax.axvline(
    R_FID_MAX,
    linestyle=":",
    linewidth=1.5,
    color="0.30",
    label=(
        r"fiducial analysis limit: "
        r"$R=1332\,h^{-1}{\rm Mpc}$"
    )
)

# ------------------------------------------------------------
# First central finite-volume departure
# ------------------------------------------------------------

ax.axvline(
    R_CENTRAL_EXIT,
    linestyle=":",
    linewidth=1.5,
    color="0.55"
)

ax.text(
    R_CENTRAL_EXIT + 18,
    2.86,
    (
        "first central exit\n"
        r"$R=1884\,h^{-1}{\rm Mpc}$"
    ),
    fontsize=8,
    rotation=90,
    va="bottom",
    color="0.35"
)

# ------------------------------------------------------------
# Annotation
# ------------------------------------------------------------

ax.text(
    0.985, 0.955,
    "MuchoUchuu:\n"
    r"$R_{\rm hom}=446.3^{+41.7}_{-27.7}\ h^{-1}{\rm Mpc}$"
    "\n"
    r"$d_s^{\rm late}=3.011^{+0.025}_{-0.025}$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xlabel(
    r"physical diffusion scale $R\ [h^{-1}\,{\rm Mpc}]$"
)
ax.set_ylabel(
    r"effective spectral dimension $d_s$"
)
ax.set_title("Euclidean transport in MuchoUchuu")

ax.set_xlim(250, 2000)
ax.set_ylim(2.85, 3.11)

ax.legend(
    loc="lower left",
    fontsize=8.5,
    ncol=2,
    frameon=True
)

fig.tight_layout()

fig.savefig(OUT_PNG, dpi=250)
fig.savefig(OUT_PDF)

print("Wrote:")
print(OUT_PNG)
print(OUT_PDF)
