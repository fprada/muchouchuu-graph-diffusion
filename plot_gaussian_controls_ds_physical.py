#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

MUCHO_CSV = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584/"
    "spectral_dimension_sliding_bootstrap.csv"
)

GRF_CSV = Path(
    "gaussian_pk_control_seed1001/"
    "grf_step6_window7_s64_strictcentered_tmin3584/"
    "spectral_dimension_sliding_bootstrap.csv"
)

CAMB_CSV = Path(
    "gaussian_camb_z0_seed3001/"
    "camb_step6_window7_s64_strictcentered_tmin3584/"
    "spectral_dimension_sliding_bootstrap.csv"
)


# ------------------------------------------------------------
# Independently calibrated physical-scale mappings
#
# R(t) = A_RMS sqrt(t)
# ------------------------------------------------------------

A_MUCHO = 10.409557615168367
A_GRF   = 13.528985
A_CAMB  = 13.276739


OUT = Path("gaussian_controls_ds_physical_comparison.pdf")
OUT_PNG = Path("gaussian_controls_ds_physical_comparison.png")


def load_curve(path, A):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "diffusion_time",
        "central_ds",
        "bootstrap_ds_q16",
        "bootstrap_ds_q84",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path}: missing columns {missing}"
        )

    df = df.copy()

    df["R_mpc_h"] = (
        A * np.sqrt(df["diffusion_time"].to_numpy(dtype=float))
    )

    # Keep only strict-centred points for which the spectral
    # dimension is actually defined.
    df = df[
        np.isfinite(df["central_ds"])
        & np.isfinite(df["bootstrap_ds_q16"])
        & np.isfinite(df["bootstrap_ds_q84"])
    ].copy()

    return df


mucho = load_curve(MUCHO_CSV, A_MUCHO)
grf   = load_curve(GRF_CSV, A_GRF)
camb  = load_curve(CAMB_CSV, A_CAMB)


# ------------------------------------------------------------
# Print the plotted support for reproducibility
# ------------------------------------------------------------

for name, df in [
    ("MuchoUchuu", mucho),
    ("P_h(k)-GRF", grf),
    ("CAMB Gaussian", camb),
]:
    print(
        f"{name:16s}: "
        f"N={len(df):2d}, "
        f"R=[{df['R_mpc_h'].min():.1f}, "
        f"{df['R_mpc_h'].max():.1f}] h^-1 Mpc"
    )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 4.8))


# One-percent Euclidean band
ax.axhspan(
    2.97,
    3.03,
    alpha=0.12,
    label=r"1\% band around $d_s=3$",
)

# Exact Euclidean value
ax.axhline(
    3.0,
    linestyle="--",
    linewidth=1.2,
    label=r"$d_s=3$",
)


# ------------------------------------------------------------
# MuchoUchuu
# ------------------------------------------------------------

line_m, = ax.plot(
    mucho["R_mpc_h"],
    mucho["central_ds"],
    marker="o",
    markersize=3.5,
    linewidth=1.7,
    label="MuchoUchuu",
)

ax.fill_between(
    mucho["R_mpc_h"],
    mucho["bootstrap_ds_q16"],
    mucho["bootstrap_ds_q84"],
    color=line_m.get_color(),
    alpha=0.18,
)


# ------------------------------------------------------------
# P_h(k)-matched Gaussian control
# ------------------------------------------------------------

line_g, = ax.plot(
    grf["R_mpc_h"],
    grf["central_ds"],
    marker="s",
    markersize=3.5,
    linewidth=1.7,
    label=r"$P_h(k)$-matched Gaussian",
)

ax.fill_between(
    grf["R_mpc_h"],
    grf["bootstrap_ds_q16"],
    grf["bootstrap_ds_q84"],
    color=line_g.get_color(),
    alpha=0.18,
)


# ------------------------------------------------------------
# CAMB Gaussian matter control
# ------------------------------------------------------------

line_c, = ax.plot(
    camb["R_mpc_h"],
    camb["central_ds"],
    marker="^",
    markersize=3.8,
    linewidth=1.7,
    label="CAMB Gaussian matter",
)

ax.fill_between(
    camb["R_mpc_h"],
    camb["bootstrap_ds_q16"],
    camb["bootstrap_ds_q84"],
    color=line_c.get_color(),
    alpha=0.18,
)


# ------------------------------------------------------------
# Axes
#
# Concentrate on the common scientifically useful pre-mixing range.
# MuchoUchuu's fiducial strict-centred trajectory terminates near
# R = 1053 h^-1 Mpc.
# ------------------------------------------------------------

ax.set_xlim(280, 1400)

ax.set_xlabel(
    r"physical diffusion scale $R\ [h^{-1}\,\mathrm{Mpc}]$"
)

ax.set_ylabel(
    r"effective spectral dimension $d_s$"
)


ax.legend(
    loc="lower left",
    frameon=True,
    fontsize=8.5,
)

ax.grid(
    True,
    alpha=0.20,
)


# Final axis formatting
ax.set_ylim(2.90, 3.10)

ax.minorticks_on()

# Ticks on all four sides, pointing inward.
ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
)

ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=3,
)

ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")

fig.tight_layout()


fig.savefig(
    OUT,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PNG,
    dpi=250,
    bbox_inches="tight",
)

print()
print("Wrote:", OUT)
print("Wrote:", OUT_PNG)
