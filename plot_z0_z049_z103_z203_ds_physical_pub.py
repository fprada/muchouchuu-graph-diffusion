#!/usr/bin/env python

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Input products
# ============================================================

Z0_DIR = Path(
    "muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full"
)

Z049_DIR = Path(
    "growth_test/z049/"
    "step6_window7_s64_strictcentered_tmin3584_fixed"
)

Z1_DIR = Path(
    "growth_test/z103/"
    "step6_window7_s64_strictcentered_tmin3584_fixed"
)

Z2_DIR = Path(
    "growth_test/z203/"
    "step6_window7_s64_strictcentered_tmin3584_fixed"
)

# Fourier calibrations
A_Z0 = 10.4095576
A_Z049 = 10.438294184707798
A_Z1 = 10.4894608905163
A_Z2 = 10.598993479525339

# Physical crossing medians
RHOM_Z0 = 446.108
RHOM_Z1 = 452.176006
RHOM_Z2 = 453.702

OUT_PNG = "figure_z0_z103_z203_ds_physical_scale_with_delta.png"
OUT_PDF = "figure_z0_z103_z203_ds_physical_scale_with_delta.pdf"


# ============================================================
# Helpers
# ============================================================

def load_step6(root):
    df = pd.read_csv(
        root / "spectral_dimension_sliding_bootstrap.csv"
    )

    boot = np.load(
        root / "bootstrap_spectral_dimension_curves.npy"
    )

    if "diffusion_time" not in df.columns:
        raise RuntimeError(
            f"{root}: diffusion_time column not found"
        )

    if "central_ds" not in df.columns:
        raise RuntimeError(
            f"{root}: central_ds column not found"
        )

    t = df["diffusion_time"].to_numpy(float)
    ds = df["central_ds"].to_numpy(float)

    if boot.ndim != 2:
        raise RuntimeError(
            f"{root}: bootstrap array has shape {boot.shape}"
        )

    if boot.shape[1] != len(df):
        raise RuntimeError(
            f"{root}: bootstrap has {boot.shape[1]} time columns "
            f"but CSV has {len(df)} rows"
        )

    return df, t, ds, boot


def index_for_times(t, common):
    out = []

    for x in common:
        hit = np.where(np.isclose(t, x))[0]

        if len(hit) != 1:
            raise RuntimeError(
                f"Could not uniquely locate t={x}"
            )

        out.append(hit[0])

    return np.asarray(out, dtype=int)


# ============================================================
# Load all three redshifts
# ============================================================

df0, t0, ds0_all, boot0_all = load_step6(Z0_DIR)
df049, t049, ds049_all, boot049_all = load_step6(Z049_DIR)
df1, t1, ds1_all, boot1_all = load_step6(Z1_DIR)

# z=2.03 special case:
# the authoritative fixed Step-6 directory has a valid bootstrap NPY,
# but its spectral_dimension_sliding_bootstrap.csv is header-only.
# Use the non-fixed CSV for the central curve and the fixed NPY for bootstrap.
Z2_CENTRAL_DIR = Path(
    "growth_test/z203/"
    "step6_window7_s64_strictcentered_tmin3584"
)

df2 = pd.read_csv(
    Z2_CENTRAL_DIR / "spectral_dimension_sliding_bootstrap.csv"
)

t2 = df2["diffusion_time"].to_numpy(float)
ds2_all = df2["central_ds"].to_numpy(float)

boot2_all = np.load(
    Z2_DIR / "bootstrap_spectral_dimension_curves.npy"
)

if boot2_all.ndim != 2:
    raise RuntimeError(
        f"z203 fixed bootstrap has shape {boot2_all.shape}"
    )

if boot2_all.shape[1] != 21:
    raise RuntimeError(
        f"z203 fixed bootstrap expected 21 time columns, "
        f"found {boot2_all.shape[1]}"
    )

print(
    "z=2.03: central curve from non-fixed CSV; "
    "bootstrap from fixed Step-6 NPY"
)


# ============================================================
# Find common strict-centered valid times
# ============================================================

valid0 = t0[np.isfinite(ds0_all)]
valid049 = t049[np.isfinite(ds049_all)]
valid1 = t1[np.isfinite(ds1_all)]
valid2 = t2[np.isfinite(ds2_all)]

common_t = np.array(
    sorted(
        set(valid0.astype(int))
        & set(valid049.astype(int))
        & set(valid1.astype(int))
        & set(valid2.astype(int))
    ),
    dtype=float,
)

print("Common strict-centered times:")
print(common_t.astype(int))
print("Number of common points =", len(common_t))

if len(common_t) != 15:
    print(
        "WARNING: expected 15 strict-centered points, "
        f"found {len(common_t)}"
    )

i0 = index_for_times(t0, common_t)
i049 = index_for_times(t049, common_t)
i1 = index_for_times(t1, common_t)
i2 = index_for_times(t2, common_t)

ds0 = ds0_all[i0]
ds049 = ds049_all[i049]
ds1 = ds1_all[i1]
ds2 = ds2_all[i2]

boot0 = boot0_all[:, i0]
boot049 = boot049_all[:, i049]
boot1 = boot1_all[:, i1]
boot2 = boot2_all[:, i2]

print("Bootstrap shapes:")
print("z=0    :", boot0.shape)
print("z=0.49 :", boot049.shape)
print("z=1.03 :", boot1.shape)
print("z=2.03 :", boot2.shape)


# ============================================================
# Bootstrap uncertainty bands
# ============================================================

q16_0, q50_0, q84_0 = np.nanpercentile(
    boot0, [16, 50, 84], axis=0
)

q16_049, q50_049, q84_049 = np.nanpercentile(
    boot049, [16, 50, 84], axis=0
)

q16_1, q50_1, q84_1 = np.nanpercentile(
    boot1, [16, 50, 84], axis=0
)

q16_2, q50_2, q84_2 = np.nanpercentile(
    boot2, [16, 50, 84], axis=0
)


# ============================================================
# Physical scale R = A_RMS sqrt(t)
# ============================================================

R0 = A_Z0 * np.sqrt(common_t)
R049 = A_Z049 * np.sqrt(common_t)
R1 = A_Z1 * np.sqrt(common_t)
R2 = A_Z2 * np.sqrt(common_t)


# ============================================================
# Paired residuals at matched diffusion time
#
# Same global probe ordering and same bootstrap seed means
# corresponding bootstrap rows can be subtracted directly.
# ============================================================

if not (
    boot0.shape[0]
    == boot049.shape[0]
    == boot1.shape[0]
    == boot2.shape[0]
):
    raise RuntimeError(
        "Bootstrap replicate counts do not match"
    )

delta0490_boot = boot049 - boot0
delta10_boot = boot1 - boot0
delta20_boot = boot2 - boot0

d049_q16, d049_med, d049_q84 = np.nanpercentile(
    delta0490_boot, [16, 50, 84], axis=0
)

d10_q16, d10_med, d10_q84 = np.nanpercentile(
    delta10_boot, [16, 50, 84], axis=0
)

d20_q16, d20_med, d20_q84 = np.nanpercentile(
    delta20_boot, [16, 50, 84], axis=0
)

delta049_central = ds049 - ds0
delta10_central = ds1 - ds0
delta20_central = ds2 - ds0


# ============================================================
# Figure
# ============================================================
# ============================================================
# Publication figure
# ============================================================

# Transport homogeneity scales, h^-1 Mpc
RHOM0_MED = 446.108
RHOM0_Q16 = 418.555
RHOM0_Q84 = 486.916

RHOM049_MED = 449.783921
RHOM049_Q16 = 426.771553
RHOM049_Q84 = 481.479664

RHOM1_MED = 452.176006
RHOM1_Q16 = 424.028350
RHOM1_Q84 = 500.799341

RHOM2_MED = 453.702
RHOM2_Q16 = 428.439
RHOM2_Q84 = 488.538


fig = plt.figure(figsize=(8.2, 7.4))

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[3.15, 1.35],
    hspace=0.055,
)

ax = fig.add_subplot(gs[0])
axd = fig.add_subplot(gs[1], sharex=ax)


# ------------------------------------------------------------
# Top panel: spectral dimension versus physical scale
# ------------------------------------------------------------

# Euclidean target and 1% homogeneity band
ax.axhspan(
    2.97,
    3.03,
    alpha=0.10,
    zorder=0,
)

ax.axhline(
    3.0,
    linestyle="--",
    linewidth=1.0,
    zorder=1,
)


# z = 0
line0, = ax.plot(
    R0,
    ds0,
    marker="o",
    markersize=4.4,
    linewidth=1.9,
    label=r"$z=0$",
    zorder=4,
)

ax.fill_between(
    R0,
    q16_0,
    q84_0,
    color=line0.get_color(),
    alpha=0.18,
    linewidth=0,
    zorder=2,
)


# z = 0.49
line049, = ax.plot(
    R049,
    ds049,
    marker="D",
    markersize=4.2,
    linewidth=1.9,
    label=r"$z=0.49$",
    zorder=5,
)

ax.fill_between(
    R049,
    q16_049,
    q84_049,
    color=line049.get_color(),
    alpha=0.18,
    linewidth=0,
    zorder=2,
)


# z = 1.03
line1, = ax.plot(
    R1,
    ds1,
    marker="s",
    markersize=4.3,
    linewidth=1.9,
    label=r"$z=1.03$",
    zorder=5,
)

ax.fill_between(
    R1,
    q16_1,
    q84_1,
    color=line1.get_color(),
    alpha=0.18,
    linewidth=0,
    zorder=2,
)


# z = 2.03
line2, = ax.plot(
    R2,
    ds2,
    marker="^",
    markersize=4.7,
    linewidth=1.9,
    label=r"$z=2.03$",
    zorder=6,
)

ax.fill_between(
    R2,
    q16_2,
    q84_2,
    color=line2.get_color(),
    alpha=0.18,
    linewidth=0,
    zorder=2,
)


# ------------------------------------------------------------
# Transport homogeneity scale medians
# ------------------------------------------------------------

# Median Rhom vertical lines removed because the three values
# are nearly coincident and visually merge.


# Small horizontal 68% indicators near lower edge of top panel.
# These are preferable to three heavily overlapping vertical bands.
y_indicator = 2.935

ax.plot(
    [RHOM0_Q16, RHOM0_Q84],
    [y_indicator + 0.006, y_indicator + 0.006],
    color=line0.get_color(),
    linewidth=2.2,
    solid_capstyle="round",
)

ax.plot(
    [RHOM049_Q16, RHOM049_Q84],
    [y_indicator + 0.002, y_indicator + 0.002],
    color=line049.get_color(),
    linewidth=2.2,
    solid_capstyle="round",
)

ax.plot(
    [RHOM1_Q16, RHOM1_Q84],
    [y_indicator - 0.002, y_indicator - 0.002],
    color=line1.get_color(),
    linewidth=2.2,
    solid_capstyle="round",
)

ax.plot(
    [RHOM2_Q16, RHOM2_Q84],
    [y_indicator - 0.006, y_indicator - 0.006],
    color=line2.get_color(),
    linewidth=2.2,
    solid_capstyle="round",
)


ax.text(
    0.56,
    0.042,
    r"68\% intervals of $R_{\rm hom}^{(t)}$",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.5,
)


# ------------------------------------------------------------
# Top-panel labels
# ------------------------------------------------------------

ax.set_ylabel(
    r"Spectral dimension $d_s$",
    fontsize=12,
)

ax.set_ylim(2.92, 3.085)

ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=10,
)

ax.text(
    0.018,
    0.975,
    "(a)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
)

ax.text(
    0.025,
    0.885,
    r"$|d_s-3|\leq0.03$",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
)


# ------------------------------------------------------------
# Bottom panel: paired residuals at matched diffusion time
#
# The bootstrap rows are paired because the same global probe
# ordering and bootstrap seed are used at all redshifts.
#
# The horizontal coordinate uses the z=0 physical mapping R0(t).
# ------------------------------------------------------------

axd.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)


ld049, = axd.plot(
    R0,
    delta049_central,
    marker="D",
    markersize=3.9,
    linewidth=1.7,
    color=line049.get_color(),
    label=r"$z=0.49-z=0$",
    zorder=4,
)

axd.fill_between(
    R0,
    d049_q16,
    d049_q84,
    color=ld049.get_color(),
    alpha=0.18,
    linewidth=0,
)


ld1, = axd.plot(
    R0,
    delta10_central,
    marker="s",
    markersize=4.0,
    linewidth=1.7,
    color=line1.get_color(),
    label=r"$z=1.03-z=0$",
    zorder=4,
)

axd.fill_between(
    R0,
    d10_q16,
    d10_q84,
    color=ld1.get_color(),
    alpha=0.18,
    linewidth=0,
)


ld2, = axd.plot(
    R0,
    delta20_central,
    marker="^",
    markersize=4.2,
    linewidth=1.7,
    color=line2.get_color(),
    label=r"$z=2.03-z=0$",
    zorder=5,
)

axd.fill_between(
    R0,
    d20_q16,
    d20_q84,
    color=ld2.get_color(),
    alpha=0.18,
    linewidth=0,
)


axd.set_ylabel(
    r"$\Delta d_s$",
    fontsize=12,
)

axd.set_xlabel(
    r"Physical diffusion scale "
    r"$R=A_{\rm RMS}\sqrt{t}\;[h^{-1}\,\mathrm{Mpc}]$",
    fontsize=12,
)

axd.set_ylim(-0.04, 0.03)

axd.legend(
    frameon=False,
    loc="upper right",
    fontsize=9.5,
)

axd.text(
    0.025,
    0.90,
    "(b)",
    transform=axd.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
)


# ------------------------------------------------------------
# Shared formatting
# ------------------------------------------------------------

# The strict-centered physical range is roughly 300--1100 h^-1 Mpc.
axd.set_xlim(280, 1120)

for a in (ax, axd):
    a.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=10,
    )

    a.minorticks_on()

# Hide top-panel x tick labels.
plt.setp(ax.get_xticklabels(), visible=False)

fig.align_ylabels([ax, axd])

fig.subplots_adjust(
    left=0.115,
    right=0.985,
    bottom=0.105,
    top=0.985,
)


# ------------------------------------------------------------
# Print numerical values used in the figure
# ------------------------------------------------------------

print()
print("Transport homogeneity radii used in figure:")
print(
    f"z=0    : {RHOM0_MED:.3f} "
    f"[{RHOM0_Q16:.3f}, {RHOM0_Q84:.3f}] h^-1 Mpc"
)
print(
    f"z=0.49 : {RHOM049_MED:.3f} "
    f"[{RHOM049_Q16:.3f}, {RHOM049_Q84:.3f}] h^-1 Mpc"
)
print(
    f"z=1.03 : {RHOM1_MED:.3f} "
    f"[{RHOM1_Q16:.3f}, {RHOM1_Q84:.3f}] h^-1 Mpc"
)
print(
    f"z=2.03 : {RHOM2_MED:.3f} "
    f"[{RHOM2_Q16:.3f}, {RHOM2_Q84:.3f}] h^-1 Mpc"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT_PNG = (
    "figure_z0_z049_z103_z203_ds_physical_scale_pub_v2.png"
)

OUT_PDF = (
    "figure_z0_z049_z103_z203_ds_physical_scale_pub_v2.pdf"
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

print()
print("Wrote:", OUT_PNG)
print("Wrote:", OUT_PDF)

plt.close(fig)
