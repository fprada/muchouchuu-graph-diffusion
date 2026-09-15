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
valid1 = t1[np.isfinite(ds1_all)]
valid2 = t2[np.isfinite(ds2_all)]

common_t = np.array(
    sorted(
        set(valid0.astype(int))
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
i1 = index_for_times(t1, common_t)
i2 = index_for_times(t2, common_t)

ds0 = ds0_all[i0]
ds1 = ds1_all[i1]
ds2 = ds2_all[i2]

boot0 = boot0_all[:, i0]
boot1 = boot1_all[:, i1]
boot2 = boot2_all[:, i2]

print("Bootstrap shapes:")
print("z=0    :", boot0.shape)
print("z=1.03 :", boot1.shape)
print("z=2.03 :", boot2.shape)


# ============================================================
# Bootstrap uncertainty bands
# ============================================================

q16_0, q50_0, q84_0 = np.nanpercentile(
    boot0, [16, 50, 84], axis=0
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
    == boot1.shape[0]
    == boot2.shape[0]
):
    raise RuntimeError(
        "Bootstrap replicate counts do not match"
    )

delta10_boot = boot1 - boot0
delta20_boot = boot2 - boot0

d10_q16, d10_med, d10_q84 = np.nanpercentile(
    delta10_boot, [16, 50, 84], axis=0
)

d20_q16, d20_med, d20_q84 = np.nanpercentile(
    delta20_boot, [16, 50, 84], axis=0
)

delta10_central = ds1 - ds0
delta20_central = ds2 - ds0


# ============================================================
# Figure
# ============================================================

fig, (ax, axd) = plt.subplots(
    2,
    1,
    figsize=(8.0, 8.2),
    sharex=False,
    gridspec_kw={
        "height_ratios": [2.35, 1.0],
        "hspace": 0.08,
    },
)


# ------------------------------------------------------------
# Top panel: d_s versus physical scale
# ------------------------------------------------------------

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


line0, = ax.plot(
    R0,
    ds0,
    marker="o",
    markersize=4.5,
    linewidth=1.8,
    label=r"$z=0$",
)

ax.fill_between(
    R0,
    q16_0,
    q84_0,
    color=line0.get_color(),
    alpha=0.18,
    linewidth=0,
)


line1, = ax.plot(
    R1,
    ds1,
    marker="s",
    markersize=4.5,
    linewidth=1.8,
    label=r"$z=1.03$",
)

ax.fill_between(
    R1,
    q16_1,
    q84_1,
    color=line1.get_color(),
    alpha=0.18,
    linewidth=0,
)


line2, = ax.plot(
    R2,
    ds2,
    marker="^",
    markersize=4.8,
    linewidth=1.8,
    label=r"$z=2.03$",
)

ax.fill_between(
    R2,
    q16_2,
    q84_2,
    color=line2.get_color(),
    alpha=0.18,
    linewidth=0,
)


# Transport-scale medians
ax.axvline(
    RHOM_Z0,
    color=line0.get_color(),
    linestyle=":",
    linewidth=1.1,
    alpha=0.8,
)

ax.axvline(
    RHOM_Z1,
    color=line1.get_color(),
    linestyle=":",
    linewidth=1.1,
    alpha=0.8,
)

ax.axvline(
    RHOM_Z2,
    color=line2.get_color(),
    linestyle=":",
    linewidth=1.1,
    alpha=0.8,
)


ax.set_ylabel(r"Spectral dimension $d_s$")
ax.legend(frameon=False, loc="best")

ax.text(
    0.02,
    0.96,
    r"$|d_s-3|\leq0.03$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)


# ------------------------------------------------------------
# Bottom panel: paired residuals
#
# These are differences at matched diffusion time.
# We display them against the z=0 physical mapping R0(t).
# ------------------------------------------------------------

axd.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)


ld1, = axd.plot(
    R0,
    delta10_central,
    marker="s",
    markersize=4.0,
    linewidth=1.6,
    label=r"$z=1.03-z=0$",
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
    linewidth=1.6,
    label=r"$z=2.03-z=0$",
)

axd.fill_between(
    R0,
    d20_q16,
    d20_q84,
    color=ld2.get_color(),
    alpha=0.18,
    linewidth=0,
)


axd.set_ylabel(r"$\Delta d_s$")
axd.set_xlabel(
    r"Physical diffusion scale "
    r"$R=A_{\rm RMS}\sqrt{t}\;[h^{-1}\,\mathrm{Mpc}]$"
)

axd.legend(
    frameon=False,
    loc="best",
)


# ============================================================
# Finish
# ============================================================

for a in (ax, axd):
    a.tick_params(direction="in", which="both")

fig.align_ylabels([ax, axd])

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
