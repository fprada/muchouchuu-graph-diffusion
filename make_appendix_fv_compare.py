from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUTS
# ============================================================

MU_FILE = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace_t131072/combined_64probe/"
    "muchouchuu_extended_64probe_bootstrap_ds.csv"
)

BIG_FILE = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/"
    "biguchuu_diffusion_reproduction_v2/"
    "biguchuu_extended_centered_w7_s64_b10000/"
    "spectral_dimension_sliding_bootstrap.csv"
)

OUT_PNG = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_vs_biguchuu_extended_finite_volume_ds.png"
)

OUT_PDF = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_vs_biguchuu_extended_finite_volume_ds.pdf"
)

# Physical diffusion calibrations
A_MU  = 10.4095576   # h^-1 Mpc step^-1/2
A_BIG = 10.35        # h^-1 Mpc step^-1/2

# Box sizes
L_MU  = 6000.0
L_BIG = 4000.0

# ============================================================
# LOAD MUCHOUCHUU
# ============================================================

mu = pd.read_csv(MU_FILE)

# Use physical scale already saved if present; otherwise reconstruct
if "R_mpc_h" not in mu.columns:
    mu["R_mpc_h"] = A_MU * np.sqrt(mu["diffusion_time"])

# Keep valid strict-centred estimates
mu = mu[
    np.isfinite(mu["central_ds"]) &
    np.isfinite(mu["ds_q16"]) &
    np.isfinite(mu["ds_q84"])
].copy()

mu = mu.sort_values("diffusion_time")

# ============================================================
# LOAD BIGUCHUU
# ============================================================

bg = pd.read_csv(BIG_FILE)

# Convert diffusion time to physical scale
bg["R_mpc_h"] = A_BIG * np.sqrt(bg["diffusion_time"])
bg["R_over_L"] = bg["R_mpc_h"] / L_BIG

# Rename bootstrap columns for consistency
bg = bg.rename(columns={
    "bootstrap_ds_q025": "ds_q025",
    "bootstrap_ds_q16":  "ds_q16",
    "bootstrap_ds_q84":  "ds_q84",
    "bootstrap_ds_q975": "ds_q975",
})

# Keep valid strict-centred estimates
bg = bg[
    np.isfinite(bg["central_ds"]) &
    np.isfinite(bg["ds_q16"]) &
    np.isfinite(bg["ds_q84"])
].copy()

bg = bg.sort_values("diffusion_time")

# ============================================================
# DIAGNOSTIC THRESHOLDS
# ============================================================

def first_row(df, condition):
    q = df.loc[condition]
    if len(q) == 0:
        return None
    return q.iloc[0]


# Central curve below lower 1% boundary
mu_central = first_row(mu, mu["central_ds"] < 2.97)
bg_central = first_row(bg, bg["central_ds"] < 2.97)

# Entire 68% interval below lower 1% boundary
mu_68 = first_row(mu, mu["ds_q84"] < 2.97)
bg_68 = first_row(bg, bg["ds_q84"] < 2.97)

# Conservative diagnostics
mu_95_3 = first_row(mu, mu["ds_q975"] < 3.0)
bg_95_3 = first_row(bg, bg["ds_q975"] < 3.0)

mu_95_297 = first_row(mu, mu["ds_q975"] < 2.97)
bg_95_297 = first_row(bg, bg["ds_q975"] < 2.97)


def report(name, row, L):
    if row is None:
        print(name, ": not reached")
        return

    print(
        f"{name:36s}"
        f" t={int(row['diffusion_time']):6d}"
        f"  R={row['R_mpc_h']:8.2f}"
        f"  R/L={row['R_mpc_h']/L:.3f}"
        f"  ds={row['central_ds']:.4f}"
    )


print("\n============================================================")
print("FINITE-VOLUME DIAGNOSTICS")
print("============================================================")

report("Mucho central < 2.97", mu_central, L_MU)
report("Mucho q84 < 2.97",     mu_68,      L_MU)
report("Mucho q975 < 3",       mu_95_3,    L_MU)
report("Mucho q975 < 2.97",    mu_95_297,  L_MU)

print()

report("Big central < 2.97",    bg_central, L_BIG)
report("Big q84 < 2.97",       bg_68,      L_BIG)
report("Big q975 < 3",         bg_95_3,    L_BIG)
report("Big q975 < 2.97",      bg_95_297,  L_BIG)

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7.4, 5.2))

# Euclidean 1% band
ax.axhspan(
    2.97, 3.03,
    alpha=0.12,
    label=r"1% band around $d_s=3$"
)

ax.axhline(
    3.0,
    linestyle="--",
    linewidth=1.1,
    label=r"$d_s=3$"
)

# ------------------------------------------------------------
# MuchoUchuu
# ------------------------------------------------------------

mu_line, = ax.plot(
    mu["R_mpc_h"],
    mu["central_ds"],
    linewidth=2.0,
    marker="o",
    markersize=3.5,
    label="MuchoUchuu"
)

mu_color = mu_line.get_color()

ax.fill_between(
    mu["R_mpc_h"].to_numpy(),
    mu["ds_q16"].to_numpy(),
    mu["ds_q84"].to_numpy(),
    alpha=0.22,
    color=mu_color
)

# ------------------------------------------------------------
# BigUchuu
# ------------------------------------------------------------

bg_line, = ax.plot(
    bg["R_mpc_h"],
    bg["central_ds"],
    linewidth=2.0,
    marker="s",
    markersize=3.5,
    label="BigUchuu"
)

bg_color = bg_line.get_color()

ax.fill_between(
    bg["R_mpc_h"].to_numpy(),
    bg["ds_q16"].to_numpy(),
    bg["ds_q84"].to_numpy(),
    alpha=0.22,
    color=bg_color
)

# ------------------------------------------------------------
# Mark first central departures
# ------------------------------------------------------------

if bg_central is not None:
    ax.axvline(
        bg_central["R_mpc_h"],
        linestyle=":",
        linewidth=1.3,
        color=bg_color
    )

if mu_central is not None:
    ax.axvline(
        mu_central["R_mpc_h"],
        linestyle=":",
        linewidth=1.3,
        color=mu_color
    )

# Labels next to the vertical markers
if bg_central is not None:
    ax.text(
        bg_central["R_mpc_h"] + 25,
        2.30,
        rf"BigUchuu central exit"
        "\n"
        rf"$R={bg_central['R_mpc_h']:.0f}\,h^{{-1}}{{\rm Mpc}}$",
        fontsize=8,
        rotation=90,
        va="bottom",
        color=bg_color
    )

if mu_central is not None:
    ax.text(
        mu_central["R_mpc_h"] + 25,
        2.30,
        rf"MuchoUchuu central exit"
        "\n"
        rf"$R={mu_central['R_mpc_h']:.0f}\,h^{{-1}}{{\rm Mpc}}$",
        fontsize=8,
        rotation=90,
        va="bottom",
        color=mu_color
    )

# ------------------------------------------------------------
# Main labels
# ------------------------------------------------------------

ax.set_xlabel(
    r"physical diffusion scale $R\,[h^{-1}\,\mathrm{Mpc}]$"
)

ax.set_ylabel(
    r"effective spectral dimension $d_s$"
)

ax.set_title(
    "Finite-volume suppression in MuchoUchuu and BigUchuu"
)

# Start sufficiently early to connect visually with Fig. 3,
# but concentrate on the late regime.
ax.set_xlim(220, 3200)
ax.set_ylim(1.8, 3.10)

ax.legend(
    loc="lower left",
    fontsize=9,
    frameon=True
)

fig.tight_layout()

fig.savefig(OUT_PNG, dpi=250)
fig.savefig(OUT_PDF)

print("\nWrote:")
print(OUT_PNG)
print(OUT_PDF)
