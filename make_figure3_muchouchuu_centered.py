from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace/step6_centered_w7_s64_b10000/"
    "spectral_dimension_sliding_bootstrap.csv"
)

OUT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "figure3_muchouchuu_centered"
)

A_RMS = 10.4095576

df = pd.read_csv(INPUT)
df = df[np.isfinite(df["central_ds"])].copy()

df["R"] = A_RMS * np.sqrt(df["diffusion_time"].astype(float))

fig, ax = plt.subplots(figsize=(7.0, 5.2))

# 1% Euclidean band
ax.axhspan(2.97, 3.03, alpha=0.12)

# Euclidean reference
ax.axhline(3.0, linestyle="--", linewidth=1.2)

# 68% bootstrap interval
ax.fill_between(
    df["R"],
    df["bootstrap_ds_q16"],
    df["bootstrap_ds_q84"],
    alpha=0.25,
)

# Central spectral dimension
ax.plot(
    df["R"],
    df["central_ds"],
    marker="o",
    linewidth=1.8,
    markersize=4.5,
)

ax.set_xlabel(r"$R\ [h^{-1}\,\mathrm{Mpc}]$")
ax.set_ylabel(r"$d_s(R)$")

ax.tick_params(direction="in", top=True, right=True)

fig.tight_layout()

fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight")

print(df[[
    "diffusion_time",
    "R",
    "central_ds",
    "bootstrap_ds_q16",
    "bootstrap_ds_q84"
]].to_string(index=False))

print("\nWrote:")
print(str(OUT) + ".pdf")
print(str(OUT) + ".png")
