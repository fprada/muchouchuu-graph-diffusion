import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
FN = "muchouchuu_halo_pk_shells_nmesh256.csv"

L = 6000.0
V = L**3

# Measured value used ONLY as a comparison marker
RHOM = 446.3
RHOM_LO = 446.3 - 27.7
RHOM_HI = 446.3 + 41.7

# Physical diffusion-scale grid
R = np.geomspace(30.0, 1500.0, 1200)

# ============================================================
# LOAD EXACT-SHELL POWER SPECTRUM
# ============================================================
df = pd.read_csv(FN)

k = df["k_h_mpc"].to_numpy(dtype=float)
Praw = df["P_raw_mpc_h3"].to_numpy(dtype=float)
Pclust = df["P_shot_subtracted_mpc_h3"].to_numpy(dtype=float)
Nm = df["Nmodes_full"].to_numpy(dtype=float)

Pshot_value = float(df["P_shot_mpc_h3"].iloc[0])
Pshot = np.full_like(k, Pshot_value)

print("Loaded shells:", len(k))
print("k range:", k.min(), k.max())
print("shot noise:", Pshot_value)

# ============================================================
# DISCRETE PERIODIC-BOX DIFFUSION VARIANCE
# ============================================================
#
# sigma^2(R)
# = 1/V sum_k P(k) exp[-k^2 R^2 / 3]
#
# Using exact shell multiplicities:
#
# = 1/V sum_shell Nmodes(shell) P(shell) W^2(shell,R)
#
# ============================================================
def diffusion_variance(P):
    sigma2 = np.empty_like(R)

    for i, r in enumerate(R):
        W2 = np.exp(-(k * r)**2 / 6.0)
        sigma2[i] = np.sum(
            Nm * P * W2
        ) / V

    return sigma2


sig_raw = diffusion_variance(Praw)
sig_clust = diffusion_variance(Pclust)
sig_shot = diffusion_variance(Pshot)

# ============================================================
# LOGARITHMIC DERIVATIVE PROXIES
# ============================================================
lnR = np.log(R)

X_raw = -np.gradient(sig_raw, lnR)
X_clust = -np.gradient(sig_clust, lnR)
X_shot = -np.gradient(sig_shot, lnR)

# Fractional logarithmic slopes
gamma_raw = -np.gradient(
    np.log(sig_raw),
    lnR
)

gamma_clust = -np.gradient(
    np.log(sig_clust),
    lnR
)

# ============================================================
# VALUES AT THE MEASURED GRAPH Rhom
# ============================================================
def at_rhom(arr):
    return np.interp(
        RHOM,
        R,
        arr
    )


print()
print("=" * 72)
print("VALUES AT MEASURED GRAPH HOMOGENEITY SCALE")
print("=" * 72)

for name, arr in [
    ("sigma2 raw", sig_raw),
    ("sigma2 clustering", sig_clust),
    ("sigma2 shot", sig_shot),
    ("X raw", X_raw),
    ("X clustering", X_clust),
    ("X shot", X_shot),
    ("gamma raw", gamma_raw),
    ("gamma clustering", gamma_clust),
]:
    print(
        f"{name:22s} = {at_rhom(arr):.10e}"
    )

print()

xraw_r = at_rhom(X_raw)
xcl_r = at_rhom(X_clust)
xshot_r = at_rhom(X_shot)

print(
    "X_clustering / X_raw =",
    xcl_r / xraw_r
)

print(
    "X_shot / X_raw       =",
    xshot_r / xraw_r
)

# ============================================================
# IDENTIFY WHICH FOURIER SHELLS CONTRIBUTE AT Rhom
# ============================================================
W2_rhom = np.exp(
    -(k * RHOM)**2 / 3.0
)

contrib_raw = (
    Nm * Praw * W2_rhom / V
)

contrib_clust = (
    Nm * Pclust * W2_rhom / V
)

tmp = pd.DataFrame({
    "n2": df["n2"].to_numpy(),
    "k_h_mpc": k,
    "Nmodes": Nm.astype(int),
    "P_raw": Praw,
    "W2_at_Rhom": W2_rhom,
    "contribution_raw": contrib_raw,
    "contribution_clust": contrib_clust,
})

tmp["fraction_raw"] = (
    tmp["contribution_raw"]
    / tmp["contribution_raw"].sum()
)

tmp = tmp.sort_values(
    "contribution_raw",
    ascending=False
)

tmp.to_csv(
    "muchouchuu_shell_contributions_at_Rhom.csv",
    index=False
)

print()
print("=" * 72)
print("TOP FOURIER-SHELL CONTRIBUTIONS AT R = 446.3")
print("=" * 72)

print(
    tmp.head(20).to_string(
        index=False,
        formatters={
            "k_h_mpc": "{:.6f}".format,
            "P_raw": "{:.2f}".format,
            "W2_at_Rhom": "{:.6f}".format,
            "contribution_raw": "{:.6e}".format,
            "contribution_clust": "{:.6e}".format,
            "fraction_raw": "{:.5f}".format,
        }
    )
)

# ============================================================
# SAVE MAIN CURVES
# ============================================================
out = pd.DataFrame({
    "R_hmpc": R,
    "sigma2_raw": sig_raw,
    "sigma2_clustering": sig_clust,
    "sigma2_shot": sig_shot,
    "X_raw": X_raw,
    "X_clustering": X_clust,
    "X_shot": X_shot,
    "gamma_raw": gamma_raw,
    "gamma_clustering": gamma_clust,
})

out.to_csv(
    "muchouchuu_heatkernel_pk_proxy.csv",
    index=False
)

# ============================================================
# FIGURE 1: DIFFUSION VARIANCE
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.8))

ax.plot(
    R,
    sig_raw,
    label="raw tracer P(k)"
)

ax.plot(
    R,
    sig_clust,
    label="shot-noise subtracted"
)

ax.plot(
    R,
    sig_shot,
    label="shot noise only"
)

ax.axvline(
    RHOM,
    linestyle="--"
)

ax.axvspan(
    RHOM_LO,
    RHOM_HI,
    alpha=0.15
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"$R\ [h^{-1}\,\mathrm{Mpc}]$"
)

ax.set_ylabel(
    r"$\sigma_{\rm diff}^2(R)$"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "muchouchuu_heatkernel_variance_from_pk.png",
    dpi=220
)

# ============================================================
# FIGURE 2: DERIVATIVE PROXY
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.8))

ax.plot(
    R,
    X_raw,
    label="raw tracer P(k)"
)

ax.plot(
    R,
    X_clust,
    label="shot-noise subtracted"
)

ax.plot(
    R,
    X_shot,
    label="shot noise only"
)

ax.axvline(
    RHOM,
    linestyle="--"
)

ax.axvspan(
    RHOM_LO,
    RHOM_HI,
    alpha=0.15
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    r"$R\ [h^{-1}\,\mathrm{Mpc}]$"
)

ax.set_ylabel(
    r"$-d\sigma_{\rm diff}^2/d\ln R$"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "muchouchuu_heatkernel_derivative_proxy.png",
    dpi=220
)

print()
print("Wrote:")
print("  muchouchuu_diffusion_pk_proxy.csv")
print("  muchouchuu_shell_contributions_at_Rhom.csv")
print("  muchouchuu_diffusion_variance_from_pk.png")
print("  muchouchuu_diffusion_derivative_proxy.png")
