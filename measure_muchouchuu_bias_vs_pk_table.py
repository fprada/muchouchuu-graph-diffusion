import numpy as np
import pandas as pd

PK_FILE = "PkTable.dat"
HALO_FILE = "muchouchuu_halo_pk_shells_nmesh256.csv"
OUT = "muchouchuu_bias_vs_pk_table.csv"

# ------------------------------------------------------------
# Input linear matter P(k) table
# ------------------------------------------------------------
a = np.loadtxt(PK_FILE, comments="#")

k_lin = a[:, 0].astype(float)
P_lin = a[:, 1].astype(float)

good = (
    np.isfinite(k_lin)
    & np.isfinite(P_lin)
    & (k_lin > 0.0)
    & (P_lin > 0.0)
)

k_lin = k_lin[good]
P_lin = P_lin[good]

order = np.argsort(k_lin)
k_lin = k_lin[order]
P_lin = P_lin[order]

# ------------------------------------------------------------
# MuchoUchuu halo P(k)
# ------------------------------------------------------------
d = pd.read_csv(HALO_FILE)

k = d["k_h_mpc"].to_numpy(float)
Ph = d["P_shot_subtracted_mpc_h3"].to_numpy(float)
nm = d["Nmodes_full"].to_numpy(float)

inside = (
    (k >= k_lin.min())
    & (k <= k_lin.max())
    & np.isfinite(Ph)
    & (Ph > 0.0)
    & (nm > 0.0)
)

# Log-log interpolation of the supplied linear P(k)
Pmm_lin = np.full_like(k, np.nan, dtype=float)

Pmm_lin[inside] = np.exp(
    np.interp(
        np.log(k[inside]),
        np.log(k_lin),
        np.log(P_lin),
    )
)

bias_lin = np.full_like(k, np.nan, dtype=float)

bias_lin[inside] = np.sqrt(
    Ph[inside] / Pmm_lin[inside]
)

out = d.copy()
out["P_mm_linear_from_PkTable"] = Pmm_lin
out["bias_vs_linear"] = bias_lin

out.to_csv(OUT, index=False)

# ------------------------------------------------------------
# Low-k summaries
# ------------------------------------------------------------
bands = [
    (0.0010, 0.0040),
    (0.0020, 0.0060),
    (0.0025, 0.0080),
    (0.0040, 0.0100),
    (0.0020, 0.0100),
]

print("=" * 80)
print("MUCHOUCHUU HALO BIAS VS SUPPLIED LINEAR P(k)")
print("=" * 80)

for lo, hi in bands:

    m = (
        inside
        & (k >= lo)
        & (k <= hi)
    )

    if not np.any(m):
        continue

    # Prefer ratio of mode-weighted powers over
    # averaging noisy shell-by-shell bias values.
    b2_mode = (
        np.sum(nm[m] * Ph[m])
        / np.sum(nm[m] * Pmm_lin[m])
    )

    b_shell = bias_lin[m]

    print()
    print(f"{lo:.4f} <= k <= {hi:.4f}")
    print("  shells              :", int(np.sum(m)))
    print("  b_eff mode-weighted :", float(np.sqrt(b2_mode)))
    print("  median shell bias   :", float(np.median(b_shell)))
    print(
        "  q16/q84 shell bias :",
        np.quantile(b_shell, [0.16, 0.84]),
    )

# ------------------------------------------------------------
# Heat-kernel-weighted effective bias
# ------------------------------------------------------------
print()
print("=" * 80)
print("HEAT-KERNEL-WEIGHTED EFFECTIVE BIAS")
print("=" * 80)

Rvals = [
    300.0,
    350.0,
    400.0,
    414.5,
    425.0,
    446.1,
    475.0,
    500.0,
    550.0,
    600.0,
]

m = inside

for R in Rvals:

    W = np.exp(-(k[m] * R)**2 / 6.0)

    Xh = np.sum(
        nm[m] * Ph[m] * W
    )

    Xm = np.sum(
        nm[m] * Pmm_lin[m] * W
    )

    beff = np.sqrt(Xh / Xm)

    print(
        f"R={R:6.1f} h^-1 Mpc : "
        f"b_eff_linear = {beff:.6f}"
    )

print()
print("Wrote:", OUT)
