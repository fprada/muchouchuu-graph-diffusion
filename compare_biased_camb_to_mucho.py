import numpy as np
import pandas as pd

CAMB_FILE = "PkTable.dat"

MUCHO_FILE = (
    "muchouchuu_halo_pk_shells_nmesh256.csv"
)

BIASED_FILE = (
    "gaussian_camb_z0_seed3001/"
    "camb_biased_bE1p566_N108000000_pk_shells_nmesh256.csv"
)

OUT = (
    "gaussian_camb_z0_seed3001/"
    "biased_camb_vs_muchouchuu.csv"
)

# ------------------------------------------------------------
# Linear CAMB P(k)
# ------------------------------------------------------------
a = np.loadtxt(CAMB_FILE, comments="#")

kc = a[:, 0].astype(float)
Pc = a[:, 1].astype(float)

good = (
    np.isfinite(kc)
    & np.isfinite(Pc)
    & (kc > 0.0)
    & (Pc > 0.0)
)

kc = kc[good]
Pc = Pc[good]

order = np.argsort(kc)
kc = kc[order]
Pc = Pc[order]

# ------------------------------------------------------------
# Point-process spectra
# ------------------------------------------------------------
m = pd.read_csv(MUCHO_FILE)
b = pd.read_csv(BIASED_FILE)

x = m[
    [
        "n2",
        "k_h_mpc",
        "P_shot_subtracted_mpc_h3",
        "Nmodes_full",
    ]
].merge(
    b[
        [
            "n2",
            "k_h_mpc",
            "P_shot_subtracted_mpc_h3",
            "Nmodes_full",
        ]
    ],
    on="n2",
    suffixes=("_mucho", "_biased"),
)

# Exact shell geometry check.
if not np.allclose(
    x["k_h_mpc_mucho"],
    x["k_h_mpc_biased"],
    rtol=0.0,
    atol=1e-12,
):
    raise RuntimeError("k-shell mismatch")

if not np.array_equal(
    x["Nmodes_full_mucho"].to_numpy(),
    x["Nmodes_full_biased"].to_numpy(),
):
    raise RuntimeError("mode-count mismatch")

k = x["k_h_mpc_mucho"].to_numpy(float)
nm = x["Nmodes_full_mucho"].to_numpy(float)

Pm = x[
    "P_shot_subtracted_mpc_h3_mucho"
].to_numpy(float)

Pb = x[
    "P_shot_subtracted_mpc_h3_biased"
].to_numpy(float)

# Linear CAMB spectrum on exact shell k values.
Plinear = np.exp(
    np.interp(
        np.log(k),
        np.log(kc),
        np.log(Pc),
    )
)

valid = (
    np.isfinite(k)
    & np.isfinite(Pm)
    & np.isfinite(Pb)
    & np.isfinite(Plinear)
    & (Pm > 0.0)
    & (Pb > 0.0)
    & (Plinear > 0.0)
    & (nm > 0.0)
)

k = k[valid]
nm = nm[valid]
Pm = Pm[valid]
Pb = Pb[valid]
Plinear = Plinear[valid]

ratio_bm = Pb / Pm
bias_realized = np.sqrt(Pb / Plinear)

out = pd.DataFrame({
    "k_h_mpc": k,
    "Nmodes_full": nm.astype(np.int64),
    "P_muchouchuu": Pm,
    "P_biased_camb": Pb,
    "P_linear_camb": Plinear,
    "biased_over_mucho": ratio_bm,
    "realized_bias_vs_linear": bias_realized,
})

out.to_csv(OUT, index=False)

# ------------------------------------------------------------
# Low-k band summaries
# ------------------------------------------------------------
bands = [
    (0.0010, 0.0040),
    (0.0020, 0.0060),
    (0.0025, 0.0080),
    (0.0040, 0.0100),
    (0.0020, 0.0100),
]

print("=" * 90)
print("BIASED CAMB TRACERS VS MUCHOUCHUU")
print("=" * 90)

for lo, hi in bands:

    q = (
        (k >= lo)
        & (k <= hi)
    )

    # Mode-weighted power ratios.
    Pb_band = np.sum(
        nm[q] * Pb[q]
    )

    Pm_band = np.sum(
        nm[q] * Pm[q]
    )

    Pl_band = np.sum(
        nm[q] * Plinear[q]
    )

    ratio = Pb_band / Pm_band
    b_real = np.sqrt(
        Pb_band / Pl_band
    )

    b_mucho = np.sqrt(
        Pm_band / Pl_band
    )

    print()
    print(f"{lo:.4f} <= k <= {hi:.4f}")
    print("  shells                  :", int(q.sum()))
    print("  biased/Mucho mode-wtd   :", float(ratio))
    print("  realized b_E vs linear  :", float(b_real))
    print("  Mucho b_eff vs linear   :", float(b_mucho))
    print(
        "  b_realized / b_Mucho    :",
        float(b_real / b_mucho),
    )

# ------------------------------------------------------------
# Heat-kernel weighted comparison
# ------------------------------------------------------------
print()
print("=" * 90)
print("HEAT-KERNEL-WEIGHTED COMPARISON")
print("=" * 90)
print()
print(
    "R[h^-1Mpc]   biased/Mucho   b_realized   b_Mucho   b_ratio"
)

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

for R in Rvals:

    W = np.exp(
        -(k * R)**2 / 6.0
    )

    Xm = np.sum(
        nm * Pm * W
    )

    Xb = np.sum(
        nm * Pb * W
    )

    Xl = np.sum(
        nm * Plinear * W
    )

    ratio = Xb / Xm

    b_real = np.sqrt(
        Xb / Xl
    )

    b_mucho = np.sqrt(
        Xm / Xl
    )

    print(
        f"{R:10.1f}   "
        f"{ratio:12.6f}   "
        f"{b_real:10.6f}   "
        f"{b_mucho:8.6f}   "
        f"{b_real/b_mucho:8.6f}"
    )

print()
print("Wrote:", OUT)
