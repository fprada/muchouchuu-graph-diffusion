import numpy as np
import pandas as pd

CAMB = "PkTable.dat"
POINT = (
    "gaussian_camb_z0_seed3001/"
    "camb_displaced_N108000000_pk_shells_nmesh256.csv"
)

OUT = (
    "gaussian_camb_z0_seed3001/"
    "camb_point_pk_vs_input.csv"
)

# CAMB:
# column 0 = k [h/Mpc]
# column 1 = P(k) [(Mpc/h)^3]
c = np.loadtxt(CAMB, comments="#")
kc = c[:, 0].astype(float)
Pc = c[:, 1].astype(float)

good = (
    np.isfinite(kc)
    & np.isfinite(Pc)
    & (kc > 0)
    & (Pc > 0)
)

kc = kc[good]
Pc = Pc[good]

order = np.argsort(kc)
kc = kc[order]
Pc = Pc[order]

p = pd.read_csv(POINT)

k = p["k_h_mpc"].to_numpy(float)
Ppoint = p["P_shot_subtracted_mpc_h3"].to_numpy(float)
nm = p["Nmodes_full"].to_numpy(float)

# Log-log interpolation of input CAMB spectrum.
inside = (
    (k >= kc.min())
    & (k <= kc.max())
    & np.isfinite(Ppoint)
    & (Ppoint > 0)
    & (nm > 0)
)

k = k[inside]
Ppoint = Ppoint[inside]
nm = nm[inside]

Pinput = np.exp(
    np.interp(
        np.log(k),
        np.log(kc),
        np.log(Pc),
    )
)

ratio = Ppoint / Pinput

out = pd.DataFrame({
    "k_h_mpc": k,
    "P_camb_input_mpc_h3": Pinput,
    "P_point_shot_subtracted_mpc_h3": Ppoint,
    "Nmodes_full": nm.astype(np.int64),
    "ratio_point_to_camb": ratio,
})

out.to_csv(OUT, index=False)

print("=" * 76)
print("CAMB-DISPLACED POINT PROCESS VS INPUT LINEAR CAMB P(k)")
print("=" * 76)

bands = [
    (0.0010, 0.0100),
    (0.0010, 0.0040),
    (0.0020, 0.0060),
    (0.0025, 0.0080),
    (0.0040, 0.0100),
]

for lo, hi in bands:
    m = (k >= lo) & (k <= hi)

    r = ratio[m]
    w = nm[m]

    print()
    print(f"{lo:.4f} <= k <= {hi:.4f}")
    print("  shells        :", int(m.sum()))
    print("  median ratio  :", float(np.median(r)))
    print(
        "  q16/q84       :",
        np.quantile(r, [0.16, 0.84]),
    )
    print(
        "  exp(mean ln)  :",
        float(np.exp(np.mean(np.log(r)))),
    )
    print(
        "  mode-weighted :",
        float(np.sum(w * Ppoint[m]) / np.sum(w * Pinput[m])),
    )

print()
print("=" * 76)
print("HEAT-KERNEL-WEIGHTED COMPARISON")
print("=" * 76)

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
    W = np.exp(-(k * R)**2 / 6.0)

    Xin = np.sum(
        nm * Pinput * W
    )

    Xpt = np.sum(
        nm * Ppoint * W
    )

    print(
        f"R={R:6.1f} h^-1 Mpc : "
        f"X_point/X_CAMB = {Xpt/Xin:.6f}"
    )

print()
print("Wrote:", OUT)
