import numpy as np
import pandas as pd

MUCHO = "muchouchuu_halo_pk_shells_nmesh256.csv"
GRF_HALO = (
    "gaussian_pk_control_seed1001/"
    "grf_displaced_N108000000_pk_shells_nmesh256.csv"
)
GRF_CAMB = (
    "gaussian_camb_z0_seed3001/"
    "camb_displaced_N108000000_pk_shells_nmesh256.csv"
)

OUT = "three_way_heatkernel_pk_comparison.csv"

def load(path, suffix):
    d = pd.read_csv(path)
    return d[
        [
            "n2",
            "k_h_mpc",
            "P_shot_subtracted_mpc_h3",
            "Nmodes_full",
        ]
    ].rename(
        columns={
            "k_h_mpc": f"k_{suffix}",
            "P_shot_subtracted_mpc_h3": f"P_{suffix}",
            "Nmodes_full": f"N_{suffix}",
        }
    )

m = load(MUCHO, "mucho")
g = load(GRF_HALO, "halo_grf")
c = load(GRF_CAMB, "camb_grf")

x = m.merge(g, on="n2").merge(c, on="n2")

# Exact shell geometry should be identical.
for a, b in [
    ("k_mucho", "k_halo_grf"),
    ("k_mucho", "k_camb_grf"),
    ("N_mucho", "N_halo_grf"),
    ("N_mucho", "N_camb_grf"),
]:
    if not np.allclose(x[a], x[b], rtol=0, atol=1e-12):
        raise RuntimeError(f"Shell mismatch: {a} vs {b}")

k = x["k_mucho"].to_numpy(float)
nm = x["N_mucho"].to_numpy(float)

Pm = x["P_mucho"].to_numpy(float)
Ph = x["P_halo_grf"].to_numpy(float)
Pc = x["P_camb_grf"].to_numpy(float)

valid = (
    np.isfinite(k)
    & np.isfinite(Pm)
    & np.isfinite(Ph)
    & np.isfinite(Pc)
    & (Pm > 0)
    & (Ph > 0)
    & (Pc > 0)
    & (nm > 0)
)

k = k[valid]
nm = nm[valid]
Pm = Pm[valid]
Ph = Ph[valid]
Pc = Pc[valid]

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

rows = []

print("=" * 100)
print("THREE-WAY HEAT-KERNEL-WEIGHTED P(k) COMPARISON")
print("=" * 100)
print()
print(
    "R[h^-1Mpc]    X_Mucho          X_haloGRF        X_CAMBGRF        "
    "halo/Mucho   CAMB/Mucho   CAMB/halo"
)

for R in Rvals:
    W = np.exp(-(k * R)**2 / 6.0)

    Xm = np.sum(nm * Pm * W)
    Xh = np.sum(nm * Ph * W)
    Xc = np.sum(nm * Pc * W)

    rows.append({
        "R_hmpc": R,
        "X_muchouchuu": Xm,
        "X_halo_grf": Xh,
        "X_camb_grf": Xc,
        "halo_grf_over_mucho": Xh / Xm,
        "camb_grf_over_mucho": Xc / Xm,
        "camb_grf_over_halo_grf": Xc / Xh,
    })

    print(
        f"{R:10.1f}  "
        f"{Xm:14.6e}  "
        f"{Xh:14.6e}  "
        f"{Xc:14.6e}  "
        f"{Xh/Xm:11.6f}  "
        f"{Xc/Xm:11.6f}  "
        f"{Xc/Xh:10.6f}"
    )

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print()
print("Transition-scale summary:")
for R in [414.5, 446.1]:
    r = out[np.isclose(out["R_hmpc"], R)].iloc[0]
    print(
        f"R={R:6.1f}: "
        f"haloGRF/Mucho={r['halo_grf_over_mucho']:.6f}, "
        f"CAMBGRF/Mucho={r['camb_grf_over_mucho']:.6f}, "
        f"CAMBGRF/haloGRF={r['camb_grf_over_halo_grf']:.6f}"
    )

print()
print("Wrote:", OUT)
