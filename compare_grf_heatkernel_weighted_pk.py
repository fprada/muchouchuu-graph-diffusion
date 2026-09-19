from pathlib import Path

import numpy as np
import pandas as pd


MUCHO_FILE = Path(
    "muchouchuu_halo_pk_shells_nmesh256.csv"
)

GRF_FILE = Path(
    "gaussian_pk_control_seed1001/"
    "grf_displaced_N108000000_pk_shells_nmesh256.csv"
)

OUTFILE = Path(
    "gaussian_pk_control_seed1001/"
    "grf_vs_mucho_heatkernel_weighted_pk.csv"
)


mucho = pd.read_csv(MUCHO_FILE)
grf = pd.read_csv(GRF_FILE)

m = mucho.merge(
    grf[
        [
            "n2",
            "P_shot_subtracted_mpc_h3",
            "Nmodes_full",
        ]
    ],
    on="n2",
    suffixes=("_mucho", "_grf"),
    how="inner",
)

# Use k from the MuchoUchuu file. Shell geometry is identical.
k = m["k_h_mpc"].to_numpy(float)

Pm = m["P_shot_subtracted_mpc_h3_mucho"].to_numpy(float)
Pg = m["P_shot_subtracted_mpc_h3_grf"].to_numpy(float)

nm = m["Nmodes_full_mucho"].to_numpy(float)
ng = m["Nmodes_full_grf"].to_numpy(float)

if not np.array_equal(nm, ng):
    raise RuntimeError(
        "MuchoUchuu and GRF shell multiplicities differ."
    )

# Only use positive clustering-power estimates in both catalogues.
valid = (
    np.isfinite(k)
    & np.isfinite(Pm)
    & np.isfinite(Pg)
    & (Pm > 0.0)
    & (Pg > 0.0)
    & (nm > 0.0)
)

k = k[valid]
Pm = Pm[valid]
Pg = Pg[valid]
nm = nm[valid]


def filtered_power(R, P):
    """
    Shell-summed heat-kernel-filtered clustering power.

    W(k,R) = exp[-k^2 R^2 / 6].
    Any common Fourier-volume normalization cancels in the
    GRF/MuchoUchuu ratio.
    """
    W = np.exp(-(k * R)**2 / 6.0)

    return np.sum(
        nm * P * W
    )


R_values = np.array(
    [
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
    ],
    dtype=float,
)

rows = []

for R in R_values:

    Xm = filtered_power(R, Pm)
    Xg = filtered_power(R, Pg)

    rows.append(
        {
            "R_hmpc": R,
            "X_mucho": Xm,
            "X_grf": Xg,
            "ratio_grf_to_mucho": Xg / Xm,
        }
    )

out = pd.DataFrame(rows)

out.to_csv(
    OUTFILE,
    index=False,
)

print("=" * 76)
print("HEAT-KERNEL-WEIGHTED P(k): GRF VS MUCHOUCHUU")
print("=" * 76)
print()
print(
    out.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("Transition-scale diagnostics:")

for R in [414.5, 446.1]:

    row = out[np.isclose(out["R_hmpc"], R)].iloc[0]

    print(
        f"R={R:6.1f} h^-1 Mpc : "
        f"X_GRf/X_Mucho = "
        f"{row['ratio_grf_to_mucho']:.6f}"
    )

print()
print("Wrote:", OUTFILE)
