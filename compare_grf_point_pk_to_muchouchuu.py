from pathlib import Path
import numpy as np
import pandas as pd

TARGET_FILE = Path(
    "muchouchuu_halo_pk_shells_nmesh256.csv"
)

GRF_FILE = Path(
    "gaussian_pk_control_seed1001/"
    "grf_displaced_N16777216_pk_shells_nmesh256.csv"
)

OUTFILE = Path(
    "gaussian_pk_control_seed1001/"
    "grf_point_pk_vs_muchouchuu.csv"
)

target = pd.read_csv(TARGET_FILE)
grf = pd.read_csv(GRF_FILE)

m = target.merge(
    grf[
        [
            "n2",
            "P_raw_mpc_h3",
            "P_shot_mpc_h3",
            "P_shot_subtracted_mpc_h3",
            "Nmodes_full",
        ]
    ],
    on="n2",
    suffixes=("_mucho", "_grf"),
    how="inner",
)

m["ratio_clustering_grf_to_mucho"] = (
    m["P_shot_subtracted_mpc_h3_grf"]
    / m["P_shot_subtracted_mpc_h3_mucho"]
)

m.to_csv(
    OUTFILE,
    index=False,
)

def summarize(klo, khi):
    x = m[
        (m["k_h_mpc"] >= klo)
        & (m["k_h_mpc"] <= khi)
        & (m["P_shot_subtracted_mpc_h3_mucho"] > 0.0)
        & (m["P_shot_subtracted_mpc_h3_grf"] > 0.0)
    ].copy()

    r = x["ratio_clustering_grf_to_mucho"].to_numpy(float)
    w = x["Nmodes_full_mucho"].to_numpy(float)

    weighted = np.sum(
        w * x["P_shot_subtracted_mpc_h3_grf"]
    ) / np.sum(
        w * x["P_shot_subtracted_mpc_h3_mucho"]
    )

    print(
        f"{klo:.4f} <= k <= {khi:.4f}"
    )
    print("  shells        :", len(x))
    print("  median ratio  :", np.median(r))
    print(
        "  q16/q84       :",
        np.quantile(r, [0.16, 0.84]),
    )
    print(
        "  exp(mean ln)  :",
        np.exp(np.mean(np.log(r))),
    )
    print("  mode-weighted :", weighted)
    print()

print("=" * 76)
print("GRF DISPLACED-POINT P(k) VS MUCHOUCHUU")
print("=" * 76)
print("matched shells:", len(m))
print()

summarize(0.001, 0.010)
summarize(0.001, 0.004)
summarize(0.002, 0.006)
summarize(0.0025, 0.008)
summarize(0.004, 0.010)

print("First 20 low-k shells:")

x = m[
    (m["k_h_mpc"] >= 0.001)
    & (m["k_h_mpc"] <= 0.010)
].copy()

print(
    x[
        [
            "n2",
            "k_h_mpc",
            "Nmodes_full_mucho",
            "P_shot_subtracted_mpc_h3_mucho",
            "P_shot_subtracted_mpc_h3_grf",
            "ratio_clustering_grf_to_mucho",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda z: f"{z:.6g}",
    )
)

print()
print("Wrote:", OUTFILE)
