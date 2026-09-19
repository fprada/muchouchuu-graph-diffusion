import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
STEP6_DIR = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)

DS_CSV = STEP6_DIR / "spectral_dimension_sliding_bootstrap.csv"
DS_BOOT = STEP6_DIR / "bootstrap_spectral_dimension_curves.npy"

PK_PROXY_FILE = Path("muchouchuu_heatkernel_pk_proxy.csv")

OUTDIR = Path("pk_heatkernel_bootstrap")
OUTDIR.mkdir(exist_ok=True)

A_RMS = 10.409558

T_FIT_MIN = 896
T_FIT_MAX = 1792

DS_LIMIT = 3.03

RHOM_OBS = 446.1084358874493
RHOM_OBS_LO = 418.5554238785571
RHOM_OBS_HI = 486.91627649785516


# ---------------------------------------------------------------------
# Read strict-centred ds bootstrap curves
# ---------------------------------------------------------------------
tab = pd.read_csv(DS_CSV)

t = tab["diffusion_time"].to_numpy(dtype=float)
Rds = A_RMS * np.sqrt(t)

quality = tab["fit_quality_ok"].to_numpy(dtype=bool)

boot = np.load(DS_BOOT)

if boot.ndim != 2:
    raise RuntimeError(f"Expected 2D bootstrap array, got {boot.shape}")

if boot.shape[1] != len(t):
    raise RuntimeError(
        f"Bootstrap/time mismatch: {boot.shape} versus {len(t)} CSV times"
    )

nboot = boot.shape[0]


# ---------------------------------------------------------------------
# Read P(k) heat-kernel proxy
# ---------------------------------------------------------------------
pk = pd.read_csv(PK_PROXY_FILE)

Rp = pk["R_hmpc"].to_numpy(dtype=float)
X_clust_full = pk["X_clustering"].to_numpy(dtype=float)
X_raw_full = pk["X_raw"].to_numpy(dtype=float)
X_shot_full = pk["X_shot"].to_numpy(dtype=float)

# Interpolate proxy onto ds sampling radii
X_clust = np.interp(Rds, Rp, X_clust_full)
X_raw = np.interp(Rds, Rp, X_raw_full)
X_shot = np.interp(Rds, Rp, X_shot_full)


# ---------------------------------------------------------------------
# Fiducial pre-transition fit mask
# ---------------------------------------------------------------------
fit = (
    (t >= T_FIT_MIN)
    & (t <= T_FIT_MAX)
    & quality
    & np.isfinite(X_clust)
)

fit_idx = np.where(fit)[0]

if len(fit_idx) != 5:
    raise RuntimeError(
        f"Expected 5 fiducial fit points, found {len(fit_idx)}: "
        f"{t[fit_idx]}"
    )

print("=" * 72)
print("BOOTSTRAP P(k) HOMOGENEITY PROPAGATION")
print("=" * 72)
print("bootstrap realizations :", nboot)
print("fit times              :", t[fit_idx].astype(int))
print("fit radii              :", np.round(Rds[fit_idx], 3))
print()


# ---------------------------------------------------------------------
# Fit C realization by realization
#
# Model:
#       ds - 3 = C X_clustering
#
# We intentionally use the same unweighted fixed-intercept regression
# as the fiducial deterministic fit. Cross-time covariance is preserved
# because each complete bootstrap ds(t) curve is fitted as a unit.
# ---------------------------------------------------------------------
C_boot = np.full(nboot, np.nan)

for b in range(nboot):
    y = boot[b, fit_idx] - 3.0
    x = X_clust[fit_idx]

    good = np.isfinite(y) & np.isfinite(x)

    if np.sum(good) != len(fit_idx):
        continue

    denom = np.sum(x[good] ** 2)

    if denom <= 0:
        continue

    C_boot[b] = np.sum(x[good] * y[good]) / denom


# ---------------------------------------------------------------------
# Convert each C into a continuous ds=3.03 crossing
#
# We solve:
#       C X_clustering(R) = 0.03
#
# The proxy decreases with R over the relevant range.
# We sort by R and search directly for the first sign change.
# ---------------------------------------------------------------------
order = np.argsort(Rp)

Rgrid = Rp[order]
Xgrid = X_clust_full[order]
Xraw_grid = X_raw_full[order]
Xshot_grid = X_shot_full[order]

Rhom_boot = np.full(nboot, np.nan)
clust_frac_boot = np.full(nboot, np.nan)
shot_frac_boot = np.full(nboot, np.nan)

for b, C in enumerate(C_boot):

    if not np.isfinite(C) or C <= 0:
        continue

    f = C * Xgrid - (DS_LIMIT - 3.0)

    finite = np.isfinite(f) & np.isfinite(Rgrid)

    rr = Rgrid[finite]
    ff = f[finite]

    if len(rr) < 2:
        continue

    crossing = None

    for j in range(len(rr) - 1):
        if ff[j] == 0:
            crossing = rr[j]
            break

        if ff[j] * ff[j + 1] < 0:
            # Linear interpolation in R.
            crossing = (
                rr[j]
                + (0.0 - ff[j])
                * (rr[j + 1] - rr[j])
                / (ff[j + 1] - ff[j])
            )
            break

    if crossing is None:
        continue

    Rhom_boot[b] = crossing

    xraw = np.interp(crossing, Rgrid, Xraw_grid)
    xshot = np.interp(crossing, Rgrid, Xshot_grid)
    xclust = np.interp(crossing, Rgrid, Xgrid)

    denom = xclust + xshot

    if np.isfinite(denom) and denom != 0:
        clust_frac_boot[b] = xclust / denom
        shot_frac_boot[b] = xshot / denom


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
valid_C = np.isfinite(C_boot)
valid_R = np.isfinite(Rhom_boot)

C_good = C_boot[valid_C]
R_good = Rhom_boot[valid_R]

if len(R_good) == 0:
    raise RuntimeError("No valid bootstrap Rhom crossings obtained")


def qs(x):
    return np.percentile(x, [2.5, 16, 50, 84, 97.5])


C_q025, C_q16, C_med, C_q84, C_q975 = qs(C_good)
R_q025, R_q16, R_med, R_q84, R_q975 = qs(R_good)

cf_good = clust_frac_boot[np.isfinite(clust_frac_boot)]
sf_good = shot_frac_boot[np.isfinite(shot_frac_boot)]

cf_q = qs(cf_good)
sf_q = qs(sf_good)

inside_obs68 = np.mean(
    (R_good >= RHOM_OBS_LO)
    & (R_good <= RHOM_OBS_HI)
)

below_obs_median = np.mean(R_good < RHOM_OBS)

print("=" * 72)
print("BOOTSTRAP FIT AMPLITUDE")
print("=" * 72)
print(f"valid C realizations  : {valid_C.sum()} / {nboot}")
print(f"C median              : {C_med:.8e}")
print(f"C q16                 : {C_q16:.8e}")
print(f"C q84                 : {C_q84:.8e}")
print(
    "C = "
    f"{C_med:.6f} "
    f"+{C_q84 - C_med:.6f} "
    f"-{C_med - C_q16:.6f}"
)
print()

print("=" * 72)
print("BOOTSTRAP PREDICTED HOMOGENEITY SCALE")
print("=" * 72)
print(f"valid Rhom crossings  : {valid_R.sum()} / {nboot}")
print(f"success fraction      : {valid_R.mean():.6f}")
print(f"Rhom q025             : {R_q025:.3f}")
print(f"Rhom q16              : {R_q16:.3f}")
print(f"Rhom median           : {R_med:.3f}")
print(f"Rhom q84              : {R_q84:.3f}")
print(f"Rhom q975             : {R_q975:.3f}")
print(
    "Rhom_Pk = "
    f"{R_med:.1f} "
    f"+{R_q84 - R_med:.1f} "
    f"-{R_med - R_q16:.1f} h^-1 Mpc"
)
print()

print(f"Observed Rhom median  : {RHOM_OBS:.3f}")
print(
    f"Median prediction/obs : {R_med / RHOM_OBS:.4f}"
)
print(
    f"Median prediction-obs : {R_med - RHOM_OBS:.3f} h^-1 Mpc"
)
print(
    f"P(prediction in observed 68% interval) : "
    f"{inside_obs68:.4f}"
)
print(
    f"P(prediction below observed median)    : "
    f"{below_obs_median:.4f}"
)
print()

print("=" * 72)
print("AT THE PREDICTED CROSSING")
print("=" * 72)
print(
    "clustering fraction median/q16/q84 : "
    f"{cf_q[2]:.5f} "
    f"[{cf_q[1]:.5f}, {cf_q[3]:.5f}]"
)
print(
    "shot fraction median/q16/q84       : "
    f"{sf_q[2]:.5f} "
    f"[{sf_q[1]:.5f}, {sf_q[3]:.5f}]"
)


# ---------------------------------------------------------------------
# Save realization-level data
# ---------------------------------------------------------------------
out = pd.DataFrame({
    "bootstrap_index": np.arange(nboot),
    "C": C_boot,
    "Rhom_pk_hmpc": Rhom_boot,
    "clustering_fraction": clust_frac_boot,
    "shot_fraction": shot_frac_boot,
})

out.to_csv(
    OUTDIR / "bootstrap_pk_Rhom_samples.csv",
    index=False,
)

np.save(
    OUTDIR / "bootstrap_C_samples.npy",
    C_boot,
)

np.save(
    OUTDIR / "bootstrap_Rhom_pk_samples.npy",
    Rhom_boot,
)


summary = {
    "n_bootstrap": int(nboot),
    "n_valid_C": int(valid_C.sum()),
    "n_valid_Rhom": int(valid_R.sum()),
    "success_fraction": float(valid_R.mean()),
    "fit_t_min": int(T_FIT_MIN),
    "fit_t_max": int(T_FIT_MAX),
    "fit_times": [int(x) for x in t[fit_idx]],
    "C": {
        "q025": float(C_q025),
        "q16": float(C_q16),
        "median": float(C_med),
        "q84": float(C_q84),
        "q975": float(C_q975),
    },
    "Rhom_pk_hmpc": {
        "q025": float(R_q025),
        "q16": float(R_q16),
        "median": float(R_med),
        "q84": float(R_q84),
        "q975": float(R_q975),
    },
    "observed_Rhom_hmpc": {
        "median": float(RHOM_OBS),
        "q16": float(RHOM_OBS_LO),
        "q84": float(RHOM_OBS_HI),
    },
    "prob_prediction_inside_observed_68": float(inside_obs68),
    "prob_prediction_below_observed_median": float(below_obs_median),
    "clustering_fraction": {
        "q16": float(cf_q[1]),
        "median": float(cf_q[2]),
        "q84": float(cf_q[3]),
    },
    "shot_fraction": {
        "q16": float(sf_q[1]),
        "median": float(sf_q[2]),
        "q84": float(sf_q[3]),
    },
}

with open(
    OUTDIR / "bootstrap_pk_Rhom_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print()
print("Wrote:")
print(" ", OUTDIR / "bootstrap_pk_Rhom_samples.csv")
print(" ", OUTDIR / "bootstrap_C_samples.npy")
print(" ", OUTDIR / "bootstrap_Rhom_pk_samples.npy")
print(" ", OUTDIR / "bootstrap_pk_Rhom_summary.json")
