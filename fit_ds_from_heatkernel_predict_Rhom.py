import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DS_FILE = (
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584/"
    "spectral_dimension_sliding_bootstrap.csv"
)

PK_PROXY_FILE = "muchouchuu_heatkernel_pk_proxy.csv"

A_RMS = 10.409558

T_FIT_MIN = 512
T_FIT_MAX = 1280

DS_LIMIT = 3.03

RHOM_OBS = 446.1084358874493
RHOM_OBS_LO = 418.5554238785571
RHOM_OBS_HI = 486.91627649785516

ds = pd.read_csv(DS_FILE)

t = ds["diffusion_time"].to_numpy(dtype=float)
Rds = A_RMS * np.sqrt(t)

ds_med = ds["bootstrap_ds_median"].to_numpy(dtype=float)
ds_q16 = ds["bootstrap_ds_q16"].to_numpy(dtype=float)
ds_q84 = ds["bootstrap_ds_q84"].to_numpy(dtype=float)

quality = ds["fit_quality_ok"].to_numpy(dtype=bool)

pk = pd.read_csv(PK_PROXY_FILE)

Rp = pk["R_hmpc"].to_numpy(dtype=float)

X_clust_full = pk["X_clustering"].to_numpy(dtype=float)
X_raw_full = pk["X_raw"].to_numpy(dtype=float)
X_shot_full = pk["X_shot"].to_numpy(dtype=float)

X_clust = np.interp(Rds, Rp, X_clust_full)
X_raw = np.interp(Rds, Rp, X_raw_full)
X_shot = np.interp(Rds, Rp, X_shot_full)

fit = (
    (t >= T_FIT_MIN)
    & (t <= T_FIT_MAX)
    & quality
    & np.isfinite(ds_med)
    & np.isfinite(X_clust)
)

y = ds_med[fit] - 3.0
x = X_clust[fit]

C = np.sum(x * y) / np.sum(x * x)

pred_sampled = 3.0 + C * X_clust

resid = ds_med[fit] - pred_sampled[fit]

rmse = np.sqrt(np.mean(resid**2))

ss_res = np.sum(resid**2)
ss_tot = np.sum(
    (ds_med[fit] - np.mean(ds_med[fit]))**2
)

R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print("=" * 72)
print("PRE-HOMOGENEITY FIT")
print("=" * 72)

print("fit t range          :", T_FIT_MIN, "-", T_FIT_MAX)
print(
    "fit R range          :",
    f"{Rds[fit].min():.3f} - {Rds[fit].max():.3f} h^-1 Mpc"
)
print("number of fit points :", fit.sum())
print("C                    :", f"{C:.8e}")
print("fit RMSE             :", f"{rmse:.8e}")
print("fit R^2              :", f"{R2:.6f}")

print()
print("Fit points:")

for tt, rr, dd, xx, pp in zip(
    t[fit],
    Rds[fit],
    ds_med[fit],
    X_clust[fit],
    pred_sampled[fit]
):
    print(
        f"t={tt:7.0f} "
        f"R={rr:8.3f} "
        f"ds={dd:.6f} "
        f"X={xx:.8e} "
        f"pred={pp:.6f}"
    )

ds_pred_cont = 3.0 + C * X_clust_full

cross_cont = np.where(
    (ds_pred_cont[:-1] > DS_LIMIT)
    & (ds_pred_cont[1:] <= DS_LIMIT)
)[0]

if len(cross_cont):
    i = cross_cont[0]

    r0, r1 = Rp[i], Rp[i+1]
    d0, d1 = ds_pred_cont[i], ds_pred_cont[i+1]

    R_cross_cont = (
        r0
        + (DS_LIMIT - d0)
        * (r1 - r0)
        / (d1 - d0)
    )
else:
    R_cross_cont = np.nan

within = (
    np.abs(pred_sampled - 3.0) <= 0.03
) & quality

persistent_index = None

for i in range(len(t) - 2):
    if within[i] and within[i+1] and within[i+2]:
        persistent_index = i
        break

if persistent_index is not None:
    i = persistent_index

    if (
        i > 0
        and pred_sampled[i-1] > DS_LIMIT
        and pred_sampled[i] <= DS_LIMIT
    ):
        r0 = Rds[i-1]
        r1 = Rds[i]
        d0 = pred_sampled[i-1]
        d1 = pred_sampled[i]

        R_cross_persistent = (
            r0
            + (DS_LIMIT-d0)
            * (r1-r0)
            / (d1-d0)
        )
    else:
        R_cross_persistent = Rds[i]

    t_persistent = t[i]
else:
    R_cross_persistent = np.nan
    t_persistent = np.nan

print()
print("=" * 72)
print("PREDICTED HOMOGENEITY SCALE")
print("=" * 72)

print(
    "Continuous ds=3.03 crossing :",
    f"{R_cross_cont:.3f} h^-1 Mpc"
)

print(
    "Persistent-grid crossing    :",
    f"{R_cross_persistent:.3f} h^-1 Mpc"
)

print(
    "First persistent grid time  :",
    t_persistent
)

print()
print(
    "Observed bootstrap median   :",
    f"{RHOM_OBS:.3f} h^-1 Mpc"
)

if np.isfinite(R_cross_persistent):
    print(
        "Prediction / observed      :",
        f"{R_cross_persistent/RHOM_OBS:.4f}"
    )

    print(
        "Prediction - observed      :",
        f"{R_cross_persistent-RHOM_OBS:+.3f} h^-1 Mpc"
    )

if np.isfinite(R_cross_persistent):

    xcl = np.interp(
        R_cross_persistent,
        Rp,
        X_clust_full
    )

    xraw = np.interp(
        R_cross_persistent,
        Rp,
        X_raw_full
    )

    xshot = np.interp(
        R_cross_persistent,
        Rp,
        X_shot_full
    )

    print()
    print("At predicted crossing:")
    print(
        "  clustering fraction =",
        f"{xcl/xraw:.5f}"
    )
    print(
        "  shot fraction       =",
        f"{xshot/xraw:.5f}"
    )

comparison = pd.DataFrame({
    "diffusion_time": t,
    "R_hmpc": Rds,
    "ds_measured_median": ds_med,
    "ds_q16": ds_q16,
    "ds_q84": ds_q84,
    "X_clustering": X_clust,
    "X_raw": X_raw,
    "X_shot": X_shot,
    "ds_model": pred_sampled,
    "used_for_fit": fit,
    "model_within_1pct": within,
})

comparison.to_csv(
    "muchouchuu_ds_heatkernel_prediction_comparison.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(7.2, 5.0))

ax.fill_between(
    Rds,
    ds_q16,
    ds_q84,
    alpha=0.18,
    label="measured 68% bootstrap"
)

ax.plot(
    Rds,
    ds_med,
    marker="o",
    label="measured median $d_s$"
)

ax.plot(
    Rp,
    ds_pred_cont,
    linewidth=2.0,
    label=r"$3+C\,X_{\rm clustering}(R)$"
)

ax.scatter(
    Rds[fit],
    ds_med[fit],
    s=65,
    marker="s",
    label="points used to fit $C$"
)

ax.axhline(3.0, linestyle="--", linewidth=1.0)
ax.axhline(3.03, linestyle=":", linewidth=1.0)
ax.axhline(2.97, linestyle=":", linewidth=1.0)

ax.axvspan(
    RHOM_OBS_LO,
    RHOM_OBS_HI,
    alpha=0.10
)

ax.axvline(
    RHOM_OBS,
    linestyle="--",
    linewidth=1.2,
    label=r"observed $R_{\rm hom}$"
)

if np.isfinite(R_cross_persistent):
    ax.axvline(
        R_cross_persistent,
        linestyle="-.",
        linewidth=1.4,
        label=r"predicted $R_{\rm hom}$"
    )

ax.set_xlim(200, 900)
ax.set_ylim(2.95, 3.09)

ax.set_xlabel(
    r"$R\ [h^{-1}\,\mathrm{Mpc}]$"
)

ax.set_ylabel(r"$d_s$")

ax.legend(fontsize=8)

fig.tight_layout()

fig.savefig(
    "muchouchuu_ds_heatkernel_prediction.png",
    dpi=220
)

print()
print("Wrote:")
print("  muchouchuu_ds_pk_prediction_comparison.csv")
print("  muchouchuu_ds_pk_prediction.png")
