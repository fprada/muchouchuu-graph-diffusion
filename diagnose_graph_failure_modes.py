from pathlib import Path
import numpy as np
import pandas as pd

STEP6 = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)

TAB = pd.read_csv(
    STEP6 / "spectral_dimension_sliding_bootstrap.csv"
)
BOOT = np.load(
    STEP6 / "bootstrap_spectral_dimension_curves.npy"
)
TCROSS = np.load(
    STEP6 / "bootstrap_crossing_times_interpolated.npy"
)

t = TAB["diffusion_time"].to_numpy(float)
quality = TAB["fit_quality_ok"].to_numpy(bool)

# Same homogeneity criterion
DS_LIMIT_LO = 2.97
DS_LIMIT_HI = 3.03
PERSISTENCE = 3

valid_centres = (
    quality
    & np.isfinite(TAB["bootstrap_ds_median"].to_numpy(float))
)

idx = np.where(valid_centres)[0]

fail = ~np.isfinite(TCROSS)
succ = np.isfinite(TCROSS)

print("=" * 72)
print("GRAPH FAILURE-MODE DIAGNOSTIC")
print("=" * 72)
print("total             :", len(TCROSS))
print("successes         :", succ.sum())
print("failures          :", fail.sum())
print("valid time centres:", t[idx].astype(int))
print()

counts = {
    "never_enters_band": 0,
    "enters_but_not_persistent": 0,
    "persistent_found_recheck": 0,
    "insufficient_finite_values": 0,
}

last_ds = []
min_abs3 = []
first_band_time = []

for b in np.where(fail)[0]:
    y = BOOT[b, idx]
    tt = t[idx]

    finite = np.isfinite(y)

    if finite.sum() < PERSISTENCE:
        counts["insufficient_finite_values"] += 1
        last_ds.append(np.nan)
        min_abs3.append(np.nan)
        first_band_time.append(np.nan)
        continue

    yf = y[finite]
    tf = tt[finite]

    last_ds.append(yf[-1])
    min_abs3.append(np.min(np.abs(yf - 3.0)))

    inside = (yf >= DS_LIMIT_LO) & (yf <= DS_LIMIT_HI)

    if not np.any(inside):
        counts["never_enters_band"] += 1
        first_band_time.append(np.nan)
        continue

    first_band_time.append(tf[np.where(inside)[0][0]])

    found = False
    for j in range(len(inside) - PERSISTENCE + 1):
        if np.all(inside[j:j + PERSISTENCE]):
            found = True
            break

    if found:
        counts["persistent_found_recheck"] += 1
    else:
        counts["enters_but_not_persistent"] += 1

print("Failure classification:")
for k, v in counts.items():
    print(f"  {k:28s}: {v}")
print()

last_ds = np.asarray(last_ds, float)
min_abs3 = np.asarray(min_abs3, float)
first_band_time = np.asarray(first_band_time, float)

def q(x):
    x = x[np.isfinite(x)]
    return np.percentile(x, [16, 50, 84]) if len(x) else [np.nan]*3

qlast = q(last_ds)
qmin = q(min_abs3)
qfirst = q(first_band_time)

print("Failed realizations:")
print(
    "  final available ds q16/med/q84 = "
    f"{qlast[0]:.5f} {qlast[1]:.5f} {qlast[2]:.5f}"
)
print(
    "  min |ds-3| q16/med/q84        = "
    f"{qmin[0]:.5f} {qmin[1]:.5f} {qmin[2]:.5f}"
)
print(
    "  first band-entry time q16/med/q84 = "
    f"{qfirst[0]:.1f} {qfirst[1]:.1f} {qfirst[2]:.1f}"
)

out = pd.DataFrame({
    "bootstrap_index": np.where(fail)[0],
    "last_ds": last_ds,
    "min_abs_ds_minus_3": min_abs3,
    "first_band_time": first_band_time,
})

out.to_csv(
    STEP6 / "failed_crossing_mode_diagnostic.csv",
    index=False
)

print()
print(
    "Wrote:",
    STEP6 / "failed_crossing_mode_diagnostic.csv"
)
