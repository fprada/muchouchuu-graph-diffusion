from pathlib import Path
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
STEP6 = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)

TAB_FILE = STEP6 / "spectral_dimension_sliding_bootstrap.csv"
BOOT_FILE = STEP6 / "bootstrap_spectral_dimension_curves.npy"

OUTDIR = Path("graph_crossing_tmax_robustness")
OUTDIR.mkdir(exist_ok=True)

A_RMS = 10.409558
TOLERANCE = 0.03
PERSISTENCE = 3

TMAX_LIST = [
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
    10240,
]


# ---------------------------------------------------------------------
# Exact interpolation used by the production crossing code
# ---------------------------------------------------------------------
def interpolate_zero_logtime(
    t_left,
    t_right,
    margin_left,
    margin_right,
):
    if not (
        np.isfinite(t_left)
        and np.isfinite(t_right)
        and np.isfinite(margin_left)
        and np.isfinite(margin_right)
    ):
        return np.nan

    if t_left <= 0.0 or t_right <= t_left:
        return np.nan

    if not (margin_left < 0.0 <= margin_right):
        return np.nan

    denom = margin_right - margin_left

    if denom <= 0.0:
        return np.nan

    frac = float(
        np.clip(
            -margin_left / denom,
            0.0,
            1.0,
        )
    )

    return float(
        np.exp(
            np.log(t_left)
            + frac * (
                np.log(t_right)
                - np.log(t_left)
            )
        )
    )


def first_persistent_time_interpolated(
    margin,
    times,
    auxiliary_ok,
    persistence,
):
    """
    Exact logic used by the production strict-centred crossing code.
    """
    margin = np.asarray(margin, dtype=float)
    times = np.asarray(times, dtype=float)
    auxiliary_ok = np.asarray(auxiliary_ok, dtype=bool)

    if not (
        len(margin)
        == len(times)
        == len(auxiliary_ok)
    ):
        raise ValueError(
            "margin, times, and auxiliary_ok must have equal length"
        )

    persistence = int(persistence)

    passes = (
        (margin >= 0.0)
        & auxiliary_ok
    )

    for j in range(len(times)):

        stop = j + persistence

        if stop > len(times):
            break

        if not np.all(passes[j:stop]):
            continue

        if j == 0:
            return float(times[j])

        if (
            auxiliary_ok[j - 1]
            and auxiliary_ok[j]
            and margin[j - 1] < 0.0 <= margin[j]
        ):
            t_cross = interpolate_zero_logtime(
                times[j - 1],
                times[j],
                margin[j - 1],
                margin[j],
            )

            if np.isfinite(t_cross):
                return t_cross

        return float(times[j])

    return math.nan


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
tab = pd.read_csv(TAB_FILE)
boot = np.load(BOOT_FILE)

times_full = tab["diffusion_time"].to_numpy(float)
quality_full = tab["fit_quality_ok"].to_numpy(bool)

if boot.ndim != 2:
    raise RuntimeError(
        f"Expected 2D bootstrap array; got {boot.shape}"
    )

if boot.shape[1] != len(times_full):
    raise RuntimeError(
        f"Bootstrap/time mismatch: "
        f"{boot.shape} vs {len(times_full)} times"
    )

nboot = boot.shape[0]


# ---------------------------------------------------------------------
# Truncation experiment
# ---------------------------------------------------------------------
summary_rows = []
sample_rows = []

print("=" * 78)
print("STRICT-CENTRED GRAPH CROSSING: ARTIFICIAL TMAX TRUNCATION")
print("=" * 78)
print("bootstrap realizations :", nboot)
print("persistence            :", PERSISTENCE)
print("tolerance              :", TOLERANCE)
print()


for tmax in TMAX_LIST:

    keep = times_full <= tmax

    times = times_full[keep]
    quality = quality_full[keep]
    ds_boot = boot[:, keep]

    crossing_t = np.full(
        nboot,
        np.nan,
        dtype=float,
    )

    for b in range(nboot):

        margin = (
            TOLERANCE
            - np.abs(ds_boot[b] - 3.0)
        )

        crossing_t[b] = (
            first_persistent_time_interpolated(
                margin=margin,
                times=times,
                auxiliary_ok=quality,
                persistence=PERSISTENCE,
            )
        )

    detected = np.isfinite(crossing_t)
    valid_t = crossing_t[detected]

    crossing_R = (
        A_RMS * np.sqrt(crossing_t)
    )

    valid_R = crossing_R[detected]

    success = detected.mean()

    if len(valid_R) > 0:
        q025, q16, med, q84, q975 = (
            np.percentile(
                valid_R,
                [2.5, 16, 50, 84, 97.5],
            )
        )
    else:
        q025 = q16 = med = q84 = q975 = np.nan

    # Latest possible START of a 3-point persistent run
    # among strict-centred quality-valid times available at this tmax.
    valid_quality_times = times[quality]

    if len(valid_quality_times) >= PERSISTENCE:
        latest_detectable_start = (
            valid_quality_times[-PERSISTENCE]
        )
        latest_detectable_R = (
            A_RMS
            * np.sqrt(latest_detectable_start)
        )
    else:
        latest_detectable_start = np.nan
        latest_detectable_R = np.nan

    summary_rows.append({
        "tmax": int(tmax),
        "n_sampled_times": int(len(times)),
        "n_quality_times": int(np.sum(quality)),
        "latest_detectable_start_t":
            latest_detectable_start,
        "latest_detectable_start_R_hmpc":
            latest_detectable_R,
        "n_detected": int(detected.sum()),
        "n_failed": int((~detected).sum()),
        "success_fraction": float(success),
        "Rhom_q025_hmpc": q025,
        "Rhom_q16_hmpc": q16,
        "Rhom_median_hmpc": med,
        "Rhom_q84_hmpc": q84,
        "Rhom_q975_hmpc": q975,
    })

    for b in range(nboot):
        sample_rows.append({
            "tmax": int(tmax),
            "bootstrap_index": b,
            "crossing_detected":
                bool(detected[b]),
            "crossing_time":
                crossing_t[b],
            "Rhom_hmpc":
                crossing_R[b],
        })

    print("-" * 78)
    print(f"TMAX = {tmax}")
    print(
        f"quality-valid sampled times : "
        f"{int(np.sum(quality))}"
    )
    print(
        f"latest detectable start     : "
        f"t={latest_detectable_start:.0f}, "
        f"R={latest_detectable_R:.3f} h^-1 Mpc"
    )
    print(
        f"crossings                   : "
        f"{detected.sum()} / {nboot}"
    )
    print(
        f"success fraction            : "
        f"{success:.4f}"
    )

    if len(valid_R):
        print(
            f"Rhom detected median        : "
            f"{med:.3f} "
            f"+{q84-med:.3f} "
            f"-{med-q16:.3f} h^-1 Mpc"
        )
        print(
            f"Rhom detected 95% interval  : "
            f"[{q025:.3f}, {q975:.3f}]"
        )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
summary = pd.DataFrame(summary_rows)
samples = pd.DataFrame(sample_rows)

summary.to_csv(
    OUTDIR / "graph_crossing_tmax_summary.csv",
    index=False,
)

samples.to_csv(
    OUTDIR / "graph_crossing_tmax_samples.csv",
    index=False,
)

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    summary[
        [
            "tmax",
            "latest_detectable_start_t",
            "success_fraction",
            "Rhom_q16_hmpc",
            "Rhom_median_hmpc",
            "Rhom_q84_hmpc",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print()
print("Wrote:")
print(
    " ",
    OUTDIR / "graph_crossing_tmax_summary.csv",
)
print(
    " ",
    OUTDIR / "graph_crossing_tmax_samples.csv",
)
