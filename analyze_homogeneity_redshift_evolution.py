#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Inputs
# ============================================================

TRANSPORT = {
    "z0": {
        "z": 0.00,
        "crossings": Path(
            "muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/"
            "bootstrap_crossing_times_interpolated.npy"
        ),
        "A": 10.4095576,
    },
    "z049": {
        "z": 0.49,
        "crossings": Path(
            "growth_test/z049/"
            "step6_window7_s64_strictcentered_tmin3584_fixed/"
            "bootstrap_crossing_times_interpolated.npy"
        ),
        "A": 10.438294184707798,
    },
    "z103": {
        "z": 1.03,
        "crossings": Path(
            "growth_test/z103/"
            "step6_window7_s64_strictcentered_tmin3584_fixed/"
            "bootstrap_crossing_times_interpolated.npy"
        ),
        "A": 10.4894608905163,
    },
    "z203": {
        "z": 2.03,
        "crossings": Path(
            "growth_test/z203/"
            "step6_window7_s64_strictcentered_tmin3584_fixed/"
            "bootstrap_crossing_times_interpolated.npy"
        ),
        "A": 10.598993479525339,
    },
}


DENSITY = {
    "z0": {
        "z": 0.00,
        "crossings": Path(
            "muchouchuu_density_transport_ratio_bootstrap_1pct_final/"
            "density_crossing_radii_interpolated.npy"
        ),
    },
    "z049": {
        "z": 0.49,
        "crossings": Path(
            "growth_test/z049/"
            "muchouchuu_density_transport_ratio_bootstrap_1pct/"
            "density_crossing_radii_interpolated.npy"
        ),
    },
    "z103": {
        "z": 1.03,
        "crossings": Path(
            "growth_test/z103/"
            "muchouchuu_density_transport_ratio_bootstrap_1pct/"
            "density_crossing_radii_interpolated.npy"
        ),
    },
    "z203": {
        "z": 2.03,
        "crossings": Path(
            "growth_test/z203/"
            "muchouchuu_density_transport_ratio_bootstrap_1pct_10000/"
            "density_crossing_radii_interpolated.npy"
        ),
    },
}


OUT_JSON = Path("homogeneity_redshift_evolution_bootstrap_summary.json")
OUT_CSV = Path("homogeneity_redshift_evolution_bootstrap_summary.csv")


# ============================================================
# Utilities
# ============================================================

def qsummary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    q025, q16, q50, q84, q975 = np.percentile(
        x, [2.5, 16, 50, 84, 97.5]
    )

    return {
        "n": int(len(x)),
        "q025": float(q025),
        "q16": float(q16),
        "median": float(q50),
        "q84": float(q84),
        "q975": float(q975),
        "probability_gt_0": float(np.mean(x > 0.0)),
        "probability_lt_0": float(np.mean(x < 0.0)),
    }


def all_pairs_difference_cdf(delta, x, y):
    """
    CDF of D = x - y for every independent pair (x_i, y_j).

    Returns P(D <= delta) without explicitly constructing the
    len(x) x len(y) difference matrix.
    """
    x = np.asarray(x, dtype=float)
    y = np.sort(np.asarray(y, dtype=float))

    # x - y <= delta  <=>  y >= x - delta
    idx = np.searchsorted(
        y,
        x - delta,
        side="left",
    )

    count = np.sum(len(y) - idx, dtype=np.int64)
    return float(count / (len(x) * len(y)))


def all_pairs_difference_quantile(p, x, y):
    """
    Numerically invert the exact all-pairs difference CDF.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo = float(np.min(x) - np.max(y))
    hi = float(np.max(x) - np.min(y))

    for _ in range(80):
        mid = 0.5 * (lo + hi)

        if all_pairs_difference_cdf(mid, x, y) < p:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def independent_difference_summary(x, y):
    """
    Exact all-pairs independent-bootstrap comparison for D=x-y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    qs = {}
    for name, p in [
        ("q025", 0.025),
        ("q16", 0.16),
        ("median", 0.50),
        ("q84", 0.84),
        ("q975", 0.975),
    ]:
        qs[name] = float(
            all_pairs_difference_quantile(p, x, y)
        )

    # P(x-y < 0) = P(x < y)
    ys = np.sort(y)

    n_lt = 0
    n_gt = 0

    for value in x:
        # number y > x -> x-y < 0
        n_lt += len(ys) - np.searchsorted(
            ys, value, side="right"
        )

        # number y < x -> x-y > 0
        n_gt += np.searchsorted(
            ys, value, side="left"
        )

    npairs = len(x) * len(y)

    qs["n_x"] = int(len(x))
    qs["n_y"] = int(len(y))
    qs["number_of_independent_pairs"] = int(npairs)
    qs["probability_lt_0"] = float(n_lt / npairs)
    qs["probability_gt_0"] = float(n_gt / npairs)

    return qs


# ============================================================
# Transport: paired probe bootstrap
# ============================================================

t0 = np.load(
    TRANSPORT["z0"]["crossings"]
).astype(float)

A0 = TRANSPORT["z0"]["A"]

R0 = A0 * np.sqrt(t0)

transport_results = {}

for key in ["z049", "z103", "z203"]:

    tz = np.load(
        TRANSPORT[key]["crossings"]
    ).astype(float)

    if tz.shape != t0.shape:
        raise RuntimeError(
            f"{key}: transport bootstrap shape {tz.shape} "
            f"does not match z0 {t0.shape}"
        )

    Az = TRANSPORT[key]["A"]

    Rz = Az * np.sqrt(tz)

    valid = (
        np.isfinite(R0)
        & np.isfinite(Rz)
    )

    delta = Rz[valid] - R0[valid]
    frac = delta / R0[valid]

    transport_results[key] = {
        "z": TRANSPORT[key]["z"],
        "comparison": f"{key}-z0",
        "construction": (
            "paired probe bootstrap; identical bootstrap row "
            "corresponds to identical resampled global probe indices"
        ),
        "paired_valid_replicates": int(valid.sum()),
        "delta_R_mpc_h": qsummary(delta),
        "fractional_delta_R": qsummary(frac),
    }


# ============================================================
# Density: independent bootstrap draws
# ============================================================

rho0 = np.load(
    DENSITY["z0"]["crossings"]
).astype(float)

rho0 = rho0[np.isfinite(rho0)]

density_results = {}

for key in ["z049", "z103", "z203"]:

    rhoz = np.load(
        DENSITY[key]["crossings"]
    ).astype(float)

    rhoz = rhoz[np.isfinite(rhoz)]

    summary = independent_difference_summary(
        rhoz,
        rho0,
    )

    density_results[key] = {
        "z": DENSITY[key]["z"],
        "comparison": f"{key}-z0",
        "construction": (
            "all pairwise combinations of independent "
            "density bootstrap crossing radii"
        ),
        "delta_R_mpc_h": summary,
    }


# ============================================================
# Save combined summary
# ============================================================

result = {
    "transport": transport_results,
    "density": density_results,
}

OUT_JSON.write_text(
    json.dumps(result, indent=2) + "\n"
)


# Flat CSV summary
rows = []

for key in ["z049", "z103", "z203"]:

    t = transport_results[key]["delta_R_mpc_h"]

    rows.append({
        "observable": "transport",
        "z": transport_results[key]["z"],
        "comparison": f"{key}-z0",
        "n_or_pairs":
            transport_results[key]["paired_valid_replicates"],
        "q025": t["q025"],
        "q16": t["q16"],
        "median": t["median"],
        "q84": t["q84"],
        "q975": t["q975"],
        "probability_gt_0": t["probability_gt_0"],
        "probability_lt_0": t["probability_lt_0"],
    })

    d = density_results[key]["delta_R_mpc_h"]

    rows.append({
        "observable": "density",
        "z": density_results[key]["z"],
        "comparison": f"{key}-z0",
        "n_or_pairs":
            d["number_of_independent_pairs"],
        "q025": d["q025"],
        "q16": d["q16"],
        "median": d["median"],
        "q84": d["q84"],
        "q975": d["q975"],
        "probability_gt_0": d["probability_gt_0"],
        "probability_lt_0": d["probability_lt_0"],
    })


pd.DataFrame(rows).to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\nPAIRED TRANSPORT EVOLUTION")
print("==========================")

for key in ["z049", "z103", "z203"]:

    r = transport_results[key]
    q = r["delta_R_mpc_h"]
    f = r["fractional_delta_R"]

    print(f"\n{key} - z0")
    print(
        "paired valid =",
        r["paired_valid_replicates"]
    )
    print(
        "delta R median =",
        q["median"],
        "68% =",
        [q["q16"], q["q84"]],
        "95% =",
        [q["q025"], q["q975"]],
    )
    print(
        "P(delta R > 0) =",
        q["probability_gt_0"]
    )
    print(
        "fractional median =",
        f["median"],
        "68% =",
        [f["q16"], f["q84"]],
    )


print("\nINDEPENDENT DENSITY EVOLUTION")
print("=============================")

for key in ["z049", "z103", "z203"]:

    r = density_results[key]
    q = r["delta_R_mpc_h"]

    print(f"\n{key} - z0")
    print(
        "independent pairs =",
        q["number_of_independent_pairs"]
    )
    print(
        "delta R median =",
        q["median"],
        "68% =",
        [q["q16"], q["q84"]],
        "95% =",
        [q["q025"], q["q975"]],
    )
    print(
        "P(delta R < 0) =",
        q["probability_lt_0"]
    )


print()
print("Wrote:", OUT_JSON)
print("Wrote:", OUT_CSV)
