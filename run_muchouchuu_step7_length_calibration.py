#!/usr/bin/env python3
"""
Step 7: calibrate MuchoUchuu diffusion time to a physical length.

Definition
----------
For each diffusion time, estimate the large-separation plateau of the median
diffusion distance from the highest-distance bins. Define ell_f(t) as the
physical separation where the monotone median curve first reaches fraction f
of that plateau. Fit

    ell_f(t) = A_f * sqrt(t)

through the origin. The default f=0.5 is the fiducial calibration; additional
fractions provide a definition/systematic sensitivity check.

Important
---------
The q16/q84 columns in the metric-recovery table describe pair-to-pair scatter,
not the standard error of the binned median. This script therefore does not
misuse them as statistical errors. It reports regression scatter and
fraction-choice sensitivity instead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQ = {
    "diffusion_time",
    "distance_bin_center_mpc_h",
    "pair_count",
    "median_diffusion_distance",
}


def first_crossing(r: np.ndarray, y: np.ndarray, target: float) -> float:
    """Linear interpolation of first y >= target."""
    idx = np.flatnonzero(y >= target)
    if idx.size == 0:
        return math.nan
    j = int(idx[0])
    if j == 0:
        return float(r[0])
    x0, x1 = float(r[j - 1]), float(r[j])
    y0, y1 = float(y[j - 1]), float(y[j])
    if y1 <= y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def fit_origin(t: np.ndarray, ell: np.ndarray) -> dict:
    x = np.sqrt(t.astype(float))
    y = ell.astype(float)
    a = float(np.dot(x, y) / np.dot(x, x))
    pred = a * x
    resid = y - pred
    dof = max(len(y) - 1, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    se_a = float(np.sqrt(sigma2 / np.dot(x, x)))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else math.nan
    rel_rms = float(np.sqrt(np.mean((resid / y) ** 2)))
    return {
        "A_mpc_h_per_sqrt_step": a,
        "A_standard_error_regression": se_a,
        "r2": r2,
        "relative_rms_scatter": rel_rms,
        "n_times": int(len(y)),
    }


def convert_crossing(crossing_json: str, A: float) -> dict:
    d = json.loads(Path(crossing_json).read_text())
    q = d.get("bootstrap_crossing_time_quantiles") or {}

    def cv(v):
        return None if v is None else float(A * math.sqrt(float(v)))

    return {
        "source": str(Path(crossing_json).resolve()),
        "A_used_mpc_h_per_sqrt_step": A,
        "first_central_persistent_time": d.get("first_central_persistent_time"),
        "first_central_persistent_length_mpc_h": cv(
            d.get("first_central_persistent_time")
        ),
        "bootstrap_crossing_success_fraction":
            d.get("bootstrap_crossing_success_fraction"),
        "bootstrap_time_quantiles": q,
        "bootstrap_length_quantiles_mpc_h": {
            k: cv(v) for k, v in q.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("curve_csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--fractions", nargs="+", type=float, default=[0.4, 0.5, 0.6],
        help="Fractions of the large-distance plateau; 0.5 is fiducial."
    )
    ap.add_argument("--fiducial-fraction", type=float, default=0.5)
    ap.add_argument(
        "--tail-fraction", type=float, default=0.20,
        help="Highest fraction of distance bins used to estimate the plateau."
    )
    ap.add_argument(
        "--max-tail-rise", type=float, default=0.15,
        help="Maximum fractional rise across the tail for accepting a plateau."
    )
    ap.add_argument(
        "--min-pairs-per-bin", type=int, default=100,
        help="Discard bins with fewer sampled pairs."
    )
    ap.add_argument(
        "--min-times", type=int, default=3,
        help="Minimum valid diffusion times required for a fit."
    )
    ap.add_argument(
        "--crossing-json", default=None,
        help="Optional Step-6 crossing_summary.json to convert into Mpc/h."
    )
    args = ap.parse_args()

    if not (0 < args.tail_fraction < 1):
        raise ValueError("--tail-fraction must be between 0 and 1")
    if any(not (0 < f < 1) for f in args.fractions):
        raise ValueError("Every plateau fraction must be between 0 and 1")
    if args.fiducial_fraction not in args.fractions:
        raise ValueError("--fiducial-fraction must be included in --fractions")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.curve_csv)
    missing = REQ - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "graph" in df.columns and df["graph"].nunique() > 1:
        raise ValueError("Input contains multiple graphs; provide one graph at a time")

    rows = []
    for t, g in df.groupby("diffusion_time", sort=True):
        g = g[g["pair_count"] >= args.min_pairs_per_bin].copy()
        g = g.sort_values("distance_bin_center_mpc_h")
        r = g["distance_bin_center_mpc_h"].to_numpy(float)
        yraw = g["median_diffusion_distance"].to_numpy(float)
        finite = np.isfinite(r) & np.isfinite(yraw)
        r, yraw = r[finite], yraw[finite]
        if len(r) < 8:
            continue

        # Simple monotone regularization appropriate for a saturating median curve.
        y = np.maximum.accumulate(yraw)

        ntail = max(4, int(math.ceil(args.tail_fraction * len(r))))
        tail_y = y[-ntail:]
        plateau = float(np.median(tail_y))
        tail_rise = float((tail_y[-1] - tail_y[0]) / plateau) if plateau > 0 else math.inf
        plateau_ok = bool(
            np.isfinite(plateau) and plateau > 0 and
            tail_rise <= args.max_tail_rise
        )

        base = {
            "diffusion_time": int(t),
            "sqrt_time": float(math.sqrt(float(t))),
            "n_bins": int(len(r)),
            "max_distance_mpc_h": float(r[-1]),
            "plateau_estimate": plateau,
            "tail_fraction": args.tail_fraction,
            "tail_bins": ntail,
            "tail_fractional_rise": tail_rise,
            "plateau_accepted": plateau_ok,
        }
        for f in args.fractions:
            target = f * plateau
            ell = first_crossing(r, y, target) if plateau_ok else math.nan
            # Require crossing to occur before the plateau-estimation tail.
            if np.isfinite(ell) and ell >= r[-ntail]:
                ell = math.nan
            base[f"ell_f{f:.3f}_mpc_h"] = ell
            base[f"target_f{f:.3f}"] = target
        rows.append(base)

    cal = pd.DataFrame(rows)
    cal.to_csv(out / "muchouchuu_diffusion_length_calibration.csv", index=False)

    fits = {}
    for f in args.fractions:
        col = f"ell_f{f:.3f}_mpc_h"
        good = cal["plateau_accepted"] & np.isfinite(cal[col])
        sub = cal.loc[good, ["diffusion_time", col]]
        if len(sub) < args.min_times:
            fits[f"{f:.3f}"] = {
                "error": f"Only {len(sub)} valid times; need {args.min_times}"
            }
            continue
        fit = fit_origin(
            sub["diffusion_time"].to_numpy(float),
            sub[col].to_numpy(float),
        )
        fit["fraction"] = f
        fit["valid_times"] = sub["diffusion_time"].astype(int).tolist()
        fits[f"{f:.3f}"] = fit

    fidkey = f"{args.fiducial_fraction:.3f}"
    if "A_mpc_h_per_sqrt_step" not in fits.get(fidkey, {}):
        raise RuntimeError(
            "Fiducial calibration failed. Extend the metric-recovery curve to "
            "larger separations or relax plateau checks only after inspection."
        )

    A = float(fits[fidkey]["A_mpc_h_per_sqrt_step"])
    summary = {
        "curve_csv": str(Path(args.curve_csv).resolve()),
        "definition": "ell_f is first separation reaching fraction f of tail plateau",
        "fiducial_fraction": args.fiducial_fraction,
        "tail_fraction": args.tail_fraction,
        "maximum_tail_fractional_rise": args.max_tail_rise,
        "minimum_pairs_per_bin": args.min_pairs_per_bin,
        "fits": fits,
    }

    valid_As = [
        v["A_mpc_h_per_sqrt_step"]
        for v in fits.values()
        if isinstance(v, dict) and "A_mpc_h_per_sqrt_step" in v
    ]
    if valid_As:
        summary["definition_sensitivity_A_min"] = float(min(valid_As))
        summary["definition_sensitivity_A_max"] = float(max(valid_As))

    if args.crossing_json:
        summary["physical_crossing"] = convert_crossing(args.crossing_json, A)

    (out / "muchouchuu_diffusion_length_fit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    import matplotlib.pyplot as plt

    # Calibration plot.
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    xgrid = np.linspace(0, math.sqrt(cal["diffusion_time"].max()) * 1.05, 300)
    for f in args.fractions:
        col = f"ell_f{f:.3f}_mpc_h"
        good = cal["plateau_accepted"] & np.isfinite(cal[col])
        ax.scatter(
            cal.loc[good, "sqrt_time"], cal.loc[good, col],
            label=f"f={f:g} valid points"
        )
        fit = fits.get(f"{f:.3f}", {})
        if "A_mpc_h_per_sqrt_step" in fit:
            ax.plot(
                xgrid, fit["A_mpc_h_per_sqrt_step"] * xgrid,
                label=f"f={f:g}: A={fit['A_mpc_h_per_sqrt_step']:.3f}"
            )
    ax.set_xlabel(r"$\sqrt{t}$")
    ax.set_ylabel(r"Characteristic length $\ell_f(t)$ [$h^{-1}$ Mpc]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "muchouchuu_diffusion_length_calibration.png", dpi=180)
    plt.close(fig)

    # Plateau-quality plot.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        cal["diffusion_time"], cal["tail_fractional_rise"],
        marker="o"
    )
    ax.axhline(args.max_tail_rise, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time")
    ax.set_ylabel("Fractional rise across plateau-estimation tail")
    fig.tight_layout()
    fig.savefig(out / "muchouchuu_plateau_quality.png", dpi=180)
    plt.close(fig)

    print(cal.to_string(index=False))
    print("\nFiducial A =", A, "h^-1 Mpc / sqrt(step)")
    print("Wrote:", out / "muchouchuu_diffusion_length_calibration.csv")
    print("Wrote:", out / "muchouchuu_diffusion_length_fit.json")
    print("Wrote:", out / "muchouchuu_diffusion_length_calibration.png")
    print("Wrote:", out / "muchouchuu_plateau_quality.png")


if __name__ == "__main__":
    main()
