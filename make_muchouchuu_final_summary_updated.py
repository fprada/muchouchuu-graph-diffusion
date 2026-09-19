#!/usr/bin/env python3
"""
Create the final MuchoUchuu s64 window-7 numerical summary.

This version propagates the regression uncertainty in the Fourier/RMS
calibration,

    R(t) = A_RMS * sqrt(t),

and reports the calibration contribution separately from the bootstrap
crossing-time uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_A_RMS = 10.409557615168367
DEFAULT_SIGMA_A_RMS = 0.0008030807269589261
DEFAULT_T_MAX = 16384


def radius(t: float, a_rms: float) -> float:
    return a_rms * math.sqrt(t)


def calibration_sigma_radius(t: float, sigma_a_rms: float) -> float:
    return math.sqrt(t) * sigma_a_rms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("muchouchuu_step6_window7_s64_interpolated_plateau"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("muchouchuu_final_summary_s64_window7"),
    )
    parser.add_argument("--a-rms", type=float, default=DEFAULT_A_RMS)
    parser.add_argument(
        "--sigma-a-rms",
        type=float,
        default=DEFAULT_SIGMA_A_RMS,
    )
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    args = parser.parse_args()

    crossing_path = args.analysis_dir / "crossing_summary.json"
    plateau_path = args.analysis_dir / "late_time_plateau_summary.json"

    crossing = json.loads(crossing_path.read_text())
    plateau = json.loads(plateau_path.read_text())

    cq = crossing["bootstrap_crossing_time_quantiles"]
    pq = plateau["bootstrap_plateau_quantiles"]
    sq = plateau["bootstrap_slope_quantiles"]

    r_q025 = radius(cq["q025"], args.a_rms)
    r_q16 = radius(cq["q16"], args.a_rms)
    r_med = radius(cq["median"], args.a_rms)
    r_q84 = radius(cq["q84"], args.a_rms)
    r_q975 = radius(cq["q975"], args.a_rms)
    r_max = radius(args.t_max, args.a_rms)

    r_med_cal_sigma = calibration_sigma_radius(
        cq["median"], args.sigma_a_rms
    )
    r_max_cal_sigma = calibration_sigma_radius(
        args.t_max, args.sigma_a_rms
    )

    rows = [
        {
            "quantity": "transport_homogeneity_time",
            "central_or_median": cq["median"],
            "q16": cq["q16"],
            "q84": cq["q84"],
            "q025": cq["q025"],
            "q975": cq["q975"],
            "units": "diffusion steps",
            "calibration_sigma": float("nan"),
        },
        {
            "quantity": "transport_homogeneity_scale",
            "central_or_median": r_med,
            "q16": r_q16,
            "q84": r_q84,
            "q025": r_q025,
            "q975": r_q975,
            "units": "h^-1 Mpc",
            "calibration_sigma": r_med_cal_sigma,
        },
        {
            "quantity": "late_time_plateau_ds",
            "central_or_median": pq["median"],
            "q16": pq["q16"],
            "q84": pq["q84"],
            "q025": pq["q025"],
            "q975": pq["q975"],
            "units": "dimensionless",
            "calibration_sigma": float("nan"),
        },
        {
            "quantity": "late_time_slope",
            "central_or_median": sq["median"],
            "q16": sq["q16"],
            "q84": sq["q84"],
            "q025": sq["q025"],
            "q975": sq["q975"],
            "units": "d(ds)/d(log t)",
            "calibration_sigma": float("nan"),
        },
    ]

    df = pd.DataFrame(rows)

    csv_path = args.output_prefix.with_suffix(".csv")
    txt_path = args.output_prefix.with_suffix(".txt")

    df.to_csv(csv_path, index=False)

    relative_calibration_uncertainty = (
        args.sigma_a_rms / args.a_rms
    )

    with txt_path.open("w") as f:
        f.write("MuchoUchuu final s64 window-7 summary\n\n")

        f.write(
            "Transport homogeneity time:\n"
            f"  median = {cq['median']:.6f}\n"
            f"  68% = [{cq['q16']:.6f}, {cq['q84']:.6f}]\n"
            f"  95% = [{cq['q025']:.6f}, {cq['q975']:.6f}]\n\n"
        )

        f.write(
            "Transport homogeneity scale:\n"
            f"  median = {r_med:.3f} h^-1 Mpc\n"
            f"  68% = [{r_q16:.3f}, {r_q84:.3f}] h^-1 Mpc\n"
            f"  95% = [{r_q025:.3f}, {r_q975:.3f}] h^-1 Mpc\n"
            f"  calibration sigma at median = "
            f"{r_med_cal_sigma:.3f} h^-1 Mpc\n\n"
        )

        f.write(
            "Late-time spectral-dimension plateau:\n"
            f"  t_min = {plateau['plateau_tmin']:.0f}\n"
            f"  number of late-time points = "
            f"{plateau['number_of_late_points']}\n"
            f"  median = {pq['median']:.6f}\n"
            f"  68% = [{pq['q16']:.6f}, {pq['q84']:.6f}]\n"
            f"  95% = [{pq['q025']:.6f}, {pq['q975']:.6f}]\n"
            f"  P(ds >= 3) = "
            f"{plateau['probability_plateau_ge_3']:.4f}\n\n"
        )

        f.write(
            "Late-time slope:\n"
            f"  median = {sq['median']:.6f}\n"
            f"  68% = [{sq['q16']:.6f}, {sq['q84']:.6f}]\n"
            f"  95% = [{sq['q025']:.6f}, {sq['q975']:.6f}]\n"
            f"  P(slope > 0) = "
            f"{plateau['probability_slope_gt_0']:.4f}\n\n"
        )

        f.write(
            "Calibration:\n"
            f"  A_RMS = {args.a_rms:.9f} +/- "
            f"{args.sigma_a_rms:.9f} "
            "h^-1 Mpc step^(-1/2)\n"
            f"  relative calibration uncertainty = "
            f"{relative_calibration_uncertainty:.3e}\n"
            f"  calibration contribution at R_hom = "
            f"{r_med_cal_sigma:.3f} h^-1 Mpc\n"
            f"  calibration contribution at R_max = "
            f"{r_max_cal_sigma:.3f} h^-1 Mpc\n"
            "  This is negligible relative to the crossing-time "
            "bootstrap uncertainty.\n\n"
        )

        f.write(
            f"Crossing success fraction = "
            f"{crossing['bootstrap_crossing_success_fraction']:.4f}\n"
        )
        f.write(
            f"Maximum tested RMS scale = {r_max:.3f} h^-1 Mpc\n"
        )
        f.write(
            f"Extent beyond median onset = "
            f"{r_max - r_med:.3f} h^-1 Mpc\n"
        )
        f.write(
            f"Maximum/onset ratio = {r_max / r_med:.4f}\n"
        )

    print(df.to_string(index=False))
    print("\nWrote:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
