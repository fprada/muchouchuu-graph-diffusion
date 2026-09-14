#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crossing_json", type=Path)
    ap.add_argument(
        "--calibration-json",
        type=Path,
        default=Path(
            "muchouchuu_fourier_calibration/"
            "fourier_diffusion_calibration.json"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "muchouchuu_interpolated_transport_uncertainty_1pct.json"
        ),
    )
    args = ap.parse_args()

    crossing = json.loads(args.crossing_json.read_text())
    calibration = json.loads(args.calibration_json.read_text())

    A = float(calibration["A_RMS_mpc_h_per_sqrt_step"])
    sigma_A = float(
        calibration.get("A_RMS_standard_error_regression", 0.0)
    )
    q = crossing["bootstrap_crossing_time_quantiles"]

    def radius(t):
        return A * math.sqrt(float(t))

    radii = {key: radius(value) for key, value in q.items()}
    r50 = radii["median"]

    result = {
        "simulation": "MuchoUchuu",
        "crossing_tolerance": crossing.get("tolerance", 0.01),
        "interpolation": crossing.get(
            "crossing_time_interpolation",
            "log_time_linear_margin",
        ),
        "A_RMS_mpc_h_per_sqrt_step": A,
        "A_RMS_standard_error_regression": sigma_A,
        "bootstrap_crossing_success_fraction":
            crossing["bootstrap_crossing_success_fraction"],
        "crossing_time_quantiles_interpolated_steps": q,
        "transport_length_quantiles_mpc_h": radii,
        "transport_68_percent": {
            "median": r50,
            "minus": r50 - radii["q16"],
            "plus": radii["q84"] - r50,
            "interval": [radii["q16"], radii["q84"]],
        },
        "transport_95_percent": {
            "median": r50,
            "minus": r50 - radii["q025"],
            "plus": radii["q975"] - r50,
            "interval": [radii["q025"], radii["q975"]],
        },
        "calibration_error_at_median_mpc_h":
            math.sqrt(float(q["median"])) * sigma_A,
    }

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
