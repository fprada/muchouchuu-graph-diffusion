#!/usr/bin/env python3
"""
MuchoUchuu-only Step 8.

Combines:
  - Step 6 crossing summaries for windows 3, 5, and 7
  - Step 7 diffusion-length calibration
  - Step 6 spectral-dimension table for the fiducial window

Outputs:
  - muchouchuu_transport_summary.csv
  - muchouchuu_transport_summary.json
  - muchouchuu_transport_scale.png
  - muchouchuu_spectral_dimension_summary.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def qget(d: dict, key: str):
    q = d.get("bootstrap_crossing_time_quantiles") or {}
    return q.get(key)


def to_length(A: float, t):
    return None if t is None else float(A * math.sqrt(float(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window3-json", required=True)
    ap.add_argument("--window5-json", required=True)
    ap.add_argument("--window7-json", required=True)
    ap.add_argument("--window5-csv", required=True)
    ap.add_argument("--calibration-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fiducial-fraction", type=float, default=0.5)
    ap.add_argument("--density-homogeneity-scale", type=float, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    w3 = load_json(args.window3_json)
    w5 = load_json(args.window5_json)
    w7 = load_json(args.window7_json)
    cal = load_json(args.calibration_json)

    fkey = f"{args.fiducial_fraction:.3f}"
    fits = cal.get("fits", {})
    if fkey not in fits or "A_mpc_h_per_sqrt_step" not in fits[fkey]:
        raise ValueError(f"No valid calibration for fraction {fkey}")

    fid_fit = fits[fkey]
    A = float(fid_fit["A_mpc_h_per_sqrt_step"])
    Ase = float(fid_fit.get("A_standard_error_regression", np.nan))

    central_onsets = [
        w3.get("first_central_persistent_time"),
        w5.get("first_central_persistent_time"),
        w7.get("first_central_persistent_time"),
    ]
    successes = [
        w3.get("bootstrap_crossing_success_fraction"),
        w5.get("bootstrap_crossing_success_fraction"),
        w7.get("bootstrap_crossing_success_fraction"),
    ]

    # Fiducial statistical result comes from window 5.
    t_central = w5.get("first_central_persistent_time")
    t_q025 = qget(w5, "q025")
    t_q16 = qget(w5, "q16")
    t_med = qget(w5, "median")
    t_q84 = qget(w5, "q84")
    t_q975 = qget(w5, "q975")

    # Definition sensitivity from all available fractions.
    def_lengths = {}
    for key, fit in fits.items():
        if "A_mpc_h_per_sqrt_step" in fit and t_med is not None:
            def_lengths[key] = to_length(
                float(fit["A_mpc_h_per_sqrt_step"]), t_med
            )

    summary = {
        "simulation": "MuchoUchuu",
        "box_size_mpc_h": 6000.0,
        "number_of_objects": 108000000,
        "number_density_h3_mpc3": 5e-4,
        "graph_k": 12,
        "fiducial_step6_window_points": 5,
        "window_central_onsets": {
            "3": central_onsets[0],
            "5": central_onsets[1],
            "7": central_onsets[2],
        },
        "window_bootstrap_crossing_success_fractions": {
            "3": successes[0],
            "5": successes[1],
            "7": successes[2],
        },
        "smoothing_onset_min": min(central_onsets),
        "smoothing_onset_max": max(central_onsets),
        "crossing_time": {
            "central_onset": t_central,
            "q025": t_q025,
            "q16": t_q16,
            "median": t_med,
            "q84": t_q84,
            "q975": t_q975,
        },
        "fiducial_fraction": args.fiducial_fraction,
        "diffusion_length_calibration": {
            "A_mpc_h_per_sqrt_step": A,
            "A_standard_error_regression": Ase,
            "r2": fid_fit.get("r2"),
            "relative_rms_scatter": fid_fit.get("relative_rms_scatter"),
            "valid_times": fid_fit.get("valid_times"),
        },
        "transport_length_mpc_h": {
            "central_onset": to_length(A, t_central),
            "q025": to_length(A, t_q025),
            "q16": to_length(A, t_q16),
            "median": to_length(A, t_med),
            "q84": to_length(A, t_q84),
            "q975": to_length(A, t_q975),
        },
        "definition_sensitivity_length_at_median_time_mpc_h": def_lengths,
        "density_homogeneity_scale_mpc_h":
            args.density_homogeneity_scale,
    }

    if args.density_homogeneity_scale is not None and t_med is not None:
        summary["transport_to_density_scale_ratio"] = (
            to_length(A, t_med) / args.density_homogeneity_scale
        )

    (out / "muchouchuu_transport_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    row = {
        "simulation": "MuchoUchuu",
        "box_size_mpc_h": 6000.0,
        "number_of_objects": 108000000,
        "number_density_h3_mpc3": 5e-4,
        "graph_k": 12,
        "fiducial_window_points": 5,
        "central_onset_time": t_central,
        "crossing_time_median": t_med,
        "crossing_time_q16": t_q16,
        "crossing_time_q84": t_q84,
        "crossing_time_q025": t_q025,
        "crossing_time_q975": t_q975,
        "crossing_success_fraction_window3": successes[0],
        "crossing_success_fraction_window5": successes[1],
        "crossing_success_fraction_window7": successes[2],
        "A_fiducial_mpc_h_per_sqrt_step": A,
        "A_fiducial_regression_se": Ase,
        "calibration_r2": fid_fit.get("r2"),
        "transport_length_median_mpc_h": to_length(A, t_med),
        "transport_length_q16_mpc_h": to_length(A, t_q16),
        "transport_length_q84_mpc_h": to_length(A, t_q84),
        "transport_length_q025_mpc_h": to_length(A, t_q025),
        "transport_length_q975_mpc_h": to_length(A, t_q975),
        "definition_length_f0.4_mpc_h": def_lengths.get("0.400"),
        "definition_length_f0.5_mpc_h": def_lengths.get("0.500"),
        "definition_length_f0.6_mpc_h": def_lengths.get("0.600"),
        "density_homogeneity_scale_mpc_h":
            args.density_homogeneity_scale,
    }
    pd.DataFrame([row]).to_csv(
        out / "muchouchuu_transport_summary.csv", index=False
    )

    # Fiducial spectral-dimension plot.
    d = pd.read_csv(args.window5_csv)
    req = {
        "diffusion_time", "central_ds",
        "bootstrap_ds_q16", "bootstrap_ds_q84",
        "bootstrap_ds_q025", "bootstrap_ds_q975",
    }
    missing = req - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns in window-5 CSV: {sorted(missing)}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = d["diffusion_time"].to_numpy(float)
    ax.fill_between(
        x, d["bootstrap_ds_q025"], d["bootstrap_ds_q975"],
        alpha=0.18, label="95% bootstrap interval"
    )
    ax.fill_between(
        x, d["bootstrap_ds_q16"], d["bootstrap_ds_q84"],
        alpha=0.30, label="68% bootstrap interval"
    )
    ax.plot(x, d["central_ds"], marker="o", label="Window-5 central estimate")
    ax.axhline(3.0, linestyle="--", label=r"$d_s=3$")
    ax.axhspan(2.97, 3.03, alpha=0.10, label="1% tolerance band")
    if t_med is not None:
        ax.axvline(float(t_med), linestyle=":", label="Bootstrap median onset")
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time")
    ax.set_ylabel(r"Effective spectral dimension $d_s$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "muchouchuu_spectral_dimension_summary.png", dpi=180)
    plt.close(fig)

    # Transport-scale uncertainty plot.
    labels = ["2.5%", "16%", "median", "84%", "97.5%"]
    vals = [
        to_length(A, t_q025),
        to_length(A, t_q16),
        to_length(A, t_med),
        to_length(A, t_q84),
        to_length(A, t_q975),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(labels, vals, marker="o")
    ax.set_ylabel(r"Transport length [$h^{-1}$ Mpc]")
    ax.set_xlabel("Bootstrap crossing-time quantile")
    fig.tight_layout()
    fig.savefig(out / "muchouchuu_transport_scale.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nWrote:", out / "muchouchuu_transport_summary.csv")
    print("Wrote:", out / "muchouchuu_transport_summary.json")
    print("Wrote:", out / "muchouchuu_transport_scale.png")
    print("Wrote:", out / "muchouchuu_spectral_dimension_summary.png")


if __name__ == "__main__":
    main()
