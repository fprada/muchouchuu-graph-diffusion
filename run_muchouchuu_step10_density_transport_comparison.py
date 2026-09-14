#!/usr/bin/env python3
"""
MuchoUchuu Step 10: final density-versus-transport comparison.

Inputs
------
1. Step 8 transport summary JSON
2. Step 9 density-homogeneity summary JSON

Outputs
-------
- muchouchuu_density_vs_transport_summary.csv
- muchouchuu_density_vs_transport_summary.json
- muchouchuu_density_vs_transport_comparison.png
- muchouchuu_density_vs_transport_intervals.png
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


def require(mapping: dict, path: list[str]):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing JSON field: " + ".".join(path))
        cur = cur[key]
    return cur


def ratio_interval_independent(
    t_q025, t_q16, t_med, t_q84, t_q975,
    d_q025, d_q16, d_med, d_q84, d_q975,
):
    # Conservative quantile-ratio bounds, without assuming covariance.
    return {
        "q025_conservative": t_q025 / d_q975,
        "q16_conservative": t_q16 / d_q84,
        "median_ratio": t_med / d_med,
        "q84_conservative": t_q84 / d_q16,
        "q975_conservative": t_q975 / d_q025,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport-json", required=True)
    ap.add_argument("--density-json", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    transport = load_json(args.transport_json)
    density = load_json(args.density_json)

    tlen = require(transport, ["transport_length_mpc_h"])
    dquant = require(density, ["bootstrap_crossing_quantiles_mpc_h"])

    t_med = float(require(tlen, ["median"]))
    t_q16 = float(require(tlen, ["q16"]))
    t_q84 = float(require(tlen, ["q84"]))
    t_q025 = float(require(tlen, ["q025"]))
    t_q975 = float(require(tlen, ["q975"]))

    d_med = float(require(dquant, ["median"]))
    d_q16 = float(require(dquant, ["q16"]))
    d_q84 = float(require(dquant, ["q84"]))
    d_q025 = float(require(dquant, ["q025"]))
    d_q975 = float(require(dquant, ["q975"]))

    ratio = ratio_interval_independent(
        t_q025, t_q16, t_med, t_q84, t_q975,
        d_q025, d_q16, d_med, d_q84, d_q975,
    )

    definition = require(
        transport,
        ["definition_sensitivity_length_at_median_time_mpc_h"],
    )

    summary = {
        "simulation": "MuchoUchuu",
        "transport": {
            "fiducial_definition_fraction":
                require(transport, ["fiducial_fraction"]),
            "median_mpc_h": t_med,
            "q16_mpc_h": t_q16,
            "q84_mpc_h": t_q84,
            "q025_mpc_h": t_q025,
            "q975_mpc_h": t_q975,
            "definition_sensitivity_mpc_h": definition,
        },
        "density": {
            "central_first_persistent_radius_mpc_h":
                require(
                    density,
                    ["central_first_persistent_radius_mpc_h"],
                ),
            "median_mpc_h": d_med,
            "q16_mpc_h": d_q16,
            "q84_mpc_h": d_q84,
            "q025_mpc_h": d_q025,
            "q975_mpc_h": d_q975,
            "bootstrap_crossing_success_fraction":
                require(
                    density,
                    ["bootstrap_crossing_success_fraction"],
                ),
        },
        "transport_to_density_ratio": ratio,
        "difference_mpc_h": {
            "median": t_med - d_med,
            "conservative_q16": t_q16 - d_q84,
            "conservative_q84": t_q84 - d_q16,
            "conservative_q025": t_q025 - d_q975,
            "conservative_q975": t_q975 - d_q025,
        },
        "interpretation": (
            "The density and transport scales are operationally different "
            "estimators. The ratio quantile bounds are conservative because "
            "the two bootstrap distributions were not jointly resampled."
        ),
        "source_files": {
            "transport_json": str(Path(args.transport_json).resolve()),
            "density_json": str(Path(args.density_json).resolve()),
        },
    }

    (out / "muchouchuu_density_vs_transport_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    rows = [
        {
            "quantity": "density_homogeneity",
            "median_mpc_h": d_med,
            "q16_mpc_h": d_q16,
            "q84_mpc_h": d_q84,
            "q025_mpc_h": d_q025,
            "q975_mpc_h": d_q975,
            "definition": (
                "first persistent radius satisfying scaled-count and D2 "
                "tolerances"
            ),
        },
        {
            "quantity": "transport_homogeneity",
            "median_mpc_h": t_med,
            "q16_mpc_h": t_q16,
            "q84_mpc_h": t_q84,
            "q025_mpc_h": t_q025,
            "q975_mpc_h": t_q975,
            "definition": "f=0.5 diffusion-length calibration",
        },
    ]
    pd.DataFrame(rows).to_csv(
        out / "muchouchuu_density_vs_transport_summary.csv",
        index=False,
    )

    import matplotlib.pyplot as plt

    # Horizontal interval comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["Density homogeneity", "Transport homogeneity"]
    med = np.array([d_med, t_med], dtype=float)
    lo68 = med - np.array([d_q16, t_q16], dtype=float)
    hi68 = np.array([d_q84, t_q84], dtype=float) - med
    lo95 = med - np.array([d_q025, t_q025], dtype=float)
    hi95 = np.array([d_q975, t_q975], dtype=float) - med
    y = np.arange(2)

    ax.errorbar(
        med, y, xerr=np.vstack([lo95, hi95]),
        fmt="none", capsize=5, label="95% interval"
    )
    ax.errorbar(
        med, y, xerr=np.vstack([lo68, hi68]),
        fmt="o", capsize=5, label="68% interval"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Scale [$h^{-1}$ Mpc]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out / "muchouchuu_density_vs_transport_intervals.png",
        dpi=180,
    )
    plt.close(fig)

    # Scale comparison with definition sensitivity.
    f04 = float(definition.get("0.400", np.nan))
    f05 = float(definition.get("0.500", np.nan))
    f06 = float(definition.get("0.600", np.nan))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        ["Density", "Transport f=0.5"],
        [d_med, t_med],
    )
    if np.isfinite(f04) and np.isfinite(f06):
        ax.errorbar(
            [1],
            [f05],
            yerr=[[f05 - f04], [f06 - f05]],
            fmt="none",
            capsize=6,
            label="Transport definition sensitivity f=0.4–0.6",
        )
    ax.set_ylabel(r"Scale [$h^{-1}$ Mpc]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out / "muchouchuu_density_vs_transport_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nWrote:", out / "muchouchuu_density_vs_transport_summary.csv")
    print("Wrote:", out / "muchouchuu_density_vs_transport_summary.json")
    print("Wrote:", out / "muchouchuu_density_vs_transport_comparison.png")
    print("Wrote:", out / "muchouchuu_density_vs_transport_intervals.png")


if __name__ == "__main__":
    main()
