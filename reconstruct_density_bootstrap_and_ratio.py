#!/usr/bin/env python3
"""
Reconstruct the MuchoUchuu density-homogeneity bootstrap from saved
counts-per-center, interpolate each persistent crossing continuously in
log(radius), and combine it with the interpolated diffusion-transport
bootstrap to estimate R_transport / R_density.

The density criterion is the original joint condition:
    abs(Nbar - 1) <= scaled_count_tolerance
    abs(D2 - 3)   <= dimension_tolerance

For interpolation, define the signed joint margin
    m(r) = min(
        scaled_count_tolerance - abs(Nbar - 1),
        dimension_tolerance    - abs(D2 - 3)
    )
so that m >= 0 exactly when both criteria pass.  The first persistent pass
is retained, and its leading boundary is interpolated linearly in m versus
log(r).

The transport and density bootstraps are statistically independent, so the
ratio distribution is formed from all pairwise combinations of their valid
bootstrap samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def local_dimension(
    radii: np.ndarray,
    mean_counts: np.ndarray,
    window_points: int,
) -> np.ndarray:
    n = len(radii)
    out = np.full(n, np.nan, dtype=float)
    half = window_points // 2
    xall = np.log(radii)
    yall = np.log(mean_counts)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < window_points:
            if lo == 0:
                hi = min(n, window_points)
            else:
                lo = max(0, n - window_points)

        x = xall[lo:hi]
        y = yall[lo:hi]
        if np.all(np.isfinite(y)) and len(x) >= 3:
            out[i] = np.polyfit(x, y, 1)[0]

    return out


def first_persistent_index(mask: np.ndarray, persistence: int) -> int | None:
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if bool(ok) else 0
        if run >= persistence:
            return i - persistence + 1
    return None


def interpolate_zero_logx(
    x_left: float,
    x_right: float,
    margin_left: float,
    margin_right: float,
) -> float:
    values = (x_left, x_right, margin_left, margin_right)
    if not all(np.isfinite(v) for v in values):
        return math.nan
    if x_left <= 0.0 or x_right <= x_left:
        return math.nan
    if not (margin_left < 0.0 <= margin_right):
        return math.nan

    denom = margin_right - margin_left
    if denom <= 0.0:
        return math.nan

    frac = float(np.clip(-margin_left / denom, 0.0, 1.0))
    return float(
        np.exp(
            np.log(x_left)
            + frac * (np.log(x_right) - np.log(x_left))
        )
    )


def first_persistent_interpolated(
    radii: np.ndarray,
    joint_margin: np.ndarray,
    persistence: int,
) -> float:
    passes = np.asarray(joint_margin >= 0.0, dtype=bool)
    j = first_persistent_index(passes, persistence)
    if j is None:
        return math.nan
    if j == 0:
        return float(radii[0])

    crossing = interpolate_zero_logx(
        radii[j - 1],
        radii[j],
        joint_margin[j - 1],
        joint_margin[j],
    )
    if np.isfinite(crossing):
        return crossing

    return float(radii[j])


def quantiles(values: np.ndarray) -> dict[str, float]:
    q025, q16, q50, q84, q975 = np.quantile(
        values,
        [0.025, 0.16, 0.50, 0.84, 0.975],
    )
    return {
        "q025": float(q025),
        "q16": float(q16),
        "median": float(q50),
        "q84": float(q84),
        "q975": float(q975),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--density-dir",
        type=Path,
        default=Path("muchouchuu_density_homogeneity"),
    )
    ap.add_argument(
        "--transport-crossings",
        type=Path,
        default=Path(
            "muchouchuu_step6_window5_rel001_interpolated/"
            "bootstrap_crossing_times_interpolated.npy"
        ),
    )
    ap.add_argument(
        "--fourier-json",
        type=Path,
        default=Path(
            "muchouchuu_fourier_calibration/"
            "fourier_diffusion_calibration.json"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "muchouchuu_density_transport_ratio_bootstrap_1pct"
        ),
    )
    ap.add_argument("--number-density", type=float, default=5e-4)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=13579)
    ap.add_argument("--dimension-window-points", type=int, default=5)
    ap.add_argument("--scaled-count-tolerance", type=float, default=0.01)
    ap.add_argument("--dimension-tolerance", type=float, default=0.03)
    ap.add_argument("--persistence", type=int, default=3)
    args = ap.parse_args()

    density_dir = args.density_dir.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    counts = np.load(
        density_dir / "counts_per_center.npy",
        mmap_mode="r",
    )
    radii = np.load(density_dir / "radii_mpc_h.npy").astype(float)

    if counts.ndim != 2:
        raise ValueError("counts_per_center.npy must be a 2D array")
    if counts.shape[1] != len(radii):
        raise ValueError("Counts radius dimension does not match radii array")
    if args.dimension_window_points < 3 or args.dimension_window_points % 2 == 0:
        raise ValueError("dimension-window-points must be odd and >= 3")

    n_centers = counts.shape[0]
    volume = 4.0 * np.pi * radii**3 / 3.0
    poisson_counts = args.number_density * volume

    rng = np.random.default_rng(args.bootstrap_seed)
    density_cross = np.full(args.bootstrap, np.nan, dtype=float)
    density_cross_discrete = np.full(args.bootstrap, np.nan, dtype=float)

    for b in range(args.bootstrap):
        idx = rng.integers(0, n_centers, size=n_centers)
        mean_counts = np.asarray(counts[idx].mean(axis=0), dtype=float)
        scaled = mean_counts / poisson_counts
        d2 = local_dimension(
            radii,
            mean_counts,
            args.dimension_window_points,
        )

        count_margin = (
            args.scaled_count_tolerance - np.abs(scaled - 1.0)
        )
        dimension_margin = (
            args.dimension_tolerance - np.abs(d2 - 3.0)
        )
        joint_margin = np.minimum(count_margin, dimension_margin)

        density_cross[b] = first_persistent_interpolated(
            radii,
            joint_margin,
            args.persistence,
        )

        j = first_persistent_index(
            joint_margin >= 0.0,
            args.persistence,
        )
        if j is not None:
            density_cross_discrete[b] = radii[j]

        if (b + 1) % 100 == 0 or b + 1 == args.bootstrap:
            print(
                f"density bootstrap {b + 1}/{args.bootstrap}",
                flush=True,
            )

    valid_density = density_cross[np.isfinite(density_cross)]
    valid_density_discrete = density_cross_discrete[
        np.isfinite(density_cross_discrete)
    ]

    np.save(
        outdir / "density_crossing_radii_interpolated.npy",
        density_cross,
    )
    np.save(
        outdir / "density_crossing_radii_discrete.npy",
        density_cross_discrete,
    )
    pd.DataFrame({
        "bootstrap_index": np.arange(args.bootstrap),
        "density_crossing_radius_interpolated_mpc_h": density_cross,
        "density_crossing_radius_discrete_mpc_h": density_cross_discrete,
        "crossing_detected": np.isfinite(density_cross),
    }).to_csv(
        outdir / "density_crossing_radii_bootstrap.csv",
        index=False,
    )

    transport_t = np.load(args.transport_crossings).astype(float)
    valid_transport_t = transport_t[np.isfinite(transport_t)]

    calibration = json.loads(args.fourier_json.read_text())
    A = float(calibration["A_RMS_mpc_h_per_sqrt_step"])
    valid_transport_r = A * np.sqrt(valid_transport_t)

    if valid_density.size == 0:
        raise RuntimeError("No valid density crossings")
    if valid_transport_r.size == 0:
        raise RuntimeError("No valid transport crossings")

    # Exact empirical independent-bootstrap ratio distribution.
    ratio = (
        valid_transport_r[:, None]
        / valid_density[None, :]
    ).reshape(-1)

    np.save(outdir / "transport_to_density_ratio_all_pairs.npy", ratio)

    density_q = quantiles(valid_density)
    density_discrete_q = quantiles(valid_density_discrete)
    transport_q = quantiles(valid_transport_r)
    ratio_q = quantiles(ratio)

    summary = {
        "simulation": "MuchoUchuu",
        "density_bootstrap": {
            "replicates": int(args.bootstrap),
            "seed": int(args.bootstrap_seed),
            "sampled_centers": int(n_centers),
            "dimension_window_points":
                int(args.dimension_window_points),
            "scaled_count_tolerance":
                float(args.scaled_count_tolerance),
            "dimension_tolerance":
                float(args.dimension_tolerance),
            "persistence": int(args.persistence),
            "crossing_interpolation":
                "linear joint margin versus log(radius)",
            "joint_margin_definition": (
                "min(scaled_count_tolerance - abs(Nbar - 1), "
                "dimension_tolerance - abs(D2 - 3))"
            ),
            "success_fraction":
                float(valid_density.size / args.bootstrap),
            "interpolated_crossing_quantiles_mpc_h": density_q,
            "discrete_crossing_quantiles_mpc_h":
                density_discrete_q,
        },
        "transport_bootstrap": {
            "input_replicates": int(transport_t.size),
            "valid_replicates": int(valid_transport_t.size),
            "success_fraction":
                float(valid_transport_t.size / transport_t.size),
            "A_RMS_mpc_h_per_sqrt_step": A,
            "transport_radius_quantiles_mpc_h": transport_q,
        },
        "transport_to_density_ratio": {
            "construction": (
                "all pairwise combinations of independent valid "
                "transport and density bootstrap samples"
            ),
            "number_of_ratio_samples": int(ratio.size),
            "quantiles": ratio_q,
            "68_percent": {
                "median": ratio_q["median"],
                "minus": ratio_q["median"] - ratio_q["q16"],
                "plus": ratio_q["q84"] - ratio_q["median"],
                "interval": [ratio_q["q16"], ratio_q["q84"]],
            },
            "95_percent": {
                "median": ratio_q["median"],
                "minus": ratio_q["median"] - ratio_q["q025"],
                "plus": ratio_q["q975"] - ratio_q["median"],
                "interval": [ratio_q["q025"], ratio_q["q975"]],
            },
        },
        "source_files": {
            "counts_per_center":
                str((density_dir / "counts_per_center.npy").resolve()),
            "radii":
                str((density_dir / "radii_mpc_h.npy").resolve()),
            "transport_crossings":
                str(args.transport_crossings.resolve()),
            "fourier_calibration":
                str(args.fourier_json.resolve()),
        },
    }

    (outdir / "density_transport_ratio_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    pd.DataFrame([
        {
            "quantity": "density_homogeneity_mpc_h",
            **density_q,
        },
        {
            "quantity": "transport_homogeneity_mpc_h",
            **transport_q,
        },
        {
            "quantity": "transport_to_density_ratio",
            **ratio_q,
        },
    ]).to_csv(
        outdir / "density_transport_ratio_quantiles.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2))
    print(
        "\nWrote:",
        outdir / "density_transport_ratio_summary.json",
    )


if __name__ == "__main__":
    main()
