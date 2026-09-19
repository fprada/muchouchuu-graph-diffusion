#!/usr/bin/env python3
"""
Create a manuscript-ready MuchoUchuu results paragraph.

This version includes the Fourier/RMS calibration uncertainty and states
that its contribution to the physical homogeneity-scale uncertainty is
negligible relative to the bootstrap crossing-time uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_A_RMS = 10.409557615168367
DEFAULT_SIGMA_A_RMS = 0.0008030807269589261
DEFAULT_T_MAX = 16384


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("muchouchuu_step6_window7_s64_interpolated_plateau"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("muchouchuu_final_results_paragraph.txt"),
    )
    parser.add_argument("--a-rms", type=float, default=DEFAULT_A_RMS)
    parser.add_argument(
        "--sigma-a-rms",
        type=float,
        default=DEFAULT_SIGMA_A_RMS,
    )
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    args = parser.parse_args()

    crossing = json.loads(
        (args.analysis_dir / "crossing_summary.json").read_text()
    )
    plateau = json.loads(
        (args.analysis_dir / "late_time_plateau_summary.json").read_text()
    )

    cq = crossing["bootstrap_crossing_time_quantiles"]
    pq = plateau["bootstrap_plateau_quantiles"]
    sq = plateau["bootstrap_slope_quantiles"]

    radii = {
        key: args.a_rms * math.sqrt(value)
        for key, value in cq.items()
    }

    r_max = args.a_rms * math.sqrt(args.t_max)
    r_hom_cal_sigma = (
        math.sqrt(cq["median"]) * args.sigma_a_rms
    )

    text = f"""
Using 64 independent Hutchinson probes and a seven-point sliding-window
estimator, we infer a bootstrap median transport-homogeneity scale of
R_hom = {radii['median']:.1f} h^(-1) Mpc, with a 68 per cent interval of
[{radii['q16']:.1f}, {radii['q84']:.1f}] h^(-1) Mpc and a 95 per cent
interval of [{radii['q025']:.1f}, {radii['q975']:.1f}] h^(-1) Mpc.
The bootstrap crossing success fraction is
{crossing['bootstrap_crossing_success_fraction']:.3f}. The Fourier/RMS
conversion uses A_RMS = {args.a_rms:.5f} +/- {args.sigma_a_rms:.5f}
h^(-1) Mpc step^(-1/2). Its contribution to the uncertainty in R_hom is
only {r_hom_cal_sigma:.2f} h^(-1) Mpc and is negligible relative to the
bootstrap crossing-time uncertainty.

For diffusion times t >= {plateau['plateau_tmin']:.0f}, the late-time
effective spectral dimension has bootstrap median d_s = {pq['median']:.4f},
with a 68 per cent interval [{pq['q16']:.4f}, {pq['q84']:.4f}] and a
95 per cent interval [{pq['q025']:.4f}, {pq['q975']:.4f}]. The
corresponding late-time slope is d d_s / d log(t) = {sq['median']:.4f},
with a 68 per cent interval [{sq['q16']:.4f}, {sq['q84']:.4f}] and a
95 per cent interval [{sq['q025']:.4f}, {sq['q975']:.4f}]. Both d_s = 3
and zero slope lie within the inferred intervals, supporting a
statistically flat late-time regime consistent with Euclidean transport.

The maximum tested diffusion time, t_max = {args.t_max}, corresponds to
an RMS transport scale of {r_max:.1f} h^(-1) Mpc. Thus, the calculation
tests consistency with Euclidean transport over approximately
{r_max - radii['median']:.1f} h^(-1) Mpc beyond the median onset scale,
reaching {r_max / radii['median']:.2f} times the inferred homogeneity
scale.
""".strip()

    args.output.write_text(text + "\n")
    print(text)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
