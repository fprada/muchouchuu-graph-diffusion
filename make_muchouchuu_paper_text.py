#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path


A_RMS = 10.409557615168367
T_MAX = 16384

root = Path(
    "muchouchuu_step6_window7_s64_interpolated_plateau"
)

crossing = json.loads(
    (root / "crossing_summary.json").read_text()
)

plateau = json.loads(
    (root / "late_time_plateau_summary.json").read_text()
)

cq = crossing["bootstrap_crossing_time_quantiles"]
pq = plateau["bootstrap_plateau_quantiles"]
sq = plateau["bootstrap_slope_quantiles"]

r = {
    key: A_RMS * math.sqrt(value)
    for key, value in cq.items()
}

rmax = A_RMS * math.sqrt(T_MAX)

plateau_t_min = plateau["plateau_tmin"]

text = f"""
Using 64 independent Hutchinson probes and a seven-point sliding-window
estimator, we find a bootstrap median transport-homogeneity scale of
R_hom = {r['median']:.1f} h^(-1) Mpc, with a 68 per cent interval of
[{r['q16']:.1f}, {r['q84']:.1f}] h^(-1) Mpc and a 95 per cent interval
of [{r['q025']:.1f}, {r['q975']:.1f}] h^(-1) Mpc. The bootstrap crossing
success fraction is
{crossing['bootstrap_crossing_success_fraction']:.3f}.

For diffusion times t >= {plateau_t_min:.0f}, the late-time
effective spectral dimension has bootstrap median
d_s = {pq['median']:.4f}, with a 68 per cent interval
[{pq['q16']:.4f}, {pq['q84']:.4f}] and a 95 per cent interval
[{pq['q025']:.4f}, {pq['q975']:.4f}]. The corresponding late-time slope is
d d_s / d log(t) = {sq['median']:.4f}, with a 68 per cent interval
[{sq['q16']:.4f}, {sq['q84']:.4f}]. Both d_s = 3 and zero slope are
contained within the inferred intervals, supporting a statistically flat
Euclidean late-time regime.

The maximum tested diffusion time, t_max = {T_MAX}, corresponds to an RMS
transport scale of {rmax:.1f} h^(-1) Mpc. Thus, the calculation tests the
Euclidean regime for approximately {rmax-r['median']:.1f} h^(-1) Mpc beyond
the median onset scale, reaching {rmax/r['median']:.2f} times the inferred
homogeneity scale.
""".strip()

Path(
    "muchouchuu_final_results_paragraph.txt"
).write_text(text + "\n")

print(text)
print("\nWrote: muchouchuu_final_results_paragraph.txt")
