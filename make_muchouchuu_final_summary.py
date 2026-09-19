#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


A_RMS = 10.409557615168367
T_MAX = 16384

analysis = Path(
    "muchouchuu_step6_window7_s64_interpolated_plateau"
)

crossing = json.loads(
    (analysis / "crossing_summary.json").read_text()
)

plateau = json.loads(
    (analysis / "late_time_plateau_summary.json").read_text()
)

cq = crossing["bootstrap_crossing_time_quantiles"]
pq = plateau["bootstrap_plateau_quantiles"]
sq = plateau["bootstrap_slope_quantiles"]

def radius(t: float) -> float:
    return A_RMS * math.sqrt(t)

r_q025 = radius(cq["q025"])
r_q16 = radius(cq["q16"])
r_med = radius(cq["median"])
r_q84 = radius(cq["q84"])
r_q975 = radius(cq["q975"])

r_max = radius(T_MAX)

rows = [
    {
        "quantity": "transport_homogeneity_time",
        "central_or_median": cq["median"],
        "q16": cq["q16"],
        "q84": cq["q84"],
        "q025": cq["q025"],
        "q975": cq["q975"],
        "units": "diffusion steps",
    },
    {
        "quantity": "transport_homogeneity_scale",
        "central_or_median": r_med,
        "q16": r_q16,
        "q84": r_q84,
        "q025": r_q025,
        "q975": r_q975,
        "units": "h^-1 Mpc",
    },
    {
        "quantity": "late_time_plateau_ds",
        "central_or_median": pq["median"],
        "q16": pq["q16"],
        "q84": pq["q84"],
        "q025": pq["q025"],
        "q975": pq["q975"],
        "units": "dimensionless",
    },
    {
        "quantity": "late_time_slope",
        "central_or_median": sq["median"],
        "q16": sq["q16"],
        "q84": sq["q84"],
        "q025": sq["q025"],
        "q975": sq["q975"],
        "units": "d(ds)/d(log t)",
    },
]

df = pd.DataFrame(rows)

df.to_csv(
    "muchouchuu_final_summary_s64_window7.csv",
    index=False,
)

with open(
    "muchouchuu_final_summary_s64_window7.txt",
    "w",
) as f:
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
        f"  95% = [{r_q025:.3f}, {r_q975:.3f}] h^-1 Mpc\n\n"
    )
    f.write(
        "Late-time spectral-dimension plateau:\n"
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
        f"Crossing success fraction = "
        f"{crossing['bootstrap_crossing_success_fraction']:.4f}\n"
    )
    f.write(
        f"Maximum tested RMS scale = {r_max:.3f} h^-1 Mpc\n"
    )
    f.write(
        f"Extent beyond median onset = {r_max-r_med:.3f} h^-1 Mpc\n"
    )
    f.write(
        f"Maximum/onset ratio = {r_max/r_med:.4f}\n"
    )

print(df.to_string(index=False))
print("\nWrote:")
print("  muchouchuu_final_summary_s64_window7.csv")
print("  muchouchuu_final_summary_s64_window7.txt")
