#!/usr/bin/env python3
"""
Build a Euclidean null-control probe dataset for the MuchoUchuu Step-6 pipeline.

The control enforces an exact mean return-probability law

    P_return(t) ∝ t^(-3/2)

while preserving the observed probe-level fractional fluctuations and their
cross-time trajectory structure:

    synthetic_probe_i(t)
      = target_mean(t) * observed_probe_i(t) / observed_mean(t)

This isolates estimator and bootstrap bias while retaining the empirical noise
pattern as closely as possible.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TIME_CANDIDATES = (
    "diffusion_time", "time", "t", "step", "diffusion_steps"
)
MEAN_CANDIDATES = (
    "mean_return_probability", "return_probability",
    "mean_return_prob", "p_return", "mean_trace_per_node"
)


def pick_column(df: pd.DataFrame, candidates, label: str) -> str:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(
        f"Could not identify {label} column. Available columns: {list(df.columns)}"
    )


def extract_time(path: str) -> int:
    match = re.search(r"[tT](\d{3,})", Path(path).name)
    if not match:
        raise ValueError(f"Cannot extract diffusion time from {path}")
    return int(match.group(1))


def extract_probe_values(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric columns in {path}")

    drop = []
    for col in numeric.columns:
        name = str(col).lower()
        if (
            name in TIME_CANDIDATES
            or any(
                key in name
                for key in ("time", "step", "probe_id", "index")
            )
        ):
            drop.append(col)

    numeric = numeric.drop(columns=drop, errors="ignore")
    if numeric.empty:
        raise ValueError(f"No usable probe columns in {path}")

    preferred = [
        col for col in numeric.columns
        if any(
            key in str(col).lower()
            for key in ("return", "trace", "estimate", "value")
        )
        and not any(
            key in str(col).lower()
            for key in ("se", "error", "std")
        )
    ]
    if preferred:
        numeric = numeric[preferred]

    values = numeric.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0 or np.any(values <= 0):
        raise ValueError(f"Probe values in {path} must be finite and positive")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--probe-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--normalization-time",
        type=float,
        default=None,
        help=(
            "Reference time used to normalize the t^-3/2 control. "
            "Default: first available time."
        ),
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    time_col = pick_column(summary, TIME_CANDIDATES, "diffusion-time")
    mean_col = pick_column(
        summary,
        MEAN_CANDIDATES,
        "mean return-probability",
    )
    summary = summary.sort_values(time_col).reset_index(drop=True)

    times = summary[time_col].to_numpy(dtype=float)
    observed_summary_mean = summary[mean_col].to_numpy(dtype=float)

    files = sorted(glob.glob(args.probe_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matched --probe-glob={args.probe_glob!r}"
        )

    probe_by_time = {
        extract_time(path): extract_probe_values(path)
        for path in files
    }

    missing = [int(t) for t in times if int(t) not in probe_by_time]
    if missing:
        raise ValueError(f"Missing probe files for times: {missing}")

    counts = [probe_by_time[int(t)].size for t in times]
    if len(set(counts)) != 1:
        raise ValueError(
            "Probe counts differ by time; cannot preserve trajectories. "
            f"Counts: {dict(zip(map(int, times), counts))}"
        )

    reference_time = (
        float(times[0])
        if args.normalization_time is None
        else float(args.normalization_time)
    )

    if reference_time not in set(times):
        raise ValueError(
            f"normalization-time={reference_time} is not in the time grid"
        )

    ref_index = int(np.where(times == reference_time)[0][0])
    reference_mean = float(observed_summary_mean[ref_index])

    target_mean = (
        reference_mean
        * (times / reference_time) ** (-1.5)
    )

    synthetic_summary_rows = []
    diagnostic_rows = []

    for time, target in zip(times, target_mean):
        observed = probe_by_time[int(time)]
        observed_mean = float(np.mean(observed))

        relative_trajectory = observed / observed_mean
        synthetic = target * relative_trajectory

        filename = (
            output
            / f"trace_probe_estimates_t{int(time):07d}.csv"
        )
        pd.DataFrame({
            "return_probability": synthetic,
        }).to_csv(filename, index=False)

        synthetic_mean = float(np.mean(synthetic))
        synthetic_se = float(
            np.std(synthetic, ddof=1) / np.sqrt(synthetic.size)
        )

        synthetic_summary_rows.append({
            "diffusion_time": int(time),
            "mean_return_probability": synthetic_mean,
            "mean_return_probability_se": synthetic_se,
        })

        diagnostic_rows.append({
            "diffusion_time": int(time),
            "observed_probe_mean": observed_mean,
            "target_euclidean_mean": float(target),
            "synthetic_probe_mean": synthetic_mean,
            "number_of_probes": int(synthetic.size),
            "relative_mean_error":
                synthetic_mean / target - 1.0,
        })

    synthetic_summary = pd.DataFrame(synthetic_summary_rows)
    summary_path = output / "return_probability_heat_trace_euclidean_control.csv"
    synthetic_summary.to_csv(summary_path, index=False)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(
        output / "euclidean_control_diagnostics.csv",
        index=False,
    )

    metadata = {
        "control_definition": "P_return(t) proportional to t^(-3/2)",
        "target_spectral_dimension": 3.0,
        "noise_preservation": (
            "Observed probe-level fractional trajectories preserved at "
            "each diffusion time"
        ),
        "reference_time": reference_time,
        "reference_mean_return_probability": reference_mean,
        "number_of_times": int(len(times)),
        "number_of_probes": int(counts[0]),
        "summary_csv": str(summary_path),
        "probe_glob": str(
            output / "trace_probe_estimates_t*.csv"
        ),
    }
    (output / "euclidean_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"Wrote: {summary_path}")
    print(
        "Probe glob:",
        output / "trace_probe_estimates_t*.csv",
    )
    print(
        "Maximum absolute relative mean error:",
        float(np.max(np.abs(
            diagnostics["relative_mean_error"].to_numpy()
        ))),
    )


if __name__ == "__main__":
    main()
