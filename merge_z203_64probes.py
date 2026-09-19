#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion"
)

INPUT_ROOT = ROOT / "growth_test/z203/trace_batches"
OUTDIR = ROOT / "growth_test/z203/canonical_64x21"

NVERT = 108_000_000

TIMES = [
    512, 640, 768, 896, 1024, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096, 5120, 6144,
    7168, 8192, 10240, 12288, 14336, 16384,
]

SEEDS = [
    12345,
    41001,
    41002,
    41003,
    41004,
    41005,
    41006,
    41007,
]

OUTDIR.mkdir(parents=True, exist_ok=True)

rows = []

for batch_id in range(8):

    batch_number = batch_id + 1

    batch_dir = (
        INPUT_ROOT /
        f"batch_{batch_number:02d}_trace"
    )

    validated = batch_dir / "BATCH_VALIDATED.txt"

    if not validated.exists():
        raise FileNotFoundError(
            f"Batch {batch_number} is not validated: {validated}"
        )

    for t in TIMES:

        candidates = [
            batch_dir / f"trace_probe_estimates_t{t}.csv",
            batch_dir / f"trace_probe_estimates_t{t:07d}.csv",
        ]

        path = None
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

        if path is None:
            # Fallback in case filenames contain another zero-padding scheme.
            matches = list(
                batch_dir.glob(
                    f"trace_probe_estimates_t*{t}*.csv"
                )
            )

            good = []
            for m in matches:
                hit = re.search(r"[tT](\d+)", m.name)
                if hit and int(hit.group(1)) == t:
                    good.append(m)

            if len(good) != 1:
                raise FileNotFoundError(
                    f"Cannot identify unique t={t} file in {batch_dir}"
                )

            path = good[0]

        df = pd.read_csv(path)

        required = {
            "probe_index",
            "diffusion_time",
            "individual_trace_estimate",
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{path}: missing columns {sorted(missing)}"
            )

        if len(df) != 8:
            raise RuntimeError(
                f"{path}: expected 8 probes, found {len(df)}"
            )

        df = (
            df.sort_values("probe_index")
            .reset_index(drop=True)
        )

        local_indices = df["probe_index"].to_numpy(int)

        if not np.array_equal(
            local_indices,
            np.arange(8),
        ):
            raise RuntimeError(
                f"{path}: local probe indices are not 0..7"
            )

        if not np.all(
            df["diffusion_time"].to_numpy(int) == t
        ):
            raise RuntimeError(
                f"{path}: diffusion_time mismatch"
            )

        for local_probe in range(8):

            global_probe = (
                batch_id * 8 + local_probe
            )

            trace = float(
                df.loc[
                    local_probe,
                    "individual_trace_estimate",
                ]
            )

            rows.append(
                {
                    "batch_id": batch_id,
                    "batch_number": batch_number,
                    "seed": SEEDS[batch_id],
                    "local_probe_index": local_probe,
                    "global_probe_index": global_probe,
                    "diffusion_time": t,
                    "individual_trace_estimate": trace,
                    "P_return_probe": trace / NVERT,
                }
            )

# ---------------------------------------------------------
# Canonical probe-major table
# ---------------------------------------------------------

out = pd.DataFrame(rows)

out = out.sort_values(
    ["global_probe_index", "diffusion_time"]
).reset_index(drop=True)

expected_rows = 64 * len(TIMES)

if len(out) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, found {len(out)}"
    )

if sorted(
    out["global_probe_index"].unique()
) != list(range(64)):
    raise RuntimeError(
        "Global probe numbering is not 0..63"
    )

for t in TIMES:

    sub = out[out["diffusion_time"] == t]

    if len(sub) != 64:
        raise RuntimeError(
            f"t={t}: found {len(sub)} probes instead of 64"
        )

trajectory_path = (
    OUTDIR /
    "z203_probe_trajectories_s64_t21.csv"
)

out.to_csv(
    trajectory_path,
    index=False,
)

# ---------------------------------------------------------
# Time-major table
# ---------------------------------------------------------

time_major = out.sort_values(
    ["diffusion_time", "global_probe_index"]
).reset_index(drop=True)

time_major_path = (
    OUTDIR /
    "z203_probe_values_by_time_s64_t21.csv"
)

time_major.to_csv(
    time_major_path,
    index=False,
)

# ---------------------------------------------------------
# 64-probe mean heat trace
# ---------------------------------------------------------

summary_rows = []

for t in TIMES:

    sub = out[
        out["diffusion_time"] == t
    ].sort_values("global_probe_index")

    trace_values = (
        sub["individual_trace_estimate"]
        .to_numpy(float)
    )

    trace_mean = trace_values.mean()
    trace_std = trace_values.std(ddof=1)
    trace_se = trace_std / np.sqrt(64)

    p_return = trace_mean / NVERT
    p_return_se = trace_se / NVERT

    summary_rows.append(
        {
            "diffusion_time": t,
            "n_probes": 64,
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "trace_se": trace_se,
            "P_return": p_return,
            "P_return_se": p_return_se,
            "relative_se": p_return_se / p_return,
        }
    )

    print(
        f"t={t:5d} "
        f"Tr={trace_mean:.10e} +/- {trace_se:.3e} "
        f"P={p_return:.10e} +/- {p_return_se:.3e} "
        f"relSE={p_return_se/p_return:.4f}"
    )

summary = pd.DataFrame(summary_rows)

summary_path = (
    OUTDIR /
    "z203_return_probability_heat_trace_s64.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print()
print("Validated:", len(TIMES), "times x 64 probes")
print("Total rows:", len(out))
print()
print("Wrote:")
print(trajectory_path)
print(time_major_path)
print(summary_path)
