#!/usr/bin/env python3

from pathlib import Path
import subprocess
import tempfile

import pandas as pd


OLD_DIR = Path("biguchuu_knn_return_trace_t32768")
NEW_DIR = Path("biguchuu_knn_return_trace_t32768_extra32_seed52001")

STEP6 = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "upgrade_step6_sliding_bootstrap_interpolated_plateau.py"
)

TMAX_VALUES = [
    7168,
    8192,
    10240,
    12288,
    16384,
    32768,
]

SEED = 12345
BOOTSTRAP = 10000
WINDOW = 7
TOLERANCE = 0.03
PERSISTENCE = 3
MIN_R2 = 0.98
PLATEAU_TMIN = 3584


def available_times(directory):
    result = []

    for path in directory.glob("trace_probe_estimates_t*.csv"):
        stem = path.stem
        t = int(stem.split("_t")[-1])
        result.append(t)

    return sorted(result)


old_times = available_times(OLD_DIR)
new_times = available_times(NEW_DIR)

common_times = sorted(set(old_times) & set(new_times))

print("Old times:", old_times)
print("New times:", new_times)
print("Common times:", common_times)

if not common_times:
    raise RuntimeError("No common diffusion times found.")


for tmax in TMAX_VALUES:

    times = [
        t for t in common_times
        if t <= tmax
    ]

    if not times:
        raise RuntimeError(f"No times available for tmax={tmax}")

    print()
    print("=" * 72)
    print(f"Running 64-probe finite-volume analysis: tmax={tmax}")
    print("Times:", times)
    print("=" * 72)

    output_dir = Path(
        f"biguchuu_step6_s64_finite_volume_tmax{tmax}"
    )

    with tempfile.TemporaryDirectory(
        prefix=f"biguchuu_s64_tmax{tmax}_"
    ) as tmpname:

        tmp = Path(tmpname)

        summary_rows = []

        for t in times:

            old_path = (
                OLD_DIR /
                f"trace_probe_estimates_t{t:07d}.csv"
            )

            new_path = (
                NEW_DIR /
                f"trace_probe_estimates_t{t:07d}.csv"
            )

            old = pd.read_csv(old_path)
            new = pd.read_csv(new_path)

            if len(old) != 32 or len(new) != 32:
                raise RuntimeError(
                    f"t={t}: expected 32+32 probes, "
                    f"got {len(old)}+{len(new)}"
                )

            old = old.copy()
            new = new.copy()

            old["probe_index"] = range(32)
            new["probe_index"] = range(32, 64)

            merged = pd.concat(
                [old, new],
                ignore_index=True,
            )

            merged_path = (
                tmp /
                f"trace_probe_estimates_t{t:07d}.csv"
            )
            merged.to_csv(merged_path, index=False)

            values = merged[
                "individual_trace_estimate"
            ].astype(float)

            trace_mean = values.mean()
            trace_std = values.std(ddof=1)
            trace_se = trace_std / (64 ** 0.5)

            N = 32_000_000

            summary_rows.append({
                "graph": "knn_periodic",
                "diffusion_time": t,
                "number_of_probes": 64,
                "trace_mean": trace_mean,
                "trace_std": trace_std,
                "trace_se": trace_se,
                "return_probability": trace_mean / N,
                "return_probability_se": trace_se / N,
                "relative_se": trace_se / trace_mean,
            })

        summary = pd.DataFrame(summary_rows)

        summary_path = (
            tmp /
            f"return_trace_summary_t{tmax}.csv"
        )

        summary.to_csv(summary_path, index=False)

        cmd = [
            "python",
            "-u",
            str(STEP6),
            str(summary_path),
            "--probe-glob",
            str(tmp / "trace_probe_estimates_t*.csv"),
            "--output-dir",
            str(output_dir),
            "--window-points",
            str(WINDOW),
            "--bootstrap",
            str(BOOTSTRAP),
            "--seed",
            str(SEED),
            "--tolerance",
            str(TOLERANCE),
            "--persistence",
            str(PERSISTENCE),
            "--min-r2",
            str(MIN_R2),
            "--plateau-tmin",
            str(PLATEAU_TMIN),
        ]

        print("Command:")
        print(" ".join(cmd))

        subprocess.run(
            cmd,
            check=True,
        )

print()
print("All six 64-probe finite-volume analyses completed.")
