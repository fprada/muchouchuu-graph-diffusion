from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion")
NVERT = 108_000_000

old_times = [
    512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
    2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
    10240, 12288, 14336, 16384
]

new_times = [20480, 24576, 32768]

all_times = old_times + new_times

# Canonical 8-probe groups.
groups = [
    {
        "name": "original_seed_12345",
        "global_start": 0,
        "old_dir": ROOT / "muchouchuu_knn_return_trace_t16384_s8",
        "new_dir": ROOT / "muchouchuu_extended_trace/original_seed_12345",
    },
    {
        "name": "batch_01",
        "global_start": 8,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_01_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_01",
    },
    {
        "name": "batch_02_rebuilt",
        "global_start": 16,
        "old_dir": ROOT / "muchouchuu_batch_02_rebuilt/trace",
        "new_dir": ROOT / "muchouchuu_batch_02_rebuilt/trace",
    },
    {
        "name": "batch_03",
        "global_start": 24,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_03_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_03",
    },
    {
        "name": "batch_04",
        "global_start": 32,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_04_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_04",
    },
    {
        "name": "batch_05",
        "global_start": 40,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_05_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_05",
    },
    {
        "name": "batch_06",
        "global_start": 48,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_06_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_06",
    },
    {
        "name": "batch_07",
        "global_start": 56,
        "old_dir": ROOT / "muchouchuu_additional_trace_batches/batch_07_trace",
        "new_dir": ROOT / "muchouchuu_extended_trace/batch_07",
    },
]

rows = []

expected_cols = [
    "graph",
    "diffusion_time",
    "probe_index",
    "individual_trace_estimate",
]

for group in groups:
    for t in all_times:
        tag = f"{t:07d}"

        src_dir = group["old_dir"] if t in old_times else group["new_dir"]
        path = src_dir / f"trace_probe_estimates_t{tag}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if list(df.columns) != expected_cols:
            raise RuntimeError(
                f"Unexpected columns in {path}: {list(df.columns)}"
            )

        if len(df) != 8:
            raise RuntimeError(
                f"{path}: expected 8 probes, found {len(df)}"
            )

        df = df.sort_values("probe_index").reset_index(drop=True)

        local = df["probe_index"].to_numpy(int)

        if not np.array_equal(local, np.arange(8)):
            raise RuntimeError(
                f"{path}: local probe indices are not 0..7"
            )

        if not np.all(df["diffusion_time"].to_numpy(int) == t):
            raise RuntimeError(
                f"{path}: diffusion_time mismatch"
            )

        for i in range(8):
            trace = float(df.loc[i, "individual_trace_estimate"])

            rows.append({
                "batch": group["name"],
                "local_probe_index": int(i),
                "global_probe_index": int(group["global_start"] + i),
                "diffusion_time": int(t),
                "individual_trace_estimate": trace,
                "P_return_probe": trace / NVERT,
            })

out = pd.DataFrame(rows)

# Canonical ordering: probe trajectory first, then time.
out = out.sort_values(
    ["global_probe_index", "diffusion_time"]
).reset_index(drop=True)

# Hard integrity checks.
if len(out) != 64 * 24:
    raise RuntimeError(
        f"Expected {64*24} rows, found {len(out)}"
    )

if sorted(out["global_probe_index"].unique()) != list(range(64)):
    raise RuntimeError("Global probe indices are not 0..63")

if sorted(out["diffusion_time"].unique()) != sorted(all_times):
    raise RuntimeError("Diffusion-time grid mismatch")

counts_probe = out.groupby("global_probe_index").size()
if not np.all(counts_probe.values == 24):
    raise RuntimeError("Not every probe has 24 epochs")

counts_time = out.groupby("diffusion_time").size()
if not np.all(counts_time.values == 64):
    raise RuntimeError("Not every time has 64 probes")

outdir = ROOT / "muchouchuu_extended_trace/canonical_64x24"
outdir.mkdir(parents=True, exist_ok=True)

traj_path = outdir / "muchouchuu_probe_trajectories_s64_t24.csv"
out.to_csv(traj_path, index=False)

# Also write a time-major table.
time_major = out.sort_values(
    ["diffusion_time", "global_probe_index"]
).reset_index(drop=True)

time_major_path = outdir / "muchouchuu_probe_values_by_time_s64_t24.csv"
time_major.to_csv(time_major_path, index=False)

# Summary per diffusion time.
summary = (
    out.groupby("diffusion_time")
    .agg(
        n_probes=("global_probe_index", "count"),
        trace_mean=("individual_trace_estimate", "mean"),
        trace_std=("individual_trace_estimate", "std"),
    )
    .reset_index()
)

summary["trace_se"] = summary["trace_std"] / np.sqrt(summary["n_probes"])
summary["P_return"] = summary["trace_mean"] / NVERT
summary["P_return_se"] = summary["trace_se"] / NVERT
summary["relative_se"] = summary["P_return_se"] / summary["P_return"]

summary_path = outdir / "muchouchuu_return_probability_s64_t24.csv"
summary.to_csv(summary_path, index=False)

print("SUCCESS")
print("rows =", len(out))
print("probes =", out["global_probe_index"].nunique())
print("times =", out["diffusion_time"].nunique())
print()
print("Wrote:")
print(traj_path)
print(time_major_path)
print(summary_path)
print()
print(summary.tail(6).to_string(index=False))
