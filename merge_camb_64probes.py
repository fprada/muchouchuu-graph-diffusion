from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion")
N = 108_000_000

sources = [
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_00_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_01_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_02_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_03_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_04_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_05_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_06_trace",
    ROOT / "gaussian_camb_z0_seed3001/camb_trace_batches/batch_07_trace",
]

times = [
    512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560,
    3072, 3584, 4096, 5120, 6144,
    7168, 8192, 10240, 12288, 14336, 16384,
]

outdir = ROOT / "gaussian_camb_z0_seed3001/camb_trace_merged_s64"
outdir.mkdir(parents=True, exist_ok=True)

summary_rows = []

for t in times:
    tag = f"{t:07d}"
    pieces = []

    for batch_id, src in enumerate(sources):
        path = src / f"trace_probe_estimates_t{tag}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        expected = {
            "graph",
            "diffusion_time",
            "probe_index",
            "individual_trace_estimate",
        }

        if set(df.columns) != expected:
            raise RuntimeError(
                f"Unexpected columns in {path}: {list(df.columns)}"
            )

        if len(df) != 8:
            raise RuntimeError(
                f"{path} contains {len(df)} probes; expected 8"
            )

        if not np.all(df["diffusion_time"].values == t):
            raise RuntimeError(
                f"Wrong diffusion_time values in {path}"
            )

        df = df.copy()
        df["batch_id"] = batch_id
        df["local_probe_index"] = df["probe_index"].astype(int)
        df["probe_index"] = (
            batch_id * 8 + df["local_probe_index"]
        )

        pieces.append(df)

    merged = pd.concat(pieces, ignore_index=True)
    merged = merged.sort_values("probe_index").reset_index(drop=True)

    if len(merged) != 64:
        raise RuntimeError(
            f"t={t}: merged {len(merged)} probes; expected 64"
        )

    if merged["probe_index"].tolist() != list(range(64)):
        raise RuntimeError(
            f"t={t}: global probe numbering is not 0..63"
        )

    trace_values = merged["individual_trace_estimate"].to_numpy(float)

    trace_mean = trace_values.mean()
    trace_se = trace_values.std(ddof=1) / np.sqrt(len(trace_values))

    p_return = trace_mean / N
    p_return_se = trace_se / N

    merged_path = outdir / f"trace_probe_estimates_t{tag}_s64.csv"
    merged.to_csv(merged_path, index=False)

    summary_rows.append({
        "diffusion_time": t,
        "n_probes": 64,
        "trace_mean": trace_mean,
        "trace_se": trace_se,
        "P_return": p_return,
        "P_return_se": p_return_se,
        "relative_se": p_return_se / p_return,
    })

    print(
        f"t={t:5d} "
        f"Tr={trace_mean:.10e} +/- {trace_se:.3e} "
        f"P={p_return:.10e} +/- {p_return_se:.3e} "
        f"relSE={p_return_se/p_return:.4f}"
    )

summary = pd.DataFrame(summary_rows)

summary_path = outdir / "return_probability_heat_trace_extended_s64.csv"
summary.to_csv(summary_path, index=False)

print()
print("Wrote:")
print(summary_path)
