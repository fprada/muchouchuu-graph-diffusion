#!/usr/bin/env python
import json
from pathlib import Path

import pandas as pd

import measure_diffusion_length_fourier as base


ROOT = Path(
    "gaussian_pk_control_seed1001/"
    "grf_diffusion_metric_parallel"
)

HARMONICS = [1, 2, 3, 4]
MIN_CORRELATION = 0.05

frames = []

for h in HARMONICS:
    f = ROOT / f"h{h}" / "fourier_mode_correlations.csv"

    if not f.exists():
        raise FileNotFoundError(f)

    df = pd.read_csv(f)

    present = sorted(df["harmonic"].unique())
    if present != [h]:
        raise RuntimeError(
            f"Unexpected harmonics in {f}: {present}"
        )

    frames.append(df)

all_corr = pd.concat(frames, ignore_index=True)

all_corr = all_corr.sort_values(
    ["diffusion_time", "harmonic", "axis"]
)

all_corr.to_csv(
    ROOT / "fourier_mode_correlations.csv",
    index=False,
)

length_rows = []

times = sorted(
    all_corr["diffusion_time"].unique()
)

for t in times:
    frame = all_corr[
        all_corr["diffusion_time"] == t
    ].copy()

    present = sorted(
        frame["harmonic"].unique()
    )

    if present != HARMONICS:
        raise RuntimeError(
            f"t={t}: harmonics={present}, expected={HARMONICS}"
        )

    fit, used_shells = base.fit_diffusion_length(
        frame,
        MIN_CORRELATION,
    )

    if fit is None:
        print(
            f"Unable to fit diffusion length at t={t}",
            flush=True,
        )
        continue

    fit["diffusion_time"] = int(t)
    fit["backend"] = "mkl_parallel_harmonics"
    fit["harmonics_requested"] = "1,2,3,4"

    length_rows.append(fit)

    used = ",".join(
        str(int(x))
        for x in used_shells["harmonic"].to_numpy()
    )

    print(
        f"t={int(t):5d} "
        f"ell={fit['physical_diffusion_length_mpc_h']:.6f} "
        f"MSD={fit['mean_squared_displacement_mpc_h2']:.6f} "
        f"shells={fit['n_fourier_shells_used']} "
        f"used=[{used}] "
        f"rmse={fit['relative_log_fit_rmse']:.4e}",
        flush=True,
    )

if not length_rows:
    raise RuntimeError("No diffusion-length fits produced")

length_frame = pd.DataFrame(
    length_rows
).sort_values(
    "diffusion_time"
)

length_frame.to_csv(
    ROOT / "physical_diffusion_length.csv",
    index=False,
)

base.plot_lengths(
    length_frame,
    ROOT / "physical_diffusion_length.png",
)

metadata = {
    "harmonics": HARMONICS,
    "min_correlation": MIN_CORRELATION,
    "strategy": (
        "four independent concurrent 6-column harmonic "
        "propagations followed by original joint "
        "four-harmonic k^2 fit"
    ),
}

(
    ROOT / "physical_diffusion_length_metadata.json"
).write_text(
    json.dumps(metadata, indent=2)
)

print()
print(
    "Wrote:",
    ROOT / "physical_diffusion_length.csv",
)
