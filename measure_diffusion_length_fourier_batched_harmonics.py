#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import measure_diffusion_length_fourier as base


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("xyz_binary")
    p.add_argument("n", type=int)
    p.add_argument("graph")

    p.add_argument("--box-size", type=float, default=4000.0)

    p.add_argument(
        "--times",
        nargs="+",
        type=int,
        required=True,
    )

    p.add_argument(
        "--harmonics",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
    )

    p.add_argument(
        "--min-correlation",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--backend",
        choices=["auto", "mkl", "scipy"],
        default="auto",
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=16,
    )

    p.add_argument(
        "--io-chunk-rows",
        type=int,
        default=500000,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_times = sorted(set(args.times))
    max_time = max(requested_times)

    # --------------------------------------------------------
    # Existing full 24-column Fourier basis
    # --------------------------------------------------------
    modes_path = output_dir / "fourier_modes_t0.npy"

    if not modes_path.exists():
        raise FileNotFoundError(
            "Expected completed Fourier basis does not exist: {}".format(
                modes_path
            )
        )

    initial_modes = np.load(
        str(modes_path),
        mmap_mode="r",
    )

    ncols_expected = 2 * 3 * len(args.harmonics)

    if initial_modes.shape != (args.n, ncols_expected):
        raise ValueError(
            "Fourier basis shape {} != expected {}".format(
                initial_modes.shape,
                (args.n, ncols_expected),
            )
        )

    if initial_modes.dtype != np.float32:
        raise ValueError(
            "Fourier basis dtype {} != float32".format(
                initial_modes.dtype
            )
        )

    print(
        "Reusing Fourier basis:",
        modes_path,
        initial_modes.shape,
        initial_modes.dtype,
        flush=True,
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------
    print("Loading graph...", flush=True)

    graph = sparse.load_npz(args.graph).tocsr()
    graph = graph.astype(np.float32, copy=False)
    graph.sort_indices()

    if graph.shape != (args.n, args.n):
        raise ValueError(
            "Graph shape {} does not match N={}".format(
                graph.shape,
                args.n,
            )
        )

    print(
        "Graph:",
        graph.shape,
        "nnz=",
        graph.nnz,
        flush=True,
    )

    degree = np.asarray(
        graph.sum(axis=1)
    ).ravel().astype(np.float32)

    degree = np.maximum(
        degree,
        np.float32(1e-20),
    )

    backend = args.backend
    if backend == "auto":
        backend = "mkl" if base.HAVE_MKL else "scipy"

    print("Backend:", backend, flush=True)

    multiplier = base.Multiplier(
        graph,
        backend,
    )

    # --------------------------------------------------------
    # Only SIX columns are propagated at once:
    #
    # x cos/sin
    # y cos/sin
    # z cos/sin
    #
    # for one harmonic.
    # --------------------------------------------------------
    work_current_path = (
        output_dir / "fourier_work_current_harmonic.npy"
    )

    work_next_path = (
        output_dir / "fourier_work_next_harmonic.npy"
    )

    current = np.lib.format.open_memmap(
        str(work_current_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, 6),
    )

    next_state = np.lib.format.open_memmap(
        str(work_next_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, 6),
    )

    # Map diffusion time -> list of one-harmonic correlation frames.
    frames_by_time = {
        t: [] for t in requested_times
    }

    total_start = time.time()

    # --------------------------------------------------------
    # Propagate harmonics independently.
    # Linearity makes this algebraically identical to
    # propagating all 24 columns simultaneously.
    # --------------------------------------------------------
    for h_index, harmonic in enumerate(args.harmonics):

        col0 = 6 * h_index
        col1 = col0 + 6

        print()
        print("=" * 80, flush=True)
        print(
            "HARMONIC {} ({}/{}) columns {}:{}".format(
                harmonic,
                h_index + 1,
                len(args.harmonics),
                col0,
                col1,
            ),
            flush=True,
        )
        print("=" * 80, flush=True)

        initial_h = initial_modes[:, col0:col1]

        # Reset t=0 state from the immutable full Fourier basis.
        for start in range(
            0,
            args.n,
            args.io_chunk_rows,
        ):
            stop = min(
                start + args.io_chunk_rows,
                args.n,
            )

            current[start:stop] = initial_h[start:stop]

        current.flush()

        # Clear other buffer before starting this harmonic.
        next_state[:] = 0.0
        next_state.flush()

        harmonic_start = time.time()

        for step in range(1, max_time + 1):

            step_start = time.time()

            multiplier.multiply(
                current,
                next_state,
            )

            next_state /= degree[:, None]

            current, next_state = (
                next_state,
                current,
            )

            if (
                step % args.progress_every == 0
                or step in requested_times
            ):
                elapsed = time.time() - harmonic_start
                mean_seconds = elapsed / step

                print(
                    "h={} t={} step_seconds={:.3f} "
                    "mean_seconds={:.3f} "
                    "estimated_remaining_seconds={:.1f}".format(
                        harmonic,
                        step,
                        time.time() - step_start,
                        mean_seconds,
                        mean_seconds * (max_time - step),
                    ),
                    flush=True,
                )

            if step not in requested_times:
                continue

            frame = base.evaluate_correlations(
                initial_h,
                current,
                [harmonic],
                args.box_size,
                args.io_chunk_rows,
            )

            frame.insert(
                0,
                "diffusion_time",
                step,
            )

            frames_by_time[step].append(frame)

            shell_c = float(
                frame["axis_mean_correlation"].iloc[0]
            )

            print(
                "correlation h={} t={} C={:.10g}".format(
                    harmonic,
                    step,
                    shell_c,
                ),
                flush=True,
            )

        print(
            "Finished harmonic {} in {:.1f} s".format(
                harmonic,
                time.time() - harmonic_start,
            ),
            flush=True,
        )

    # --------------------------------------------------------
    # Reassemble EXACT original 4-harmonic estimator at each t.
    # --------------------------------------------------------
    correlation_rows = []
    length_rows = []

    print()
    print("=" * 80, flush=True)
    print("JOINT FOUR-HARMONIC FITS", flush=True)
    print("=" * 80, flush=True)

    for step in requested_times:

        pieces = frames_by_time[step]

        if len(pieces) != len(args.harmonics):
            raise RuntimeError(
                "t={} has {} harmonic frames, expected {}".format(
                    step,
                    len(pieces),
                    len(args.harmonics),
                )
            )

        frame = pd.concat(
            pieces,
            ignore_index=True,
        )

        correlation_rows.append(frame)

        # Remove diffusion_time only if the original fitter does
        # not care about the extra column; it selects explicit
        # columns internally, so passing it is safe.
        fit, used_shells = base.fit_diffusion_length(
            frame,
            args.min_correlation,
        )

        if fit is None:
            print(
                "Unable to fit diffusion length at t={}".format(
                    step
                ),
                flush=True,
            )
            continue

        fit["diffusion_time"] = step
        fit["backend"] = backend
        fit["harmonics_requested"] = ",".join(
            str(value) for value in args.harmonics
        )

        length_rows.append(fit)

        print(
            "length t={} ell={:.6f} h^-1 Mpc "
            "MSD={:.6f} shells={} relative_rmse={:.4e}".format(
                step,
                fit["physical_diffusion_length_mpc_h"],
                fit["mean_squared_displacement_mpc_h2"],
                fit["n_fourier_shells_used"],
                fit["relative_log_fit_rmse"],
            ),
            flush=True,
        )

    correlation_frame = pd.concat(
        correlation_rows,
        ignore_index=True,
    )

    correlation_frame.to_csv(
        output_dir / "fourier_mode_correlations.csv",
        index=False,
    )

    if not length_rows:
        raise RuntimeError(
            "No diffusion-length fits were produced."
        )

    length_frame = pd.DataFrame(
        length_rows
    ).sort_values(
        "diffusion_time"
    )

    length_frame.to_csv(
        output_dir / "physical_diffusion_length.csv",
        index=False,
    )

    base.plot_lengths(
        length_frame,
        output_dir / "physical_diffusion_length.png",
    )

    metadata = vars(args).copy()
    metadata["selected_backend"] = backend
    metadata["n_fourier_columns"] = int(ncols_expected)
    metadata["propagation_columns_at_once"] = 6
    metadata["propagation_strategy"] = (
        "harmonic-batched; four independent 6-column propagations; "
        "joint four-harmonic k^2 fit reconstructed at each requested time"
    )
    metadata["reused_fourier_modes_t0"] = True
    metadata["wall_seconds"] = float(
        time.time() - total_start
    )

    (
        output_dir
        / "physical_diffusion_length_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("Finished:", output_dir, flush=True)


if __name__ == "__main__":
    main(parse_args())
