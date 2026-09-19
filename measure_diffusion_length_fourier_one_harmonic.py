#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import measure_diffusion_length_fourier as base


ALL_HARMONICS = [1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("n", type=int)
    p.add_argument("graph")

    p.add_argument("--basis-file", required=True)
    p.add_argument("--harmonic", type=int, choices=ALL_HARMONICS, required=True)
    p.add_argument("--box-size", type=float, default=6000.0)

    p.add_argument(
        "--times",
        nargs="+",
        type=int,
        required=True,
    )

    p.add_argument(
        "--backend",
        choices=["auto", "mkl", "scipy"],
        default="mkl",
    )

    p.add_argument("--progress-every", type=int, default=16)
    p.add_argument("--io-chunk-rows", type=int, default=500000)
    p.add_argument("--output-dir", required=True)

    return p.parse_args()


def main(args):

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    requested_times = sorted(set(args.times))
    max_time = max(requested_times)

    h = args.harmonic
    h_index = ALL_HARMONICS.index(h)

    col0 = 6 * h_index
    col1 = col0 + 6

    # --------------------------------------------------------
    # Read only the relevant six columns from the completed
    # immutable 24-column t=0 Fourier basis.
    # --------------------------------------------------------
    basis_path = Path(args.basis_file)

    initial_all = np.load(
        str(basis_path),
        mmap_mode="r",
    )

    expected_shape = (args.n, 24)

    if initial_all.shape != expected_shape:
        raise ValueError(
            "Basis shape {} != {}".format(
                initial_all.shape,
                expected_shape,
            )
        )

    if initial_all.dtype != np.float32:
        raise ValueError(
            "Basis dtype {} != float32".format(
                initial_all.dtype
            )
        )

    initial = initial_all[:, col0:col1]

    print(
        "harmonic={} basis columns={}:{}".format(
            h, col0, col1
        ),
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
            "Graph shape {} != ({},{})".format(
                graph.shape,
                args.n,
                args.n,
            )
        )

    print(
        "Graph shape={} nnz={}".format(
            graph.shape,
            graph.nnz,
        ),
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
    # Two 6-column states in RAM.
    #
    # Each array is:
    #   108,000,000 x 6 x 4 bytes = 2.592 GB
    #
    # Keeping both states in RAM removes disk-backed memmap I/O
    # from every diffusion step.
    # --------------------------------------------------------
    state_gb = args.n * 6 * np.dtype(np.float32).itemsize / 1e9

    print(
        "Allocating two in-RAM float32 states: "
        "{:.3f} GB each, {:.3f} GB total".format(
            state_gb,
            2.0 * state_gb,
        ),
        flush=True,
    )

    current = np.empty(
        (args.n, 6),
        dtype=np.float32,
        order="C",
    )

    next_state = np.empty(
        (args.n, 6),
        dtype=np.float32,
        order="C",
    )

    print("Initializing t=0 state in RAM...", flush=True)

    for start in range(
        0,
        args.n,
        args.io_chunk_rows,
    ):
        stop = min(
            start + args.io_chunk_rows,
            args.n,
        )

        current[start:stop, :] = initial[start:stop, :]

    print(
        "t=0 state initialized; "
        "current C-contiguous={} next C-contiguous={}".format(
            current.flags.c_contiguous,
            next_state.flags.c_contiguous,
        ),
        flush=True,
    )

    frames = []
    propagation_start = time.time()

    # --------------------------------------------------------
    # Diffusion
    # --------------------------------------------------------
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
            elapsed = time.time() - propagation_start
            mean_seconds = elapsed / step

            print(
                "h={} t={} step_seconds={:.3f} "
                "mean_seconds={:.3f} "
                "estimated_remaining_seconds={:.1f}".format(
                    h,
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
            initial,
            current,
            [h],
            args.box_size,
            args.io_chunk_rows,
        )

        frame.insert(
            0,
            "diffusion_time",
            step,
        )

        frames.append(frame)

        print(
            "h={} t={} C={:.10g}".format(
                h,
                step,
                float(
                    frame[
                        "axis_mean_correlation"
                    ].iloc[0]
                ),
            ),
            flush=True,
        )

    import pandas as pd

    correlation_frame = pd.concat(
        frames,
        ignore_index=True,
    )

    correlation_file = (
        out
        / "fourier_mode_correlations.csv"
    )

    correlation_frame.to_csv(
        correlation_file,
        index=False,
    )

    metadata = vars(args).copy()
    metadata["basis_columns"] = [col0, col1]
    metadata["selected_backend"] = backend
    metadata["wall_seconds"] = float(
        time.time() - propagation_start
    )

    (
        out / "metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2)
    )

    print(
        "Finished harmonic {}: {}".format(
            h,
            correlation_file,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
