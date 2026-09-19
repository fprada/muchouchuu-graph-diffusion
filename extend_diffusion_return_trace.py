#!/usr/bin/env python3
"""
Extend an existing BigUchuu random-walk diffusion run with stochastic estimates
of the discrete-time heat trace and mean return probability.

For P = D^{-1} A and a Gaussian probe matrix R whose entries have variance 1/m,

    heat_trace(t) = Tr(P^t)
                  ≈ Tr(R^T P^t R)
                  = sum_a R[:,a]^T Y_t[:,a],

where Y_t = P^t R and m is the number of probe columns.

The mean return probability averaged over graph vertices is

    P_return(t) = Tr(P^t) / N.

The script:
  * regenerates the original probes from --seed and --sketch-dim;
  * evaluates all existing state_tXXXXXXX.npy checkpoints;
  * optionally continues propagation to larger times with MKL;
  * writes return_probability_heat_trace.csv;
  * preserves existing checkpoints by using separate mutable buffers;
  * estimates Monte Carlo standard errors from the scatter among probes.

Important:
  This is the DISCRETE-TIME random-walk heat trace Tr(P^t), not the continuous
  Laplacian heat trace Tr(exp(-tau L)).
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from sparse_dot_mkl import dot_product_mkl
    HAVE_MKL = True
except Exception:
    dot_product_mkl = None
    HAVE_MKL = False


CHECKPOINT_RE = re.compile(r"state_t(\d{7})\.npy$")


def checkpoint_step(path):
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def list_checkpoints(directory):
    found = []
    for path in directory.glob("state_t*.npy"):
        step = checkpoint_step(path)
        if step is not None:
            found.append((step, path))
    return sorted(found)


def copy_array_chunked(source, destination, chunk_rows):
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        destination[start:stop] = source[start:stop]
    destination.flush()


def create_gaussian_probes(path, n, sketch_dim, seed, chunk_rows):
    """
    Reproduce the probe convention used by the fast diffusion runner:
    Gaussian entries scaled by 1/sqrt(sketch_dim).

    Then E[R R^T] = I, so Tr(R^T P^t R) is unbiased for Tr(P^t).
    """
    print("Creating immutable trace probes:", path, flush=True)
    rng = np.random.RandomState(seed)
    probes = np.lib.format.open_memmap(
        str(path),
        mode="w+",
        dtype=np.float32,
        shape=(n, sketch_dim),
    )
    scale = np.float32(1.0 / np.sqrt(sketch_dim))

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = rng.standard_normal(
            (stop - start, sketch_dim)
        ).astype(np.float32)
        block *= scale
        probes[start:stop] = block

        if stop == n or stop % (10 * chunk_rows) == 0:
            print(
                "probe rows {}/{}".format(stop, n),
                flush=True,
            )

    probes.flush()
    return probes


def load_or_create_probes(
    path,
    n,
    sketch_dim,
    seed,
    chunk_rows,
):
    if path.exists():
        probes = np.load(str(path), mmap_mode="r")
        if probes.shape != (n, sketch_dim):
            raise ValueError(
                "Probe shape {} does not match ({}, {}).".format(
                    probes.shape, n, sketch_dim
                )
            )
        print("Reusing immutable trace probes:", path, flush=True)
        return probes

    probes = create_gaussian_probes(
        path,
        n,
        sketch_dim,
        seed,
        chunk_rows,
    )
    del probes
    return np.load(str(path), mmap_mode="r")


def estimate_trace(probes, state, n, sketch_dim, chunk_rows):
    """
    Return:
      stochastic_trace
      stochastic_trace_standard_error
      mean_return_probability
      mean_return_probability_standard_error
      per_probe_trace_estimates

    probes[:,a] has variance 1/m. For each column,
      m * probes[:,a]^T state[:,a]
    is an individual trace estimate. Their mean equals
      sum_a probes[:,a]^T state[:,a].
    """
    if state.shape != probes.shape:
        raise ValueError(
            "State shape {} differs from probe shape {}.".format(
                state.shape, probes.shape
            )
        )

    column_quadratic_forms = np.zeros(sketch_dim, dtype=np.float64)

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)

        # Promote products and accumulation to float64.
        probe_block = np.asarray(
            probes[start:stop],
            dtype=np.float64,
        )
        state_block = np.asarray(
            state[start:stop],
            dtype=np.float64,
        )
        column_quadratic_forms += np.sum(
            probe_block * state_block,
            axis=0,
            dtype=np.float64,
        )

    per_probe_trace = (
        float(sketch_dim) * column_quadratic_forms
    )
    trace_estimate = float(np.mean(per_probe_trace))

    if sketch_dim > 1:
        trace_se = float(
            np.std(per_probe_trace, ddof=1)
            / np.sqrt(sketch_dim)
        )
    else:
        trace_se = np.nan

    return_probability = trace_estimate / float(n)
    return_probability_se = trace_se / float(n)

    return (
        trace_estimate,
        trace_se,
        return_probability,
        return_probability_se,
        per_probe_trace,
    )


def completed_trace_times(path):
    if not path.exists():
        return set()
    frame = pd.read_csv(str(path))
    return set(frame["diffusion_time"].astype(int).tolist())


def append_trace_row(path, row):
    fields = [
        "graph",
        "diffusion_time",
        "n_vertices",
        "n_probes",
        "stochastic_heat_trace",
        "stochastic_heat_trace_se",
        "mean_return_probability",
        "mean_return_probability_se",
        "relative_trace_se",
        "backend",
        "source_state",
    ]

    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_probe_estimates(path, graph, step, values):
    frame = pd.DataFrame(
        {
            "graph": graph,
            "diffusion_time": step,
            "probe_index": np.arange(len(values), dtype=np.int64),
            "individual_trace_estimate": values,
        }
    )
    frame.to_csv(str(path), index=False)


def save_checkpoint(state, step, output_dir, chunk_rows):
    final_path = output_dir / "state_t{:07d}.npy".format(step)
    temporary_path = output_dir / (
        "state_t{:07d}.tmp.npy".format(step)
    )

    print("Saving checkpoint:", final_path, flush=True)
    output = np.lib.format.open_memmap(
        str(temporary_path),
        mode="w+",
        dtype=np.float32,
        shape=state.shape,
    )
    copy_array_chunked(state, output, chunk_rows)
    del output
    os.replace(str(temporary_path), str(final_path))


class Multiplier(object):
    def __init__(self, adjacency, backend):
        self.adjacency = adjacency
        self.backend = backend

        if backend == "mkl" and not HAVE_MKL:
            raise RuntimeError(
                "MKL requested, but sparse_dot_mkl is unavailable."
            )

    def multiply(self, source, destination):
        if self.backend == "mkl":
            destination.fill(0.0)
            result = dot_product_mkl(
                self.adjacency,
                source,
                cast=False,
                out=destination,
                out_scalar=0.0,
            )
            if result is not destination:
                destination[:] = result
        else:
            destination[:] = self.adjacency @ source


def record_trace(
    graph_name,
    step,
    source_state_name,
    probes,
    state,
    args,
    trace_csv,
    backend,
):
    start = time.time()
    (
        trace_estimate,
        trace_se,
        return_probability,
        return_probability_se,
        per_probe_trace,
    ) = estimate_trace(
        probes,
        state,
        args.n,
        args.sketch_dim,
        args.io_chunk_rows,
    )

    relative_se = (
        abs(trace_se / trace_estimate)
        if trace_estimate != 0.0
        else np.inf
    )

    append_trace_row(
        trace_csv,
        {
            "graph": graph_name,
            "diffusion_time": step,
            "n_vertices": args.n,
            "n_probes": args.sketch_dim,
            "stochastic_heat_trace": trace_estimate,
            "stochastic_heat_trace_se": trace_se,
            "mean_return_probability": return_probability,
            "mean_return_probability_se": return_probability_se,
            "relative_trace_se": relative_se,
            "backend": backend,
            "source_state": source_state_name,
        },
    )

    if args.save_per_probe:
        save_probe_estimates(
            Path(args.output_dir)
            / "trace_probe_estimates_t{:07d}.csv".format(step),
            graph_name,
            step,
            per_probe_trace,
        )

    print(
        "trace t={} Tr(P^t)={:.8e} +/- {:.3e} "
        "P_return={:.8e} +/- {:.3e} relative_se={:.3e} "
        "seconds={:.2f}".format(
            step,
            trace_estimate,
            trace_se,
            return_probability,
            return_probability_se,
            relative_se,
            time.time() - start,
        ),
        flush=True,
    )


def main(args):
    state_dir = Path(args.state_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = Path(args.graph).expanduser().resolve()
    graph_name = graph_path.stem
    trace_csv = output_dir / "return_probability_heat_trace.csv"

    backend = args.backend
    if backend == "auto":
        backend = "mkl" if HAVE_MKL else "scipy"

    if backend == "mkl" and not HAVE_MKL:
        raise RuntimeError(
            "Backend mkl selected but sparse_dot_mkl is unavailable."
        )

    probe_path = output_dir / "trace_probes.npy"
    probes = load_or_create_probes(
        probe_path,
        args.n,
        args.sketch_dim,
        args.seed,
        args.io_chunk_rows,
    )

    requested_times = sorted(set(args.times))
    maximum_time = max(requested_times)
    done = completed_trace_times(trace_csv)

    existing = list_checkpoints(state_dir)
    if not existing:
        raise RuntimeError(
            "No state_tXXXXXXX.npy checkpoints found in {}".format(
                state_dir
            )
        )

    existing_by_step = dict(existing)

    # First evaluate any exact existing checkpoints, without propagation.
    for step in requested_times:
        if step in done:
            continue
        checkpoint = existing_by_step.get(step)
        if checkpoint is None:
            continue

        state = np.load(str(checkpoint), mmap_mode="r")
        record_trace(
            graph_name,
            step,
            checkpoint.name,
            probes,
            state,
            args,
            trace_csv,
            backend="checkpoint",
        )
        del state
        done.add(step)

    # Start from the latest checkpoint not later than the EARLIEST missing
    # requested time. This is essential when requested times such as 640 or
    # 896 lie between coarse diffusion checkpoints. Choosing the newest
    # checkpoint before maximum_time would silently skip those earlier times.
    missing_requested = [
        step for step in requested_times if step not in done
    ]

    if not missing_requested:
        print(
            "All requested trace times were obtained from existing "
            "checkpoints.",
            flush=True,
        )
        return

    earliest_missing = min(missing_requested)
    eligible = [
        (step, path)
        for step, path in existing
        if step <= earliest_missing
    ]
    if not eligible:
        raise RuntimeError(
            "No checkpoint is available at or before earliest missing "
            "time t={}.".format(earliest_missing)
        )

    start_step, start_checkpoint = max(
        eligible,
        key=lambda item: item[0],
    )

    missing_future = [
        step
        for step in requested_times
        if step > start_step and step not in done
    ]

    if not missing_future:
        print(
            "All requested trace times were obtained from existing "
            "checkpoints.",
            flush=True,
        )
        return

    print("Loading graph:", graph_path, flush=True)
    adjacency = sparse.load_npz(str(graph_path)).tocsr()
    adjacency = adjacency.astype(np.float32, copy=False)
    adjacency.sort_indices()

    if adjacency.shape != (args.n, args.n):
        raise ValueError(
            "Graph shape {} does not match N={}".format(
                adjacency.shape, args.n
            )
        )

    degree = np.asarray(
        adjacency.sum(axis=1)
    ).ravel().astype(np.float32)
    degree = np.maximum(degree, np.float32(1e-20))

    # Preserve the original checkpoint. Copy it into mutable buffer A.
    checkpoint_state = np.load(
        str(start_checkpoint),
        mmap_mode="r",
    )
    if checkpoint_state.shape != (args.n, args.sketch_dim):
        raise ValueError(
            "Checkpoint shape {} does not match ({}, {}).".format(
                checkpoint_state.shape,
                args.n,
                args.sketch_dim,
            )
        )

    buffer_a_path = output_dir / "trace_work_a.npy"
    buffer_b_path = output_dir / "trace_work_b.npy"

    buffer_a = np.lib.format.open_memmap(
        str(buffer_a_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, args.sketch_dim),
    )
    buffer_b = np.lib.format.open_memmap(
        str(buffer_b_path),
        mode="w+",
        dtype=np.float32,
        shape=(args.n, args.sketch_dim),
    )

    print(
        "Copying checkpoint t={} into mutable work buffer.".format(
            start_step
        ),
        flush=True,
    )
    copy_array_chunked(
        checkpoint_state,
        buffer_a,
        args.io_chunk_rows,
    )
    del checkpoint_state

    multiplier = Multiplier(adjacency, backend)
    source = buffer_a
    destination = buffer_b

    metadata = vars(args).copy()
    metadata.update(
        {
            "graph_name": graph_name,
            "selected_backend": backend,
            "mkl_available": HAVE_MKL,
            "start_step": start_step,
            "start_checkpoint": str(start_checkpoint),
            "estimator": "Gaussian Hutchinson trace estimator",
            "operator": "P = D^-1 A",
            "heat_trace_definition": "Tr(P^t)",
            "return_probability_definition": "Tr(P^t)/N",
        }
    )
    (output_dir / "return_trace_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    propagation_start = time.time()

    for step in range(start_step + 1, maximum_time + 1):
        step_start = time.time()

        multiplier.multiply(source, destination)
        destination /= degree[:, None]
        source, destination = destination, source

        if (
            step % args.progress_every == 0
            or step in requested_times
        ):
            mean_seconds = (
                time.time() - propagation_start
            ) / (step - start_step)
            remaining = mean_seconds * (maximum_time - step)

            print(
                "t={} step_seconds={:.3f} mean_seconds={:.3f} "
                "estimated_remaining_seconds={:.1f}".format(
                    step,
                    time.time() - step_start,
                    mean_seconds,
                    remaining,
                ),
                flush=True,
            )

        if step in requested_times and step not in done:
            record_trace(
                graph_name,
                step,
                "propagated_state",
                probes,
                source,
                args,
                trace_csv,
                backend,
            )
            done.add(step)

        should_checkpoint = (
            args.checkpoint_every > 0
            and step % args.checkpoint_every == 0
        ) or step == maximum_time

        if should_checkpoint:
            save_checkpoint(
                source,
                step,
                output_dir,
                args.io_chunk_rows,
            )

    print("Finished:", output_dir, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("graph")
    parser.add_argument("n", type=int)

    parser.add_argument(
        "--state-dir",
        required=True,
        help=(
            "Directory containing existing state_tXXXXXXX.npy "
            "checkpoints."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--sketch-dim",
        type=int,
        default=32,
        help=(
            "Must match the sketch dimension of the existing "
            "checkpoints."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=(
            "Must match the seed used to generate the original "
            "diffusion sketch."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mkl", "scipy"],
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--io-chunk-rows",
        type=int,
        default=500000,
    )
    parser.add_argument(
        "--save-per-probe",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
