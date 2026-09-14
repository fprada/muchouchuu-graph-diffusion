#!/usr/bin/env python3
"""
Step 3: validate the MuchoUchuu coordinates and periodic kNN graph.

The default validation is designed for the 108-million-halo graph and avoids
forming A - A.T, which may require excessive memory. It performs:

Coordinate checks
-----------------
- binary file size and inferred shape
- finite values
- coordinate range [0, L)
- sampled coordinate statistics
- optional comparison with selector metadata

Graph checks
------------
- CSR shape and dtype
- finite graph data
- canonical/sorted CSR structure
- self-loop count
- degree min, max, mean, standard deviation, and percentiles
- consistency with k=12 symmetrized kNN expectations
- sampled reverse-edge symmetry
- sampled periodic edge lengths
- optional comparison with graph parameters.json

Optional exact symmetry
-----------------------
Use --full-symmetry-check only on a node with enough memory. It evaluates
(A != A.T).nnz and can require substantial temporary storage.

Example
-------
python -u validate_muchouchuu_coordinates_and_knn.py \
  muchouchuu_xyz_f32.bin \
  muchouchuu_graphs_csr_safe/knn_periodic.npz \
  --box-size 6000 \
  --k 12 \
  --coordinate-metadata muchouchuu_xyz_f32.metadata.json \
  --graph-metadata muchouchuu_graphs_csr_safe/parameters.json \
  --report validation_muchouchuu_knn.json \
  2>&1 | tee validate_muchouchuu_knn.log
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return json.load(handle)


def infer_coordinate_count(path: Path) -> tuple[int, int]:
    byte_count = path.stat().st_size
    bytes_per_row = 3 * np.dtype(np.float32).itemsize

    if byte_count % bytes_per_row != 0:
        raise ValueError(
            f"Coordinate binary has {byte_count:,} bytes, which is not "
            f"divisible by {bytes_per_row}. Expected raw float32 XYZ rows."
        )

    return byte_count // bytes_per_row, byte_count


def validate_coordinates(
    coordinate_path: Path,
    box_size: float,
    requested_n: int | None,
    chunk_rows: int,
    sample_size: int,
    seed: int,
) -> tuple[np.memmap, dict]:
    inferred_n, byte_count = infer_coordinate_count(coordinate_path)

    if requested_n is not None and requested_n != inferred_n:
        raise ValueError(
            f"--n={requested_n:,}, but coordinate file size implies "
            f"N={inferred_n:,}."
        )

    n = inferred_n
    xyz = np.memmap(
        coordinate_path,
        dtype=np.float32,
        mode="r",
        shape=(n, 3),
    )

    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    finite_count = 0
    below_zero = 0
    at_or_above_box = 0

    print(f"Coordinate rows: {n:,}")
    print(f"Coordinate bytes: {byte_count:,}")
    print(f"Coordinate box: [0, {box_size:g}) h^-1 Mpc")

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = np.asarray(xyz[start:stop])

        finite = np.isfinite(block)
        finite_count += int(finite.sum())
        below_zero += int(np.count_nonzero(block < 0.0))
        at_or_above_box += int(np.count_nonzero(block >= box_size))

        block_min = np.nanmin(block, axis=0)
        block_max = np.nanmax(block, axis=0)
        global_min = np.minimum(global_min, block_min)
        global_max = np.maximum(global_max, block_max)

        if start % (10 * chunk_rows) == 0:
            print(f"Coordinate scan: {start:,}-{stop:,}")

    expected_values = n * 3
    all_finite = finite_count == expected_values
    in_box = below_zero == 0 and at_or_above_box == 0

    rng = np.random.default_rng(seed)
    m = min(sample_size, n)
    sample_rows = rng.choice(n, size=m, replace=False)
    sample = np.asarray(xyz[sample_rows], dtype=np.float64)

    coordinate_report = {
        "path": str(coordinate_path),
        "dtype": "float32",
        "shape": [n, 3],
        "bytes": byte_count,
        "box_size_mpc_h": box_size,
        "minimum_xyz_mpc_h": global_min.tolist(),
        "maximum_xyz_mpc_h": global_max.tolist(),
        "finite_values": finite_count,
        "expected_values": expected_values,
        "all_finite": all_finite,
        "values_below_zero": below_zero,
        "values_at_or_above_box_size": at_or_above_box,
        "all_coordinates_in_periodic_box": in_box,
        "sample_size": m,
        "sample_mean_xyz_mpc_h": sample.mean(axis=0).tolist(),
        "sample_std_xyz_mpc_h": sample.std(axis=0).tolist(),
    }

    if not all_finite:
        raise ValueError("Coordinate file contains NaN or infinite values.")

    if not in_box:
        raise ValueError(
            "Coordinates are not all in the expected periodic interval "
            f"[0, {box_size:g})."
        )

    return xyz, coordinate_report


def degree_statistics(graph: sp.csr_matrix) -> tuple[np.ndarray, dict]:
    degrees = np.diff(graph.indptr).astype(np.int64, copy=False)

    percentiles = np.percentile(
        degrees,
        [0, 1, 5, 16, 50, 84, 95, 99, 100],
    )

    stats = {
        "minimum": int(degrees.min()),
        "maximum": int(degrees.max()),
        "mean": float(degrees.mean()),
        "standard_deviation": float(degrees.std()),
        "percentiles": {
            "p00": float(percentiles[0]),
            "p01": float(percentiles[1]),
            "p05": float(percentiles[2]),
            "p16": float(percentiles[3]),
            "p50": float(percentiles[4]),
            "p84": float(percentiles[5]),
            "p95": float(percentiles[6]),
            "p99": float(percentiles[7]),
            "p100": float(percentiles[8]),
        },
    }

    return degrees, stats


def sampled_reverse_edge_check(
    graph: sp.csr_matrix,
    sample_edges: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    n = graph.shape[0]
    degrees = np.diff(graph.indptr)
    nonempty_rows = np.flatnonzero(degrees > 0)

    m = min(sample_edges, int(graph.nnz))
    failures = []
    tested = 0

    while tested < m:
        batch = min(100_000, m - tested)
        rows = rng.choice(nonempty_rows, size=batch, replace=True)

        for i in rows:
            start = graph.indptr[i]
            stop = graph.indptr[i + 1]
            pos = int(rng.integers(start, stop))
            j = int(graph.indices[pos])

            j_start = graph.indptr[j]
            j_stop = graph.indptr[j + 1]
            j_neighbors = graph.indices[j_start:j_stop]
            location = np.searchsorted(j_neighbors, i)
            reverse_exists = (
                location < j_neighbors.size
                and int(j_neighbors[location]) == int(i)
            )

            if not reverse_exists and len(failures) < 20:
                failures.append([int(i), j])

        tested += batch
        if tested % 1_000_000 == 0 or tested == m:
            print(f"Sampled symmetry: {tested:,}/{m:,} edges")

    # Count failures again exactly for the stored sample is not possible
    # without retaining all sampled pairs, so detect all failures in-loop.
    # The list is capped, but this counter is not.
    # Repeat using vector batches would add memory and complexity; instead
    # track a separate counter.
    #
    # This placeholder is replaced by the accurate implementation below.
    raise RuntimeError("Internal sampled symmetry implementation placeholder")


def sampled_reverse_edge_check(
    graph: sp.csr_matrix,
    sample_edges: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    degrees = np.diff(graph.indptr)
    nonempty_rows = np.flatnonzero(degrees > 0)
    m = min(sample_edges, int(graph.nnz))

    failure_count = 0
    examples = []
    tested = 0

    while tested < m:
        batch = min(100_000, m - tested)
        rows = rng.choice(nonempty_rows, size=batch, replace=True)

        for i_value in rows:
            i = int(i_value)
            start = int(graph.indptr[i])
            stop = int(graph.indptr[i + 1])
            pos = int(rng.integers(start, stop))
            j = int(graph.indices[pos])

            j_start = int(graph.indptr[j])
            j_stop = int(graph.indptr[j + 1])
            j_neighbors = graph.indices[j_start:j_stop]
            location = int(np.searchsorted(j_neighbors, i))

            reverse_exists = (
                location < j_neighbors.size
                and int(j_neighbors[location]) == i
            )

            if not reverse_exists:
                failure_count += 1
                if len(examples) < 20:
                    examples.append([i, j])

        tested += batch

        if tested % 1_000_000 == 0 or tested == m:
            print(f"Sampled symmetry: {tested:,}/{m:,} edges")

    return {
        "sampled_edges": m,
        "reverse_edge_failures": failure_count,
        "reverse_edge_failure_fraction": (
            failure_count / m if m else 0.0
        ),
        "first_failure_examples": examples,
        "passed": failure_count == 0,
    }


def sampled_periodic_edge_lengths(
    graph: sp.csr_matrix,
    xyz: np.memmap,
    box_size: float,
    sample_edges: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    degrees = np.diff(graph.indptr)
    nonempty_rows = np.flatnonzero(degrees > 0)
    m = min(sample_edges, int(graph.nnz))

    distances = np.empty(m, dtype=np.float64)
    self_edges = 0

    for q in range(m):
        i = int(rng.choice(nonempty_rows))
        start = int(graph.indptr[i])
        stop = int(graph.indptr[i + 1])
        j = int(graph.indices[int(rng.integers(start, stop))])

        if i == j:
            self_edges += 1

        delta = np.abs(
            np.asarray(xyz[i], dtype=np.float64)
            - np.asarray(xyz[j], dtype=np.float64)
        )
        delta = np.minimum(delta, box_size - delta)
        distances[q] = math.sqrt(float(np.dot(delta, delta)))

        if (q + 1) % 1_000_000 == 0 or q + 1 == m:
            print(f"Sampled edge lengths: {q + 1:,}/{m:,}")

    percentiles = np.percentile(
        distances,
        [0, 1, 5, 16, 50, 84, 95, 99, 100],
    )

    return {
        "sampled_edges": m,
        "sampled_self_edges": self_edges,
        "minimum_mpc_h": float(percentiles[0]),
        "p01_mpc_h": float(percentiles[1]),
        "p05_mpc_h": float(percentiles[2]),
        "p16_mpc_h": float(percentiles[3]),
        "median_mpc_h": float(percentiles[4]),
        "p84_mpc_h": float(percentiles[5]),
        "p95_mpc_h": float(percentiles[6]),
        "p99_mpc_h": float(percentiles[7]),
        "maximum_mpc_h": float(percentiles[8]),
        "all_sampled_distances_finite": bool(
            np.isfinite(distances).all()
        ),
        "all_sampled_distances_positive": bool(
            np.all(distances > 0.0)
        ),
    }


def validate_graph(
    graph_path: Path,
    xyz: np.memmap,
    box_size: float,
    expected_k: int,
    sample_edges: int,
    seed: int,
    full_symmetry_check: bool,
) -> tuple[sp.csr_matrix, dict]:
    print(f"Loading graph: {graph_path}")
    graph = sp.load_npz(graph_path)

    if not sp.isspmatrix_csr(graph):
        print(f"Converting {type(graph).__name__} to CSR")
        graph = graph.tocsr()

    n = xyz.shape[0]

    if graph.shape != (n, n):
        raise ValueError(
            f"Graph shape is {graph.shape}, expected ({n}, {n})."
        )

    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.sort_indices()

    diagonal = graph.diagonal()
    diagonal_nonzero = int(np.count_nonzero(diagonal))
    finite_data = bool(np.isfinite(graph.data).all())

    degrees, degree_report = degree_statistics(graph)

    sampled_symmetry = sampled_reverse_edge_check(
        graph,
        sample_edges=sample_edges,
        seed=seed + 1,
    )

    edge_lengths = sampled_periodic_edge_lengths(
        graph,
        xyz,
        box_size=box_size,
        sample_edges=sample_edges,
        seed=seed + 2,
    )

    exact_asymmetry_nnz = None
    if full_symmetry_check:
        print(
            "Running exact symmetry check. This may require substantial "
            "temporary memory."
        )
        exact_asymmetry_nnz = int((graph != graph.T).nnz)

    graph_report = {
        "path": str(graph_path),
        "format": "csr",
        "shape": list(graph.shape),
        "dtype": str(graph.dtype),
        "nnz": int(graph.nnz),
        "undirected_edges_if_symmetric": int(graph.nnz // 2),
        "has_sorted_indices": bool(graph.has_sorted_indices),
        "has_canonical_format": bool(graph.has_canonical_format),
        "finite_data": finite_data,
        "diagonal_nonzero_count": diagonal_nonzero,
        "expected_knn_k": expected_k,
        "all_degrees_at_least_k": bool(degrees.min() >= expected_k),
        "degree_statistics": degree_report,
        "sampled_symmetry": sampled_symmetry,
        "sampled_periodic_edge_lengths": edge_lengths,
        "full_symmetry_check_requested": full_symmetry_check,
        "exact_asymmetry_nnz": exact_asymmetry_nnz,
    }

    if not finite_data:
        raise ValueError("Graph data contain NaN or infinite values.")

    if diagonal_nonzero != 0:
        raise ValueError(
            f"Graph contains {diagonal_nonzero:,} nonzero self-loops."
        )

    if degrees.min() < expected_k:
        raise ValueError(
            f"Minimum graph degree is {degrees.min()}, below k={expected_k}."
        )

    if not sampled_symmetry["passed"]:
        raise ValueError(
            "Sampled reverse-edge symmetry check failed. "
            f"Failures={sampled_symmetry['reverse_edge_failures']:,}."
        )

    if full_symmetry_check and exact_asymmetry_nnz != 0:
        raise ValueError(
            f"Exact symmetry check found {exact_asymmetry_nnz:,} "
            "asymmetric entries."
        )

    return graph, graph_report


def compare_metadata(
    coordinate_report: dict,
    graph_report: dict,
    coordinate_metadata: dict | None,
    graph_metadata: dict | None,
) -> dict:
    checks = {}

    if coordinate_metadata is not None:
        checks["coordinate_metadata_n_matches"] = (
            coordinate_metadata.get("written_objects")
            == coordinate_report["shape"][0]
        )
        checks["coordinate_metadata_shape_matches"] = (
            coordinate_metadata.get("xyz_binary_shape")
            == coordinate_report["shape"]
        )
        checks["coordinate_metadata_dtype_matches"] = (
            coordinate_metadata.get("xyz_binary_dtype")
            == coordinate_report["dtype"]
        )

    if graph_metadata is not None:
        checks["graph_metadata_n_matches"] = (
            graph_metadata.get("n") == graph_report["shape"][0]
        )
        checks["graph_metadata_k_matches"] = (
            graph_metadata.get("k") == graph_report["expected_knn_k"]
        )
        checks["graph_metadata_validation_passed"] = bool(
            graph_metadata.get("validation_passed")
        )

        stored_mean = graph_metadata.get(
            "mean_symmetrized_knn_degree"
        )
        if stored_mean is not None:
            measured_mean = graph_report[
                "degree_statistics"
            ]["mean"]
            checks["graph_metadata_mean_degree_difference"] = (
                measured_mean - float(stored_mean)
            )

    checks["all_boolean_checks_pass"] = all(
        value
        for value in checks.values()
        if isinstance(value, bool)
    )

    return checks


def main(args: argparse.Namespace) -> None:
    coordinate_path = Path(args.xyz_binary).expanduser().resolve()
    graph_path = Path(args.graph_npz).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    coordinate_metadata_path = (
        Path(args.coordinate_metadata).expanduser().resolve()
        if args.coordinate_metadata
        else None
    )
    graph_metadata_path = (
        Path(args.graph_metadata).expanduser().resolve()
        if args.graph_metadata
        else None
    )

    if not coordinate_path.exists():
        raise FileNotFoundError(coordinate_path)
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    xyz, coordinate_report = validate_coordinates(
        coordinate_path=coordinate_path,
        box_size=args.box_size,
        requested_n=args.n,
        chunk_rows=args.coordinate_chunk_rows,
        sample_size=args.coordinate_sample,
        seed=args.seed,
    )

    graph, graph_report = validate_graph(
        graph_path=graph_path,
        xyz=xyz,
        box_size=args.box_size,
        expected_k=args.k,
        sample_edges=args.edge_sample,
        seed=args.seed,
        full_symmetry_check=args.full_symmetry_check,
    )

    coordinate_metadata = load_json(coordinate_metadata_path)
    graph_metadata = load_json(graph_metadata_path)

    metadata_checks = compare_metadata(
        coordinate_report,
        graph_report,
        coordinate_metadata,
        graph_metadata,
    )

    overall_passed = (
        coordinate_report["all_finite"]
        and coordinate_report["all_coordinates_in_periodic_box"]
        and graph_report["finite_data"]
        and graph_report["diagonal_nonzero_count"] == 0
        and graph_report["all_degrees_at_least_k"]
        and graph_report["sampled_symmetry"]["passed"]
        and metadata_checks["all_boolean_checks_pass"]
        and (
            graph_report["exact_asymmetry_nnz"] in (None, 0)
        )
    )

    report = {
        "overall_validation_passed": overall_passed,
        "coordinates": coordinate_report,
        "graph": graph_report,
        "metadata_checks": metadata_checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            default=json_scalar,
        )

    print("\nValidation summary")
    print("------------------")
    print(f"Overall passed: {overall_passed}")
    print(
        "Coordinate range: "
        f"min={coordinate_report['minimum_xyz_mpc_h']}, "
        f"max={coordinate_report['maximum_xyz_mpc_h']}"
    )
    print(f"Graph shape: {tuple(graph_report['shape'])}")
    print(f"Graph nnz: {graph_report['nnz']:,}")
    print(
        "Mean symmetrized degree: "
        f"{graph_report['degree_statistics']['mean']:.6f}"
    )
    print(
        "Degree range: "
        f"{graph_report['degree_statistics']['minimum']} to "
        f"{graph_report['degree_statistics']['maximum']}"
    )
    print(
        "Sampled reverse-edge failures: "
        f"{graph_report['sampled_symmetry']['reverse_edge_failures']}"
    )
    print(
        "Sampled periodic edge-length median/max: "
        f"{graph_report['sampled_periodic_edge_lengths']['median_mpc_h']:.6f} / "
        f"{graph_report['sampled_periodic_edge_lengths']['maximum_mpc_h']:.6f} "
        "h^-1 Mpc"
    )
    print(f"Wrote report: {report_path}")

    if not overall_passed:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("xyz_binary")
    parser.add_argument("graph_npz")

    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--box-size", type=float, default=6000.0)
    parser.add_argument("--k", type=int, default=12)

    parser.add_argument(
        "--coordinate-metadata",
        default=None,
    )
    parser.add_argument(
        "--graph-metadata",
        default=None,
    )
    parser.add_argument(
        "--coordinate-chunk-rows",
        type=int,
        default=5_000_000,
    )
    parser.add_argument(
        "--coordinate-sample",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--edge-sample",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--full-symmetry-check",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--report",
        default="validation_muchouchuu_knn.json",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
