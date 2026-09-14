#!/usr/bin/env python3
"""
Select the exact top-Vpeak MuchoUchuu halo population from one HDF5 file.

The standard MuchoUchuu HDF5 layout is column-oriented, with separate
one-dimensional datasets named x, y, z, and Vpeak.

Example
-------
python select_muchouchuu_vpeak_v4.py \
  /work/ishiyama/MuchoUchuu6G/00030/hlist_0.99944.list.h5 \
  --graph-sample 0 \
  --output muchouchuu_xyz_f32.bin

The main output is a raw C-order float32 array with shape (N, 3):
x0, y0, z0, x1, y1, z1, ...

The exact top-Vpeak threshold is calculated in the native precision of the
input Vpeak dataset. For the standard MuchoUchuu catalogue this is float64.
This prevents the threshold mismatch caused by ranking float64 values through
a float32 temporary array.

The XYZ output is directly compatible with the periodic graph-building
scripts used for the BigUchuu diffusion analysis.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np


def list_datasets(h5: h5py.File) -> list[str]:
    names: list[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5.visititems(visitor)
    return names


def resolve_dataset(
    h5: h5py.File,
    explicit_path: str | None,
    leaf_name: str,
) -> h5py.Dataset:
    if explicit_path is not None:
        for candidate in (explicit_path, explicit_path.lstrip("/")):
            if candidate in h5 and isinstance(h5[candidate], h5py.Dataset):
                return h5[candidate]
        raise KeyError(
            f"Could not find dataset {explicit_path!r}. "
            f"Available datasets include: {list_datasets(h5)[:100]}"
        )

    matches: list[str] = []

    def visitor(name, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and name.split("/")[-1].lower() == leaf_name.lower()
        ):
            matches.append(name)

    h5.visititems(visitor)

    if len(matches) == 1:
        return h5[matches[0]]

    if len(matches) == 0:
        raise KeyError(
            f"No dataset named {leaf_name!r} found. "
            f"Available datasets include: {list_datasets(h5)[:100]}"
        )

    raise ValueError(
        f"Multiple datasets match {leaf_name!r}: {matches}. "
        f"Specify one explicitly with --{leaf_name.lower()}-dataset."
    )


def validate_columns(columns: dict[str, h5py.Dataset]) -> int:
    lengths: dict[str, int] = {}

    for name, dataset in columns.items():
        if dataset.ndim != 1:
            raise ValueError(
                f"{name} dataset {dataset.name} has shape {dataset.shape}; "
                "expected a one-dimensional column."
            )
        lengths[name] = dataset.shape[0]

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Column datasets have inconsistent row counts: "
            f"{lengths}"
        )

    return next(iter(lengths.values()))


def write_vpeak_memmap(
    vpeak_dataset: h5py.Dataset,
    path: Path,
    chunk_size: int,
) -> np.memmap:
    n_rows = vpeak_dataset.shape[0]
    input_dtype = np.dtype(vpeak_dataset.dtype)

    if input_dtype.kind != "f":
        raise TypeError(
            "Vpeak must be stored as a floating-point dataset; "
            f"found dtype={input_dtype}."
        )

    # Use native byte order while preserving the input precision. Thus a
    # float64 MuchoUchuu Vpeak column remains float64 during partitioning.
    memmap_dtype = input_dtype.newbyteorder("=")

    values = np.memmap(
        path,
        mode="w+",
        dtype=memmap_dtype,
        shape=(n_rows,),
    )

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        values[start:stop] = vpeak_dataset[start:stop]

        if start % (10 * chunk_size) == 0:
            print(f"Copied Vpeak rows {start:,}-{stop:,}")

    values.flush()
    return values


def exact_threshold(values: np.memmap, n_target: int) -> float:
    n_total = len(values)

    if not 0 < n_target <= n_total:
        raise ValueError(
            f"Requested {n_target:,} objects from a catalogue containing "
            f"{n_total:,} rows."
        )

    kth = n_total - n_target

    print(
        f"Partitioning {n_total:,} values to identify "
        f"the top {n_target:,} halos..."
    )

    values.partition(kth)
    threshold = float(values[kth])
    values.flush()
    return threshold


def count_threshold_members(
    vpeak_dataset: h5py.Dataset,
    threshold: float,
    chunk_size: int,
) -> tuple[int, int]:
    greater = 0
    equal = 0

    for start in range(0, vpeak_dataset.shape[0], chunk_size):
        stop = min(start + chunk_size, vpeak_dataset.shape[0])
        values = vpeak_dataset[start:stop]

        greater += int(np.count_nonzero(values > threshold))
        equal += int(np.count_nonzero(values == threshold))

    return greater, equal


def stream_selection(
    columns: dict[str, h5py.Dataset],
    threshold: float,
    n_target: int,
    n_greater: int,
    graph_sample: int,
    output_path: Path,
    vpeak_output_path: Path | None,
    chunk_size: int,
    seed: int,
) -> dict:
    equals_needed = n_target - n_greater
    equals_used = 0
    selected_seen = 0

    rng = np.random.default_rng(seed)

    if graph_sample < 0:
        raise ValueError("--graph-sample must be zero or positive.")

    if graph_sample > 0:
        sample_size = min(graph_sample, n_target)
        reservoir_xyz = np.empty((sample_size, 3), dtype=np.float32)
        reservoir_vpeak = np.empty(
            sample_size,
            dtype=np.dtype(columns["Vpeak"].dtype).newbyteorder("="),
        )
        xyz_stream = None
        vpeak_stream = None
    else:
        sample_size = n_target
        xyz_stream = open(output_path, "wb")
        vpeak_stream = (
            open(vpeak_output_path, "wb")
            if vpeak_output_path is not None
            else None
        )

    try:
        n_rows = columns["Vpeak"].shape[0]

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)

            vpeak = columns["Vpeak"][start:stop]

            xyz = np.column_stack([
                columns["x"][start:stop].astype(np.float32, copy=False),
                columns["y"][start:stop].astype(np.float32, copy=False),
                columns["z"][start:stop].astype(np.float32, copy=False),
            ])

            greater_mask = vpeak > threshold
            equal_indices = np.flatnonzero(vpeak == threshold)

            if equals_used < equals_needed and equal_indices.size:
                take = min(
                    equals_needed - equals_used,
                    equal_indices.size,
                )

                equal_mask = np.zeros(vpeak.size, dtype=bool)
                equal_mask[equal_indices[:take]] = True
                equals_used += take
                selected_mask = greater_mask | equal_mask
            else:
                selected_mask = greater_mask

            # Keep Vpeak at native input precision. Only XYZ is converted
            # to float32 for compatibility with the graph pipeline.
            selected_vpeak = np.asarray(vpeak[selected_mask])
            selected_xyz = xyz[selected_mask]

            if graph_sample > 0:
                for point, value in zip(selected_xyz, selected_vpeak):
                    if selected_seen < sample_size:
                        reservoir_xyz[selected_seen] = point
                        reservoir_vpeak[selected_seen] = value
                    else:
                        j = int(rng.integers(0, selected_seen + 1))
                        if j < sample_size:
                            reservoir_xyz[j] = point
                            reservoir_vpeak[j] = value

                    selected_seen += 1
            else:
                selected_xyz.astype(
                    np.float32,
                    copy=False,
                ).tofile(xyz_stream)

                if vpeak_stream is not None:
                    selected_vpeak.astype(
                        np.dtype(columns["Vpeak"].dtype).newbyteorder("="),
                        copy=False,
                    ).tofile(vpeak_stream)

                selected_seen += selected_vpeak.size

            if start % (10 * chunk_size) == 0:
                print(
                    f"Selection rows {start:,}-{stop:,}; "
                    f"selected={selected_seen:,}"
                )

        if selected_seen != n_target:
            raise RuntimeError(
                f"Expected {n_target:,} selected halos, "
                f"but streamed {selected_seen:,}."
            )

        if graph_sample > 0:
            reservoir_xyz.astype(
                np.float32,
                copy=False,
            ).tofile(output_path)

            if vpeak_output_path is not None:
                reservoir_vpeak.astype(
                    np.dtype(columns["Vpeak"].dtype).newbyteorder("="),
                    copy=False,
                ).tofile(vpeak_output_path)

    finally:
        if xyz_stream is not None:
            xyz_stream.close()
        if vpeak_stream is not None:
            vpeak_stream.close()

    return {
        "written_objects": sample_size,
        "equal_threshold_objects_used": equals_used,
    }


def main(args: argparse.Namespace) -> None:
    catalogue_path = Path(args.catalogue_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    vpeak_output_path = (
        Path(args.vpeak_output).expanduser().resolve()
        if args.vpeak_output is not None
        else None
    )

    if not catalogue_path.exists():
        raise FileNotFoundError(catalogue_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if vpeak_output_path is not None:
        vpeak_output_path.parent.mkdir(parents=True, exist_ok=True)

    target_count = int(
        round(args.number_density * args.box_size**3)
    )

    with h5py.File(catalogue_path, "r") as h5:
        columns = {
            "x": resolve_dataset(h5, args.x_dataset, "x"),
            "y": resolve_dataset(h5, args.y_dataset, "y"),
            "z": resolve_dataset(h5, args.z_dataset, "z"),
            "Vpeak": resolve_dataset(
                h5,
                args.vpeak_dataset,
                "Vpeak",
            ),
        }

        n_total = validate_columns(columns)

        if target_count > n_total:
            raise ValueError(
                f"Target density requires {target_count:,} halos, "
                f"but the file contains only {n_total:,}. "
                "Reduce --number-density or verify --box-size."
            )

        print(f"Catalogue: {catalogue_path}")
        print(f"Rows: {n_total:,}")
        print(
            "Resolved datasets: "
            + ", ".join(
                f"{name}={dataset.name}"
                for name, dataset in columns.items()
            )
        )
        print(f"Target selected count: {target_count:,}")

        with tempfile.TemporaryDirectory(
            dir=args.temp_dir
        ) as temporary_directory:
            memmap_path = (
                Path(temporary_directory) / "vpeak_native_precision.dat"
            )

            values = write_vpeak_memmap(
                columns["Vpeak"],
                memmap_path,
                args.chunk_size,
            )

            threshold = exact_threshold(values, target_count)
            del values

            n_greater, n_equal = count_threshold_members(
                columns["Vpeak"],
                threshold,
                args.chunk_size,
            )

            print(f"Vpeak input dtype: {columns['Vpeak'].dtype}")
            print(f"Vpeak threshold: {threshold:.17g}")
            print(f"Vpeak > threshold: {n_greater:,}")
            print(f"Vpeak = threshold: {n_equal:,}")

            selection = stream_selection(
                columns=columns,
                threshold=threshold,
                n_target=target_count,
                n_greater=n_greater,
                graph_sample=args.graph_sample,
                output_path=output_path,
                vpeak_output_path=vpeak_output_path,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )

        dataset_paths = {
            name: dataset.name
            for name, dataset in columns.items()
        }

    written = selection["written_objects"]
    written_density = written / args.box_size**3

    metadata = {
        "catalogue_file": str(catalogue_path),
        "input_datasets": dataset_paths,
        "box_size_mpc_h": args.box_size,
        "target_number_density_h3_mpc_minus3": args.number_density,
        "target_selected_population": target_count,
        "Vpeak_input_dtype": str(columns["Vpeak"].dtype),
        "Vpeak_threshold_dtype": str(
            np.dtype(columns["Vpeak"].dtype).newbyteorder("=")
        ),
        "Vpeak_threshold": threshold,
        "objects_strictly_above_threshold": n_greater,
        "objects_equal_to_threshold": n_equal,
        "equal_threshold_objects_used": (
            selection["equal_threshold_objects_used"]
        ),
        "xyz_binary_file": str(output_path),
        "xyz_binary_dtype": "float32",
        "xyz_binary_shape": [written, 3],
        "xyz_binary_layout": "C-order interleaved x,y,z",
        "vpeak_binary_file": (
            str(vpeak_output_path)
            if vpeak_output_path is not None
            else None
        ),
        "vpeak_binary_dtype": (
            str(np.dtype(columns["Vpeak"].dtype).newbyteorder("="))
            if vpeak_output_path is not None
            else None
        ),
        "written_objects": written,
        "written_sample_number_density_h3_mpc_minus3": (
            written_density
        ),
        "is_full_density_graph_catalogue": written == target_count,
        "seed": args.seed,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    expected_xyz_bytes = written * 3 * np.dtype(np.float32).itemsize
    actual_xyz_bytes = output_path.stat().st_size

    if actual_xyz_bytes != expected_xyz_bytes:
        raise RuntimeError(
            f"XYZ binary size mismatch: expected {expected_xyz_bytes:,} "
            f"bytes, found {actual_xyz_bytes:,}."
        )

    print(f"\nWrote XYZ binary: {output_path}")
    print(
        f"XYZ shape=({written:,}, 3), dtype=float32, "
        f"bytes={actual_xyz_bytes:,}"
    )

    if vpeak_output_path is not None:
        vpeak_output_dtype = np.dtype(
            metadata["vpeak_binary_dtype"]
        )
        expected_vpeak_bytes = written * vpeak_output_dtype.itemsize
        actual_vpeak_bytes = vpeak_output_path.stat().st_size
        if actual_vpeak_bytes != expected_vpeak_bytes:
            raise RuntimeError(
                f"Vpeak binary size mismatch: expected "
                f"{expected_vpeak_bytes:,} bytes, found "
                f"{actual_vpeak_bytes:,}."
            )
        print(f"Wrote Vpeak binary: {vpeak_output_path}")

    print(f"Wrote metadata: {metadata_path}")
    print(
        "Written sample graph density: "
        f"{written_density:.6e} h^3 Mpc^-3"
    )

    if written != target_count:
        print(
            "\nWARNING: the reservoir sample spans the full box and "
            "preserves the top-Vpeak selection, but it does not preserve "
            "the target graph number density."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "catalogue_file",
        help="Path and filename of the MuchoUchuu HDF5 catalogue.",
    )

    parser.add_argument("--x-dataset", default=None)
    parser.add_argument("--y-dataset", default=None)
    parser.add_argument("--z-dataset", default=None)
    parser.add_argument("--vpeak-dataset", default=None)

    parser.add_argument(
        "--box-size",
        type=float,
        default=6000.0,
    )
    parser.add_argument(
        "--number-density",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--graph-sample",
        type=int,
        default=32000,
        help="Use 0 to write all selected halos.",
    )
    parser.add_argument(
        "--output",
        default="muchouchuu_xyz_f32.bin",
        help=(
            "Raw float32 XYZ binary output. The file contains N x 3 "
            "interleaved values in C order."
        ),
    )
    parser.add_argument(
        "--vpeak-output",
        default=None,
        help=(
            "Optional raw Vpeak binary output for the selected halos, "
            "written in the native precision of the input Vpeak dataset. "
            "It is not required by the graph builder."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
