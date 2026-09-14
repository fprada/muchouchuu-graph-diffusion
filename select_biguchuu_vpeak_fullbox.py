
#!/usr/bin/env python3
"""
Select a Vpeak-limited BigUchuu population from two row-aligned HDF5 files.

Files
-----
BigUchuu_hlist_050_2.h5, data_id=2:
    structured field Vpeak

BigUchuu_hlist_050_4.h5, data_id=4:
    structured fields x, y, z

The target population is the top N objects by Vpeak, where

    N = number_density * box_size**3.

For L=4000 h^-1 Mpc and n=5e-4 h^3 Mpc^-3:

    N = 32,000,000.

The program finds the exact Vpeak threshold using a disk-backed temporary
memmap, then performs a second streaming pass through the original files.

Output modes
------------
1. --graph-sample 0
   Write every selected halo. This can be very large.

2. --graph-sample N
   Reservoir-sample N halos from the full Vpeak-selected population while
   preserving coverage of the full 4 Gpc/h box.

Important scientific caveat
---------------------------
A random graph sample with N << 32,000,000 does NOT retain the target graph
number density. It retains the Vpeak selection function and full-box coverage,
but its graph density is N / L^3. For diffusion geometry at the original
density, the graph itself must contain all selected halos, or one must use
subvolumes / distributed graph construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def resolve_dataset(h5: h5py.File, data_id: str) -> h5py.Dataset:
    candidates = [str(data_id), f"data_{data_id}"]

    for key in candidates:
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return h5[key]

    matches = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            if leaf in candidates:
                matches.append(name)

    h5.visititems(visitor)

    if len(matches) == 1:
        return h5[matches[0]]

    available = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            available.append(name)

    h5.visititems(collect)

    raise KeyError(
        f"Could not resolve data_id={data_id}. "
        f"Available datasets include: {available[:50]}"
    )


def validate_datasets(vpeak_ds: h5py.Dataset, xyz_ds: h5py.Dataset) -> None:
    if vpeak_ds.shape[0] != xyz_ds.shape[0]:
        raise ValueError(
            "The Vpeak and coordinate datasets have different row counts: "
            f"{vpeak_ds.shape[0]} versus {xyz_ds.shape[0]}. "
            "They cannot be joined row-by-row."
        )

    if vpeak_ds.dtype.names is None or "Vpeak" not in vpeak_ds.dtype.names:
        raise ValueError(
            f"Vpeak dataset fields are {vpeak_ds.dtype.names}; "
            "expected a field named 'Vpeak'."
        )

    if xyz_ds.dtype.names is None:
        raise ValueError("Coordinate dataset is not a structured array.")

    missing = {"x", "y", "z"} - set(xyz_ds.dtype.names)
    if missing:
        raise ValueError(
            f"Coordinate dataset is missing fields: {sorted(missing)}."
        )


def write_vpeak_memmap(
    dataset: h5py.Dataset,
    memmap_path: Path,
    chunk_size: int,
) -> np.memmap:
    n_total = dataset.shape[0]
    values = np.memmap(
        memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_total,),
    )

    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        values[start:stop] = dataset[start:stop]["Vpeak"]
        if start % (10 * chunk_size) == 0:
            print(f"Copied Vpeak rows {start:,}–{stop:,}")

    values.flush()
    return values


def exact_top_threshold(
    values: np.memmap,
    n_target: int,
) -> float:
    """
    Find the exact kth-order threshold in place on the temporary memmap.

    The memmap is temporary, so modifying its order is harmless.
    """
    n_total = len(values)
    if not 0 < n_target <= n_total:
        raise ValueError(
            f"n_target must be in [1, {n_total}], received {n_target}."
        )

    kth = n_total - n_target
    print(
        f"Partitioning {n_total:,} Vpeak values to identify "
        f"the top {n_target:,}..."
    )
    values.partition(kth)
    threshold = float(values[kth])
    values.flush()
    return threshold


def count_threshold_members(
    dataset: h5py.Dataset,
    threshold: float,
    chunk_size: int,
) -> tuple[int, int]:
    greater = 0
    equal = 0

    for start in range(0, dataset.shape[0], chunk_size):
        stop = min(start + chunk_size, dataset.shape[0])
        v = dataset[start:stop]["Vpeak"]
        greater += int(np.count_nonzero(v > threshold))
        equal += int(np.count_nonzero(v == threshold))

    return greater, equal


def xyz_from_structured(block: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            block["x"].astype(np.float32, copy=False),
            block["y"].astype(np.float32, copy=False),
            block["z"].astype(np.float32, copy=False),
        ]
    )


def reservoir_or_full_selection(
    vpeak_ds: h5py.Dataset,
    xyz_ds: h5py.Dataset,
    threshold: float,
    n_target: int,
    n_greater: int,
    graph_sample: int,
    output_path: Path,
    chunk_size: int,
    seed: int,
) -> dict:
    """
    Stream the exact selected population.

    Objects with Vpeak > threshold are always eligible. Enough objects with
    Vpeak == threshold are then included to make the population exactly
    n_target. Equal-threshold rows are accepted in row order, which is
    deterministic.

    If graph_sample > 0, reservoir sampling is applied to this exact selected
    stream. Otherwise all selected coordinates and Vpeak values are written.
    """
    equals_needed = n_target - n_greater
    equals_used = 0
    selected_seen = 0

    rng = np.random.default_rng(seed)

    if graph_sample > 0:
        sample_size = min(graph_sample, n_target)
        reservoir_xyz = np.empty((sample_size, 3), dtype=np.float32)
        reservoir_vpeak = np.empty(sample_size, dtype=np.float32)
    else:
        sample_size = n_target
        out_h5 = h5py.File(output_path, "w")
        out_xyz = out_h5.create_dataset(
            "xyz",
            shape=(n_target, 3),
            dtype="f4",
            chunks=(min(chunk_size, n_target), 3),
            compression="gzip",
            compression_opts=1,
        )
        out_vpeak = out_h5.create_dataset(
            "Vpeak",
            shape=(n_target,),
            dtype="f4",
            chunks=(min(chunk_size, n_target),),
            compression="gzip",
            compression_opts=1,
        )
        write_position = 0

    for start in range(0, vpeak_ds.shape[0], chunk_size):
        stop = min(start + chunk_size, vpeak_ds.shape[0])

        v = vpeak_ds[start:stop]["Vpeak"]
        coords = xyz_from_structured(xyz_ds[start:stop])

        greater_mask = v > threshold
        equal_indices = np.flatnonzero(v == threshold)

        if equals_used < equals_needed and len(equal_indices):
            take_equal = min(
                equals_needed - equals_used,
                len(equal_indices),
            )
            equal_mask = np.zeros(len(v), dtype=bool)
            equal_mask[equal_indices[:take_equal]] = True
            equals_used += take_equal
            selected_mask = greater_mask | equal_mask
        else:
            selected_mask = greater_mask

        selected_v = v[selected_mask].astype(np.float32, copy=False)
        selected_xyz = coords[selected_mask]

        if graph_sample > 0:
            for point, vp in zip(selected_xyz, selected_v):
                if selected_seen < sample_size:
                    reservoir_xyz[selected_seen] = point
                    reservoir_vpeak[selected_seen] = vp
                else:
                    j = int(rng.integers(0, selected_seen + 1))
                    if j < sample_size:
                        reservoir_xyz[j] = point
                        reservoir_vpeak[j] = vp
                selected_seen += 1
        else:
            m = len(selected_v)
            out_xyz[write_position:write_position + m] = selected_xyz
            out_vpeak[write_position:write_position + m] = selected_v
            write_position += m
            selected_seen += m

        if start % (10 * chunk_size) == 0:
            print(
                f"Selection pass rows {start:,}–{stop:,}; "
                f"selected stream count={selected_seen:,}"
            )

    if selected_seen != n_target:
        raise RuntimeError(
            f"Expected exactly {n_target:,} selected objects, "
            f"but streamed {selected_seen:,}."
        )

    if graph_sample > 0:
        with h5py.File(output_path, "w") as out_h5:
            out_h5.create_dataset(
                "xyz",
                data=reservoir_xyz,
                compression="gzip",
                compression_opts=1,
            )
            out_h5.create_dataset(
                "Vpeak",
                data=reservoir_vpeak,
                compression="gzip",
                compression_opts=1,
            )
    else:
        out_h5.close()

    return {
        "full_selected_population": n_target,
        "written_objects": sample_size,
        "equal_threshold_objects_used": equals_used,
    }


def main(args: argparse.Namespace) -> None:
    vpeak_path = Path(args.vpeak_file).expanduser().resolve()
    xyz_path = Path(args.xyz_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_count = int(
        round(args.number_density * args.box_size**3)
    )

    with h5py.File(vpeak_path, "r") as fv, h5py.File(xyz_path, "r") as fx:
        vpeak_ds = resolve_dataset(fv, args.vpeak_data_id)
        xyz_ds = resolve_dataset(fx, args.xyz_data_id)
        validate_datasets(vpeak_ds, xyz_ds)

        n_total = vpeak_ds.shape[0]
        if target_count > n_total:
            raise ValueError(
                f"Target density requires {target_count:,} halos, "
                f"but the catalogue contains only {n_total:,}."
            )

        print(f"Vpeak dataset: {vpeak_ds.name}, rows={n_total:,}")
        print(f"XYZ dataset: {xyz_ds.name}, rows={xyz_ds.shape[0]:,}")
        print(f"Target selected count: {target_count:,}")

        with tempfile.TemporaryDirectory(
            dir=args.temp_dir
        ) as temp_dir:
            mmap_path = Path(temp_dir) / "vpeak_float32.dat"
            values = write_vpeak_memmap(
                vpeak_ds,
                mmap_path,
                args.chunk_size,
            )
            threshold = exact_top_threshold(
                values,
                target_count,
            )
            del values

            n_greater, n_equal = count_threshold_members(
                vpeak_ds,
                threshold,
                args.chunk_size,
            )

            print(f"Vpeak threshold: {threshold:.8g}")
            print(f"Objects with Vpeak > threshold: {n_greater:,}")
            print(f"Objects with Vpeak = threshold: {n_equal:,}")

            selection_info = reservoir_or_full_selection(
                vpeak_ds=vpeak_ds,
                xyz_ds=xyz_ds,
                threshold=threshold,
                n_target=target_count,
                n_greater=n_greater,
                graph_sample=args.graph_sample,
                output_path=output_path,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )

    written = selection_info["written_objects"]
    graph_density = written / args.box_size**3

    metadata = {
        "vpeak_file": str(vpeak_path),
        "xyz_file": str(xyz_path),
        "vpeak_data_id": str(args.vpeak_data_id),
        "xyz_data_id": str(args.xyz_data_id),
        "box_size_mpc_h": args.box_size,
        "target_number_density_h3_mpc_minus3": args.number_density,
        "target_selected_population": target_count,
        "Vpeak_threshold": threshold,
        "objects_strictly_above_threshold": n_greater,
        "objects_equal_to_threshold": n_equal,
        "written_objects": written,
        "written_sample_number_density_h3_mpc_minus3": graph_density,
        "is_full_density_graph_catalogue": written == target_count,
        "selection": (
            "Exact top-Vpeak population, with deterministic tie handling; "
            "optional reservoir sample from that population"
        ),
        "seed": args.seed,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nWrote: {output_path}")
    print(f"Wrote: {metadata_path}")
    print(
        f"Written sample graph density: {graph_density:.6e} "
        "h^3 Mpc^-3"
    )

    if written != target_count:
        print(
            "\nWARNING: this sample spans the full box and preserves the "
            "Vpeak selection, but its graph density is lower than the target. "
            "Do not interpret its diffusion graph as a graph built at "
            "5e-4 h^3 Mpc^-3."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("vpeak_file")
    p.add_argument("xyz_file")
    p.add_argument(
        "--vpeak-data-id",
        default="2",
    )
    p.add_argument(
        "--xyz-data-id",
        default="4",
    )
    p.add_argument(
        "--box-size",
        type=float,
        default=4000.0,
    )
    p.add_argument(
        "--number-density",
        type=float,
        default=5e-4,
    )
    p.add_argument(
        "--graph-sample",
        type=int,
        default=32000,
        help=(
            "Number of selected halos to reservoir-sample for a pilot graph. "
            "Use 0 to write all selected halos."
        ),
    )
    p.add_argument(
        "--output",
        default="biguchuu_vpeak_selected.h5",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for the temporary disk-backed Vpeak array.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
