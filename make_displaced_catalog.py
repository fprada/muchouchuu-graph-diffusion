from pathlib import Path
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "--field-dir",
    default="gaussian_pk_control_seed1001",
)

parser.add_argument(
    "--npoints",
    type=int,
    default=16777216,
)

parser.add_argument(
    "--seed",
    type=int,
    default=2001,
)

parser.add_argument(
    "--chunk",
    type=int,
    default=1000000,
)

parser.add_argument(
    "--output",
    required=True,
)

args = parser.parse_args()


FIELD_DIR = Path(args.field_dir)
OUTFILE = Path(args.output)

LBOX = 6000.0

px = np.load(
    FIELD_DIR / "psi_x_nmesh256_f32.npy",
    mmap_mode="r",
)

py = np.load(
    FIELD_DIR / "psi_y_nmesh256_f32.npy",
    mmap_mode="r",
)

pz = np.load(
    FIELD_DIR / "psi_z_nmesh256_f32.npy",
    mmap_mode="r",
)

if px.shape != py.shape or px.shape != pz.shape:
    raise RuntimeError(
        f"Inconsistent displacement shapes: "
        f"{px.shape}, {py.shape}, {pz.shape}"
    )

if (
    px.ndim != 3
    or px.shape[0] != px.shape[1]
    or px.shape[0] != px.shape[2]
):
    raise RuntimeError(
        f"Expected a cubic 3D displacement grid, got {px.shape}"
    )

NMESH = px.shape[0]
DX = LBOX / NMESH

print("=" * 72)
print("GAUSSIAN DISPLACED POINT CATALOGUE")
print("=" * 72)
print("field directory :", FIELD_DIR)
print("Nmesh           :", NMESH)
print("Lbox            :", LBOX)
print("dx              :", DX)
print("Npoints         :", args.npoints)
print("point seed      :", args.seed)
print("chunk size      :", args.chunk)
print("output          :", OUTFILE)

nbar = args.npoints / LBOX**3
shot = 1.0 / nbar

print("nbar            :", nbar)
print("shot noise      :", shot)
print()


def interpolate_periodic(field, q):
    """
    Periodic trilinear interpolation.

    field : (N,N,N)
    q     : (M,3), positions in [0,LBOX)
    """

    u = q / DX

    i0 = np.floor(u).astype(np.int64)
    frac = u - i0

    i0 %= NMESH
    i1 = (i0 + 1) % NMESH

    ix0 = i0[:, 0]
    iy0 = i0[:, 1]
    iz0 = i0[:, 2]

    ix1 = i1[:, 0]
    iy1 = i1[:, 1]
    iz1 = i1[:, 2]

    fx = frac[:, 0]
    fy = frac[:, 1]
    fz = frac[:, 2]

    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz

    wx1 = fx
    wy1 = fy
    wz1 = fz

    result = (
        field[ix0, iy0, iz0] * wx0 * wy0 * wz0
        + field[ix1, iy0, iz0] * wx1 * wy0 * wz0
        + field[ix0, iy1, iz0] * wx0 * wy1 * wz0
        + field[ix0, iy0, iz1] * wx0 * wy0 * wz1
        + field[ix1, iy1, iz0] * wx1 * wy1 * wz0
        + field[ix1, iy0, iz1] * wx1 * wy0 * wz1
        + field[ix0, iy1, iz1] * wx0 * wy1 * wz1
        + field[ix1, iy1, iz1] * wx1 * wy1 * wz1
    )

    return result


rng = np.random.default_rng(args.seed)

OUTFILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

written = 0

sum_disp2 = 0.0
sum_disp = np.zeros(3, dtype=np.float64)

with open(OUTFILE, "wb") as fout:

    while written < args.npoints:

        m = min(
            args.chunk,
            args.npoints - written,
        )

        # Homogeneous Poisson reference catalogue.
        q = rng.uniform(
            0.0,
            LBOX,
            size=(m, 3),
        )

        dxp = interpolate_periodic(px, q)
        dyp = interpolate_periodic(py, q)
        dzp = interpolate_periodic(pz, q)

        sum_disp[0] += np.sum(dxp, dtype=np.float64)
        sum_disp[1] += np.sum(dyp, dtype=np.float64)
        sum_disp[2] += np.sum(dzp, dtype=np.float64)

        sum_disp2 += (
            np.sum(dxp.astype(np.float64)**2)
            + np.sum(dyp.astype(np.float64)**2)
            + np.sum(dzp.astype(np.float64)**2)
        )

        q[:, 0] += dxp
        q[:, 1] += dyp
        q[:, 2] += dzp

        # Periodic wrapping.
        q %= LBOX

        q32 = q.astype(
            np.float32,
            copy=False,
        )

        # Values infinitesimally below LBOX can round to exactly
        # LBOX when converted to float32. Map those periodically to zero.
        q32[q32 >= np.float32(LBOX)] = np.float32(0.0)

        q32.tofile(fout)

        written += m

        print(
            f"{written:12d} / {args.npoints:12d}",
            flush=True,
        )


mean_disp = sum_disp / args.npoints

rms_3d = np.sqrt(
    sum_disp2 / args.npoints
)

print()
print("Mean interpolated displacement:")
print(
    f"  x={mean_disp[0]:.6e} "
    f"y={mean_disp[1]:.6e} "
    f"z={mean_disp[2]:.6e}"
)

print(
    "3D rms interpolated displacement:",
    rms_3d,
)

expected_bytes = args.npoints * 3 * 4
actual_bytes = OUTFILE.stat().st_size

print()
print("expected bytes :", expected_bytes)
print("actual bytes   :", actual_bytes)

if actual_bytes != expected_bytes:
    raise RuntimeError(
        "Output file size is inconsistent with "
        "the requested N x 3 float32 catalogue."
    )

metadata = {
    "npoints": int(args.npoints),
    "Lbox_hmpc": LBOX,
    "nbar_h3mpc3": nbar,
    "shot_noise_hmpc3": shot,
    "point_seed": int(args.seed),
    "field_directory": str(FIELD_DIR),
    "field_nmesh": int(NMESH),
    "field_dx_hmpc": DX,
    "dtype": "float32",
    "shape": [int(args.npoints), 3],
    "three_dimensional_rms_displacement_hmpc": float(rms_3d),
}

with open(
    str(OUTFILE) + ".json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

print()
print("Wrote:", OUTFILE)
print("Wrote:", str(OUTFILE) + ".json")
