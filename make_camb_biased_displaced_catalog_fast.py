from pathlib import Path
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "--field-dir",
    default="gaussian_camb_z0_seed3001",
)
parser.add_argument(
    "--npoints",
    type=int,
    default=108000000,
)
parser.add_argument(
    "--seed",
    type=int,
    default=5001,
)
parser.add_argument(
    "--eulerian-bias",
    type=float,
    default=1.566,
)
parser.add_argument(
    "--chunk",
    type=int,
    default=1000000,
)
parser.add_argument(
    "--output",
    default=(
        "gaussian_camb_z0_seed3001/"
        "camb_biased_bE1p566_N108000000_xyz_f32.bin"
    ),
)

args = parser.parse_args()

FIELD = Path(args.field_dir)
OUT = Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)

L = 6000.0
NMESH = 256
DX = L / NMESH
NCELL = NMESH**3

bE = float(args.eulerian_bias)
bL = bE - 1.0

delta = np.load(
    FIELD / "delta_linear_nmesh256_f32.npy",
    mmap_mode="r",
)

psi_x = np.load(
    FIELD / "psi_x_nmesh256_f32.npy",
    mmap_mode="r",
)
psi_y = np.load(
    FIELD / "psi_y_nmesh256_f32.npy",
    mmap_mode="r",
)
psi_z = np.load(
    FIELD / "psi_z_nmesh256_f32.npy",
    mmap_mode="r",
)

sigma2 = float(
    np.var(delta, dtype=np.float64)
)

print("=" * 76, flush=True)
print("FAST CAMB BIASED LAGRANGIAN TRACER CATALOGUE", flush=True)
print("=" * 76, flush=True)
print("Lbox                  :", L, flush=True)
print("NMESH                 :", NMESH, flush=True)
print("Npoints               :", args.npoints, flush=True)
print("point seed            :", args.seed, flush=True)
print("target Eulerian bias  :", bE, flush=True)
print("Lagrangian bias       :", bL, flush=True)
print("sigma_delta           :", np.sqrt(sigma2), flush=True)
print("output                :", OUT, flush=True)

# ------------------------------------------------------------
# Cell probabilities
# ------------------------------------------------------------
logw = (
    bL * np.asarray(delta, dtype=np.float64)
    - 0.5 * bL*bL*sigma2
)

logw -= np.max(logw)

w = np.exp(logw).reshape(-1)
p = w / w.sum(dtype=np.float64)

print()
print("weight diagnostics:", flush=True)
print("min weight         :", float(w.min()), flush=True)
print("max weight         :", float(w.max()), flush=True)
print("max/min            :", float(w.max()/w.min()), flush=True)
print(
    "effective cells    :",
    float(1.0 / np.sum(p*p)),
    flush=True,
)

rng = np.random.default_rng(args.seed)

print()
print("Drawing fixed-N multinomial counts...", flush=True)

counts = rng.multinomial(
    args.npoints,
    p,
)

del p
del w
del logw

print(
    "nonempty cells     :",
    int(np.count_nonzero(counts)),
    flush=True,
)
print(
    "maximum cell count :",
    int(counts.max()),
    flush=True,
)
print(
    "sum counts         :",
    int(counts.sum()),
    flush=True,
)

# ------------------------------------------------------------
# Expand counts once.
# int32 is enough because NCELL = 16,777,216.
# ------------------------------------------------------------
print()
print("Expanding cell IDs...", flush=True)

cell_ids = np.repeat(
    np.arange(NCELL, dtype=np.int32),
    counts,
)

del counts

if len(cell_ids) != args.npoints:
    raise RuntimeError("cell-id expansion mismatch")

print(
    "cell-id array MB   :",
    cell_ids.nbytes / 1024**2,
    flush=True,
)

# Shuffle so chunks sample the full volume rather than
# marching through neighboring mesh cells.
rng.shuffle(cell_ids)

# ------------------------------------------------------------
# Periodic trilinear interpolation
# ------------------------------------------------------------
def interp(field, q):
    u = q / DX

    i0 = np.floor(u).astype(np.int64)
    f = u - i0

    i0 %= NMESH
    i1 = (i0 + 1) % NMESH

    ix0 = i0[:, 0]
    iy0 = i0[:, 1]
    iz0 = i0[:, 2]

    ix1 = i1[:, 0]
    iy1 = i1[:, 1]
    iz1 = i1[:, 2]

    fx = f[:, 0]
    fy = f[:, 1]
    fz = f[:, 2]

    c000 = field[ix0, iy0, iz0]
    c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]
    c110 = field[ix1, iy1, iz0]

    c001 = field[ix0, iy0, iz1]
    c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]
    c111 = field[ix1, iy1, iz1]

    c00 = c000*(1-fx) + c100*fx
    c10 = c010*(1-fx) + c110*fx
    c01 = c001*(1-fx) + c101*fx
    c11 = c011*(1-fx) + c111*fx

    c0 = c00*(1-fy) + c10*fy
    c1 = c01*(1-fy) + c11*fy

    return c0*(1-fz) + c1*fz


# ------------------------------------------------------------
# Chunked catalogue generation
# ------------------------------------------------------------
sum_disp = np.zeros(3, np.float64)
sum_disp2 = 0.0

with open(OUT, "wb") as fout:

    for start in range(
        0,
        args.npoints,
        args.chunk,
    ):

        stop = min(
            start + args.chunk,
            args.npoints,
        )

        cid = cell_ids[start:stop].astype(
            np.int64,
            copy=False,
        )

        ix = cid // (NMESH * NMESH)
        rem = cid % (NMESH * NMESH)

        iy = rem // NMESH
        iz = rem % NMESH

        m = stop - start

        q = np.empty(
            (m, 3),
            dtype=np.float64,
        )

        q[:, 0] = (
            ix + rng.random(m)
        ) * DX

        q[:, 1] = (
            iy + rng.random(m)
        ) * DX

        q[:, 2] = (
            iz + rng.random(m)
        ) * DX

        dxp = interp(psi_x, q)
        dyp = interp(psi_y, q)
        dzp = interp(psi_z, q)

        disp = np.column_stack(
            (dxp, dyp, dzp)
        )

        x = q + disp
        x %= L

        x32 = x.astype(
            np.float32,
            copy=False,
        )

        x32[
            x32 >= np.float32(L)
        ] = np.float32(0.0)

        x32.tofile(fout)

        sum_disp += disp.sum(
            axis=0,
            dtype=np.float64,
        )

        sum_disp2 += float(
            np.sum(
                disp*disp,
                dtype=np.float64,
            )
        )

        print(
            f"written {stop:12d} / {args.npoints:12d}",
            flush=True,
        )

mean_disp = sum_disp / args.npoints
rms3d = np.sqrt(
    sum_disp2 / args.npoints
)

expected_bytes = args.npoints * 3 * 4
actual_bytes = OUT.stat().st_size

print()
print(
    "mean interpolated displacement:",
    mean_disp,
    flush=True,
)
print(
    "3D rms displacement          :",
    rms3d,
    flush=True,
)
print(
    "expected bytes               :",
    expected_bytes,
    flush=True,
)
print(
    "actual bytes                 :",
    actual_bytes,
    flush=True,
)

if actual_bytes != expected_bytes:
    raise RuntimeError(
        "output byte count mismatch"
    )

meta = {
    "field_dir": str(FIELD.resolve()),
    "output": str(OUT.resolve()),
    "npoints": int(args.npoints),
    "Lbox_hmpc": L,
    "nmesh": NMESH,
    "dx_hmpc": DX,
    "point_seed": int(args.seed),
    "target_eulerian_bias": bE,
    "lagrangian_bias": bL,
    "bias_relation": "b_E = 1 + b_L at linear order",
    "sampling": (
        "fixed-N multinomial over Lagrangian cells with "
        "lognormal weights"
    ),
    "displacement": "unchanged CAMB matter Psi(q)",
    "sigma_delta": float(np.sqrt(sigma2)),
    "mean_interpolated_displacement": mean_disp.tolist(),
    "rms_3d_displacement_hmpc": float(rms3d),
    "dtype": "float32",
    "shape": [int(args.npoints), 3],
}

meta_path = OUT.with_suffix(
    OUT.suffix + ".json"
)

meta_path.write_text(
    json.dumps(meta, indent=2)
)

print(
    "metadata                     :",
    meta_path,
    flush=True,
)
