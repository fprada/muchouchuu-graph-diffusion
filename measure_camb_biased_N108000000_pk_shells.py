import numpy as np
import pandas as pd
from pathlib import Path
import time

# ============================================================
# SETTINGS
# ============================================================
XYZ_FILE = "gaussian_camb_z0_seed3001/camb_biased_bE1p566_N108000000_xyz_f32.bin"

L = 6000.0
N_EXPECTED = 108_000_000

NMESH = 256
CHUNK = 2_000_000

# Exact low-k shells needed for the homogeneity calculation
KMAX_SHELL = 0.03

OUT = f"gaussian_camb_z0_seed3001/camb_biased_bE1p566_N108000000_pk_shells_nmesh{NMESH}.csv"

# ============================================================
# POSITIONS
# ============================================================
p = Path(XYZ_FILE)

if not p.exists():
    raise FileNotFoundError(p)

size = p.stat().st_size

if size % (3 * 4) != 0:
    raise RuntimeError(
        "File size is not compatible with raw float32 XYZ triples."
    )

N = size // (3 * 4)

print("File        =", p)
print("N           =", N)
print("Expected N  =", N_EXPECTED)

if N != N_EXPECTED:
    print("WARNING: N differs from expected value.")

xyz = np.memmap(
    p,
    dtype=np.float32,
    mode="r",
    shape=(N, 3)
)

# ============================================================
# MESH INFORMATION
# ============================================================
nc = NMESH**3
dx = L / NMESH
V = L**3
kf = 2.0 * np.pi / L

print()
print("NMESH       =", NMESH)
print("dx          =", dx, "h^-1 Mpc")
print("kf          =", kf, "h Mpc^-1")
print("Nyquist k   =", np.pi / dx, "h Mpc^-1")

# ============================================================
# NGP DEPOSITION
# ============================================================
counts = np.zeros(nc, dtype=np.uint32)

t0 = time.time()

print("\nDepositing tracers...")

for start in range(0, N, CHUNK):

    stop = min(start + CHUNK, N)

    pos = np.asarray(
        xyz[start:stop],
        dtype=np.float64
    )

    # Periodic wrapping
    pos %= L

    ijk = np.floor(pos / dx).astype(np.int64)
    ijk %= NMESH

    flat = (
        ijk[:, 0] * NMESH * NMESH
        + ijk[:, 1] * NMESH
        + ijk[:, 2]
    )

    bc = np.bincount(
        flat,
        minlength=nc
    )

    counts += bc.astype(np.uint32)

    del pos, ijk, flat, bc

    if (
        start == 0
        or stop == N
        or (start // CHUNK) % 5 == 0
    ):
        print(
            f"{stop:12d}/{N} "
            f"({100.0 * stop / N:6.2f}%) "
            f"elapsed={(time.time() - t0)/60.0:.2f} min"
        )

print()
print("Deposited objects =", int(counts.sum()))

# ============================================================
# DENSITY CONTRAST
# ============================================================
mean_count = N / nc

delta = counts.astype(np.float32)
delta /= mean_count
delta -= 1.0

del counts

print("Mean count/cell   =", mean_count)
print("Mean delta        =", float(delta.mean()))
print("Std delta         =", float(delta.std()))

delta = delta.reshape(
    (NMESH, NMESH, NMESH)
)

# ============================================================
# FFT
# ============================================================
print("\nComputing FFT...")

tfft = time.time()

fk = np.fft.rfftn(delta)

print(
    "FFT elapsed      =",
    (time.time() - tfft) / 60.0,
    "min"
)

del delta

# ============================================================
# POWER NORMALIZATION
# ============================================================
#
# delta_k = dx^3 FFT(delta)
#
# P(k) = |delta_k|^2 / V
#      = V / Ncell^2 |FFT(delta)|^2
#
norm = V / (nc**2)

# ============================================================
# INTEGER FOURIER COORDINATES
# ============================================================
nx = np.rint(
    np.fft.fftfreq(NMESH) * NMESH
).astype(np.int64)

ny = np.rint(
    np.fft.fftfreq(NMESH) * NMESH
).astype(np.int64)

nz = np.rint(
    np.fft.rfftfreq(NMESH) * NMESH
).astype(np.int64)

nmax = int(np.ceil(KMAX_SHELL / kf))
n2max = nmax**2

print()
print("Maximum n        =", nmax)
print("Maximum n^2      =", n2max)

# ============================================================
# HERMITIAN MULTIPLICITIES FOR rfftn
# ============================================================
#
# rfftn stores only kz >= 0.
#
# kz = 0 occurs once.
# For even NMESH, kz = Nyquist also occurs once.
# All other positive-kz modes represent both +kz and -kz.
#
herm_z = np.ones(len(nz), dtype=np.int64)

if NMESH % 2 == 0:
    mask_double = (
        (nz > 0)
        & (nz < NMESH // 2)
    )
else:
    mask_double = nz > 0

herm_z[mask_double] = 2

# ============================================================
# EXACT n^2 SHELL ACCUMULATORS
# ============================================================
sumP = np.zeros(
    n2max + 1,
    dtype=np.float64
)

count = np.zeros(
    n2max + 1,
    dtype=np.int64
)

print("\nAccumulating exact Fourier shells...")

# ============================================================
# LOOP OVER kx PLANES
# ============================================================
for ix, nxi in enumerate(nx):

    # Integer shell index n^2
    n2 = (
        nxi * nxi
        + ny[:, None]**2
        + nz[None, :]**2
    )

    valid = (
        (n2 > 0)
        & (n2 <= n2max)
    )

    # Physical Fourier components
    kxi = kf * nxi
    ky = kf * ny[:, None]
    kz = kf * nz[None, :]

    # --------------------------------------------------------
    # NGP window correction
    #
    # W_NGP(k) =
    # sinc(kx dx/2)
    # sinc(ky dx/2)
    # sinc(kz dx/2)
    #
    # np.sinc(q) = sin(pi q)/(pi q)
    # --------------------------------------------------------
    wx = np.sinc(
        kxi * dx / (2.0 * np.pi)
    )

    wy = np.sinc(
        ky * dx / (2.0 * np.pi)
    )

    wz = np.sinc(
        kz * dx / (2.0 * np.pi)
    )

    W = wx * wy * wz

    # Power in the stored rfftn modes
    pplane = norm * (
        fk[ix, :, :].real**2
        + fk[ix, :, :].imag**2
    )

    pplane /= W**2

    shell = n2[valid].astype(np.int64)
    vals = pplane[valid]

    # Hermitian multiplicity corresponding to each kz
    herm_plane = np.broadcast_to(
        herm_z[None, :],
        n2.shape
    )

    herm = herm_plane[valid]

    # Weighted power sum over the full Fourier shell
    sumP += np.bincount(
        shell,
        weights=vals * herm,
        minlength=n2max + 1
    )

    # Full number of Fourier modes in each shell
    count += np.bincount(
        shell,
        weights=herm,
        minlength=n2max + 1
    ).astype(np.int64)

    if ix % 32 == 0:
        print(
            f"ix = {ix:4d}/{NMESH}"
        )

del fk

# ============================================================
# SHELL AVERAGES
# ============================================================
n2_all = np.arange(
    n2max + 1
)

good = (
    (n2_all > 0)
    & (count > 0)
)

n2_shell = np.asarray(n2_all[good], dtype=np.int64)
nmodes_full = np.asarray(count[good], dtype=np.int64)

k_shell = kf * np.sqrt(
    n2_shell.astype(np.float64)
)

Praw = np.asarray(
    sumP[good] / nmodes_full.astype(np.float64),
    dtype=np.float64,
)

# ============================================================
# SHOT NOISE
# ============================================================
nbar = N / V
Pshot = float(1.0 / nbar)

Psub = Praw - Pshot

# ============================================================
# OUTPUT TABLE
# ============================================================
print("DEBUG shell arrays:")
print(" n2_shell     :", type(n2_shell), n2_shell.dtype, n2_shell.shape)
print(" nmodes_full  :", type(nmodes_full), nmodes_full.dtype, nmodes_full.shape)
print(" k_shell      :", type(k_shell), k_shell.dtype, k_shell.shape)
print(" Praw         :", type(Praw), Praw.dtype, Praw.shape)
print(" Pshot        :", type(Pshot), Pshot)

out = pd.DataFrame({
    "n2": n2_shell,
    "k_h_mpc": k_shell,
    "P_raw_mpc_h3": Praw,
    "P_shot_mpc_h3": np.full(Praw.shape, Pshot, dtype=np.float64),
    "P_shot_subtracted_mpc_h3": Psub,
    "Nmodes_full": nmodes_full,
})

out = out[
    out["k_h_mpc"] <= KMAX_SHELL
].copy()

out.to_csv(
    OUT,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 78)
print("EXACT-SHELL POWER-SPECTRUM SUMMARY")
print("=" * 78)

print("N               =", N)
print("nbar            =", nbar, "h^3 Mpc^-3")
print("1/nbar          =", Pshot, "(h^-1 Mpc)^3")
print("fundamental k   =", kf, "h Mpc^-1")
print()

print(
    out.head(50).to_string(index=False)
)

print()
print("First-shell checks:")

for target in [1, 2, 3, 4]:
    r = out[out["n2"] == target]

    if len(r):
        print(
            f"n2={target:2d}: "
            f"k={r.iloc[0]['k_h_mpc']:.8f}, "
            f"Nmodes_full={int(r.iloc[0]['Nmodes_full'])}, "
            f"P={r.iloc[0]['P_raw_mpc_h3']:.3f}"
        )

print()
print("Expected mode counts at lowest shells:")
print("n2=1 -> 6")
print("n2=2 -> 12")
print("n2=3 -> 8")
print("n2=4 -> 6")

print()
print("Wrote:", OUT)
