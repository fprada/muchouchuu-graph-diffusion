import numpy as np
import pandas as pd
from pathlib import Path
import time

# ============================================================
# SETTINGS
# ============================================================
XYZ_FILE = "muchouchuu_xyz_f32.bin"

L = 6000.0
N_EXPECTED = 108_000_000

# Start with 256 for the quick validation run.
NMESH = 256
CHUNK = 2_000_000

KMAX_OUTPUT = 0.05
DK = 2.0 * np.pi / L

OUT = f"muchouchuu_halo_pk_nmesh{NMESH}.csv"

# ============================================================
# READ POSITIONS
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

print("File           :", p)
print("N positions    :", N)
print("Expected N     :", N_EXPECTED)

if N != N_EXPECTED:
    print("WARNING: N differs from 108,000,000")

xyz = np.memmap(
    p,
    dtype=np.float32,
    mode="r",
    shape=(N, 3)
)

# ============================================================
# BASIC MESH INFO
# ============================================================
nc = NMESH**3
dx = L / NMESH
V = L**3

print()
print("NMESH          :", NMESH)
print("cell size      :", dx, "h^-1 Mpc")
print("N mesh cells   :", nc)
print("fundamental k  :", 2*np.pi/L, "h Mpc^-1")
print("Nyquist k      :", np.pi/dx, "h Mpc^-1")

# ============================================================
# NGP DEPOSITION
# ============================================================
#
# uint32 is safe here because the mean occupancy per cell is small.
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

    # periodic wrapping
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
        elapsed = time.time() - t0
        print(
            f"{stop:12d}/{N} "
            f"({100.0*stop/N:6.2f}%) "
            f"elapsed={elapsed/60.0:.2f} min"
        )

print()
print("Deposited objects:", int(counts.sum()))

# ============================================================
# DENSITY CONTRAST
# ============================================================
mean_count = N / nc

delta = counts.astype(np.float32)
delta /= mean_count
delta -= 1.0

del counts

print("Mean count/cell :", mean_count)
print("Mean delta      :", float(delta.mean()))
print("Std delta       :", float(delta.std()))

delta = delta.reshape(
    (NMESH, NMESH, NMESH)
)

# ============================================================
# FFT
# ============================================================
print("\nComputing rfftn...")

tfft = time.time()

fk = np.fft.rfftn(delta)

print(
    "FFT elapsed    :",
    (time.time() - tfft) / 60.0,
    "min"
)

del delta

# ============================================================
# POWER NORMALIZATION
# ============================================================
#
# Continuous Fourier transform:
#
# delta_k = dx^3 FFT(delta)
#
# P(k) = |delta_k|^2 / V
#
# which gives:
#
# P(k) = V / Ncell^2 * |FFT(delta)|^2
# ============================================================
norm = V / (nc**2)

power = norm * (
    fk.real**2 + fk.imag**2
)

del fk

# ============================================================
# FOURIER WAVENUMBERS
# ============================================================
kx = 2.0*np.pi*np.fft.fftfreq(
    NMESH,
    d=dx
)

ky = 2.0*np.pi*np.fft.fftfreq(
    NMESH,
    d=dx
)

kz = 2.0*np.pi*np.fft.rfftfreq(
    NMESH,
    d=dx
)

nbin = int(
    np.floor(KMAX_OUTPUT / DK)
) + 1

sum_pk = np.zeros(
    nbin,
    dtype=np.float64
)

sum_k = np.zeros(
    nbin,
    dtype=np.float64
)

nmodes = np.zeros(
    nbin,
    dtype=np.int64
)

# ============================================================
# SPHERICAL SHELL BINNING
# ============================================================
print("\nBinning Fourier modes...")

for ix, kxi in enumerate(kx):

    ky2 = ky[:, None]**2
    kz2 = kz[None, :]**2

    kmag = np.sqrt(
        kxi*kxi + ky2 + kz2
    )

    # --------------------------------------------------------
    # NGP Fourier-window correction
    #
    # W_NGP =
    # sinc(kx dx/2)
    # sinc(ky dx/2)
    # sinc(kz dx/2)
    #
    # np.sinc(q) = sin(pi q)/(pi q)
    # --------------------------------------------------------
    wx = np.sinc(
        kxi * dx / (2.0*np.pi)
    )

    wy = np.sinc(
        ky * dx / (2.0*np.pi)
    )[:, None]

    wz = np.sinc(
        kz * dx / (2.0*np.pi)
    )[None, :]

    W = wx * wy * wz

    pp = power[ix, :, :] / (W*W)

    valid = (
        (kmag > 0.0)
        & (kmag <= KMAX_OUTPUT)
        & np.isfinite(pp)
    )

    kval = kmag[valid]
    pval = pp[valid]

    ibin = np.floor(
        kval / DK
    ).astype(np.int64)

    good = ibin < nbin

    ibin = ibin[good]
    kval = kval[good]
    pval = pval[good]

    sum_pk += np.bincount(
        ibin,
        weights=pval,
        minlength=nbin
    )

    sum_k += np.bincount(
        ibin,
        weights=kval,
        minlength=nbin
    )

    nmodes += np.bincount(
        ibin,
        minlength=nbin
    ).astype(np.int64)

    if ix % 32 == 0:
        print(
            f"ix={ix:4d}/{NMESH}"
        )

# ============================================================
# SHELL AVERAGES
# ============================================================
good = nmodes > 0

kmean = (
    sum_k[good]
    / nmodes[good]
)

pkraw = (
    sum_pk[good]
    / nmodes[good]
)

# ============================================================
# SHOT NOISE
# ============================================================
nbar = N / V

pshot = 1.0 / nbar

pksub = pkraw - pshot

out = pd.DataFrame({
    "k_h_mpc": kmean,
    "P_raw_mpc_h3": pkraw,
    "P_shot_mpc_h3": pshot,
    "P_shot_subtracted_mpc_h3": pksub,
    "Nmodes": nmodes[good]
})

out.to_csv(
    OUT,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 70)
print("POWER-SPECTRUM SUMMARY")
print("=" * 70)

print("nbar            =", nbar, "h^3 Mpc^-3")
print("1/nbar          =", pshot, "(h^-1 Mpc)^3")
print("fundamental k   =", 2*np.pi/L, "h Mpc^-1")
print("Nyquist k       =", np.pi/dx, "h Mpc^-1")
print()

print(out.head(30).to_string(index=False))

print()
print("Wrote:", OUT)
