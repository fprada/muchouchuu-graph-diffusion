from pathlib import Path
import numpy as np

FIELD = Path("gaussian_camb_z0_seed3001")
N = 256
L = 6000.0

px = np.load(
    FIELD / "psi_x_nmesh256_f32.npy",
    mmap_mode="r",
)
py = np.load(
    FIELD / "psi_y_nmesh256_f32.npy",
    mmap_mode="r",
)
pz = np.load(
    FIELD / "psi_z_nmesh256_f32.npy",
    mmap_mode="r",
)

kf = 2.0 * np.pi / L

nx = np.fft.fftfreq(N) * N
ny = np.fft.fftfreq(N) * N
nz = np.fft.rfftfreq(N) * N

kx = kf * nx[:, None, None]
ky = kf * ny[None, :, None]
kz = kf * nz[None, None, :]

print("FFT psi_x...", flush=True)
pkx = np.fft.rfftn(px)

print("FFT psi_y...", flush=True)
pky = np.fft.rfftn(py)

print("FFT psi_z...", flush=True)
pkz = np.fft.rfftn(pz)

# Psi(k) = i k delta(k)/k^2
# therefore delta(k) = -i k.Psi(k)
delta_k = -1j * (
    kx * pkx +
    ky * pky +
    kz * pkz
)

delta_k[0, 0, 0] = 0.0

print("inverse FFT delta...", flush=True)

delta = np.fft.irfftn(
    delta_k,
    s=(N, N, N),
).real.astype(np.float32)

out = FIELD / "delta_linear_nmesh256_f32.npy"
np.save(out, delta)

print()
print("mean =", float(np.mean(delta, dtype=np.float64)))
print("std  =", float(np.std(delta, dtype=np.float64)))
print("min  =", float(delta.min()))
print("max  =", float(delta.max()))
print("wrote:", out)
