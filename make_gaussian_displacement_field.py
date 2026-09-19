from pathlib import Path
import json

import numpy as np
import pandas as pd

PK_FILE = Path("muchouchuu_halo_pk_shells_nmesh256.csv")

OUTDIR = Path("gaussian_pk_control_seed1001")
OUTDIR.mkdir(exist_ok=True)

LBOX = 6000.0
NMESH = 256
NBAR = 5.0e-4
SHOT = 1.0 / NBAR
SEED = 1001
KMAX_USER = None

df = pd.read_csv(PK_FILE)

required = [
    "n2",
    "k_h_mpc",
    "P_raw_mpc_h3",
    "P_shot_mpc_h3",
    "P_shot_subtracted_mpc_h3",
    "Nmodes_full",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

k_tab = df["k_h_mpc"].to_numpy(dtype=float)
P_raw = df["P_raw_mpc_h3"].to_numpy(dtype=float)
P_shot = df["P_shot_mpc_h3"].to_numpy(dtype=float)
P_cl = df["P_shot_subtracted_mpc_h3"].to_numpy(dtype=float)
Nmodes = df["Nmodes_full"].to_numpy(dtype=int)

good = (
    np.isfinite(k_tab)
    & np.isfinite(P_cl)
    & (k_tab > 0.0)
    & (Nmodes > 0)
)

k_tab = k_tab[good]
P_raw = P_raw[good]
P_shot = P_shot[good]
P_cl = P_cl[good]
Nmodes = Nmodes[good]

order = np.argsort(k_tab)

k_tab = k_tab[order]
P_raw = P_raw[order]
P_shot = P_shot[order]
P_cl = P_cl[order]
Nmodes = Nmodes[order]

negative = P_cl < 0.0
nneg = int(np.sum(negative))

print("Using exact MuchoUchuu shell spectrum.")
print("Number of shells:", len(k_tab))
print("Negative shot-subtracted shells:", nneg)

if nneg:
    print(
        "Negative shell estimates will be clipped to zero "
        "for construction of the Gaussian covariance."
    )

P_cl = np.maximum(P_cl, 0.0)

kmin = float(k_tab.min())
kmax_data = float(k_tab.max())

if KMAX_USER is None:
    kmax_use = kmax_data
else:
    kmax_use = min(float(KMAX_USER), kmax_data)

print()
print("Target spectrum:")
print(f"  kmin      = {kmin:.8f} h Mpc^-1")
print(f"  kmax      = {kmax_data:.8f} h Mpc^-1")
print(f"  kmax used = {kmax_use:.8f} h Mpc^-1")
print(f"  Pshot     = {SHOT:.3f} (h^-1 Mpc)^3")

print()
print("First 12 target shells:")
for i in range(min(12, len(k_tab))):
    print(
        f"  k={k_tab[i]:.8f}  "
        f"Pcl={P_cl[i]:12.3f}  "
        f"Nmodes={Nmodes[i]:4d}"
    )

N = NMESH
V = LBOX**3
dx = LBOX / N
dv = dx**3

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

print()
print("FFT geometry:")
print("  Nmesh =", N)
print("  dx =", dx)
print("  fundamental =", 2*np.pi/LBOX)
print("  axis Nyquist =", np.pi/dx)

rng = np.random.default_rng(SEED)

print()
print("Generating real-space Gaussian white noise ...")

white = rng.normal(
    0.0,
    1.0,
    size=(N, N, N),
).astype(np.float64)

print("FFT white noise ...")

delta_k = np.fft.rfftn(white)
del white

print("Building P(k) mesh ...")

kx3 = kx[:, None, None]
ky3 = ky[None, :, None]
kz3 = kz[None, None, :]

k2 = (
    kx3*kx3
    + ky3*ky3
    + kz3*kz3
)

kmag = np.sqrt(k2)

Pmesh = np.interp(
    kmag.ravel(),
    k_tab,
    P_cl,
    left=0.0,
    right=0.0,
).reshape(kmag.shape)

Pmesh[(kmag < kmin) | (kmag > kmax_use)] = 0.0
Pmesh[0, 0, 0] = 0.0

scale = np.sqrt(Pmesh / dv)
delta_k *= scale

del scale
del Pmesh

invk2 = np.zeros_like(k2)
nonzero = k2 > 0.0
invk2[nonzero] = 1.0 / k2[nonzero]

def make_component(k_component, name):
    print("Generating", name)

    psi_k = (
        1j
        * k_component
        * invk2
        * delta_k
    )

    psi = np.fft.irfftn(
        psi_k,
        s=(N, N, N),
    ).real.astype(np.float32)

    fn = OUTDIR / f"{name}_nmesh{N}_f32.npy"
    np.save(fn, psi)

    rms = float(
        np.sqrt(
            np.mean(
                psi.astype(np.float64)**2
            )
        )
    )

    print(
        f"  {fn} "
        f"rms = {rms:.6f} "
        f"min = {float(psi.min()):.6f} "
        f"max = {float(psi.max()):.6f}"
    )

    del psi_k
    del psi

make_component(kx3, "psi_x")
make_component(ky3, "psi_y")
make_component(kz3, "psi_z")

print()
print("Recovering Gaussian density field for diagnostics ...")

delta_x = np.fft.irfftn(
    delta_k,
    s=(N, N, N),
).real

print("delta Gaussian:")
print("  mean =", float(delta_x.mean()))
print("  std  =", float(delta_x.std()))
print("  min  =", float(delta_x.min()))
print("  max  =", float(delta_x.max()))

sigma_delta = float(delta_x.std())

del delta_x
del delta_k

metadata = {
    "Lbox_hmpc": LBOX,
    "Nmesh": NMESH,
    "dx_hmpc": dx,
    "nbar_h3mpc3": NBAR,
    "shot_hmpc3": SHOT,
    "seed": SEED,
    "kmin_input_hmpc": kmin,
    "kmax_input_hmpc": kmax_data,
    "kmax_used_hmpc": kmax_use,
    "sigma_delta_grid": sigma_delta,
    "construction": "Gaussian longitudinal displacement",
}

with open(
    OUTDIR / "gaussian_displacement_metadata.json",
    "w",
) as f:
    json.dump(metadata, f, indent=2)

print()
print("Wrote displacement field to:", OUTDIR)
