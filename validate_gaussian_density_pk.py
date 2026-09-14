from pathlib import Path
import numpy as np
import pandas as pd

PK_FILE = Path("muchouchuu_halo_pk_shells_nmesh256.csv")
FIELD_DIR = Path("gaussian_pk_control_seed1001")

L = 6000.0
N = 256
V = L**3
dx = L / N
dv = dx**3

target = pd.read_csv(PK_FILE)

px = np.load(FIELD_DIR / "psi_x_nmesh256_f32.npy")
py = np.load(FIELD_DIR / "psi_y_nmesh256_f32.npy")
pz = np.load(FIELD_DIR / "psi_z_nmesh256_f32.npy")

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)

PX = np.fft.rfftn(px)
PYK = np.fft.rfftn(py)
PZ = np.fft.rfftn(pz)

delta_k = -1j * (
    kx[:, None, None] * PX
    + ky[None, :, None] * PYK
    + kz[None, None, :] * PZ
)

del PX, PYK, PZ, px, py, pz

nx = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
ny = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
nz = np.rint(np.fft.rfftfreq(N) * N).astype(np.int64)

n2 = (
    nx[:, None, None]**2
    + ny[None, :, None]**2
    + nz[None, None, :]**2
)

# rFFT multiplicity for missing negative-z partners.
wz = np.ones(len(nz), dtype=np.int64)

if N % 2 == 0:
    wz[1:-1] = 2
else:
    wz[1:] = 2

weights = np.broadcast_to(
    wz[None, None, :],
    delta_k.shape,
)

# Continuous Fourier normalization:
# delta_cont(k) = dv * FFT(delta)
# P(k) = |delta_cont|^2 / V
delta_cont = dv * delta_k
power_mode = np.abs(delta_cont)**2 / V

rows = []

for n2val in target["n2"].to_numpy(dtype=np.int64):
    mask = n2 == n2val

    if not np.any(mask):
        continue

    ww = weights[mask].astype(np.float64)
    pp = power_mode[mask]

    Pmean = np.sum(ww * pp) / np.sum(ww)

    rows.append(
        (
            int(n2val),
            float(Pmean),
            int(np.sum(ww)),
        )
    )

out = pd.DataFrame(
    rows,
    columns=[
        "n2",
        "P_gaussian_mpc_h3",
        "Nmodes_full_gaussian",
    ],
)

# IMPORTANT: merge only on exact integer shell label n2.
merged = target.merge(
    out,
    on="n2",
    how="inner",
)

merged["ratio_gaussian_to_target"] = (
    merged["P_gaussian_mpc_h3"]
    / merged["P_shot_subtracted_mpc_h3"]
)

merged.to_csv(
    FIELD_DIR / "gaussian_density_pk_validation.csv",
    index=False,
)

sel = (
    (merged["k_h_mpc"] >= 0.001)
    & (merged["k_h_mpc"] <= 0.010)
    & (merged["P_shot_subtracted_mpc_h3"] > 0.0)
)

x = merged.loc[sel].copy()

print("=" * 72)
print("GAUSSIAN FIELD P(k) VALIDATION")
print("=" * 72)

print("matched target shells              :", len(merged))
print("shells in 0.001 <= k <= 0.010     :", len(x))

print(
    "median Pgen/Ptarget                :",
    np.median(x["ratio_gaussian_to_target"]),
)

# Mode-weighted ratio of total generated to target power.
num = np.sum(
    x["Nmodes_full"]
    * x["P_gaussian_mpc_h3"]
)

den = np.sum(
    x["Nmodes_full"]
    * x["P_shot_subtracted_mpc_h3"]
)

print(
    "mode-weighted Pgen/Ptarget         :",
    num / den,
)

# Also compare mean logarithmic offset, less dominated by large-P shells.
logratio = np.log(
    x["P_gaussian_mpc_h3"]
    / x["P_shot_subtracted_mpc_h3"]
)

print(
    "exp(mean ln(Pgen/Ptarget))         :",
    np.exp(np.mean(logratio)),
)

print()
print("First 20 low-k shells:")
print(
    x[
        [
            "n2",
            "k_h_mpc",
            "Nmodes_full",
            "P_shot_subtracted_mpc_h3",
            "P_gaussian_mpc_h3",
            "ratio_gaussian_to_target",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda z: f"{z:.6g}",
    )
)

print()
print(
    "Wrote:",
    FIELD_DIR / "gaussian_density_pk_validation.csv",
)
