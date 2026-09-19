from pathlib import Path
import argparse
import json

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pk-file",
        default="/mnt/data/PkTable.dat",
    )
    p.add_argument(
        "--output-dir",
        default="gaussian_camb_z0_seed3001",
    )
    p.add_argument(
        "--box-size",
        type=float,
        default=6000.0,
    )
    p.add_argument(
        "--nmesh",
        type=int,
        default=256,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=3001,
    )

    return p.parse_args()


def read_camb_pk(path):
    """
    CAMB table:
        column 1 = k [h Mpc^-1]
        column 2 = P(k) [(h^-1 Mpc)^3]
    """
    a = np.loadtxt(path, comments="#")

    if a.ndim != 2 or a.shape[1] < 2:
        raise RuntimeError("Expected a two-column CAMB P(k) table.")

    k = np.asarray(a[:, 0], dtype=np.float64)
    p = np.asarray(a[:, 1], dtype=np.float64)

    good = (
        np.isfinite(k)
        & np.isfinite(p)
        & (k > 0.0)
        & (p >= 0.0)
    )

    k = k[good]
    p = p[good]

    order = np.argsort(k)
    k = k[order]
    p = p[order]

    if np.any(np.diff(k) <= 0.0):
        raise RuntimeError("CAMB k grid is not strictly increasing.")

    return k, p


def main():
    a = parse_args()

    pk_file = Path(a.pk_file).expanduser().resolve()
    outdir = Path(a.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    L = float(a.box_size)
    nmesh = int(a.nmesh)
    seed = int(a.seed)

    dx = L / nmesh
    dv = dx**3

    k_input, p_input = read_camb_pk(pk_file)

    print("=" * 76)
    print("CAMB z=0 GAUSSIAN DISPLACEMENT FIELD")
    print("=" * 76)
    print()
    print("P(k) file       :", pk_file)
    print("L               :", L)
    print("NMESH           :", nmesh)
    print("dx              :", dx)
    print("seed            :", seed)
    print()
    print("CAMB k min/max  :", k_input.min(), k_input.max())
    print("CAMB P min/max  :", p_input.min(), p_input.max())

    kfund = 2.0 * np.pi / L
    kny = np.pi / dx

    print("fundamental k   :", kfund)
    print("axis Nyquist k  :", kny)

    # FFT mode grid.
    nx = np.fft.fftfreq(nmesh) * nmesh
    ny = np.fft.fftfreq(nmesh) * nmesh
    nz = np.fft.rfftfreq(nmesh) * nmesh

    kx = kfund * nx[:, None, None]
    ky = kfund * ny[None, :, None]
    kz = kfund * nz[None, None, :]

    k2 = kx*kx + ky*ky + kz*kz
    kmag = np.sqrt(k2)

    # Interpolate CAMB P(k) in log k - log P.
    # No extrapolated clustering power outside the supplied CAMB range.
    Pgrid = np.zeros_like(kmag, dtype=np.float64)

    m = (
        (kmag > 0.0)
        & (kmag >= k_input[0])
        & (kmag <= k_input[-1])
    )

    logk = np.log(k_input)
    logp = np.log(p_input)

    Pgrid[m] = np.exp(
        np.interp(
            np.log(kmag[m]),
            logk,
            logp,
        )
    )

    print()
    print("FFT modes inside CAMB range :", int(np.count_nonzero(m)))
    print("nonzero Pgrid modes         :", int(np.count_nonzero(Pgrid)))

    # Generate real-space unit-variance white noise.
    rng = np.random.default_rng(seed)

    white = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(nmesh, nmesh, nmesh),
    )

    # numpy FFT convention:
    #
    # For white noise with Var(w_x)=1,
    # E[|FFT(w)|^2] = Ncell.
    #
    # Multiplying by sqrt(P(k)/dV) gives a density field whose
    # Fourier covariance corresponds to the desired continuum P(k).
    wk = np.fft.rfftn(white)

    scale = np.sqrt(Pgrid / dv)
    delta_k = wk * scale

    # Explicitly remove the DC mode.
    delta_k[0, 0, 0] = 0.0

    # Linear longitudinal displacement:
    #
    #   Psi(k) = i k/k^2 delta(k)
    #
    # so that -div Psi = delta.
    invk2 = np.zeros_like(k2, dtype=np.float64)
    nzmode = k2 > 0.0
    invk2[nzmode] = 1.0 / k2[nzmode]

    psi_x_k = 1j * kx * invk2 * delta_k
    psi_y_k = 1j * ky * invk2 * delta_k
    psi_z_k = 1j * kz * invk2 * delta_k

    del wk
    del delta_k
    del scale
    del Pgrid
    del invk2

    print()
    print("Transforming displacement components to real space...")

    psi_x = np.fft.irfftn(
        psi_x_k,
        s=(nmesh, nmesh, nmesh),
    ).real.astype(np.float32)

    del psi_x_k

    psi_y = np.fft.irfftn(
        psi_y_k,
        s=(nmesh, nmesh, nmesh),
    ).real.astype(np.float32)

    del psi_y_k

    psi_z = np.fft.irfftn(
        psi_z_k,
        s=(nmesh, nmesh, nmesh),
    ).real.astype(np.float32)

    del psi_z_k

    # Also reconstruct delta for diagnostics.
    # Using -div(Psi) in Fourier space would require retaining Fourier arrays,
    # so regenerate the same white field deterministically.
    rng = np.random.default_rng(seed)
    white2 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(nmesh, nmesh, nmesh),
    )

    wk2 = np.fft.rfftn(white2)

    Pgrid2 = np.zeros_like(kmag, dtype=np.float64)
    Pgrid2[m] = np.exp(
        np.interp(
            np.log(kmag[m]),
            logk,
            logp,
        )
    )

    delta_k2 = wk2 * np.sqrt(Pgrid2 / dv)
    delta_k2[0, 0, 0] = 0.0

    delta = np.fft.irfftn(
        delta_k2,
        s=(nmesh, nmesh, nmesh),
    ).real

    del white
    del white2
    del wk2
    del delta_k2
    del Pgrid2

    print()
    print(
        "psi_x rms/min/max :",
        float(np.std(psi_x, dtype=np.float64)),
        float(np.min(psi_x)),
        float(np.max(psi_x)),
    )
    print(
        "psi_y rms/min/max :",
        float(np.std(psi_y, dtype=np.float64)),
        float(np.min(psi_y)),
        float(np.max(psi_y)),
    )
    print(
        "psi_z rms/min/max :",
        float(np.std(psi_z, dtype=np.float64)),
        float(np.min(psi_z)),
        float(np.max(psi_z)),
    )

    print(
        "delta mean/std/min/max :",
        float(np.mean(delta, dtype=np.float64)),
        float(np.std(delta, dtype=np.float64)),
        float(np.min(delta)),
        float(np.max(delta)),
    )

    px = outdir / f"psi_x_nmesh{nmesh}_f32.npy"
    py = outdir / f"psi_y_nmesh{nmesh}_f32.npy"
    pz = outdir / f"psi_z_nmesh{nmesh}_f32.npy"

    np.save(px, psi_x)
    np.save(py, psi_y)
    np.save(pz, psi_z)

    metadata = {
        "pk_file": str(pk_file),
        "pk_kind": "CAMB linear matter P(k), z=0",
        "k_units": "h Mpc^-1",
        "P_units": "(h^-1 Mpc)^3",
        "box_size_mpc_h": L,
        "nmesh": nmesh,
        "dx_mpc_h": dx,
        "seed": seed,
        "k_input_min_h_mpc": float(k_input.min()),
        "k_input_max_h_mpc": float(k_input.max()),
        "fundamental_k_h_mpc": float(kfund),
        "axis_nyquist_k_h_mpc": float(kny),
        "displacement_definition": "Psi(k)=i k delta(k)/k^2",
        "source_field": "Gaussian linear density field",
        "cosmology_from_header": {
            "Omega_b_h2": 0.021800,
            "Omega_c_h2": 0.120000,
            "Omega_nu_h2": 0.000640,
            "Omega_darkenergy": 0.688772,
            "sigma8": 0.8104,
        },
        "output_files": {
            "psi_x": str(px),
            "psi_y": str(py),
            "psi_z": str(pz),
        },
        "diagnostics": {
            "psi_x_rms_mpc_h": float(np.std(psi_x, dtype=np.float64)),
            "psi_y_rms_mpc_h": float(np.std(psi_y, dtype=np.float64)),
            "psi_z_rms_mpc_h": float(np.std(psi_z, dtype=np.float64)),
            "delta_mean": float(np.mean(delta, dtype=np.float64)),
            "delta_std": float(np.std(delta, dtype=np.float64)),
            "delta_min": float(np.min(delta)),
            "delta_max": float(np.max(delta)),
        },
    }

    meta_path = outdir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print()
    print("Wrote:")
    print(px)
    print(py)
    print(pz)
    print(meta_path)


if __name__ == "__main__":
    main()
