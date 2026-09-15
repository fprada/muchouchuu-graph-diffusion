#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER PATHS
# ============================================================

Z0_DIR = Path("muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full")
Z2_DIR = Path("growth_test/z203/step6_window7_s64_strictcentered_tmin3584_fixed")

Z0_CSV = Z0_DIR / "spectral_dimension_sliding_bootstrap.csv"
Z2_CSV = Z2_DIR / "spectral_dimension_sliding_bootstrap.csv"

Z0_CROSS = Z0_DIR / "crossing_summary.json"
Z2_CROSS = Z2_DIR / "crossing_summary.json"

OUT_PREFIX = "figure_z0_vs_z203_ds_physical_scale_with_delta"


# ============================================================
# PHYSICAL-SCALE CALIBRATIONS
# ============================================================
# R = A * sqrt(t), in h^-1 Mpc

A_Z0 = 10.4095576
A_Z2 = 10.598993479525339

TARGET_DS = 3.0
TOL = 0.03


# ============================================================
# HELPERS
# ============================================================

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            "Could not find any of these columns: {}\nAvailable columns: {}".format(
                candidates, list(df.columns)
            )
        )
    return None


def load_curve(csv_path):
    df = pd.read_csv(csv_path)

    tcol = pick_col(df, ["diffusion_time", "center_time", "time", "t"])
    dscol = pick_col(df, ["central_ds", "ds", "mean_spectral_dimension"])

    q025col = pick_col(df, ["bootstrap_ds_q025", "bootstrap_ds_p2p5"], required=False)
    q16col  = pick_col(df, ["bootstrap_ds_q16", "bootstrap_ds_p16"], required=False)
    q50col  = pick_col(df, ["bootstrap_ds_median", "bootstrap_ds_p50"], required=False)
    q84col  = pick_col(df, ["bootstrap_ds_q84", "bootstrap_ds_p84"], required=False)
    q975col = pick_col(df, ["bootstrap_ds_q975", "bootstrap_ds_p97p5"], required=False)
    r2col   = pick_col(df, ["central_r2", "mean_r2", "linear_r2"], required=False)

    out = pd.DataFrame({
        "t": pd.to_numeric(df[tcol], errors="coerce"),
        "ds": pd.to_numeric(df[dscol], errors="coerce"),
    })

    if q025col is not None:
        out["q025"] = pd.to_numeric(df[q025col], errors="coerce")
    if q16col is not None:
        out["q16"] = pd.to_numeric(df[q16col], errors="coerce")
    if q50col is not None:
        out["q50"] = pd.to_numeric(df[q50col], errors="coerce")
    if q84col is not None:
        out["q84"] = pd.to_numeric(df[q84col], errors="coerce")
    if q975col is not None:
        out["q975"] = pd.to_numeric(df[q975col], errors="coerce")
    if r2col is not None:
        out["r2"] = pd.to_numeric(df[r2col], errors="coerce")

    mask = np.isfinite(out["t"]) & np.isfinite(out["ds"])
    out = out.loc[mask].copy()
    out = out.sort_values("t").reset_index(drop=True)

    return out


def load_crossing(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)

    q = d["bootstrap_crossing_time_quantiles"]

    out = {
        "q025_t": float(q["q025"]),
        "q16_t": float(q["q16"]),
        "median_t": float(q["median"]),
        "q84_t": float(q["q84"]),
        "q975_t": float(q["q975"]),
        "success_fraction": float(d.get("bootstrap_crossing_success_fraction", np.nan)),
        "central_time": float(d.get("first_central_persistent_time", np.nan)),
    }
    return out


def time_to_scale(t, A):
    t = np.asarray(t, dtype=float)
    return A * np.sqrt(t)


def enrich_with_scale(df, A):
    out = df.copy()
    out["R"] = time_to_scale(out["t"].to_numpy(), A)
    return out


def crossing_time_to_scale(cross, A):
    c = dict(cross)
    c["q025_R"] = time_to_scale(c["q025_t"], A)
    c["q16_R"] = time_to_scale(c["q16_t"], A)
    c["median_R"] = time_to_scale(c["median_t"], A)
    c["q84_R"] = time_to_scale(c["q84_t"], A)
    c["q975_R"] = time_to_scale(c["q975_t"], A)
    c["central_R"] = time_to_scale(c["central_time"], A) if np.isfinite(c["central_time"]) else np.nan
    return c


def plot_curve(
    ax, df, label, color, marker, linestyle="-", zorder=3
):
    x = df["R"].to_numpy()
    y = df["ds"].to_numpy()

    if "q025" in df.columns and "q975" in df.columns:
        ax.fill_between(
            x, df["q025"].to_numpy(), df["q975"].to_numpy(),
            color=color, alpha=0.12, linewidth=0,
            zorder=zorder-2
        )

    if "q16" in df.columns and "q84" in df.columns:
        ax.fill_between(
            x, df["q16"].to_numpy(), df["q84"].to_numpy(),
            color=color, alpha=0.12, linewidth=0,
            zorder=zorder-2
        )

    ax.plot(
        x, y,
        marker=marker, ms=4.5, lw=2.2,
        color=color,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
        markerfacecolor=("white" if marker == "s" else color),
        markeredgewidth=(1.2 if marker == "s" else 0.8)
    )


def draw_crossing(ax, cross, label, color, linestyle, side="left"):
    ax.axvspan(
        cross["q16_R"], cross["q84_R"],
        color=color, alpha=0.06, linewidth=0
    )

    ax.axvline(
        cross["median_R"],
        color=color,
        lw=1.5,
        ls=linestyle
    )

    if side == "left":
        xtext = cross["median_R"] - 8
        ha = "right"
    else:
        xtext = cross["median_R"] + 8
        ha = "left"

    txt = (
        rf"{label}: $R_{{\rm hom}}={cross['median_R']:.0f}$"
    )

    ax.text(
        xtext,
        3.078,
        txt,
        color=color,
        rotation=90,
        va="top",
        ha=ha,
        fontsize=9
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not Z0_CSV.exists():
        raise FileNotFoundError(f"Missing file: {Z0_CSV}")
    if not Z2_CSV.exists():
        raise FileNotFoundError(f"Missing file: {Z2_CSV}")
    if not Z0_CROSS.exists():
        raise FileNotFoundError(f"Missing file: {Z0_CROSS}")
    if not Z2_CROSS.exists():
        raise FileNotFoundError(f"Missing file: {Z2_CROSS}")

    z0 = load_curve(Z0_CSV)
    z2 = load_curve(Z2_CSV)

    z0 = enrich_with_scale(z0, A_Z0)
    z2 = enrich_with_scale(z2, A_Z2)

    cross0 = crossing_time_to_scale(load_crossing(Z0_CROSS), A_Z0)
    cross2 = crossing_time_to_scale(load_crossing(Z2_CROSS), A_Z2)

    # Restrict to common diffusion-time range for a fair comparison
    tmax_common = min(z0["t"].max(), z2["t"].max())
    z0 = z0[z0["t"] <= tmax_common].copy()
    z2 = z2[z2["t"] <= tmax_common].copy()

    print("Common maximum diffusion time:", tmax_common)
    print()
    print("z=0 crossing:")
    print("  success fraction = {:.4f}".format(cross0["success_fraction"]))
    print("  t_hom median     = {:.3f}".format(cross0["median_t"]))
    print("  R_hom median     = {:.3f} h^-1 Mpc".format(cross0["median_R"]))
    print("  R_hom [q16,q84]  = [{:.3f}, {:.3f}] h^-1 Mpc".format(cross0["q16_R"], cross0["q84_R"]))
    print()
    print("z=2.03 crossing:")
    print("  success fraction = {:.4f}".format(cross2["success_fraction"]))
    print("  t_hom median     = {:.3f}".format(cross2["median_t"]))
    print("  R_hom median     = {:.3f} h^-1 Mpc".format(cross2["median_R"]))
    print("  R_hom [q16,q84]  = [{:.3f}, {:.3f}] h^-1 Mpc".format(cross2["q16_R"], cross2["q84_R"]))

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 140,
    })

    fig, (ax, axd) = plt.subplots(
        2, 1,
        figsize=(7.6, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.25]}
    )

    # Euclidean target band
    ax.axhspan(TARGET_DS - TOL, TARGET_DS + TOL, color="0.88", alpha=0.8, label=r"$|d_s-3|\leq 0.03$")
    ax.axhline(TARGET_DS, color="0.35", lw=1.2, ls="--")

    plot_curve(
        ax, z0,
        label="MuchoUchuu $z=0$",
        color="tab:blue",
        marker="o",
        linestyle="-",
        zorder=4
    )

    plot_curve(
        ax, z2,
        label="MuchoUchuu $z=2.03$",
        color="tab:orange",
        marker="s",
        linestyle="--",
        zorder=5
    )

    draw_crossing(
        ax, cross0,
        label=r"$z=0$",
        color="tab:blue",
        linestyle="--",
        side="left"
    )

    draw_crossing(
        ax, cross2,
        label=r"$z=2.03$",
        color="tab:orange",
        linestyle=":",
        side="right"
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        r"Effective spectral dimension $d_s$"
    )
    fig.suptitle(
        r"Euclidean transport in MuchoUchuu: $z=0$ vs. $z=2.03$",
        y=1.015
    )

    ax.set_xlim(300, 1100)
    ax.set_ylim(2.955, 3.085)

    ax.grid(True, alpha=0.20)
    ax.legend(loc="lower left", frameon=True, fontsize=9)


    # ========================================================
    # PAIRED REDSHIFT DIFFERENCE PANEL
    # ========================================================

    boot0 = np.load(
        Z0_DIR / "bootstrap_spectral_dimension_curves.npy"
    )

    boot2 = np.load(
        Z2_DIR / "bootstrap_spectral_dimension_curves.npy"
    )

    print("z0 bootstrap shape:", boot0.shape)
    print("z2 bootstrap shape:", boot2.shape)

    if boot0.shape != (10000, 21):
        raise RuntimeError(
            f"Unexpected z=0 bootstrap shape: {boot0.shape}"
        )

    if boot2.shape != (10000, 21):
        raise RuntimeError(
            f"Unexpected z=2.03 bootstrap shape: {boot2.shape}"
        )

    # Both redshifts use the identical canonical 21-time grid.
    all_times = np.array([
        512, 640, 768, 896, 1024, 1280, 1536,
        1792, 2048, 2560, 3072, 3584, 4096,
        5120, 6144, 7168, 8192, 10240,
        12288, 14336, 16384
    ], dtype=float)

    # Seven-point strict-centred windows have 3 unsupported
    # points at each edge, leaving 15 valid centres.
    sl = slice(3, -3)

    common_t = all_times[sl]

    # Restrict to the common top-panel range.
    keep = common_t <= tmax_common
    common_t = common_t[keep]

    d = (boot2[:, sl] - boot0[:, sl])[:, keep]

    print("paired common valid times:", common_t)
    print("paired delta array shape:", d.shape)

    dq025, dq16, dq50, dq84, dq975 = np.nanquantile(
        d,
        [0.025, 0.16, 0.50, 0.84, 0.975],
        axis=0
    )

    # Use the midpoint of the two physical calibrations for
    # the paired residual-panel x coordinate.
    A_REF = 0.5 * (A_Z0 + A_Z2)
    Rref = A_REF * np.sqrt(common_t)

    # 95% paired interval
    axd.fill_between(
        Rref, dq025, dq975,
        color="0.55",
        alpha=0.14,
        linewidth=0,
        label="95% paired interval"
    )

    # 68% paired interval
    axd.fill_between(
        Rref, dq16, dq84,
        color="tab:purple",
        alpha=0.24,
        linewidth=0,
        label="68% paired interval"
    )

    # Median paired difference
    axd.plot(
        Rref, dq50,
        color="tab:purple",
        lw=2.0,
        marker="o",
        ms=3.8,
        label=r"paired median $\Delta d_s$"
    )

    axd.axhline(
        0.0,
        color="0.3",
        lw=1.2,
        ls="--"
    )

    axd.set_ylabel(
        r"$\Delta d_s$"
        "\n"
        r"$(z=2.03)-(z=0)$"
    )

    axd.set_xlabel(
        r"Physical diffusion scale "
        r"$R\ [h^{-1}\,\mathrm{Mpc}]$"
    )

    axd.grid(alpha=0.20)

    axd.legend(
        loc="best",
        fontsize=8,
        frameon=True
    )

    axd.set_ylim(-0.055, 0.055)


    png_path = f"{OUT_PREFIX}.png"
    pdf_path = f"{OUT_PREFIX}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Wrote:", png_path)
    print("Wrote:", pdf_path)


if __name__ == "__main__":
    main()
