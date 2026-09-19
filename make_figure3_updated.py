#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT FILES
# ============================================================

CENTRAL_CSV = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace_t131072/combined_64probe/"
    "combined_64probe_extended_ds.csv"
)

BOOTSTRAP_CSV = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/"
    "muchouchuu_extended_trace_t131072/combined_64probe/"
    "muchouchuu_extended_64probe_bootstrap_ds.csv"
)

OUTDIR = Path(
    "/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion"
)

OUTPNG = OUTDIR / "figure3_muchouchuu_transport_updated.png"
OUTPDF = OUTDIR / "figure3_muchouchuu_transport_updated.pdf"


# ============================================================
# PLOT / ANALYSIS SETTINGS
# ============================================================

# Fiducial maximum scale used in the original Figure 3 analysis.
# Everything above this is plotted as an "extended" segment.
R_FID_MAX = 1332.423373

# Transport homogeneity scale shown in the current Figure 3.
R_HOM = 446.3
R_HOM_ERR_LO = 27.7
R_HOM_ERR_HI = 41.7

# Late-time plateau value shown in the Figure 3 text box.
DS_LATE = 3.011
DS_LATE_ERR_LO = 0.025
DS_LATE_ERR_HI = 0.025

# Plot range
XMIN = 220.0
XMAX = 2700.0
YMIN = 2.82
YMAX = 3.11

# Colors chosen to stay close to current Figure 3
BLUE = "#1f77b4"
ORANGE = "#d8a24a"
GREY = "0.80"


# ============================================================
# HELPERS
# ============================================================

def load_plot_dataframe(central_csv: Path, bootstrap_csv: Path) -> pd.DataFrame:
    """
    Build one plotting dataframe containing:
      diffusion_time, R_mpc_h, R_over_L,
      central_ds, central_r2,
      ds_q025, ds_q16, ds_q50, ds_q84, ds_q975,
      bootstrap_valid_fraction
    """

    c = pd.read_csv(central_csv)
    b = pd.read_csv(bootstrap_csv)

    # Merge only the extra central columns we may need from the central file.
    keep_cols = ["diffusion_time"]
    if "d_s" in c.columns:
        keep_cols.append("d_s")
    if "r2" in c.columns:
        keep_cols.append("r2")
    if "R_mpc_h" in c.columns and "R_mpc_h" not in b.columns:
        keep_cols.append("R_mpc_h")
    if "R_over_L" in c.columns and "R_over_L" not in b.columns:
        keep_cols.append("R_over_L")

    c_small = c[keep_cols].copy()
    df = pd.merge(b, c_small, on="diffusion_time", how="left")

    # Harmonize column names.
    if "central_ds" not in df.columns:
        if "d_s" in df.columns:
            df["central_ds"] = df["d_s"]
        else:
            raise RuntimeError("No central spectral-dimension column found.")

    if "central_r2" not in df.columns:
        if "r2" in df.columns:
            df["central_r2"] = df["r2"]
        else:
            df["central_r2"] = np.nan

    if "R_mpc_h" not in df.columns:
        raise RuntimeError("No R_mpc_h column found after merge.")
    if "R_over_L" not in df.columns:
        df["R_over_L"] = np.nan

    needed = [
        "diffusion_time", "R_mpc_h", "R_over_L",
        "central_ds", "central_r2",
        "ds_q025", "ds_q16", "ds_q50", "ds_q84", "ds_q975",
        "bootstrap_valid_fraction",
    ]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"Required column missing: {col}")

    df = df.sort_values("R_mpc_h").reset_index(drop=True)
    return df


def first_row_where(df: pd.DataFrame, condition) -> pd.Series | None:
    q = df.loc[condition(df)]
    if len(q) == 0:
        return None
    return q.iloc[0]


def format_diag(label: str, row: pd.Series | None):
    if row is None:
        print(f"{label:35s} none")
    else:
        print(
            f"{label:35s} "
            f"t={int(row['diffusion_time']):6d}  "
            f"R={row['R_mpc_h']:8.2f}  "
            f"R/L={row['R_over_L']:.3f}  "
            f"ds={row['central_ds']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_plot_dataframe(CENTRAL_CSV, BOOTSTRAP_CSV)

    # Split into fiducial and extended segments.
    # This guarantees a visual gap between the two.
    fid = df[df["R_mpc_h"] <= R_FID_MAX].copy()
    ext = df[df["R_mpc_h"] >  R_FID_MAX].copy()

    if len(fid) == 0:
        raise RuntimeError("No fiducial points found.")
    if len(ext) == 0:
        print("WARNING: no extended points found beyond R_FID_MAX.")

    print()
    print("============================================================")
    print("FIGURE 3 UPDATED DIAGNOSTICS")
    print("============================================================")
    print("last fiducial R =", fid["R_mpc_h"].iloc[-1])
    if len(ext) > 0:
        print("first extended R =", ext["R_mpc_h"].iloc[0])
    else:
        print("first extended R = none")

    print()
    format_diag(
        "central < 2.97",
        first_row_where(df, lambda x: x["central_ds"] < 2.97)
    )
    format_diag(
        "q84 < 2.97",
        first_row_where(df, lambda x: x["ds_q84"] < 2.97)
    )
    format_diag(
        "q975 < 3",
        first_row_where(df, lambda x: x["ds_q975"] < 3.0)
    )
    format_diag(
        "q975 < 2.97",
        first_row_where(df, lambda x: x["ds_q975"] < 2.97)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(12.8, 7.6))

    # 1% band around d_s = 3
    band1 = ax.axhspan(
        2.97, 3.03,
        color=GREY,
        alpha=0.55,
        label=r"1% band around $d_s=3$",
        zorder=0,
    )

    # d_s = 3 line
    line_ds3 = ax.axhline(
        3.0,
        color=BLUE,
        linestyle="--",
        linewidth=1.8,
        label=r"$d_s=3$",
        zorder=1,
    )

    # Transport homogeneity band
    rhom_band = ax.axvspan(
        R_HOM - R_HOM_ERR_LO,
        R_HOM + R_HOM_ERR_HI,
        color=ORANGE,
        alpha=0.35,
        label=r"MuchoUchuu: $R_{\rm hom}$ 68% interval",
        zorder=0.5,
    )

    # Transport homogeneity line
    rhom_line = ax.axvline(
        R_HOM,
        color=BLUE,
        linewidth=1.5,
        label=r"MuchoUchuu: $R_{\rm hom}$",
        zorder=2,
    )

    # --------------------------------------------------------
    # Fiducial segment (filled markers + darker band)
    # --------------------------------------------------------
    fid_band = ax.fill_between(
        fid["R_mpc_h"].to_numpy(),
        fid["ds_q16"].to_numpy(),
        fid["ds_q84"].to_numpy(),
        color=BLUE,
        alpha=0.22,
        linewidth=0,
        label=r"MuchoUchuu: probe-bootstrap 68% interval",
        zorder=1,
    )

    fid_line, = ax.plot(
        fid["R_mpc_h"].to_numpy(),
        fid["central_ds"].to_numpy(),
        color=BLUE,
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label=r"MuchoUchuu: centered 64-probe $d_s$",
        zorder=3,
    )

    # --------------------------------------------------------
    # Extended segment
    #
    # Prepend the last fiducial point so the dashed line and
    # lighter bootstrap band remain continuous across the
    # fiducial/extended transition. Open markers are plotted
    # only at genuinely extended measurement points.
    # --------------------------------------------------------
    ext_line = None
    ext_band = None

    if len(ext) > 0:

        bridge = pd.concat(
            [fid.tail(1), ext],
            ignore_index=True
        )

        ext_band = ax.fill_between(
            bridge["R_mpc_h"].to_numpy(),
            bridge["ds_q16"].to_numpy(),
            bridge["ds_q84"].to_numpy(),
            color=BLUE,
            alpha=0.10,
            linewidth=0,
            label=r"MuchoUchuu: extended 68% interval",
            zorder=0.8,
        )

        ax.plot(
            bridge["R_mpc_h"].to_numpy(),
            bridge["central_ds"].to_numpy(),
            color=BLUE,
            linestyle="--",
            linewidth=1.8,
            zorder=2.5,
        )

        ext_line, = ax.plot(
            ext["R_mpc_h"].to_numpy(),
            ext["central_ds"].to_numpy(),
            color=BLUE,
            linestyle="none",
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            markeredgewidth=1.2,
            label=r"MuchoUchuu: extended $d_s$",
            zorder=3,
        )

    # --------------------------------------------------------
    # Labels / title / limits
    # --------------------------------------------------------
    ax.set_title("Euclidean transport in MuchoUchuu")
    ax.set_xlabel(r"physical diffusion scale $R\ [h^{-1}\ \mathrm{Mpc}]$")
    ax.set_ylabel(r"effective spectral dimension $d_s$")

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # Text box in upper right
    # --------------------------------------------------------
    textbox = (
        "MuchoUchuu:\n"
        + r"$R_{\rm hom}=446.3^{+41.7}_{-27.7}\ h^{-1}\,\mathrm{Mpc}$" + "\n"
        + r"$d_s^{\rm late}=3.011^{+0.025}_{-0.025}$"
    )

    ax.text(
        0.985, 0.955, textbox,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="0.6", alpha=0.95),
        zorder=10,
    )

    # --------------------------------------------------------
    # Legend: explicit order
    # --------------------------------------------------------
    handles = [
        band1,
        line_ds3,
        fid_line,
        fid_band,
    ]
    labels = [
        r"1% band around $d_s=3$",
        r"$d_s=3$",
        r"MuchoUchuu: centered 64-probe $d_s$",
        r"MuchoUchuu: probe-bootstrap 68% interval",
    ]

    if ext_line is not None and ext_band is not None:
        handles += [ext_line, ext_band]
        labels += [
            r"MuchoUchuu: extended $d_s$",
            r"MuchoUchuu: extended 68% interval",
        ]

    handles += [rhom_line, rhom_band]
    labels += [
        r"MuchoUchuu: $R_{\rm hom}$",
        r"MuchoUchuu: $R_{\rm hom}$ 68% interval",
    ]

    ax.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.8,
        labelspacing=0.7,
    )

    fig.tight_layout()

    fig.savefig(OUTPNG, dpi=220)
    fig.savefig(OUTPDF)

    plt.close(fig)

    print()
    print("Wrote:")
    print(OUTPNG)
    print(OUTPDF)


if __name__ == "__main__":
    main()
