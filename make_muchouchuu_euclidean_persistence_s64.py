#!/usr/bin/env python3
"""Create the updated MuchoUchuu 64-probe Euclidean-persistence figure.

The script reads the sliding-window spectral-dimension curve produced by the
64-probe Step-6 analysis, rather than embedding the old 8-probe values.

Default analysis directory:
    muchouchuu_step6_window5_s64_interpolated_plateau

Expected inputs in that directory:
    crossing_summary.json
    late_time_plateau_summary.json
    a CSV containing diffusion time, central d_s, q16 and q84 columns

If the default CSV name is absent, the script scans CSV files in the analysis
directory and selects one containing recognizable spectral-dimension columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A_RMS = 10.409557615168367
T_MAX = 16384

# Density-homogeneity result retained from the previous analysis.
DENSITY_MEDIAN = 114.57295655530987
DENSITY_Q16 = 108.00050405972068
DENSITY_Q84 = 120.56992675171307

# Current 64-probe fallback values. JSON files override these when available.
CROSSING_T_MEDIAN = 1836.9896421083158
CROSSING_T_Q16 = 1616.9557402986145
CROSSING_T_Q84 = 2191.6720387810765
CROSSING_SUCCESS = 0.9375

PLATEAU_MEDIAN = 3.0123555373998103
PLATEAU_Q16 = 2.9898306180390106
PLATEAU_Q84 = 3.0353282142799545
PLATEAU_Q025 = 2.9677720299385095
PLATEAU_Q975 = 3.0560708066767677
SLOPE_MEDIAN = 0.010343030721186312
SLOPE_Q16 = -0.006150582365686598
SLOPE_Q84 = 0.026641629407613752
P_PLATEAU_GE_3 = 0.7073
P_SLOPE_GT_0 = 0.737


COLUMN_ALIASES = {
    "time": (
        "diffusion_time", "time", "t", "window_time", "center_time",
        "window_center_time", "geometric_mean_time",
    ),
    "central": (
        "central_ds", "spectral_dimension", "effective_spectral_dimension",
        "d_s", "ds", "central", "median_ds",
    ),
    "q16": (
        "bootstrap_q16", "q16", "ds_q16", "spectral_dimension_q16",
        "lower_68", "lower68",
    ),
    "q84": (
        "bootstrap_q84", "q84", "ds_q84", "spectral_dimension_q84",
        "upper_68", "upper68",
    ),
}


def normalized(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    mapping = {normalized(col): col for col in columns}
    for alias in aliases:
        key = normalized(alias)
        if key in mapping:
            return mapping[key]
    return None


def identify_spectral_columns(frame: pd.DataFrame) -> dict[str, str] | None:
    found: dict[str, str] = {}
    for role, aliases in COLUMN_ALIASES.items():
        col = find_column(frame.columns, aliases)
        if col is None:
            return None
        found[role] = col
    return found


def load_spectral_data(csv_path: Path | None, analysis_dir: Path) -> tuple[np.ndarray, Path]:
    candidates: list[Path] = []
    if csv_path is not None:
        candidates.append(csv_path)
    else:
        candidates.extend([
            analysis_dir / "sliding_window_spectral_dimension.csv",
            analysis_dir / "spectral_dimension_sliding_window.csv",
            analysis_dir / "spectral_dimension.csv",
            analysis_dir / "sliding_window_results.csv",
        ])
        candidates.extend(sorted(analysis_dir.glob("*.csv")))

    seen: set[Path] = set()
    diagnostics: list[str] = []

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)

        try:
            frame = pd.read_csv(candidate)
        except Exception as exc:
            diagnostics.append(f"{candidate}: could not read ({exc})")
            continue

        columns = identify_spectral_columns(frame)
        if columns is None:
            diagnostics.append(
                f"{candidate}: columns={list(frame.columns)!r} do not contain "
                "recognizable time/central/q16/q84 fields"
            )
            continue

        data = frame[[columns["time"], columns["central"], columns["q16"], columns["q84"]]].copy()
        data.columns = ["time", "central", "q16", "q84"]
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        data = data.sort_values("time").drop_duplicates("time", keep="last")

        if len(data) < 3:
            diagnostics.append(f"{candidate}: only {len(data)} usable rows")
            continue
        if not np.all(np.isfinite(data.to_numpy(dtype=float))):
            diagnostics.append(f"{candidate}: contains non-finite values")
            continue
        if not np.all(data["time"].to_numpy() > 0):
            diagnostics.append(f"{candidate}: diffusion times must be positive")
            continue
        if not np.all(data["q16"].to_numpy() <= data["q84"].to_numpy()):
            diagnostics.append(f"{candidate}: q16 exceeds q84")
            continue

        return data.to_numpy(dtype=float), candidate

    detail = "\n".join(diagnostics) if diagnostics else "No candidate CSV files were found."
    raise FileNotFoundError(
        "Could not locate a spectral-dimension CSV with time, central d_s, q16, and q84 columns.\n"
        "Pass it explicitly with --spectral-csv.\n" + detail
    )


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_present(mapping: dict, keys: Iterable[str], default: float) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return float(default)


def quantile(mapping: dict, key: str, default: float) -> float:
    value = mapping.get(key)
    return float(value) if value is not None else float(default)


def load_analysis_summaries(crossing_json: Path, plateau_json: Path) -> dict[str, float]:
    crossing = read_json(crossing_json)
    plateau = read_json(plateau_json)

    crossing_q = crossing.get("bootstrap_crossing_time_quantiles", {})
    plateau_q = plateau.get("bootstrap_plateau_quantiles", {})
    slope_q = plateau.get("bootstrap_slope_quantiles", {})

    return {
        "crossing_t_median": quantile(crossing_q, "median", CROSSING_T_MEDIAN),
        "crossing_t_q16": quantile(crossing_q, "q16", CROSSING_T_Q16),
        "crossing_t_q84": quantile(crossing_q, "q84", CROSSING_T_Q84),
        "crossing_success": first_present(
            crossing,
            ("bootstrap_crossing_success_fraction", "crossing_success_fraction"),
            CROSSING_SUCCESS,
        ),
        "plateau_median": quantile(plateau_q, "median", PLATEAU_MEDIAN),
        "plateau_q16": quantile(plateau_q, "q16", PLATEAU_Q16),
        "plateau_q84": quantile(plateau_q, "q84", PLATEAU_Q84),
        "plateau_q025": quantile(plateau_q, "q025", PLATEAU_Q025),
        "plateau_q975": quantile(plateau_q, "q975", PLATEAU_Q975),
        "slope_median": quantile(slope_q, "median", SLOPE_MEDIAN),
        "slope_q16": quantile(slope_q, "q16", SLOPE_Q16),
        "slope_q84": quantile(slope_q, "q84", SLOPE_Q84),
        "p_plateau_ge_3": first_present(
            plateau, ("probability_plateau_ge_3",), P_PLATEAU_GE_3
        ),
        "p_slope_gt_0": first_present(
            plateau, ("probability_slope_gt_0",), P_SLOPE_GT_0
        ),
    }


def make_figure(
    spectral_data: np.ndarray,
    stats: dict[str, float],
    output_png: Path,
    output_pdf: Path | None,
    dpi: int,
) -> None:
    t = spectral_data[:, 0]
    central_ds = spectral_data[:, 1]
    q16 = spectral_data[:, 2]
    q84 = spectral_data[:, 3]

    physical_scale = A_RMS * np.sqrt(t)
    max_tested = A_RMS * math.sqrt(T_MAX)

    transport_median = A_RMS * math.sqrt(stats["crossing_t_median"])
    transport_q16 = A_RMS * math.sqrt(stats["crossing_t_q16"])
    transport_q84 = A_RMS * math.sqrt(stats["crossing_t_q84"])

    extent_beyond_onset = max_tested - transport_median
    multiple_of_onset = max_tested / transport_median

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.4], hspace=0.25)

    ax1 = fig.add_subplot(grid[0])
    ax1.fill_between(
        physical_scale, q16, q84, alpha=0.22,
        label="Probe-bootstrap 68% interval",
    )
    ax1.plot(
        physical_scale, central_ds, marker="o", linewidth=2.2,
        markersize=4.5, label=r"64-probe sliding-window $d_s$",
    )
    ax1.axhspan(2.97, 3.03, alpha=0.12, label=r"1% band around $d_s=3$")
    ax1.axhline(3.0, linestyle="--", linewidth=1.5, label=r"$d_s=3$")

    ax1.axvspan(transport_q16, transport_q84, alpha=0.14)
    ax1.axvline(transport_median, linewidth=2.0)

    y_top = max(3.08, float(np.nanmax(q84)) + 0.012)
    y_bottom = min(2.94, float(np.nanmin(q16)) - 0.012)
    margin = 0.02 * (max_tested - 0.0)

    ax1.text(
        transport_median + 0.012 * max_tested,
        y_top - 0.015,
        "transport homogeneity\n"
        rf"$R_{{\rm hom}}={transport_median:.1f}^{{+{transport_q84-transport_median:.1f}}}_{{-{transport_median-transport_q16:.1f}}}$ "
        r"$h^{-1}$ Mpc",
        va="top", ha="left", fontsize=10,
    )

    ax1.text(
        max_tested - margin,
        y_bottom + 0.012,
        rf"$t_{{\max}}={T_MAX}$" "\n"
        rf"$\ell_{{\rm RMS}}={max_tested:.1f}\ h^{{-1}}\,\mathrm{{Mpc}}$",
        va="bottom", ha="right", fontsize=10,
    )

    plateau_text = (
        rf"late plateau: $d_s={stats['plateau_median']:.3f}"
        rf"^{{+{stats['plateau_q84']-stats['plateau_median']:.3f}}}"
        rf"_{{-{stats['plateau_median']-stats['plateau_q16']:.3f}}}$" "\n"
        rf"$P(d_s\geq3)={stats['p_plateau_ge_3']:.3f}$; "
        rf"slope $={stats['slope_median']:.3f}$"
    )
    ax1.text(
        0.985, 0.965, plateau_text, transform=ax1.transAxes,
        ha="right", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
    )

    ax1.set_xlim(0, max(1400.0, max_tested * 1.04))
    ax1.set_ylim(y_bottom, y_top)
    ax1.set_ylabel(r"effective spectral dimension $d_s$")
    ax1.set_title("MuchoUchuu: Euclidean transport from 64 independent probes")
    ax1.legend(loc="lower left", fontsize=9, frameon=False)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(grid[1])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0, 1)

    ax2.axvspan(transport_median, max_tested, alpha=0.18)
    ax2.hlines(0.50, transport_median, max_tested, linewidth=8)

    ax2.errorbar(
        DENSITY_MEDIAN, 0.72,
        xerr=np.array([[DENSITY_MEDIAN - DENSITY_Q16], [DENSITY_Q84 - DENSITY_MEDIAN]]),
        fmt="o", capsize=4,
    )
    ax2.errorbar(
        transport_median, 0.50,
        xerr=np.array([[transport_median - transport_q16], [transport_q84 - transport_median]]),
        fmt="o", capsize=4,
    )
    ax2.plot(max_tested, 0.50, marker="s", markersize=7)

    ax2.text(
        DENSITY_MEDIAN, 0.80,
        "density homogeneity\n" rf"{DENSITY_MEDIAN:.1f}",
        ha="center", va="bottom", fontsize=10,
    )
    ax2.text(
        transport_median, 0.22,
        "Euclidean onset\n" rf"{transport_median:.1f}",
        ha="center", va="top", fontsize=10,
    )
    ax2.text(
        max_tested, 0.22,
        "maximum tested\n" rf"{max_tested:.1f}",
        ha="center", va="top", fontsize=10,
    )
    ax2.text(
        0.5 * (transport_median + max_tested), 0.66,
        "64-probe analysis is consistent with a flat\n"
        r"Euclidean plateau beyond $R_{\rm hom}$",
        ha="center", va="center", fontsize=11,
    )
    ax2.text(
        0.5 * (transport_median + max_tested), 0.08,
        "extent beyond onset = "
        rf"{extent_beyond_onset:.1f} $h^{{-1}}$ Mpc"
        rf"  =  {multiple_of_onset:.2f}$\times$ onset scale"
        "\n"
        rf"crossing bootstrap success = {stats['crossing_success']:.3f}",
        ha="center", va="bottom", fontsize=10,
    )

    ax2.set_xlabel(r"physical scale [$h^{-1}$ Mpc]")
    ax2.set_yticks([])
    ax2.grid(True, axis="x", alpha=0.25)
    for side in ("left", "right", "top"):
        ax2.spines[side].set_visible(False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {output_png.resolve()}")
    if output_pdf is not None:
        print(f"Wrote: {output_pdf.resolve()}")
    print(f"Transport-homogeneity median: {transport_median:.3f} h^-1 Mpc")
    print(f"Transport 68% interval: [{transport_q16:.3f}, {transport_q84:.3f}] h^-1 Mpc")
    print(f"Maximum tested RMS scale: {max_tested:.3f} h^-1 Mpc")
    print(f"Extent beyond onset: {extent_beyond_onset:.3f} h^-1 Mpc")
    print(f"Maximum / onset ratio: {multiple_of_onset:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path("muchouchuu_step6_window5_s64_interpolated_plateau"),
    )
    parser.add_argument(
        "--spectral-csv", type=Path, default=None,
        help="CSV with diffusion time, central d_s, q16, and q84 columns.",
    )
    parser.add_argument("--crossing-json", type=Path, default=None)
    parser.add_argument("--plateau-json", type=Path, default=None)
    parser.add_argument(
        "--output-png", type=Path,
        default=Path("muchouchuu_euclidean_persistence_s64.png"),
    )
    parser.add_argument(
        "--output-pdf", type=Path,
        default=Path("muchouchuu_euclidean_persistence_s64.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    crossing_json = args.crossing_json or args.analysis_dir / "crossing_summary.json"
    plateau_json = args.plateau_json or args.analysis_dir / "late_time_plateau_summary.json"

    spectral_data, spectral_path = load_spectral_data(args.spectral_csv, args.analysis_dir)
    stats = load_analysis_summaries(crossing_json, plateau_json)

    print(f"Using spectral data: {spectral_path.resolve()}")
    print(f"Using crossing summary: {crossing_json.resolve() if crossing_json.exists() else 'fallback constants'}")
    print(f"Using plateau summary: {plateau_json.resolve() if plateau_json.exists() else 'fallback constants'}")

    make_figure(
        spectral_data=spectral_data,
        stats=stats,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
