#!/usr/bin/env python3
"""Compare MuchoUchuu and BigUchuu Euclidean-transport results.

The figure contains:
  (a) overlaid sliding-window spectral-dimension curves in physical units;
  (b) a compact scale summary for the two simulations.

By default the BigUchuu curve is expected to be the matched, pre-mixing
analysis (for example t_max=7168 or 8192).  The script reads each run's
spectral CSV, crossing_summary.json, and late_time_plateau_summary.json.

Example
-------
python make_muchouchuu_biguchuu_euclidean_comparison.py \
  --mucho-dir /work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion/muchouchuu_step6_window7_s64_interpolated_plateau \
  --big-dir /work/fprada/DIFFUSION/ANALYSIS/biguchuu_diffusion_reproduction_v2/biguchuu_step6_window7_t7168_matched_plateau_tmin3584 \
  --big-a-rms 10.35 \
  --output-pdf muchouchuu_biguchuu_transport_comparison.pdf \
  --output-png muchouchuu_biguchuu_transport_comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        "bootstrap_q16", "bootstrap_ds_q16", "q16", "ds_q16",
        "spectral_dimension_q16", "lower_68", "lower68",
    ),
    "q84": (
        "bootstrap_q84", "bootstrap_ds_q84", "q84", "ds_q84",
        "spectral_dimension_q84", "upper_68", "upper68",
    ),
}


@dataclass
class Result:
    name: str
    analysis_dir: Path
    a_rms: float
    t: np.ndarray
    central: np.ndarray
    q16: np.ndarray
    q84: np.ndarray
    crossing_t_median: float
    crossing_t_q16: float
    crossing_t_q84: float
    crossing_success: float
    plateau_median: float
    plateau_q16: float
    plateau_q84: float
    slope_median: float
    n_probes: int | None

    @property
    def scale(self) -> np.ndarray:
        return self.a_rms * np.sqrt(self.t)

    @property
    def tmax(self) -> float:
        return float(np.max(self.t))

    @property
    def max_scale(self) -> float:
        return self.a_rms * math.sqrt(self.tmax)

    @property
    def crossing_scale(self) -> float:
        return self.a_rms * math.sqrt(self.crossing_t_median)

    @property
    def crossing_scale_q16(self) -> float:
        return self.a_rms * math.sqrt(self.crossing_t_q16)

    @property
    def crossing_scale_q84(self) -> float:
        return self.a_rms * math.sqrt(self.crossing_t_q84)


def normalized(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    mapping = {normalized(col): col for col in columns}
    for alias in aliases:
        if normalized(alias) in mapping:
            return mapping[normalized(alias)]
    return None


def identify_columns(frame: pd.DataFrame) -> dict[str, str] | None:
    found: dict[str, str] = {}
    for role, aliases in COLUMN_ALIASES.items():
        col = find_column(frame.columns, aliases)
        if col is None:
            return None
        found[role] = col
    return found


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text())


def load_spectral_csv(analysis_dir: Path, explicit: Path | None = None) -> tuple[pd.DataFrame, Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        analysis_dir / "spectral_dimension_sliding_bootstrap.csv",
        analysis_dir / "sliding_window_spectral_dimension.csv",
        analysis_dir / "spectral_dimension_sliding_window.csv",
        analysis_dir / "spectral_dimension.csv",
        analysis_dir / "sliding_window_results.csv",
    ])
    candidates.extend(sorted(analysis_dir.glob("*.csv")))

    checked: set[Path] = set()
    diagnostics: list[str] = []
    for path in candidates:
        path = path.expanduser()
        if path in checked or not path.is_file():
            continue
        checked.add(path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            diagnostics.append(f"{path}: {exc}")
            continue
        cols = identify_columns(frame)
        if cols is None:
            continue
        out = frame[[cols["time"], cols["central"], cols["q16"], cols["q84"]]].copy()
        out.columns = ["time", "central", "q16", "q84"]
        out = out.apply(pd.to_numeric, errors="coerce").dropna()
        out = out.sort_values("time").drop_duplicates("time", keep="last")
        out = out[(out["time"] > 0) & (out["q16"] <= out["q84"])]
        if len(out) >= 3:
            return out, path

    raise FileNotFoundError(
        f"Could not find a suitable spectral-dimension CSV in {analysis_dir}. "
        f"Checked {len(checked)} candidate files."
    )


def load_result(
    name: str,
    analysis_dir: Path,
    a_rms: float,
    spectral_csv: Path | None,
    crossing_json: Path | None,
    plateau_json: Path | None,
) -> Result:
    frame, used_csv = load_spectral_csv(analysis_dir, spectral_csv)
    crossing_path = crossing_json or analysis_dir / "crossing_summary.json"
    plateau_path = plateau_json or analysis_dir / "late_time_plateau_summary.json"
    crossing = load_json(crossing_path)
    plateau = load_json(plateau_path)

    cq = crossing.get("bootstrap_crossing_time_quantiles", {})
    pq = plateau.get("bootstrap_plateau_quantiles", {})
    sq = plateau.get("bootstrap_slope_quantiles", {})

    required_crossing = ["median", "q16", "q84"]
    required_plateau = ["median", "q16", "q84"]
    for key in required_crossing:
        if key not in cq:
            raise KeyError(f"Missing crossing quantile '{key}' in {crossing_path}")
    for key in required_plateau:
        if key not in pq:
            raise KeyError(f"Missing plateau quantile '{key}' in {plateau_path}")
    if "median" not in sq:
        raise KeyError(f"Missing slope median in {plateau_path}")

    print(f"{name}: spectral CSV = {used_csv}")
    print(f"{name}: crossing JSON = {crossing_path}")
    print(f"{name}: plateau JSON = {plateau_path}")

    n_probes = crossing.get("number_of_probes")
    return Result(
        name=name,
        analysis_dir=analysis_dir,
        a_rms=float(a_rms),
        t=frame["time"].to_numpy(float),
        central=frame["central"].to_numpy(float),
        q16=frame["q16"].to_numpy(float),
        q84=frame["q84"].to_numpy(float),
        crossing_t_median=float(cq["median"]),
        crossing_t_q16=float(cq["q16"]),
        crossing_t_q84=float(cq["q84"]),
        crossing_success=float(crossing.get("bootstrap_crossing_success_fraction", np.nan)),
        plateau_median=float(pq["median"]),
        plateau_q16=float(pq["q16"]),
        plateau_q84=float(pq["q84"]),
        slope_median=float(sq["median"]),
        n_probes=int(n_probes) if n_probes is not None else None,
    )


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def make_figure(
    mucho: Result,
    big: Result,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    mucho_density: tuple[float, float, float] | None,
    big_density: tuple[float, float, float] | None,
) -> None:
    paper_style()
    fig = plt.figure(figsize=(11.2, 8.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.45], hspace=0.26)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])

    # Draw common Euclidean references first.
    ax1.axhspan(2.97, 3.03, alpha=0.10, label=r"1% band around $d_s=3$")
    ax1.axhline(3.0, linestyle="--", linewidth=1.3, label=r"$d_s=3$")

    # No explicit colours are hard-coded; matplotlib's default cycle distinguishes the simulations.
    for result, marker in ((mucho, "o"), (big, "s")):
        line, = ax1.plot(
            result.scale,
            result.central,
            marker=marker,
            linewidth=2.0,
            markersize=4.2,
            label=(f"{result.name}: {result.n_probes}-probe $d_s$"
                   if result.n_probes is not None else f"{result.name}: $d_s$"),
        )
        colour = line.get_color()
        ax1.fill_between(
            result.scale, result.q16, result.q84,
            alpha=0.16, color=colour,
            label=f"{result.name}: probe-bootstrap 68% interval",
        )
        ax1.axvspan(
            result.crossing_scale_q16,
            result.crossing_scale_q84,
            alpha=0.09,
            color=colour,
        )
        ax1.axvline(result.crossing_scale, linewidth=1.7, color=colour)

    all_q16 = np.concatenate([mucho.q16, big.q16])
    all_q84 = np.concatenate([mucho.q84, big.q84])
    y_bottom = min(2.82, float(np.nanmin(all_q16)) - 0.015)
    y_top = max(3.09, float(np.nanmax(all_q84)) + 0.015)
    xmax = max(mucho.max_scale, big.max_scale) * 1.05

    ax1.set_xlim(0, xmax)
    ax1.set_ylim(y_bottom, y_top)
    ax1.set_ylabel(r"effective spectral dimension $d_s$")
    ax1.set_title("Euclidean transport in MuchoUchuu and BigUchuu")
    ax1.grid(True, alpha=0.24)
    ax1.legend(loc="best", frameon=False, ncol=2)

    # Compact plateau annotations.
    mucho_text = (
        rf"{mucho.name}: $R_{{\rm hom}}={mucho.crossing_scale:.1f}"
        rf"^{{+{mucho.crossing_scale_q84-mucho.crossing_scale:.1f}}}"
        rf"_{{-{mucho.crossing_scale-mucho.crossing_scale_q16:.1f}}}$; "
        rf"$d_s^{{\rm late}}={mucho.plateau_median:.3f}$"
    )
    big_text = (
        rf"{big.name}: $R_{{\rm hom}}={big.crossing_scale:.1f}"
        rf"^{{+{big.crossing_scale_q84-big.crossing_scale:.1f}}}"
        rf"_{{-{big.crossing_scale-big.crossing_scale_q16:.1f}}}$; "
        rf"$d_s^{{\rm late}}={big.plateau_median:.3f}$"
    )
    ax1.text(
        0.012, 0.985, mucho_text + "\n" + big_text,
        transform=ax1.transAxes, ha="left", va="top", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )

    # Lower scale summary: one row per simulation.
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 1.1)
    rows = [(mucho, 0.72), (big, 0.28)]
    for result, y in rows:
        line = ax2.hlines(y, result.crossing_scale, result.max_scale, linewidth=7)
        colour = line.get_colors()[0]
        ax2.axvspan(result.crossing_scale, result.max_scale, alpha=0.08, color=colour)
        ax2.errorbar(
            result.crossing_scale, y,
            xerr=np.array([[result.crossing_scale-result.crossing_scale_q16],
                           [result.crossing_scale_q84-result.crossing_scale]]),
            fmt="o", capsize=4, color=colour,
        )
        ax2.plot(result.max_scale, y, marker="s", markersize=7, color=colour)
        ax2.text(8, y, result.name, ha="left", va="center", fontsize=10, fontweight="bold")
        ax2.text(
            result.crossing_scale, y-0.13,
            rf"onset {result.crossing_scale:.1f}",
            ha="center", va="top", fontsize=9,
        )
        ax2.text(
            result.max_scale, y-0.13,
            rf"max {result.max_scale:.1f}",
            ha="center", va="top", fontsize=9,
        )

    # Optional density-homogeneity references.
    for density, y in ((mucho_density, 0.92), (big_density, 0.48)):
        if density is None:
            continue
        med, q16, q84 = density
        ax2.errorbar(
            med, y,
            xerr=np.array([[med-q16], [q84-med]]),
            fmt="D", capsize=3,
        )
        ax2.text(med, y+0.07, f"density {med:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax2.set_xlabel(r"physical scale [$h^{-1}$ Mpc]")
    ax2.set_yticks([])
    ax2.grid(True, axis="x", alpha=0.24)
    for side in ("left", "right", "top"):
        ax2.spines[side].set_visible(False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {output_png.resolve()}")
    print(f"Wrote: {output_pdf.resolve()}")
    for r in (mucho, big):
        print(
            f"{r.name}: R_hom={r.crossing_scale:.3f} "
            f"[{r.crossing_scale_q16:.3f}, {r.crossing_scale_q84:.3f}] h^-1 Mpc; "
            f"tmax={r.tmax:g}; lmax={r.max_scale:.3f}; "
            f"plateau={r.plateau_median:.4f}; slope={r.slope_median:.4f}; "
            f"crossing_success={r.crossing_success:.4f}"
        )


def parse_density(values: list[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("Density values must be MEDIAN Q16 Q84")
    return float(values[0]), float(values[1]), float(values[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mucho-dir", type=Path, required=True)
    parser.add_argument("--big-dir", type=Path, required=True)
    parser.add_argument("--mucho-a-rms", type=float, default=10.409557615168367)
    parser.add_argument("--big-a-rms", type=float, default=10.35)
    parser.add_argument("--mucho-spectral-csv", type=Path, default=None)
    parser.add_argument("--big-spectral-csv", type=Path, default=None)
    parser.add_argument("--mucho-crossing-json", type=Path, default=None)
    parser.add_argument("--big-crossing-json", type=Path, default=None)
    parser.add_argument("--mucho-plateau-json", type=Path, default=None)
    parser.add_argument("--big-plateau-json", type=Path, default=None)
    parser.add_argument(
        "--mucho-density", type=float, nargs=3,
        metavar=("MEDIAN", "Q16", "Q84"),
        default=[114.57295655530987, 108.00050405972068, 120.56992675171307],
    )
    parser.add_argument(
        "--big-density", type=float, nargs=3,
        metavar=("MEDIAN", "Q16", "Q84"), default=None,
        help="Optional BigUchuu density-homogeneity median, q16 and q84.",
    )
    parser.add_argument("--output-png", type=Path, default=Path("muchouchuu_biguchuu_transport_comparison.png"))
    parser.add_argument("--output-pdf", type=Path, default=Path("muchouchuu_biguchuu_transport_comparison.pdf"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    mucho = load_result(
        "MuchoUchuu", args.mucho_dir, args.mucho_a_rms,
        args.mucho_spectral_csv, args.mucho_crossing_json, args.mucho_plateau_json,
    )
    big = load_result(
        "BigUchuu", args.big_dir, args.big_a_rms,
        args.big_spectral_csv, args.big_crossing_json, args.big_plateau_json,
    )

    make_figure(
        mucho=mucho,
        big=big,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        dpi=args.dpi,
        mucho_density=parse_density(args.mucho_density),
        big_density=parse_density(args.big_density),
    )


if __name__ == "__main__":
    main()
