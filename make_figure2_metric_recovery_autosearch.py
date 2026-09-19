#!/usr/bin/env python3
"""
Generate Figure 2: recovery of the Euclidean metric from graph diffusion.

This version can either:

1. Read a specific CSV file, or
2. Recursively search a directory for a CSV containing:
       - diffusion time
       - true physical separation
       - inferred graph separation

Examples
--------
Search automatically under the current directory:

    python make_figure2_metric_recovery_autosearch.py \
      --search-root . \
      --times 256 1024 4096 \
      --output figure2_metric_recovery.png

Use a specific file:

    python make_figure2_metric_recovery_autosearch.py \
      --input-csv results/pairwise_metric_table.csv \
      --times 256 1024 4096 \
      --output figure2_metric_recovery.png

The script reports all candidate files and selects the highest-ranked match.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRUE_DISTANCE_CANDIDATES = [
    "true_physical_separation",
    "true_physical_separation_mpc_h",
    "true_distance_mpc_h",
    "physical_distance_mpc_h",
    "euclidean_distance_mpc_h",
    "periodic_distance_mpc_h",
    "physical_separation_mpc_h",
    "true_separation_mpc_h",
    "pair_separation_mpc_h",
    "r_true_mpc_h",
    "r_true",
    "true_distance",
    "distance_true",
    "physical_separation",
]

INFERRED_DISTANCE_CANDIDATES = [
    "inferred_graph_separation",
    "inferred_graph_separation_mpc_h",
    "inferred_distance_mpc_h",
    "predicted_distance_mpc_h",
    "diffusion_inferred_distance_mpc_h",
    "graph_distance_mpc_h",
    "graph_inferred_separation_mpc_h",
    "predicted_separation_mpc_h",
    "r_inferred_mpc_h",
    "r_pred_mpc_h",
    "r_inferred",
    "r_pred",
    "predicted_distance",
    "inferred_distance",
    "graph_separation",
]

TIME_CANDIDATES = [
    "diffusion_time",
    "diffusion_step",
    "time",
    "t",
]

UNCERTAINTY_CANDIDATES = [
    "inferred_distance_se_mpc_h",
    "predicted_distance_se_mpc_h",
    "distance_se_mpc_h",
    "distance_error_mpc_h",
    "yerr",
]

WEIGHT_CANDIDATES = [
    "weight",
    "pair_weight",
    "count",
    "n_pairs",
]


def normalized_column_map(columns):
    return {
        str(column).strip().lower(): str(column)
        for column in columns
    }


def match_column(columns, candidates):
    normalized = normalized_column_map(columns)

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def semantic_match(columns, required_terms, forbidden_terms=None):
    forbidden_terms = forbidden_terms or []
    matches = []

    for column in columns:
        lower = str(column).strip().lower()

        if all(term in lower for term in required_terms):
            if not any(term in lower for term in forbidden_terms):
                matches.append(str(column))

    return matches[0] if matches else None


def identify_columns(frame, explicit=None):
    explicit = explicit or {}

    columns = list(frame.columns)

    time_col = explicit.get("time")
    true_col = explicit.get("true")
    inferred_col = explicit.get("inferred")
    uncertainty_col = explicit.get("uncertainty")
    weight_col = explicit.get("weight")

    if time_col is None:
        time_col = match_column(columns, TIME_CANDIDATES)

    if true_col is None:
        true_col = match_column(columns, TRUE_DISTANCE_CANDIDATES)

    if inferred_col is None:
        inferred_col = match_column(
            columns,
            INFERRED_DISTANCE_CANDIDATES,
        )

    # Semantic fallback for true physical separation.
    if true_col is None:
        true_col = semantic_match(
            columns,
            required_terms=["true", "distance"],
        )

    if true_col is None:
        true_col = semantic_match(
            columns,
            required_terms=["physical", "separation"],
            forbidden_terms=["inferred", "predicted", "graph"],
        )

    if true_col is None:
        true_col = semantic_match(
            columns,
            required_terms=["periodic", "distance"],
        )

    # Semantic fallback for inferred graph separation.
    if inferred_col is None:
        inferred_col = semantic_match(
            columns,
            required_terms=["inferred", "distance"],
        )

    if inferred_col is None:
        inferred_col = semantic_match(
            columns,
            required_terms=["predicted", "distance"],
        )

    if inferred_col is None:
        inferred_col = semantic_match(
            columns,
            required_terms=["graph", "distance"],
            forbidden_terms=["true"],
        )

    if inferred_col is None:
        inferred_col = semantic_match(
            columns,
            required_terms=["inferred", "separation"],
        )

    if uncertainty_col is None:
        uncertainty_col = match_column(
            columns,
            UNCERTAINTY_CANDIDATES,
        )

    if weight_col is None:
        weight_col = match_column(
            columns,
            WEIGHT_CANDIDATES,
        )

    return {
        "time": time_col,
        "true": true_col,
        "inferred": inferred_col,
        "uncertainty": uncertainty_col,
        "weight": weight_col,
    }


def inspect_csv_candidate(path, requested_times=None):
    try:
        sample = pd.read_csv(path, nrows=2000)
    except Exception as error:
        return {
            "path": path,
            "valid": False,
            "score": -1,
            "reason": str(error),
            "columns": [],
            "identified": {},
        }

    identified = identify_columns(sample)

    required_found = all(
        identified[key] is not None
        for key in ("time", "true", "inferred")
    )

    score = 0

    if identified["time"] is not None:
        score += 4
    if identified["true"] is not None:
        score += 4
    if identified["inferred"] is not None:
        score += 4
    if identified["uncertainty"] is not None:
        score += 1
    if identified["weight"] is not None:
        score += 1

    name = path.name.lower()

    for keyword in (
        "metric",
        "distance",
        "pair",
        "separation",
        "diffusion",
        "recovery",
    ):
        if keyword in name:
            score += 1

    available_times = []

    if identified["time"] is not None:
        try:
            available_times = sorted(
                pd.to_numeric(
                    sample[identified["time"]],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        except Exception:
            available_times = []

    if requested_times and available_times:
        overlap = len(
            set(int(x) for x in requested_times)
            & set(available_times)
        )
        score += 3 * overlap

    return {
        "path": path,
        "valid": required_found,
        "score": score,
        "reason": "",
        "columns": list(sample.columns),
        "identified": identified,
        "available_times_sample": available_times,
    }


def search_candidate_csvs(search_root, requested_times=None):
    root = Path(search_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            "Search root does not exist: {}".format(root)
        )

    candidates = []

    for path in root.rglob("*.csv"):
        result = inspect_csv_candidate(
            path,
            requested_times=requested_times,
        )

        if result["valid"]:
            candidates.append(result)

    candidates.sort(
        key=lambda result: (
            result["score"],
            str(result["path"]),
        ),
        reverse=True,
    )

    return candidates


def print_candidates(candidates):
    if not candidates:
        print(
            "No CSV file containing all three required quantities was found:"
        )
        print("  1. diffusion time")
        print("  2. true physical separation")
        print("  3. inferred graph separation")
        return

    print("\nCandidate pairwise metric files:\n")

    for rank, candidate in enumerate(candidates, start=1):
        identified = candidate["identified"]

        print(
            "[{}] score={} file={}".format(
                rank,
                candidate["score"],
                candidate["path"],
            )
        )
        print("    diffusion time:", identified["time"])
        print("    true separation:", identified["true"])
        print("    inferred separation:", identified["inferred"])

        times = candidate.get("available_times_sample", [])

        if times:
            print("    sample times:", times[:20])

        print()


def choose_input_file(args):
    if args.input_csv is not None:
        path = Path(args.input_csv).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Input CSV does not exist: {}".format(path)
            )

        return path

    candidates = search_candidate_csvs(
        args.search_root,
        requested_times=args.times,
    )

    print_candidates(candidates)

    if not candidates:
        raise FileNotFoundError(
            "No suitable pairwise metric CSV was found under {}".format(
                Path(args.search_root).resolve()
            )
        )

    selected_index = args.candidate_index - 1

    if selected_index < 0 or selected_index >= len(candidates):
        raise ValueError(
            "--candidate-index {} is outside the available range 1-{}.".format(
                args.candidate_index,
                len(candidates),
            )
        )

    selected = candidates[selected_index]["path"]

    print("Automatically selected:", selected)
    return selected


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)

    if weights is None:
        return float(np.mean(values))

    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)

    if not np.any(valid):
        return float(np.mean(values))

    return float(
        np.average(values[valid], weights=weights[valid])
    )


def weighted_std(values, weights):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return np.nan

    if weights is None:
        return float(np.std(values, ddof=1))

    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)

    if np.count_nonzero(valid) <= 1:
        return float(np.std(values, ddof=1))

    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    variance = np.average((x - mean) ** 2, weights=w)

    return float(np.sqrt(max(variance, 0.0)))


def bin_one_time(
    part,
    true_col,
    inferred_col,
    uncertainty_col,
    weight_col,
    bin_edges,
):
    true_values = part[true_col].to_numpy(float)
    inferred_values = part[inferred_col].to_numpy(float)

    uncertainties = (
        part[uncertainty_col].to_numpy(float)
        if uncertainty_col is not None
        else None
    )

    weights = (
        part[weight_col].to_numpy(float)
        if weight_col is not None
        else None
    )

    bin_index = np.digitize(true_values, bin_edges) - 1
    rows = []

    for index in range(len(bin_edges) - 1):
        mask = bin_index == index

        if np.count_nonzero(mask) == 0:
            continue

        x = true_values[mask]
        y = inferred_values[mask]
        w = weights[mask] if weights is not None else None

        mean_x = weighted_mean(x, w)
        mean_y = weighted_mean(y, w)

        residual = np.divide(
            y - x,
            x,
            out=np.full_like(y, np.nan),
            where=x != 0,
        )

        finite_residual = np.isfinite(residual)

        if uncertainty_col is not None:
            sigma = uncertainties[mask]
            sigma = sigma[np.isfinite(sigma) & (sigma >= 0)]

            y_error = (
                float(np.sqrt(np.sum(sigma ** 2)) / len(sigma))
                if len(sigma)
                else np.nan
            )
        else:
            scatter = weighted_std(y, w)
            y_error = (
                scatter / np.sqrt(np.count_nonzero(mask))
                if np.isfinite(scatter)
                else np.nan
            )

        if np.any(finite_residual):
            residual_weights = (
                w[finite_residual]
                if w is not None
                else None
            )

            residual_std = weighted_std(
                residual[finite_residual],
                residual_weights,
            )

            residual_error = (
                residual_std
                / np.sqrt(np.count_nonzero(finite_residual))
                if np.isfinite(residual_std)
                else np.nan
            )

            mean_residual = weighted_mean(
                residual[finite_residual],
                residual_weights,
            )
        else:
            mean_residual = np.nan
            residual_error = np.nan

        rows.append(
            {
                "bin_left": float(bin_edges[index]),
                "bin_right": float(bin_edges[index + 1]),
                "true_distance_mean": mean_x,
                "inferred_distance_mean": mean_y,
                "inferred_distance_error": y_error,
                "fractional_residual_mean": mean_residual,
                "fractional_residual_error": residual_error,
                "n_pairs": int(np.count_nonzero(mask)),
            }
        )

    return pd.DataFrame(rows)


def regression_metrics(true_values, inferred_values):
    x = np.asarray(true_values, dtype=float)
    y = np.asarray(inferred_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return {
            "rmse_mpc_h": np.nan,
            "fractional_rmse": np.nan,
            "r2": np.nan,
            "slope_through_origin": np.nan,
        }

    residual = y - x
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    fractional = np.divide(
        residual,
        x,
        out=np.full_like(residual, np.nan),
        where=x != 0,
    )

    fractional_rmse = float(
        np.sqrt(np.nanmean(fractional ** 2))
    )

    denominator = float(
        np.sum((x - np.mean(x)) ** 2)
    )

    r2 = (
        1.0 - float(np.sum(residual ** 2)) / denominator
        if denominator > 0
        else np.nan
    )

    slope_denominator = float(np.dot(x, x))

    slope = (
        float(np.dot(x, y) / slope_denominator)
        if slope_denominator > 0
        else np.nan
    )

    return {
        "rmse_mpc_h": rmse,
        "fractional_rmse": fractional_rmse,
        "r2": r2,
        "slope_through_origin": slope,
    }


def main(args):
    input_path = choose_input_file(args)
    frame = pd.read_csv(input_path)

    explicit = {
        "time": args.time_column,
        "true": args.true_column,
        "inferred": args.inferred_column,
        "uncertainty": args.uncertainty_column,
        "weight": args.weight_column,
    }

    identified = identify_columns(frame, explicit=explicit)

    missing = [
        key
        for key in ("time", "true", "inferred")
        if identified[key] is None
    ]

    if missing:
        raise ValueError(
            "Selected file is missing identifiable columns for: {}.\n"
            "Available columns: {}".format(
                missing,
                list(frame.columns),
            )
        )

    time_col = identified["time"]
    true_col = identified["true"]
    inferred_col = identified["inferred"]
    uncertainty_col = identified["uncertainty"]
    weight_col = identified["weight"]

    numeric_columns = [time_col, true_col, inferred_col]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    valid = (
        np.isfinite(frame[time_col].to_numpy(float))
        & np.isfinite(frame[true_col].to_numpy(float))
        & np.isfinite(frame[inferred_col].to_numpy(float))
    )

    frame = frame[valid].copy()

    if args.times:
        requested_times = [int(value) for value in args.times]
    else:
        requested_times = sorted(
            frame[time_col].astype(int).unique().tolist()
        )

    available_times = sorted(
        frame[time_col].astype(int).unique().tolist()
    )

    missing_times = sorted(
        set(requested_times) - set(available_times)
    )

    if missing_times:
        print(
            "Warning: requested times not present in the selected table:",
            missing_times,
        )
        print("Available times:", available_times)

    frame = frame[
        frame[time_col].astype(int).isin(requested_times)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No rows remain after selecting diffusion times {}.".format(
                requested_times
            )
        )

    minimum_distance = (
        args.min_distance
        if args.min_distance is not None
        else float(frame[true_col].min())
    )

    maximum_distance = (
        args.max_distance
        if args.max_distance is not None
        else float(frame[true_col].max())
    )

    frame = frame[
        (frame[true_col] >= minimum_distance)
        & (frame[true_col] <= maximum_distance)
    ].copy()

    if args.log_bins:
        if minimum_distance <= 0:
            positive = frame.loc[
                frame[true_col] > 0,
                true_col,
            ]

            if positive.empty:
                raise ValueError(
                    "Logarithmic bins require positive distances."
                )

            minimum_distance = float(positive.min())

        bin_edges = np.geomspace(
            minimum_distance,
            maximum_distance,
            args.n_bins + 1,
        )
    else:
        bin_edges = np.linspace(
            minimum_distance,
            maximum_distance,
            args.n_bins + 1,
        )

    figure, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.1, 1.35],
            "hspace": 0.06,
        },
    )

    metrics_rows = []
    binned_frames = []

    for diffusion_time in requested_times:
        part = frame[
            frame[time_col].astype(int) == int(diffusion_time)
        ].copy()

        if part.empty:
            continue

        metrics = regression_metrics(
            part[true_col].to_numpy(float),
            part[inferred_col].to_numpy(float),
        )

        metrics["diffusion_time"] = int(diffusion_time)
        metrics["n_pairs"] = int(len(part))
        metrics_rows.append(metrics)

        binned = bin_one_time(
            part,
            true_col,
            inferred_col,
            uncertainty_col,
            weight_col,
            bin_edges,
        )

        binned.insert(
            0,
            "diffusion_time",
            int(diffusion_time),
        )

        binned_frames.append(binned)

        label = (
            rf"$t={int(diffusion_time)}$"
            + "\n"
            + rf"$R^2={metrics['r2']:.3f}$"
        )

        ax_top.errorbar(
            binned["true_distance_mean"],
            binned["inferred_distance_mean"],
            yerr=binned["inferred_distance_error"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            capsize=2.5,
            label=label,
        )

        ax_bottom.errorbar(
            binned["true_distance_mean"],
            binned["fractional_residual_mean"],
            yerr=binned["fractional_residual_error"],
            marker="o",
            markersize=4.0,
            linewidth=1.4,
            capsize=2.5,
        )

    diagonal = np.array(
        [minimum_distance, maximum_distance],
        dtype=float,
    )

    ax_top.plot(
        diagonal,
        diagonal,
        linestyle="--",
        linewidth=1.3,
        label="Euclidean one-to-one relation",
    )

    ax_bottom.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax_bottom.axhline(
        0.05,
        linestyle=":",
        linewidth=1.0,
    )

    ax_bottom.axhline(
        -0.05,
        linestyle=":",
        linewidth=1.0,
        label=r"$\pm5\%$",
    )

    ax_top.set_ylabel(
        r"Graph-inferred separation "
        r"$\hat r\,[h^{-1}\,\mathrm{Mpc}]$"
    )

    ax_bottom.set_ylabel(
        r"$(\hat r-r)/r$"
    )

    ax_bottom.set_xlabel(
        r"True periodic separation "
        r"$r\,[h^{-1}\,\mathrm{Mpc}]$"
    )

    ax_top.set_xlim(
        minimum_distance,
        maximum_distance,
    )

    ax_top.set_ylim(
        minimum_distance,
        maximum_distance,
    )

    if args.log_x:
        ax_top.set_xscale("log")
        ax_bottom.set_xscale("log")

    if args.log_y:
        ax_top.set_yscale("log")

    ax_top.grid(True, alpha=0.25)
    ax_bottom.grid(True, alpha=0.25)

    ax_top.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    ax_bottom.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    ax_top.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
    )

    ax_top.set_title(
        "Recovery of the Euclidean metric from graph diffusion"
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    metrics_frame = pd.DataFrame(
        metrics_rows
    ).sort_values("diffusion_time")

    metrics_path = output.with_name(
        output.stem + "_metrics.csv"
    )

    metrics_frame.to_csv(
        metrics_path,
        index=False,
    )

    binned_path = None

    if binned_frames:
        binned_output = pd.concat(
            binned_frames,
            ignore_index=True,
        )

        binned_path = output.with_name(
            output.stem + "_binned.csv"
        )

        binned_output.to_csv(
            binned_path,
            index=False,
        )

    print("\nSelected input file:")
    print(" ", input_path)

    print("\nColumns used:")
    print("  diffusion time:", time_col)
    print("  true physical separation:", true_col)
    print("  inferred graph separation:", inferred_col)
    print("  uncertainty:", uncertainty_col)
    print("  weight:", weight_col)

    print("\nMetrics:")
    print(metrics_frame.to_string(index=False))

    print("\nWrote:", output)
    print("Wrote:", metrics_path)

    if binned_path is not None:
        print("Wrote:", binned_path)


def parse_args():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--input-csv",
        default=None,
        help="Use this specific pairwise metric CSV.",
    )

    source.add_argument(
        "--search-root",
        default=".",
        help=(
            "Recursively search this directory for a suitable CSV. "
            "Default: current directory."
        ),
    )

    parser.add_argument(
        "--candidate-index",
        type=int,
        default=1,
        help=(
            "Candidate rank to use after automatic search. "
            "Default: 1, the highest-ranked candidate."
        ),
    )

    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=None,
        help="Diffusion times to plot, e.g. 256 1024 4096.",
    )

    parser.add_argument("--true-column", default=None)
    parser.add_argument("--inferred-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--uncertainty-column", default=None)
    parser.add_argument("--weight-column", default=None)

    parser.add_argument(
        "--n-bins",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--log-bins",
        action="store_true",
    )

    parser.add_argument(
        "--log-x",
        action="store_true",
    )

    parser.add_argument(
        "--log-y",
        action="store_true",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--output",
        default="figure2_metric_recovery.png",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
