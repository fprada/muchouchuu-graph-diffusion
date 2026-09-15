#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.bbox": "tight",
})

ROOT = Path(".")

DATA = [
    {
        "z": 0.00,
        "label": "z=0",
        "ratio_summary": ROOT / "muchouchuu_density_transport_ratio_bootstrap_1pct_final/density_transport_ratio_summary.json",
        "density_npy": ROOT / "muchouchuu_density_transport_ratio_bootstrap_1pct_final/density_crossing_radii_interpolated.npy",
    },
    {
        "z": 0.49,
        "label": "z=0.49",
        "ratio_summary": ROOT / "growth_test/z049/muchouchuu_density_transport_ratio_bootstrap_1pct/density_transport_ratio_summary.json",
        "density_npy": ROOT / "growth_test/z049/muchouchuu_density_transport_ratio_bootstrap_1pct/density_crossing_radii_interpolated.npy",
    },
    {
        "z": 1.03,
        "label": "z=1.03",
        "ratio_summary": ROOT / "growth_test/z103/muchouchuu_density_transport_ratio_bootstrap_1pct/density_transport_ratio_summary.json",
        "density_npy": ROOT / "growth_test/z103/muchouchuu_density_transport_ratio_bootstrap_1pct/density_crossing_radii_interpolated.npy",
    },
    {
        "z": 2.03,
        "label": "z=2.03",
        "ratio_summary": ROOT / "growth_test/z203/muchouchuu_density_transport_ratio_bootstrap_1pct/density_transport_ratio_summary.json",
        "density_npy": ROOT / "growth_test/z203/muchouchuu_density_transport_ratio_bootstrap_1pct/density_crossing_radii_interpolated.npy",
    },
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_density_quantiles(path):
    x = np.load(path)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise RuntimeError(
            f"No finite reconstructed density crossings in {path}"
        )

    q025, q16, median, q84, q975 = np.percentile(
        x, [2.5, 16, 50, 84, 97.5]
    )

    return {
        "q025": float(q025),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
        "q975": float(q975),
    }


def get_nested(d, keys):
    x = d
    for k in keys:
        if k not in x:
            raise KeyError(f"Missing key path: {'/'.join(keys)}")
        x = x[k]
    return x


def pick_quantiles(block, candidate_paths):
    """
    Return dict with q025,q16,median,q84,q975 from the first matching path.
    candidate_paths is a list of key-path tuples.
    """
    for path in candidate_paths:
        try:
            q = get_nested(block, path)
            needed = ["q025", "q16", "median", "q84", "q975"]
            if all(k in q for k in needed):
                return {k: float(q[k]) for k in needed}
        except KeyError:
            pass
    raise KeyError(f"Could not find quantiles in any of: {candidate_paths}")


rows = []

for item in DATA:
    ratio_data = load_json(item["ratio_summary"])

    # Reconstructed/interpolated density crossing quantiles
    dens_q = load_density_quantiles(
        item["density_npy"]
    )

    # Transport quantiles
    trans_q = pick_quantiles(
        ratio_data,
        [
            ("transport_bootstrap", "transport_radius_quantiles_mpc_h"),
            ("transport_radius_quantiles_mpc_h",),
        ],
    )

    # Ratio quantiles
    ratio_q = pick_quantiles(
        ratio_data,
        [
            ("transport_to_density_ratio", "quantiles"),
            ("quantiles",),
        ],
    )

    rows.append(
        {
            "z": float(item["z"]),
            "label": item["label"],

            "density_median": dens_q["median"],
            "density_q16": dens_q["q16"],
            "density_q84": dens_q["q84"],

            "transport_median": trans_q["median"],
            "transport_q16": trans_q["q16"],
            "transport_q84": trans_q["q84"],

            "ratio_median": ratio_q["median"],
            "ratio_q16": ratio_q["q16"],
            "ratio_q84": ratio_q["q84"],
        }
    )

rows = sorted(rows, key=lambda r: r["z"])

z = np.array([r["z"] for r in rows], dtype=float)

density_med = np.array([r["density_median"] for r in rows], dtype=float)
density_lo = density_med - np.array([r["density_q16"] for r in rows], dtype=float)
density_hi = np.array([r["density_q84"] for r in rows], dtype=float) - density_med

transport_med = np.array([r["transport_median"] for r in rows], dtype=float)
transport_lo = transport_med - np.array([r["transport_q16"] for r in rows], dtype=float)
transport_hi = np.array([r["transport_q84"] for r in rows], dtype=float) - transport_med

ratio_med = np.array([r["ratio_median"] for r in rows], dtype=float)
ratio_lo = ratio_med - np.array([r["ratio_q16"] for r in rows], dtype=float)
ratio_hi = np.array([r["ratio_q84"] for r in rows], dtype=float) - ratio_med

print("Loaded values:")
for r in rows:
    print(
        f"{r['label']:>7s} : "
        f"R_density = {r['density_median']:.3f} "
        f"[{r['density_q16']:.3f}, {r['density_q84']:.3f}], "
        f"R_transport = {r['transport_median']:.3f} "
        f"[{r['transport_q16']:.3f}, {r['transport_q84']:.3f}], "
        f"ratio = {r['ratio_median']:.3f} "
        f"[{r['ratio_q16']:.3f}, {r['ratio_q84']:.3f}]"
    )

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7.6, 8.4),
    sharex=True,
    gridspec_kw={
        "height_ratios": [2.2, 1.3],
        "hspace": 0.06,
    },
)

# ---- top panel ----
ax1.errorbar(
    z, density_med,
    yerr=[density_lo, density_hi],
    fmt="o-",
    capsize=4,
    lw=1.8,
    ms=7,
    label=r"Density homogeneity $R_{\rm hom}^{(\rho)}$",
    zorder=3,
)
ax1.errorbar(
    z, transport_med,
    yerr=[transport_lo, transport_hi],
    fmt="s-",
    capsize=4,
    lw=1.8,
    ms=7,
    label=r"Transport homogeneity $R_{\rm hom}^{(t)}$",
    zorder=3,
)

ax1.set_ylabel(
    r"Homogeneity scale [$h^{-1}\,\mathrm{Mpc}$]"
)

ax1.legend(
    frameon=True,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.56),
    ncol=2,
)

ax1.grid(
    alpha=0.25,
    linewidth=0.8,
)

ax1.text(
    0.02,
    0.96,
    "(a)",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
)

# ---- bottom panel ----
ax2.errorbar(
    z, ratio_med,
    yerr=[ratio_lo, ratio_hi],
    fmt="o-",
    capsize=4,
    lw=1.8,
    ms=7,
    label=r"$R_{\rm hom}^{(t)} / R_{\rm hom}^{(\rho)}$",
    zorder=3,
)
ax2.axhline(1.0, ls="--", lw=1.0, alpha=0.7)

ax2.set_xlabel(r"Redshift $z$")
ax2.set_ylabel(
    r"$R_{\rm hom}^{(t)} / R_{\rm hom}^{(\rho)}$"
)

ax2.grid(
    alpha=0.25,
    linewidth=0.8,
)

ax2.text(
    0.02,
    0.96,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
)

# nice limits
ax2.set_xlim(-0.08, 2.12)

ax2.set_xticks([
    0.00,
    0.49,
    1.03,
    2.03,
])

ax2.set_xticklabels([
    "0",
    "0.49",
    "1.03",
    "2.03",
])
y_all = np.concatenate([density_med - density_lo, density_med + density_hi,
                        transport_med - transport_lo, transport_med + transport_hi])
pad1 = 0.06 * (y_all.max() - y_all.min())
ax1.set_ylim(y_all.min() - pad1, y_all.max() + pad1)

y2_all = np.concatenate([ratio_med - ratio_lo, ratio_med + ratio_hi])
pad2 = 0.08 * (y2_all.max() - y2_all.min())
ax2.set_ylim(y2_all.min() - pad2, y2_all.max() + pad2)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.10,
    top=0.98,
)
fig.savefig("figure_transport_density_redshift_evolution_pub.png", dpi=220)
fig.savefig("figure_transport_density_redshift_evolution_pub.pdf")
print()
print("Wrote: figure_transport_density_redshift_evolution_pub.png")
print("Wrote: figure_transport_density_redshift_evolution_pub.pdf")
