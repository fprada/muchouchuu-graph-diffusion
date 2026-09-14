from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INCSV = Path("homogeneity_redshift_evolution_bootstrap_summary.csv")
OUTPNG = Path("figure_homogeneity_redshift_evolution_bootstrap.png")
OUTPDF = Path("figure_homogeneity_redshift_evolution_bootstrap.pdf")


def load_block(df, observable):
    d = df[df["observable"] == observable].copy()
    d = d.sort_values("z").reset_index(drop=True)

    x = d["z"].to_numpy(float)
    y = d["median"].to_numpy(float)

    yerr68 = np.vstack([
        y - d["q16"].to_numpy(float),
        d["q84"].to_numpy(float) - y,
    ])

    yerr95 = np.vstack([
        y - d["q025"].to_numpy(float),
        d["q975"].to_numpy(float) - y,
    ])

    return d, x, y, yerr68, yerr95


def main():
    df = pd.read_csv(INCSV)

    required = {
        "observable", "z", "comparison", "n_or_pairs",
        "q025", "q16", "median", "q84", "q975",
        "probability_gt_0", "probability_lt_0"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {INCSV}: {sorted(missing)}")

    d_t, x_t, y_t, yerr68_t, yerr95_t = load_block(df, "transport")
    d_d, x_d, y_d, yerr68_d, yerr95_d = load_block(df, "density")

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.2, 6.6),
        sharex=True,
        constrained_layout=True
    )

    ax = axes[0]
    ax.axhline(0.0, linestyle="--", linewidth=1.5, alpha=0.9, zorder=0)

    # 95% interval
    ax.errorbar(
        x_t, y_t, yerr=yerr95_t,
        fmt="none", elinewidth=1.0, capsize=0, alpha=0.45,
        label=r"$95\%$ interval"
    )

    # 68% interval + marker
    ax.errorbar(
        x_t, y_t, yerr=yerr68_t,
        fmt="o", markersize=6, elinewidth=2.0, capsize=3,
        label=r"$68\%$ interval"
    )

    ax.set_ylabel(r"$\Delta R_{\rm hom}^{(t)}\ [h^{-1}\,{\rm Mpc}]$")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.18)

    ax = axes[1]
    ax.axhline(0.0, linestyle="--", linewidth=1.5, alpha=0.9, zorder=0)

    # 95% interval
    ax.errorbar(
        x_d, y_d, yerr=yerr95_d,
        fmt="none", elinewidth=1.0, capsize=0, alpha=0.45,
        label=r"$95\%$ interval"
    )

    # 68% interval + marker
    ax.errorbar(
        x_d, y_d, yerr=yerr68_d,
        fmt="s", markersize=6, elinewidth=2.0, capsize=3,
        label=r"$68\%$ interval"
    )

    ax.set_ylabel(r"$\Delta R_{\rm hom}^{(\rho)}\ [h^{-1}\,{\rm Mpc}]$")
    ax.set_xlabel(r"Redshift $z$")
    ax.grid(alpha=0.18)

    xticks = sorted(df["z"].unique())
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{z:.2f}" for z in xticks])

    # Optional panel labels
    axes[0].text(0.02, 0.92, "(a)", transform=axes[0].transAxes)
    axes[1].text(0.02, 0.92, "(b)", transform=axes[1].transAxes)

    fig.savefig(OUTPNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPDF, bbox_inches="tight")

    print("Wrote:", OUTPNG)
    print("Wrote:", OUTPDF)
    print()
    print("Transport points:")
    print(d_t[["z", "median", "q16", "q84", "q025", "q975"]].to_string(index=False))
    print()
    print("Density points:")
    print(d_d[["z", "median", "q16", "q84", "q025", "q975"]].to_string(index=False))


if __name__ == "__main__":
    main()
