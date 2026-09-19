from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
STEP6 = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)
PKDIR = Path("pk_heatkernel_bootstrap")

GRAPH_FILE = STEP6 / "bootstrap_crossing_times_interpolated.npy"
PK_FILE = PKDIR / "bootstrap_Rhom_pk_samples.npy"

OUTDIR = PKDIR
OUTDIR.mkdir(exist_ok=True)

A_RMS = 10.409558


# ---------------------------------------------------------------------
# Load paired bootstrap samples
# ---------------------------------------------------------------------
t_graph = np.load(GRAPH_FILE)
R_pk = np.load(PK_FILE)

if t_graph.ndim != 1 or R_pk.ndim != 1:
    raise RuntimeError(
        f"Expected 1-D arrays, got {t_graph.shape} and {R_pk.shape}"
    )

if len(t_graph) != len(R_pk):
    raise RuntimeError(
        f"Bootstrap length mismatch: graph={len(t_graph)}, pk={len(R_pk)}"
    )

nboot = len(t_graph)

# Convert interpolated graph crossing time to physical radius.
R_graph = A_RMS * np.sqrt(t_graph)

valid_graph = np.isfinite(R_graph)
valid_pk = np.isfinite(R_pk)
joint = valid_graph & valid_pk

Rg = R_graph[joint]
Rp = R_pk[joint]

# Paired differences and ratios.
delta = Rg - Rp
ratio = Rp / Rg
frac_offset = delta / Rg


def q(x):
    return np.percentile(x, [2.5, 16, 50, 84, 97.5])


gq = q(Rg)
pq = q(Rp)
dq = q(delta)
rq = q(ratio)
fq = q(frac_offset)

p_graph_gt_pk = np.mean(delta > 0)
p_graph_lt_pk = np.mean(delta < 0)
p_abs_lt_10 = np.mean(np.abs(delta) < 10.0)
p_abs_lt_20 = np.mean(np.abs(delta) < 20.0)
p_abs_lt_40 = np.mean(np.abs(delta) < 40.0)

# Two-sided sign-based probability for zero difference.
p_two_sided_sign = min(
    1.0,
    2.0 * min(p_graph_gt_pk, p_graph_lt_pk)
)

# Correlation is useful because this is a paired bootstrap.
corr = np.corrcoef(Rg, Rp)[0, 1]


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------
print("=" * 72)
print("PAIRED GRAPH-vs-P(k) BOOTSTRAP COMPARISON")
print("=" * 72)
print(f"total bootstrap realizations : {nboot}")
print(
    f"valid graph crossings        : "
    f"{valid_graph.sum()} / {nboot} "
    f"({valid_graph.mean():.4f})"
)
print(
    f"valid P(k) crossings         : "
    f"{valid_pk.sum()} / {nboot} "
    f"({valid_pk.mean():.4f})"
)
print(
    f"joint valid pairs            : "
    f"{joint.sum()} / {nboot} "
    f"({joint.mean():.4f})"
)
print()

print("Graph Rhom on joint-valid realizations:")
print(
    f"  median = {gq[2]:.3f} "
    f"+{gq[3] - gq[2]:.3f} "
    f"-{gq[2] - gq[1]:.3f} h^-1 Mpc"
)
print(
    f"  95% interval = [{gq[0]:.3f}, {gq[4]:.3f}]"
)
print()

print("P(k) Rhom on the same joint-valid realizations:")
print(
    f"  median = {pq[2]:.3f} "
    f"+{pq[3] - pq[2]:.3f} "
    f"-{pq[2] - pq[1]:.3f} h^-1 Mpc"
)
print(
    f"  95% interval = [{pq[0]:.3f}, {pq[4]:.3f}]"
)
print()

print("=" * 72)
print("PAIRED OFFSET: Delta R = R_graph - R_P(k)")
print("=" * 72)
print(
    f"Delta R median = {dq[2]:.3f} "
    f"+{dq[3] - dq[2]:.3f} "
    f"-{dq[2] - dq[1]:.3f} h^-1 Mpc"
)
print(
    f"Delta R 68% interval = [{dq[1]:.3f}, {dq[3]:.3f}]"
)
print(
    f"Delta R 95% interval = [{dq[0]:.3f}, {dq[4]:.3f}]"
)
print()

print(
    f"P(R_graph > R_Pk) = {p_graph_gt_pk:.4f}"
)
print(
    f"P(R_graph < R_Pk) = {p_graph_lt_pk:.4f}"
)
print(
    f"two-sided sign probability around Delta R=0 = "
    f"{p_two_sided_sign:.4f}"
)
print()

print(
    f"P(|Delta R| < 10 h^-1 Mpc) = {p_abs_lt_10:.4f}"
)
print(
    f"P(|Delta R| < 20 h^-1 Mpc) = {p_abs_lt_20:.4f}"
)
print(
    f"P(|Delta R| < 40 h^-1 Mpc) = {p_abs_lt_40:.4f}"
)
print()

print("=" * 72)
print("RELATIVE OFFSET")
print("=" * 72)
print(
    f"median R_Pk/R_graph = {rq[2]:.4f} "
    f"[{rq[1]:.4f}, {rq[3]:.4f}] (68%)"
)
print(
    f"median (R_graph-R_Pk)/R_graph = "
    f"{100.0*fq[2]:.2f}% "
    f"[{100.0*fq[1]:.2f}%, {100.0*fq[3]:.2f}%] (68%)"
)
print()

print(f"paired correlation corr(R_graph,R_Pk) = {corr:.4f}")


# ---------------------------------------------------------------------
# Save paired samples
# ---------------------------------------------------------------------
df = pd.DataFrame({
    "bootstrap_index": np.arange(nboot),
    "graph_crossing_time": t_graph,
    "Rhom_graph_hmpc": R_graph,
    "Rhom_pk_hmpc": R_pk,
    "joint_valid": joint,
})

df["delta_R_graph_minus_pk_hmpc"] = (
    df["Rhom_graph_hmpc"] - df["Rhom_pk_hmpc"]
)

df["ratio_pk_over_graph"] = (
    df["Rhom_pk_hmpc"] / df["Rhom_graph_hmpc"]
)

df.to_csv(
    OUTDIR / "graph_vs_pk_paired_bootstrap.csv",
    index=False
)


summary = {
    "n_bootstrap": int(nboot),
    "n_valid_graph": int(valid_graph.sum()),
    "n_valid_pk": int(valid_pk.sum()),
    "n_joint_valid": int(joint.sum()),
    "joint_valid_fraction": float(joint.mean()),
    "graph_Rhom_joint_hmpc": {
        "q025": float(gq[0]),
        "q16": float(gq[1]),
        "median": float(gq[2]),
        "q84": float(gq[3]),
        "q975": float(gq[4]),
    },
    "pk_Rhom_joint_hmpc": {
        "q025": float(pq[0]),
        "q16": float(pq[1]),
        "median": float(pq[2]),
        "q84": float(pq[3]),
        "q975": float(pq[4]),
    },
    "delta_graph_minus_pk_hmpc": {
        "q025": float(dq[0]),
        "q16": float(dq[1]),
        "median": float(dq[2]),
        "q84": float(dq[3]),
        "q975": float(dq[4]),
    },
    "ratio_pk_over_graph": {
        "q025": float(rq[0]),
        "q16": float(rq[1]),
        "median": float(rq[2]),
        "q84": float(rq[3]),
        "q975": float(rq[4]),
    },
    "p_graph_gt_pk": float(p_graph_gt_pk),
    "p_graph_lt_pk": float(p_graph_lt_pk),
    "two_sided_sign_probability": float(p_two_sided_sign),
    "p_abs_delta_lt_10": float(p_abs_lt_10),
    "p_abs_delta_lt_20": float(p_abs_lt_20),
    "p_abs_delta_lt_40": float(p_abs_lt_40),
    "paired_correlation": float(corr),
}

with open(
    OUTDIR / "graph_vs_pk_paired_bootstrap_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print()
print("Wrote:")
print(" ", OUTDIR / "graph_vs_pk_paired_bootstrap.csv")
print(" ", OUTDIR / "graph_vs_pk_paired_bootstrap_summary.json")
