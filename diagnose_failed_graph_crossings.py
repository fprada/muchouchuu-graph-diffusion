from pathlib import Path
import numpy as np
import pandas as pd

STEP6 = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)
PKDIR = Path("pk_heatkernel_bootstrap")

graph_t = np.load(
    STEP6 / "bootstrap_crossing_times_interpolated.npy"
)
pk_R = np.load(
    PKDIR / "bootstrap_Rhom_pk_samples.npy"
)

if graph_t.shape != pk_R.shape:
    raise RuntimeError(
        f"Shape mismatch: graph={graph_t.shape}, pk={pk_R.shape}"
    )

graph_ok = np.isfinite(graph_t)
graph_fail = ~graph_ok
pk_ok = np.isfinite(pk_R)

succ_pk = pk_R[graph_ok & pk_ok]
fail_pk = pk_R[graph_fail & pk_ok]

def q(x):
    return np.percentile(x, [2.5, 16, 50, 84, 97.5])

qs = q(succ_pk)
qf = q(fail_pk)

print("=" * 72)
print("FAILED GRAPH-CROSSING DIAGNOSTIC")
print("=" * 72)

print(f"total realizations       : {len(graph_t)}")
print(f"graph successes          : {graph_ok.sum()}")
print(f"graph failures           : {graph_fail.sum()}")
print(f"P(k) valid among success : {len(succ_pk)}")
print(f"P(k) valid among failure : {len(fail_pk)}")
print()

print("P(k) Rhom for GRAPH-SUCCESS realizations:")
print(
    f"  median = {qs[2]:.3f} "
    f"+{qs[3]-qs[2]:.3f} "
    f"-{qs[2]-qs[1]:.3f} h^-1 Mpc"
)
print(
    f"  95% interval = [{qs[0]:.3f}, {qs[4]:.3f}]"
)
print()

print("P(k) Rhom for GRAPH-FAILURE realizations:")
print(
    f"  median = {qf[2]:.3f} "
    f"+{qf[3]-qf[2]:.3f} "
    f"-{qf[2]-qf[1]:.3f} h^-1 Mpc"
)
print(
    f"  95% interval = [{qf[0]:.3f}, {qf[4]:.3f}]"
)
print()

print(
    "median shift failure-success = "
    f"{qf[2]-qs[2]:.3f} h^-1 Mpc"
)

# Simple rank-style comparison:
all_pk = np.concatenate([succ_pk, fail_pk])

# Probability that a random failed-case prediction exceeds
# a random successful-case prediction. Compute efficiently
# using sorted successful sample.
s = np.sort(succ_pk)
counts = np.searchsorted(s, fail_pk, side="left")
p_fail_gt_success = counts.sum() / (
    len(fail_pk) * len(succ_pk)
)

print(
    "P(Rpk_failure > Rpk_success) = "
    f"{p_fail_gt_success:.4f}"
)

# Save classification table
df = pd.DataFrame({
    "bootstrap_index": np.arange(len(graph_t)),
    "graph_crossing_detected": graph_ok,
    "graph_crossing_time": graph_t,
    "Rhom_pk_hmpc": pk_R,
})

df.to_csv(
    PKDIR / "failed_graph_crossing_diagnostic.csv",
    index=False
)

print()
print(
    "Wrote:",
    PKDIR / "failed_graph_crossing_diagnostic.csv"
)
