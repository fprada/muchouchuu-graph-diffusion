from pathlib import Path
import json

import numpy as np
import pandas as pd


STEP6 = Path(
    "muchouchuu_step6_window7_s64_strictcentered_tmin3584"
)
PKDIR = Path("pk_heatkernel_bootstrap")
PKDIR.mkdir(exist_ok=True)

GRAPH_T_FILE = STEP6 / "bootstrap_crossing_times_interpolated.npy"
PK_R_FILE = PKDIR / "bootstrap_Rhom_pk_samples.npy"

A_RMS = 10.409558

# Latest start time at which a 3-point persistent run can be verified:
T_CENSOR = 7168.0
R_CENSOR = A_RMS * np.sqrt(T_CENSOR)


graph_t = np.load(GRAPH_T_FILE)
R_pk = np.load(PK_R_FILE)

if graph_t.shape != R_pk.shape:
    raise RuntimeError(
        f"Shape mismatch: graph={graph_t.shape}, pk={R_pk.shape}"
    )

n = len(graph_t)

event = np.isfinite(graph_t)

# For successful cases use measured crossing time.
# For failures use right-censoring time.
t_obs = np.where(event, graph_t, T_CENSOR)
R_obs = A_RMS * np.sqrt(t_obs)

if not np.all(np.isfinite(R_pk)):
    raise RuntimeError("Non-finite P(k) predictions found.")


# ---------------------------------------------------------------------
# Kaplan-Meier estimator implemented directly
# ---------------------------------------------------------------------
def kaplan_meier(times, events):
    times = np.asarray(times, float)
    events = np.asarray(events, bool)

    unique_event_times = np.sort(np.unique(times[events]))

    surv = 1.0

    out_t = []
    out_s = []
    out_nrisk = []
    out_nevent = []
    out_ncensor = []

    for tt in unique_event_times:

        at_risk = np.sum(times >= tt)
        n_event = np.sum((times == tt) & events)
        n_censor = np.sum((times == tt) & (~events))

        if at_risk <= 0:
            continue

        surv *= (1.0 - n_event / at_risk)

        out_t.append(tt)
        out_s.append(surv)
        out_nrisk.append(at_risk)
        out_nevent.append(n_event)
        out_ncensor.append(n_censor)

    return pd.DataFrame({
        "time": out_t,
        "survival": out_s,
        "n_at_risk": out_nrisk,
        "n_event": out_nevent,
        "n_censor": out_ncensor,
    })


km_t = kaplan_meier(t_obs, event)

# Convert KM time grid to radius.
km_t["R_hmpc"] = A_RMS * np.sqrt(km_t["time"].to_numpy(float))


def km_quantile(km, q):
    """
    Return event-time quantile q from KM survival curve.

    CDF = 1 - S.
    Quantile occurs when S <= 1-q.
    """
    target = 1.0 - q
    hit = km["survival"].to_numpy() <= target

    if not np.any(hit):
        return np.nan

    i = np.where(hit)[0][0]
    return float(km.iloc[i]["R_hmpc"])


R_km_q16 = km_quantile(km_t, 0.16)
R_km_med = km_quantile(km_t, 0.50)
R_km_q84 = km_quantile(km_t, 0.84)
R_km_q025 = km_quantile(km_t, 0.025)
R_km_q975 = km_quantile(km_t, 0.975)


# ---------------------------------------------------------------------
# Censored paired graph-minus-P(k) offsets
#
# Success:
#     Delta = R_graph - R_pk
#
# Failure:
#     R_graph > R_censor
# therefore
#     Delta > R_censor - R_pk
#
# These are right-censored lower bounds on Delta.
# ---------------------------------------------------------------------
delta_event = R_obs - R_pk

delta_time = np.where(
    event,
    delta_event,
    R_CENSOR - R_pk,
)

delta_km = kaplan_meier(delta_time, event)

D_q16 = km_quantile(
    delta_km.assign(R_hmpc=delta_km["time"]),
    0.16
)
D_med = km_quantile(
    delta_km.assign(R_hmpc=delta_km["time"]),
    0.50
)
D_q84 = km_quantile(
    delta_km.assign(R_hmpc=delta_km["time"]),
    0.84
)
D_q025 = km_quantile(
    delta_km.assign(R_hmpc=delta_km["time"]),
    0.025
)
D_q975 = km_quantile(
    delta_km.assign(R_hmpc=delta_km["time"]),
    0.975
)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
print("=" * 72)
print("RIGHT-CENSORED GRAPH HOMOGENEITY ANALYSIS")
print("=" * 72)

print(f"total realizations       : {n}")
print(f"detected crossings       : {event.sum()}")
print(f"right-censored           : {(~event).sum()}")
print(f"crossing success fraction: {event.mean():.4f}")
print()

print(f"T_censor                 : {T_CENSOR:.1f}")
print(f"R_censor                 : {R_CENSOR:.3f} h^-1 Mpc")
print()

print("Kaplan-Meier graph Rhom:")
print(f"  q025   : {R_km_q025:.3f}")
print(f"  q16    : {R_km_q16:.3f}")
print(f"  median : {R_km_med:.3f}")
print(f"  q84    : {R_km_q84:.3f}")
print(f"  q975   : {R_km_q975:.3f}")
print()

if np.isfinite(R_km_med):
    print(
        f"  Rhom_KM = {R_km_med:.1f} "
        f"+{R_km_q84-R_km_med:.1f} "
        f"-{R_km_med-R_km_q16:.1f} h^-1 Mpc"
    )
else:
    print("  KM median not reached within observed/censored range.")

print()

print("=" * 72)
print("RIGHT-CENSORED PAIRED OFFSET")
print("=" * 72)
print("Delta R = R_graph - R_P(k)")
print()

print(f"  q025   : {D_q025:.3f}")
print(f"  q16    : {D_q16:.3f}")
print(f"  median : {D_med:.3f}")
print(f"  q84    : {D_q84:.3f}")
print(f"  q975   : {D_q975:.3f}")

if np.isfinite(D_med):
    print(
        f"  Delta_R_KM = {D_med:.1f} "
        f"+{D_q84-D_med:.1f} "
        f"-{D_med-D_q16:.1f} h^-1 Mpc"
    )
else:
    print("  KM Delta median not reached.")

print()

# Failure lower bounds on Delta
fail_lower = R_CENSOR - R_pk[~event]

print("Failed-case lower bounds on Delta R:")
print(
    f"  median lower bound = "
    f"{np.median(fail_lower):.3f} h^-1 Mpc"
)
print(
    f"  q16/q84 lower bound = "
    f"{np.percentile(fail_lower,16):.3f} / "
    f"{np.percentile(fail_lower,84):.3f}"
)
print(
    f"  minimum lower bound = "
    f"{np.min(fail_lower):.3f}"
)
print()


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
samples = pd.DataFrame({
    "bootstrap_index": np.arange(n),
    "graph_crossing_detected": event,
    "graph_time_observed_or_censored": t_obs,
    "graph_R_observed_or_censored_hmpc": R_obs,
    "Rhom_pk_hmpc": R_pk,
    "delta_observed_or_censored_hmpc": delta_time,
})

samples.to_csv(
    PKDIR / "censored_graph_vs_pk_samples.csv",
    index=False,
)

km_t.to_csv(
    PKDIR / "km_graph_Rhom_curve.csv",
    index=False,
)

delta_km.to_csv(
    PKDIR / "km_delta_graph_minus_pk_curve.csv",
    index=False,
)

summary = {
    "n_total": int(n),
    "n_detected": int(event.sum()),
    "n_censored": int((~event).sum()),
    "success_fraction": float(event.mean()),
    "t_censor": float(T_CENSOR),
    "R_censor_hmpc": float(R_CENSOR),
    "graph_Rhom_KM_hmpc": {
        "q025": None if not np.isfinite(R_km_q025) else float(R_km_q025),
        "q16": None if not np.isfinite(R_km_q16) else float(R_km_q16),
        "median": None if not np.isfinite(R_km_med) else float(R_km_med),
        "q84": None if not np.isfinite(R_km_q84) else float(R_km_q84),
        "q975": None if not np.isfinite(R_km_q975) else float(R_km_q975),
    },
    "delta_KM_hmpc": {
        "q025": None if not np.isfinite(D_q025) else float(D_q025),
        "q16": None if not np.isfinite(D_q16) else float(D_q16),
        "median": None if not np.isfinite(D_med) else float(D_med),
        "q84": None if not np.isfinite(D_q84) else float(D_q84),
        "q975": None if not np.isfinite(D_q975) else float(D_q975),
    },
}

with open(
    PKDIR / "censored_graph_vs_pk_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print("Wrote:")
print(" ", PKDIR / "censored_graph_vs_pk_samples.csv")
print(" ", PKDIR / "km_graph_Rhom_curve.csv")
print(" ", PKDIR / "km_delta_graph_minus_pk_curve.csv")
print(" ", PKDIR / "censored_graph_vs_pk_summary.json")
