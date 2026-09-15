#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

HELPERS = '''

def interpolate_zero_logtime(
    t_left: float,
    t_right: float,
    margin_left: float,
    margin_right: float,
) -> float:
    """Interpolate a negative-to-nonnegative margin crossing in log(time)."""
    values = (t_left, t_right, margin_left, margin_right)
    if not all(np.isfinite(v) for v in values):
        return np.nan
    if t_left <= 0.0 or t_right <= t_left:
        return np.nan
    if not (margin_left < 0.0 <= margin_right):
        return np.nan

    denom = margin_right - margin_left
    if denom <= 0.0:
        return np.nan

    frac = float(np.clip(-margin_left / denom, 0.0, 1.0))
    return float(
        np.exp(
            np.log(t_left)
            + frac * (np.log(t_right) - np.log(t_left))
        )
    )


def first_persistent_time_interpolated(
    margin: np.ndarray,
    times: np.ndarray,
    auxiliary_ok: np.ndarray,
    persistence: int,
) -> float:
    """Find first persistent pass and interpolate its leading boundary."""
    margin = np.asarray(margin, dtype=float)
    times = np.asarray(times, dtype=float)
    auxiliary_ok = np.asarray(auxiliary_ok, dtype=bool)

    if not (len(margin) == len(times) == len(auxiliary_ok)):
        raise ValueError(
            "margin, times, and auxiliary_ok must have equal length"
        )

    persistence = int(persistence)
    passes = (margin >= 0.0) & auxiliary_ok

    for j in range(len(times)):
        stop = j + persistence
        if stop > len(times):
            break
        if not np.all(passes[j:stop]):
            continue

        if j == 0:
            return float(times[j])

        if (
            auxiliary_ok[j - 1]
            and auxiliary_ok[j]
            and margin[j - 1] < 0.0 <= margin[j]
        ):
            t_cross = interpolate_zero_logtime(
                times[j - 1],
                times[j],
                margin[j - 1],
                margin[j],
            )
            if np.isfinite(t_cross):
                return t_cross

        return float(times[j])

    return math.nan
'''


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch anchor for {label!r} matched {count} times; expected 1."
        )
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("upgrade_step6_sliding_bootstrap_interpolated.py"),
    )
    args = ap.parse_args()

    text = args.source.read_text()
    if "first_persistent_time_interpolated" in text:
        raise RuntimeError("Source already appears patched.")

    text = replace_exact(
        text,
        "def main() -> None:\n",
        HELPERS + "\n\ndef main() -> None:\n",
        "helper insertion",
    )

    old_boot = '''    # Crossing distribution: first persistent run inside the tolerance band
    # with adequate fit quality for each bootstrap realization.
    boot_cross = np.full(args.bootstrap, np.nan, dtype=float)
    for b in range(args.bootstrap):
        mask = (np.abs(boot_ds[b] - 3.0) <= args.tolerance) & fit_quality_ok
        boot_cross[b] = first_persistent_time(mask, times, args.persistence)

    valid_cross = boot_cross[np.isfinite(boot_cross)]
'''
    new_boot = '''    # Crossing distribution with continuous log-time interpolation.
    # The bootstrap crossing retains the original hard fit-quality gate.
    boot_cross = np.full(args.bootstrap, np.nan, dtype=float)
    for b in range(args.bootstrap):
        boot_margin = args.tolerance - np.abs(boot_ds[b] - 3.0)
        boot_cross[b] = first_persistent_time_interpolated(
            margin=boot_margin,
            times=times,
            auxiliary_ok=fit_quality_ok,
            persistence=args.persistence,
        )

    valid_cross = boot_cross[np.isfinite(boot_cross)]

    np.save(
        outdir / "bootstrap_crossing_times_interpolated.npy",
        boot_cross,
    )
    pd.DataFrame({
        "bootstrap_index": np.arange(args.bootstrap),
        "crossing_time_interpolated": boot_cross,
        "crossing_detected": np.isfinite(boot_cross),
    }).to_csv(
        outdir / "bootstrap_crossing_times_interpolated.csv",
        index=False,
    )
'''
    text = replace_exact(text, old_boot, new_boot, "bootstrap crossing block")

    old_central = '''    first_central_persistent = first_persistent_time(
        qualifies, times, args.persistence
    )
'''
    new_central = '''    # Preserve the original central hard gates: CI95 contains 3 and R^2 passes.
    central_margin = args.tolerance - np.abs(central_ds - 3.0)
    central_auxiliary_ok = ci95_contains_3 & fit_quality_ok
    first_central_persistent = first_persistent_time_interpolated(
        margin=central_margin,
        times=times,
        auxiliary_ok=central_auxiliary_ok,
        persistence=args.persistence,
    )
'''
    text = replace_exact(text, old_central, new_central, "central crossing block")

    old_meta = '        "tolerance": args.tolerance,\n        "persistence": args.persistence,\n'
    new_meta = (
        '        "tolerance": args.tolerance,\n'
        '        "crossing_time_interpolation": "log_time_linear_margin",\n'
        '        "crossing_margin_definition": "tolerance - abs(d_s - 3)",\n'
        '        "interpolation_applied_before_bootstrap_quantiles": True,\n'
        '        "persistence": args.persistence,\n'
    )
    text = replace_exact(text, old_meta, new_meta, "metadata block")

    text = replace_exact(
        text,
        "            else int(first_central_persistent)\n",
        "            else float(first_central_persistent)\n",
        "central float serialization",
    )

    args.output.write_text(text)
    py_compile.compile(str(args.output), doraise=True)
    print(f"Wrote: {args.output.resolve()}")
    print("Syntax check: OK")


if __name__ == "__main__":
    main()
