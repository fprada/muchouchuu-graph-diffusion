#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import re
from pathlib import Path

HELPERS = r'''

def interpolate_zero_logtime(
    t_left: float,
    t_right: float,
    margin_left: float,
    margin_right: float,
) -> float:
    values = (t_left, t_right, margin_left, margin_right)
    if not all(np.isfinite(v) for v in values):
        return np.nan
    if t_left <= 0.0 or t_right <= t_left:
        return np.nan
    if not (margin_left < 0.0 <= margin_right):
        return np.nan

    denominator = margin_right - margin_left
    if denominator <= 0.0:
        return np.nan

    fraction = float(np.clip(-margin_left / denominator, 0.0, 1.0))
    log_crossing = (
        np.log(t_left)
        + fraction * (np.log(t_right) - np.log(t_left))
    )
    return float(np.exp(log_crossing))


def first_persistent_time_interpolated(
    margin: np.ndarray,
    times: np.ndarray,
    auxiliary_ok: np.ndarray,
    persistence: int,
) -> float:
    margin = np.asarray(margin, dtype=float)
    times = np.asarray(times, dtype=float)
    auxiliary_ok = np.asarray(auxiliary_ok, dtype=bool)

    if not (len(margin) == len(times) == len(auxiliary_ok)):
        raise ValueError(
            "margin, times, and auxiliary_ok must have equal length"
        )

    persistence = int(persistence)
    if persistence < 1:
        raise ValueError("persistence must be at least 1")

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
            crossing = interpolate_zero_logtime(
                times[j - 1],
                times[j],
                margin[j - 1],
                margin[j],
            )
            if np.isfinite(crossing):
                return crossing

        return float(times[j])

    return np.nan
'''


def replace_once(text, pattern, replacement, label, flags=0):
    new, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            f"Patch anchor for {label!r} matched {count} times; expected 1."
        )
    return new


def main():
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

    text = replace_once(
        text,
        r"(def first_persistent_time\(.*?\n)(?=def [A-Za-z_]\w*\()",
        lambda m: m.group(1) + HELPERS + "\n",
        "helper insertion",
        flags=re.S,
    )

    bootstrap_pattern = r'''(?ms)
(?P<indent>^[ \t]*)\#\s*Crossing distribution: first persistent run inside the tolerance band\s*\n
(?P=indent)boot_cross\s*=\s*np\.full\([^\n]+\)\s*\n
(?P=indent)for b in range\(args\.bootstrap\):\s*\n
(?P=indent)[ \t]+mask\s*=\s*\(np\.abs\(boot_ds\[b\]\s*-\s*3\.0\)\s*<=\s*args\.tolerance\)\s*&\s*fit_quality_ok\s*\n
(?P=indent)[ \t]+boot_cross\[b\]\s*=\s*first_persistent_time\(\s*mask,\s*times,\s*args\.persistence\s*\)\s*
'''
    bootstrap_replacement = r'''\g<indent># Continuous crossing distribution with log-time interpolation.
\g<indent>boot_cross = np.full(args.bootstrap, np.nan, dtype=float)
\g<indent>for b in range(args.bootstrap):
\g<indent>    boot_margin = args.tolerance - np.abs(boot_ds[b] - 3.0)
\g<indent>    boot_cross[b] = first_persistent_time_interpolated(
\g<indent>        margin=boot_margin,
\g<indent>        times=times,
\g<indent>        auxiliary_ok=fit_quality_ok,
\g<indent>        persistence=args.persistence,
\g<indent>    )
'''
    text = replace_once(
        text,
        bootstrap_pattern,
        bootstrap_replacement,
        "bootstrap crossing loop",
    )

    central_pattern = r'''(?ms)
(?P<indent>^[ \t]*)first_central_persistent\s*=\s*first_persistent_time\(\s*
central_in_band\s*&\s*fit_quality_ok,\s*
times,\s*
args\.persistence,\s*
\)
'''
    central_replacement = r'''\g<indent>central_margin = args.tolerance - np.abs(central_ds - 3.0)
\g<indent>first_central_persistent = first_persistent_time_interpolated(
\g<indent>    margin=central_margin,
\g<indent>    times=times,
\g<indent>    auxiliary_ok=fit_quality_ok,
\g<indent>    persistence=args.persistence,
\g<indent>)
'''
    text = replace_once(
        text,
        central_pattern,
        central_replacement,
        "central crossing",
    )

    text = replace_once(
        text,
        r"else int\(first_central_persistent\)",
        r"else float(first_central_persistent)",
        "central float serialization",
    )

    save_pattern = (
        r"(?P<line>^[ \t]*valid_cross\s*=\s*"
        r"boot_cross\[np\.isfinite\(boot_cross\)\]\s*$)"
    )
    save_replacement = r'''\g<line>
    interpolation_outdir = Path(args.output_dir)
    interpolation_outdir.mkdir(parents=True, exist_ok=True)
    np.save(
        interpolation_outdir / "bootstrap_crossing_times_interpolated.npy",
        boot_cross,
    )
    pd.DataFrame({
        "bootstrap_index": np.arange(len(boot_cross)),
        "crossing_time_interpolated": boot_cross,
        "crossing_detected": np.isfinite(boot_cross),
    }).to_csv(
        interpolation_outdir / "bootstrap_crossing_times_interpolated.csv",
        index=False,
    )'''
    text = replace_once(
        text,
        save_pattern,
        save_replacement,
        "bootstrap sample saving",
        flags=re.M,
    )

    metadata_pattern = r'(?P<line>^[ \t]*"tolerance":\s*args\.tolerance,\s*$)'
    metadata_replacement = r'''\g<line>
        "crossing_time_interpolation": "log_time_linear_margin",
        "crossing_margin_definition": "tolerance - abs(d_s - 3)",
        "interpolation_applied_before_bootstrap_quantiles": True,'''
    text = replace_once(
        text,
        metadata_pattern,
        metadata_replacement,
        "interpolation metadata",
        flags=re.M,
    )

    args.output.write_text(text)
    py_compile.compile(str(args.output), doraise=True)
    print(f"Wrote: {args.output.resolve()}")
    print("Syntax check: OK")


if __name__ == "__main__":
    main()
