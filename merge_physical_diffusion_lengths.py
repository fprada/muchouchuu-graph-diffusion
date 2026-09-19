#!/usr/bin/env python3
"""
Merge early- and late-time physical diffusion-length tables.

Rows are deduplicated by diffusion_time, keeping the last occurrence.
The merged table is sorted by diffusion_time and written to CSV.
"""

import argparse
from pathlib import Path

import pandas as pd


def main(args):
    frames = []

    for path in args.inputs:
        frame = pd.read_csv(path)

        required = {
            "diffusion_time",
            "physical_diffusion_length_mpc_h",
        }
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                "{} is missing columns {}".format(path, sorted(missing))
            )

        frame["_source_file"] = str(path)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.sort_values("diffusion_time")
        .drop_duplicates("diffusion_time", keep="last")
        .sort_values("diffusion_time")
        .reset_index(drop=True)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    print(
        merged[
            [
                "diffusion_time",
                "physical_diffusion_length_mpc_h",
                "_source_file",
            ]
        ].to_string(index=False)
    )
    print("\nWrote:", output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
