#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MuchoUchuu k=6 robustness diffusion batch
#
# Usage:
#   ./run_probe_batch_k06.sh BATCH SEED
#
# Examples:
#   ./run_probe_batch_k06.sh 00 12345
#   ./run_probe_batch_k06.sh 01 41001
# ============================================================

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 BATCH SEED"
    exit 1
fi

BATCH="$1"
SEED="$2"

ROOT="/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion"

PYTHON="${PYTHON:-python}"

XYZ="$ROOT/muchouchuu_xyz_f32.bin"

GRAPH="$ROOT/k_robustness/k06/knn_periodic.npz"

LOCALROOT="$ROOT/k_robustness/k06/batch_${BATCH}"

STATEOUT="$LOCALROOT/state"
TRACEOUT="$LOCALROOT/trace"

FINALTRACE="$ROOT/k_robustness/k06/final_batches/batch_${BATCH}_trace"


TIMES=(
    512
    640
    768
    896
    1024
    1280
    1536
    1792
    2048
    2560
    3072
    3584
    4096
    5120
    6144
    7168
    8192
    10240
    12288
    14336
    16384
    20480
    24576
    32768
)


# ============================================================
# Threading
# ============================================================

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"


mkdir -p \
    "$STATEOUT" \
    "$TRACEOUT" \
    "$FINALTRACE"


echo "============================================================"
echo "MuchoUchuu k=6 diffusion robustness batch"
echo "============================================================"
echo "Host:        $(hostname)"
echo "Batch:       $BATCH"
echo "Seed:        $SEED"
echo "Python:      $PYTHON"
echo "XYZ:         $XYZ"
echo "Graph:       $GRAPH"
echo "Local root:  $LOCALROOT"
echo "Final trace: $FINALTRACE"
echo "Started:     $(date --iso-8601=seconds)"
echo "============================================================"


if [[ -f "$FINALTRACE/BATCH_VALIDATED.txt" ]]; then
    echo "ERROR: batch already validated:"
    echo "$FINALTRACE/BATCH_VALIDATED.txt"
    exit 2
fi


# ============================================================
# STEP 1
# Generate the eight matched probes and propagate to t=512.
#
# This reproduces the original production initialization.
# ============================================================

echo
echo "STEP 1: generate probes to t=512"


/usr/bin/time -v \
"$PYTHON" -u "$ROOT/run_muchouchuu_diffusion_metric_recovery_v2.py" \
  "$XYZ" \
  "$GRAPH" \
  --box-size 6000 \
  --sketch-dim 8 \
  --seed "$SEED" \
  --times 512 \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --pair-distance-bins 12 \
  --pairs-per-bin 20 \
  --curve-distance-min 50 \
  --curve-distance-max 600 \
  --fit-windows 75 150 150 300 300 600 \
  --fit-bins 8 \
  --output-dir "$STATEOUT" \
  2>&1 | tee "$LOCALROOT/generate_to_t512.log"


# ============================================================
# Validate checkpoint
# ============================================================

test -f "$STATEOUT/state_t0000512.npy"

"$PYTHON" - "$STATEOUT/state_t0000512.npy" <<'PY'
import sys
import numpy as np

path = sys.argv[1]

a = np.load(path, mmap_mode="r")

print("checkpoint:", path)
print("checkpoint shape:", a.shape)
print("checkpoint dtype:", a.dtype)

if a.shape != (108000000, 8):
    raise ValueError(
        f"Unexpected checkpoint shape: {a.shape}"
    )

if a.dtype != np.float32:
    print(
        "WARNING: checkpoint dtype is",
        a.dtype,
        "rather than float32"
    )

print("Checkpoint validation passed.")
PY


# ============================================================
# STEP 2
# Continue the same eight probes over the complete time grid.
# ============================================================

echo
echo "STEP 2: extend return trace"


/usr/bin/time -v \
"$PYTHON" -u "$ROOT/extend_diffusion_return_trace.py" \
  "$GRAPH" \
  108000000 \
  --state-dir "$STATEOUT" \
  --output-dir "$TRACEOUT" \
  --times "${TIMES[@]}" \
  --sketch-dim 8 \
  --seed "$SEED" \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --io-chunk-rows 500000 \
  --save-per-probe \
  2>&1 | tee "$LOCALROOT/run_return_trace.log"


# ============================================================
# STEP 3
# Validate all 24 time files and all eight probes.
# ============================================================

echo
echo "STEP 3: validate outputs"


"$PYTHON" - "$TRACEOUT" "$SEED" <<'PY'
import json
import re
import sys
from pathlib import Path

import pandas as pd


root = Path(sys.argv[1])
seed = int(sys.argv[2])


expected_times = [
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
    5120,
    6144,
    7168,
    8192,
    10240,
    12288,
    14336,
    16384,
    20480,
    24576,
    32768,
]


metadata_path = root / "return_trace_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(metadata_path)


metadata = json.loads(
    metadata_path.read_text()
)


if int(metadata["seed"]) != seed:
    raise ValueError(
        f"Seed mismatch: "
        f"{metadata['seed']} != {seed}"
    )


files = sorted(
    root.glob("trace_probe_estimates_t*.csv")
)


found_times = []


for path in files:

    match = re.search(
        r"[tT](\d+)",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Could not parse time from {path.name}"
        )

    t = int(match.group(1))

    found_times.append(t)

    df = pd.read_csv(path)

    if len(df) != 8:
        raise ValueError(
            f"{path.name}: "
            f"expected 8 probe rows, "
            f"found {len(df)}"
        )


if found_times != expected_times:
    raise ValueError(
        "Time-grid mismatch.\n"
        f"Expected: {expected_times}\n"
        f"Found:    {found_times}"
    )


print(
    "Validated:",
    len(files),
    "times x 8 probes",
)

print(
    "Seed:",
    seed,
)

print(
    "First time:",
    found_times[0],
)

print(
    "Last time:",
    found_times[-1],
)
PY


# ============================================================
# STEP 4
# Copy only compact permanent outputs.
# ============================================================

echo
echo "STEP 4: copy compact outputs"


rsync -av \
  "$TRACEOUT"/trace_probe_estimates_t*.csv \
  "$FINALTRACE/"


rsync -av \
  "$TRACEOUT/return_probability_heat_trace.csv" \
  "$TRACEOUT/return_trace_metadata.json" \
  "$FINALTRACE/"


cp \
  "$LOCALROOT/generate_to_t512.log" \
  "$FINALTRACE/"


cp \
  "$LOCALROOT/run_return_trace.log" \
  "$FINALTRACE/"


cat > "$FINALTRACE/BATCH_VALIDATED.txt" <<MARKER
graph_k=6
batch=$BATCH
seed=$SEED
host=$(hostname)
time_files=24
probes_per_time=8
first_time=512
last_time=32768
graph=$GRAPH
completed=$(date --iso-8601=seconds)
MARKER


echo
echo "============================================================"
echo "Completed successfully"
echo "Permanent output:"
echo "$FINALTRACE"
echo "============================================================"
