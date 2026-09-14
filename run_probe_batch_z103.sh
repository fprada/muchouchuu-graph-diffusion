#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 BATCH_NUMBER SEED"
    exit 1
fi

BATCH_NUMBER="$1"
SEED="$2"

ROOT="/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion"
PYTHON="/home/users/dae/fprada/.conda/envs/biguchuu_mkl/bin/python"

GRAPH="$ROOT/growth_test/z103/muchouchuu_graphs_block_parallel/knn_periodic.npz"
XYZ="$ROOT/growth_test/z103/muchouchuu_xyz_f32_periodic.bin"
FINALROOT="$ROOT/growth_test/z103/trace_batches"

LOCALDISK="${LOCALDISK:?Set LOCALDISK to the node-local filesystem}"

BATCH=$(printf "batch_%02d" "$((10#$BATCH_NUMBER))")
LOCALROOT="$LOCALDISK/fprada_muchouchuu_z103_${BATCH}"
STATEOUT="$LOCALROOT/${BATCH}_state"
TRACEOUT="$LOCALROOT/${BATCH}_trace"
FINALTRACE="$FINALROOT/${BATCH}_trace"

TIMES=(
  512 640 768 896 1024 1280 1536 1792
  2048 2560 3072 3584 4096 5120 6144
  7168 8192 10240 12288 14336 16384
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"

mkdir -p "$STATEOUT" "$TRACEOUT" "$FINALTRACE"

echo "Host: $(hostname)"
echo "Batch: $BATCH"
echo "Seed: $SEED"
echo "Local root: $LOCALROOT"
echo "Final trace: $FINALTRACE"
echo "Started: $(date --iso-8601=seconds)"

if [[ -f "$FINALTRACE/BATCH_VALIDATED.txt" ]]; then
    echo "ERROR: batch already validated"
    exit 2
fi

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

test -f "$STATEOUT/state_t0000512.npy"

"$PYTHON" - "$STATEOUT/state_t0000512.npy" <<'PY'
import sys
import numpy as np

a = np.load(sys.argv[1], mmap_mode="r")
print("checkpoint shape:", a.shape)
print("checkpoint dtype:", a.dtype)

if a.shape != (108000000, 8):
    raise ValueError(f"Unexpected shape: {a.shape}")
PY

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
    512, 640, 768, 896, 1024, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096, 5120, 6144,
    7168, 8192, 10240, 12288, 14336, 16384,
]

metadata = json.loads(
    (root / "return_trace_metadata.json").read_text()
)

if metadata["seed"] != seed:
    raise ValueError("Seed mismatch")

files = sorted(root.glob("trace_probe_estimates_t*.csv"))
found_times = []

for path in files:
    match = re.search(r"[tT](\d+)", path.name)
    found_times.append(int(match.group(1)))

    df = pd.read_csv(path)
    if len(df) != 8:
        raise ValueError(
            f"{path.name}: expected 8 rows, found {len(df)}"
        )

if found_times != expected_times:
    raise ValueError(f"Time-grid mismatch: {found_times}")

print("Validated:", len(files), "times x 8 probes")
PY

echo "STEP 4: copy compact outputs"

rsync -av \
  "$TRACEOUT"/trace_probe_estimates_t*.csv \
  "$FINALTRACE/"

rsync -av \
  "$TRACEOUT/return_probability_heat_trace.csv" \
  "$TRACEOUT/return_trace_metadata.json" \
  "$FINALTRACE/"

cp "$LOCALROOT/generate_to_t512.log" "$FINALTRACE/"
cp "$LOCALROOT/run_return_trace.log" "$FINALTRACE/"

cat > "$FINALTRACE/BATCH_VALIDATED.txt" <<MARKER
batch=$BATCH
seed=$SEED
host=$(hostname)
time_files=21
probes_per_time=8
completed=$(date --iso-8601=seconds)
MARKER

echo "Completed successfully"
echo "Permanent output: $FINALTRACE"
