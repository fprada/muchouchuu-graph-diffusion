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

export LOCALDISK="${LOCALDISK:-$ROOT/local_scratch}"

BATCH=$(printf "batch_%02d" "$((10#$BATCH_NUMBER))")

GRAPH="$ROOT/growth_test/z103/muchouchuu_graphs_block_parallel/knn_periodic.npz"

LOCALROOT="$LOCALDISK/fprada_muchouchuu_z103_${BATCH}"
STATEOUT="$LOCALROOT/${BATCH}_state"
TRACEOUT="$LOCALROOT/${BATCH}_trace"

TIMES=(
  512 640 768 896 1024 1280 1536 1792
  2048 2560 3072 3584 4096 5120 6144
  7168 8192 10240 12288 14336 16384
)

CHECKPOINT="$STATEOUT/state_t0000512.npy"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: missing checkpoint $CHECKPOINT"
    exit 2
fi

echo "Resuming $BATCH seed=$SEED"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $TRACEOUT"

mkdir -p "$TRACEOUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"

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
