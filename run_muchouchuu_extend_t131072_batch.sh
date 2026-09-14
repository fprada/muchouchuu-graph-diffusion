#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 BATCH SEED STATE_DIR"
    echo "Example: $0 01 41001 muchouchuu_extended_trace/batch_01"
    exit 1
fi

BATCH="$1"
SEED="$2"
STATE_DIR="$3"

BASE="/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion"
cd "$BASE"

GRAPH="$BASE/muchouchuu_graphs/knn_periodic.npz"
N=108000000

OUTROOT="$BASE/muchouchuu_extended_trace_t131072"
OUTDIR="$OUTROOT/batch_${BATCH}"

TIMES="36864 40960 45056 49152 53248 57344 61440 65536 73728 81920 90112 98304 114688 131072"

echo "============================================================"
echo "MuchoUchuu late extension"
echo "batch       = $BATCH"
echo "seed        = $SEED"
echo "state dir   = $STATE_DIR"
echo "output dir  = $OUTDIR"
echo "graph       = $GRAPH"
echo "N           = $N"
echo "times       = $TIMES"
echo "============================================================"

if [ ! -f "$GRAPH" ]; then
    echo "ERROR: graph not found: $GRAPH"
    exit 2
fi

if [ ! -f "$STATE_DIR/state_t0032768.npy" ]; then
    echo "ERROR: missing restart state:"
    echo "$STATE_DIR/state_t0032768.npy"
    exit 3
fi

mkdir -p "$OUTDIR"

python -u "$BASE/extend_diffusion_return_trace.py" \
    "$GRAPH" "$N" \
    --state-dir "$STATE_DIR" \
    --output-dir "$OUTDIR" \
    --times $TIMES \
    --sketch-dim 8 \
    --seed "$SEED" \
    --backend mkl \
    --checkpoint-every 4096 \
    --progress-every 64 \
    --save-per-probe

echo
echo "============================================================"
echo "VALIDATING BATCH $BATCH"
echo "============================================================"

NFILES=$(find "$OUTDIR" -maxdepth 1 \
    -name 'trace_probe_estimates_t*.csv' | wc -l)

echo "Trace time files = $NFILES"

if [ "$NFILES" -ne 14 ]; then
    echo "WARNING: expected 14 trace files, found $NFILES"
    exit 4
fi

BAD=0
for f in "$OUTDIR"/trace_probe_estimates_t*.csv; do
    NPROBES=$(($(wc -l < "$f") - 1))
    if [ "$NPROBES" -ne 8 ]; then
        echo "BAD: $f has $NPROBES probes"
        BAD=1
    fi
done

if [ "$BAD" -ne 0 ]; then
    echo "ERROR: probe-count validation failed"
    exit 5
fi

echo
echo "Saved times:"
ls "$OUTDIR"/trace_probe_estimates_t*.csv \
    | sed -E 's/.*_t0*([0-9]+)\.csv/\1/' \
    | sort -n

echo
echo "Completed successfully:"
echo "$OUTDIR"
echo "============================================================"
