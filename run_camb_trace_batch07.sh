#!/bin/bash
set -euo pipefail

XYZ="gaussian_camb_z0_seed3001/camb_displaced_N108000000_xyz_f32.bin"
GRAPH="gaussian_camb_z0_seed3001/camb_graphs/knn_periodic.npz"
GMETA="gaussian_camb_z0_seed3001/camb_graphs/parameters.json"

STATE="gaussian_camb_z0_seed3001/camb_trace_batches/batch_07_state"
TRACE="gaussian_camb_z0_seed3001/camb_trace_batches/batch_07_trace"

N=108000000
SEED=41007

echo "===== STAGE 1: generate t=512 state ====="

python run_muchouchuu_diffusion_metric_recovery_v2.py \
  "$XYZ" \
  "$GRAPH" \
  --n "$N" \
  --box-size 6000 \
  --output-dir "$STATE" \
  --times 512 \
  --sketch-dim 8 \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --io-chunk-rows 500000 \
  --pair-distance-bins 12 \
  --pairs-per-bin 20 \
  --curve-distance-min 50 \
  --curve-distance-max 600 \
  --max-pair-rounds 3000 \
  --fit-windows 75 150 150 300 300 600 \
  --fit-bins 8 \
  --seed "$SEED" \
  --graph-metadata "$GMETA"

echo "===== STAGE 2: heat trace t=512..16384 ====="

python extend_diffusion_return_trace.py \
  "$GRAPH" \
  "$N" \
  --state-dir "$STATE" \
  --output-dir "$TRACE" \
  --times \
    512 640 768 896 1024 \
    1280 1536 1792 2048 2560 \
    3072 3584 4096 5120 6144 \
    7168 8192 10240 12288 14336 16384 \
  --sketch-dim 8 \
  --seed "$SEED" \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --io-chunk-rows 500000 \
  --save-per-probe

echo "===== BATCH 07 COMPLETE ====="
