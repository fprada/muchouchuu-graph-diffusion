#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/fprada/DIFFUSION/ANALYSIS/muchouchuu_diffusion
CAT=/work/ishiyama/MuchoUchuu6G/00030/hlist_0.67084.list.h5
OUTDIR=$ROOT/growth_test/z049
SCRIPT=$ROOT/select_muchouchuu_vpeak_v4.py

mkdir -p "$OUTDIR"

echo "============================================================"
echo "Starting MuchoUchuu Vpeak selection"
date
echo "Catalogue : $CAT"
echo "Output dir: $OUTDIR"
echo "============================================================"

python -u "$SCRIPT" \
  "$CAT" \
  --box-size 6000 \
  --number-density 5e-4 \
  --graph-sample 0 \
  --output "$OUTDIR/muchouchuu_xyz_f32.bin" \
  --seed 12345

echo
echo "============================================================"
echo "Selection finished"
date
echo "============================================================"

stat -c '%s %n' "$OUTDIR/muchouchuu_xyz_f32.bin"

echo
echo "Metadata:"
cat "$OUTDIR/muchouchuu_xyz_f32.metadata.json"
