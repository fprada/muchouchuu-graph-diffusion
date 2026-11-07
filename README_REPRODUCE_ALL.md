# BigUchuu graph-diffusion analysis: complete reproduction workflow

The authoritative control is documented below. Sections 0--12 preserve the
LEGACY 32-probe w=4,5,6 workflow. Those historical commands assume:

- box side: `L=4000 h^-1 Mpc`
- selected halos: `N=32,000,000`
- density: `5e-4 h^3 Mpc^-3`
- graph: periodic kNN with `k=12`
- random-walk operator: `P=D^-1 A`
- sketch dimension: `32`
- random seed: `12345`

Run from `/work/fprada/DIFFUSION` after copying all scripts there.

## BigUchuu finite-volume transport control

- Status: AUTHORITATIVE, promoted using user-supplied reproducibility evidence.
- Result directory: `biguchuu_step6_s64_finite_volume_tmax32768`.
- Probe construction: `biguchuu_knn_return_trace_t32768` supplies global probes 0--31; `biguchuu_knn_return_trace_t32768_extra32_seed52001` supplies global probes 32--63.
- driver script: `run_biguchuu_s64_finite_volume_sequence.py`; SHA256 `78def4b6645aa624776d09348dd4dffbf86dd4df4d6d211468e08b5470ad2aec`.
- step6 script: `upgrade_step6_sliding_bootstrap_interpolated_plateau.py`; SHA256 `3dc3c6c86b2a7663d1d299b3cb9e5a0529d4880d2172195882dff59814ffef3c`.
- Step-6 parameters: number_of_probes = 64; window_points = 7; bootstrap_replicates = 10000; seed = 12345; target_dimension = 3.0; tolerance = 0.03; persistence = 3; minimum_local_r2 = 0.98; late_time_plateau_tmin = 3584.
- First central persistent time: `3860.3735406999035`; bootstrap crossing success fraction: `0.8737`.
- Bootstrap crossing times: q16 = 2548.1287683604837; median = 3706.1364626319655; q84 = 6026.508082081586; q025 = 1826.782302537908; q975 = 9026.55274630243.
- Completed Fourier calibration: `biguchuu_physical_diffusion_length/physical_diffusion_length.csv`; SHA256 `d5922d1915cfcb1c0993cd17997698e2e223a4349245885e840ba3f14c9070dc`.
- Fourier script: `measure_diffusion_length_fourier.py`; SHA256 `21185509b2df3d5549964f0c538cc421163acc12e76fe18a01463c4cfd10fec3`.
- Directly calibrated physical scales (approximate, h^-1 Mpc): median 630.20, q16 522.68, q84 804.06, central persistent crossing 643.16.
- No directly calibrated q975 physical radius: q975 time 9026.55274630243 exceeds the completed Fourier endpoint t = 8192.
- `biguchuu_step6_window7_t32768_s64_centered_full` remains EXPERIMENTAL corroborating evidence, not the authoritative generator.
- Historical w=4,5,6 workflow: LEGACY. Central crossings (h^-1 Mpc) and success fractions: w4 592.7199717345741 / 0.8653; w5 619.7159219781792 / 0.8296; w6 591.5177423304661 / 0.8810.
- Historical scripts, inputs and output directories remain preserved in README_REPRODUCE_ALL.md sections 10--11 and RESULTS_REGISTRY.yaml.
- Change: historical approximate window-averaged 600 h^-1 Mpc summary replaced by the reproducible 64-probe seven-point control: central crossing approximately 643.16 and bootstrap median approximately 630.20 h^-1 Mpc (distinct estimators). Correct the former README claim of crossing fractions near 95% to the preserved legacy fractions above. No calculation was rerun.
- Documentation affected: PROJECT_LEDGER.md, RESULTS_REGISTRY.yaml, CODE_MANIFEST.tsv, README_REPRODUCE_ALL.md. No manuscript section, figure or table changed; MANUSCRIPT_MAP.md does not explicitly register BigUchuu as a control. MuchoUchuu and redshift numerical results are unchanged.


The supplied driver identifies the reproducible 64-probe workflow. Do not
substitute the historical commands below for the authoritative generator.

## 0. Environment

```bash
conda activate biguchuu_mkl
pip install -r requirements.txt

export MKL_NUM_THREADS=120
export OMP_NUM_THREADS=120
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

## 1. Select the top-Vpeak full-box halo sample

Skip this step when `biguchuu_xyz_f32.bin` already exists.

```bash
python -u select_biguchuu_vpeak_fullbox.py \
  BigUchuu_hlist_050_2.h5 \
  BigUchuu_hlist_050_4.h5 \
  --vpeak-data-id 2 \
  --xyz-data-id 4 \
  --box-size 4000 \
  --number-density 5e-4 \
  --graph-sample 0 \
  --output biguchuu_xyz_f32.bin \
  --temp-dir biguchuu_vpeak_tmp \
  2>&1 | tee select_biguchuu_vpeak_fullbox.log
```

Expected output size: `32,000,000 x 3 x 4 = 384,000,000` bytes.

## 2. Build periodic graphs

Skip this step when `biguchuu_graphs_block_parallel/knn_periodic.npz` exists.
Choose `--blocks-per-dim` and `--processes` for the machine. The values below
are a starting point, not universal requirements.

```bash
python -u build_periodic_graphs_block_parallel_updated.py \
  biguchuu_xyz_f32.bin 32000000 \
  --box-size 4000 \
  --blocks-per-dim 20 \
  --processes 120 \
  --k 12 \
  --target-degree 15 \
  --output-dir biguchuu_graphs_block_parallel \
  --work-dir biguchuu_graph_work \
  2>&1 | tee build_periodic_graphs.log
```

Primary graph used below:

```text
biguchuu_graphs_block_parallel/knn_periodic.npz
```

## 3. Validate coordinates and graph

```bash
python validate_inputs.py \
  biguchuu_xyz_f32.bin 32000000 \
  biguchuu_graphs_block_parallel/knn_periodic.npz \
  | tee validate_inputs.log
```

Save the SHA-256 values in the paper's reproducibility record.

## 4. Main diffusion sketch and Euclidean metric recovery

This revised runner preserves every checkpoint. It never opens a saved
checkpoint writable and never uses it as an alternating work buffer.

```bash
python -u run_fullbox_diffusion_safe_mkl.py \
  biguchuu_xyz_f32.bin 32000000 \
  biguchuu_graphs_block_parallel/knn_periodic.npz \
  --box-size 4000 \
  --output-dir biguchuu_knn_diffusion_safe \
  --times 16 32 64 128 256 512 1024 2048 4096 \
  --sketch-dim 32 \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --pair-distance-bins 70 \
  --pairs-per-bin 5000 \
  --curve-distance-min 50 \
  --curve-distance-max 1200 \
  --fit-windows 75 150 150 300 300 600 600 1200 \
  --fit-bins 16 \
  --seed 12345 \
  2>&1 | tee biguchuu_knn_diffusion_safe.log
```

To resume safely:

```bash
python -u run_fullbox_diffusion_safe_mkl.py \
  biguchuu_xyz_f32.bin 32000000 \
  biguchuu_graphs_block_parallel/knn_periodic.npz \
  --box-size 4000 \
  --output-dir biguchuu_knn_diffusion_safe \
  --times 16 32 64 128 256 512 1024 2048 4096 \
  --sketch-dim 32 --backend mkl --checkpoint-every 256 \
  --curve-distance-min 50 --curve-distance-max 1200 \
  --fit-windows 75 150 150 300 300 600 600 1200 \
  --seed 12345 --resume
```

Key outputs:

```text
state_t0000000.npy ... state_t0004096.npy
euclidean_diagnostics.csv
diffusion_distance_vs_physical_separation.csv
sampled_pairs.npz
```

## 5. Heat trace and return probability to t=16384

The revised trace script starts from the latest checkpoint not later than the
earliest missing requested time. This correctly handles non-checkpoint times
such as 640 and 896.

```bash
python -u extend_diffusion_return_trace.py \
  biguchuu_graphs_block_parallel/knn_periodic.npz 32000000 \
  --state-dir biguchuu_knn_diffusion_safe \
  --output-dir biguchuu_knn_return_trace_t16384 \
  --times \
    512 640 768 896 1024 1280 1536 1792 2048 \
    2560 3072 3584 4096 5120 6144 7168 8192 \
    10240 12288 14336 16384 \
  --sketch-dim 32 \
  --seed 12345 \
  --backend mkl \
  --checkpoint-every 256 \
  --progress-every 16 \
  --save-per-probe \
  2>&1 | tee biguchuu_knn_return_trace_t16384.log
```

Key outputs:

```text
return_probability_heat_trace.csv
trace_probe_estimates_t*.csv
```

## 6. Finite-difference effective spectral dimension

```bash
python compute_effective_spectral_dimension.py \
  biguchuu_knn_return_trace_t16384/return_probability_heat_trace.csv \
  --output-csv \
  biguchuu_knn_return_trace_t16384/effective_spectral_dimension.csv \
  --tolerance 0.03
```

Definition:

```text
d_s(t) = -2 d ln P_return / d ln t
```

## 7. Probe-level spectral-dimension fits

```bash
python analyze_probe_level_spectral_dimension.py \
  biguchuu_knn_return_trace_t16384 \
  --output-dir biguchuu_knn_probe_spectral_dimension \
  --fit-ranges 512 4096 1024 4096 2048 4096
```

## 8. Fourier physical-length calibration

A single run covering all requested times is simplest and avoids merging.
Matplotlib is optional; CSV outputs are always produced.

```bash
python -u measure_diffusion_length_fourier.py \
  biguchuu_xyz_f32.bin 32000000 \
  biguchuu_graphs_block_parallel/knn_periodic.npz \
  --box-size 4000 \
  --times \
    512 640 768 896 1024 1280 1536 1792 2048 \
    3072 4096 5120 6144 7168 8192 \
    10240 12288 14336 16384 \
  --harmonics 1 2 3 4 \
  --min-correlation 0.01 \
  --backend mkl \
  --spectral-dimension-csv \
    biguchuu_knn_return_trace_t16384/effective_spectral_dimension.csv \
  --output-dir biguchuu_knn_physical_length_all \
  2>&1 | tee biguchuu_knn_physical_length_all.log
```

Key outputs:

```text
fourier_mode_correlations.csv
physical_diffusion_length.csv
spectral_dimension_vs_physical_scale.csv
```

Expected calibration near the previous run:

```text
ell(2048) = 468.665599 h^-1 Mpc
ell(4096) ≈ 662.48 h^-1 Mpc
ell(16384) ≈ 1326.24 h^-1 Mpc
```

## 9. Collect all per-probe trace files

The single trace output directory already contains all times, so no symlinks
are needed. If runs were split, create links without overwriting duplicates:

```bash
mkdir -p biguchuu_knn_all_probe_traces
for d in biguchuu_knn_return_trace_corrected biguchuu_knn_return_trace_t16384; do
  for f in "$d"/trace_probe_estimates_t*.csv; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    [ -e "biguchuu_knn_all_probe_traces/$name" ] || \
      ln -s "$(readlink -f "$f")" "biguchuu_knn_all_probe_traces/$name"
  done
done
```

For the fresh single-run workflow below use:

```text
biguchuu_knn_return_trace_t16384
```

as the probe input directory.

## 10. Sliding-window probe-level analysis

```bash
for W in 4 5 6; do
  python analyze_diffusion_homogeneity_scale.py \
    biguchuu_knn_return_trace_t16384 \
    biguchuu_knn_physical_length_all/physical_diffusion_length.csv \
    --output-dir biguchuu_knn_diffusion_homogeneity_w${W} \
    --window-size ${W} \
    --step-size 1 \
    --tolerance 0.03 \
    --persistence 2
done
```

This per-probe crossing distribution is diagnostic only. Because individual
probes can fail to cross, the legacy ensemble bootstrap below supplies historical errors.

## 11. Ensemble bootstrap homogeneity scale

> **Legacy-path note.** The historical commands below wrote to
> `biguchuu_knn_bootstrap_homogeneity_all_w${W}`, but those directories
> are not present in the surviving analysis tree. The preserved LEGACY
> products are `biguchuu_knn_bootstrap_homogeneity_w4`,
> `biguchuu_knn_bootstrap_homogeneity_w5`, and
> `biguchuu_knn_bootstrap_homogeneity_w6`.
>
> The historical command also references
> `biguchuu_knn_physical_length_all/physical_diffusion_length.csv`.
> That later calibration run is incomplete and that CSV is absent.
> The surviving completed Fourier calibration is
> `biguchuu_physical_diffusion_length/physical_diffusion_length.csv`.


```bash
for W in 4 5 6; do
  python bootstrap_diffusion_homogeneity_scale.py \
    biguchuu_knn_return_trace_t16384 \
    biguchuu_knn_physical_length_all/physical_diffusion_length.csv \
    --output-dir biguchuu_knn_bootstrap_homogeneity_all_w${W} \
    --window-size ${W} \
    --step-size 1 \
    --tolerance 0.03 \
    --persistence 2 \
    --n-bootstrap 10000 \
    --seed 12345
done
```

Previously obtained central crossings were approximately:

```text
w=4: 592.7199717345741 h^-1 Mpc; success = 0.8653
w=5: 619.7159219781792 h^-1 Mpc; success = 0.8296
w=6: 591.5177423304661 h^-1 Mpc; success = 0.8810
```

The preserved crossing fractions are 86.53%, 82.96%, and 88.10%. The
approximate window-averaged `600 h^-1 Mpc` summary is LEGACY. The
authoritative central crossing is approximately `643.16 h^-1 Mpc` and
bootstrap median approximately `630.20 h^-1 Mpc`.

## 12. Essential consistency checks

### Check checkpoint uniqueness

```bash
sha256sum biguchuu_knn_diffusion_safe/state_t*.npy \
  > biguchuu_knn_diffusion_safe/checkpoint_sha256.txt
```

Different diffusion times should not have identical hashes.

### Check monotonic return probability

```bash
python - <<'PY'
import pandas as pd
p='biguchuu_knn_return_trace_t16384/return_probability_heat_trace.csv'
d=pd.read_csv(p).sort_values('diffusion_time')
print(d[['diffusion_time','mean_return_probability','relative_trace_se']].to_string(index=False))
assert (d.mean_return_probability.diff().dropna() < 0).all()
PY
```

### Check normal diffusion

```bash
python - <<'PY'
import pandas as pd, numpy as np
p='biguchuu_knn_physical_length_all/physical_diffusion_length.csv'
d=pd.read_csv(p).sort_values('diffusion_time')
d['ell_over_sqrt_t']=d.physical_diffusion_length_mpc_h/np.sqrt(d.diffusion_time)
print(d[['diffusion_time','physical_diffusion_length_mpc_h','ell_over_sqrt_t','relative_log_fit_rmse']].to_string(index=False))
PY
```

## Scientific interpretation guardrails

- The graph was built from Euclidean comoving coordinates, so recovery of a
  3D Euclidean regime is a validation of graph transport and large-scale
  averaging, not an independent measurement of cosmological curvature.
- The diffusion homogeneity scale is not the standard counts-in-spheres scale.
- The late decline of `d_s` is finite-volume mixing. Interpret conservatively
  for `ell < L/4 = 1000 h^-1 Mpc`.
- The quoted 1% band is an adopted operational definition:
  `|d_s-3| <= 0.03`, persistent for three adjacent windows in the authoritative seven-point
  analysis (two in the historical LEGACY workflow).
