# Euclidean graph diffusion project ledger

## Rule

This file records the authoritative analysis products used in the
current manuscript.

Every product should be marked as one of:

- AUTHORITATIVE
- EXPERIMENTAL
- SUPERSEDED
- LEGACY

Do not replace an authoritative product silently. When a result
changes, record the old value, new value, reason for change, and the
files/manuscript locations affected.

## Current manuscript

- Title: Euclidean graph diffusion as a probe of large-scale homogeneity in the cosmic web
- Status: current working draft
- Manuscript date: 2026-09-14

## Authoritative z=0 Figure 3 product

- Status: AUTHORITATIVE
- Preferred manuscript output:
  `figure3_muchouchuu_centered_updated_v5.pdf`
- Preferred manuscript PNG:
  `figure3_muchouchuu_centered_updated_v5.png`
- Manuscript artifact timestamp:
  `2026-08-25 14:55 +0200`
- Manuscript PDF SHA256:
  `a5a5ae43ec04b34c616542ef02fc9b11aaaff082c58aba5f0f96815ae0032175`
- Authoritative reproducible regeneration script:
  `make_figure3_muchouchuu_centered_updated_v4.py`
- Historical rendering provenance:
  `make_figure3_muchouchuu_centered_updated_v3.py` was modified at
  `2026-08-25 14:54:56 +0200`, approximately 9 seconds before the
  manuscript v5 PDF/PNG were written. This is strong evidence that the
  v5 rendering was produced with v3 or a nearly identical plotting
  state, but the exact historical command was not recovered and is
  therefore not asserted as definitive.
- Input directory:
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/`
- Inputs:
  - `spectral_dimension_sliding_bootstrap.csv`
  - `crossing_summary.json`
  - `late_time_plateau_summary.json`
- Calibration:
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`
- Scientific values:
  - first central persistent time: `2560`
  - bootstrap median crossing time: `1836.6072440054406`
  - bootstrap median transport homogeneity scale:
    `446.10841874518684 h^-1 Mpc`
  - bootstrap 68% interval:
    `[418.5554077950529, 486.91625778750154] h^-1 Mpc`
  - late-time plateau median:
    `3.009757842732382`
- Provenance note:
  Present-day regenerations with both
  `make_figure3_muchouchuu_centered_updated_v3.py` and
  `make_figure3_muchouchuu_centered_updated_v4.py` reproduce the same
  scientific values from the unchanged Step-6 inputs but are not
  pixel-identical to the 2026-08-25 v5 rendering. No change in the
  scientific result is indicated; the exact cause of the rendering
  difference is not recovered.
- Provenance-integrity correction:
  The CODE_MANIFEST.tsv entry `ouchuu_centered_updated_v3.py`, introduced
  in Git commit `5ade22f`, was malformed: no corresponding file was found
  locally or in the inspected Git history. The erroneous row was removed.
  The full v3 script remains LEGACY, and the full v4 script remains the
  AUTHORITATIVE reproducible regeneration script, as registered in
  commit `f054452`. This correction does not recover the exact historical
  v5 rendering command or change any scientific result.
- Previous manuscript-output registration:
  `figure3_muchouchuu_centered_authoritative.pdf`
- Previous PDF SHA256:
  `e0a9a32539197d124889e10d15c71646f06bb6c9eb3d46518683a918002cc5fc`
- Reason for change:
  the manuscript uses the preserved v5 publication rendering; numerical
  inference and authoritative Step-6 inputs are unchanged.
- Manuscript affected:
  Figure 3 only; no numerical manuscript text changes.

## Fiducial transport, z=0

- Status: AUTHORITATIVE
- Step6 directory:
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full`
- Gaussian probes: 64
- Bootstrap realisations: 10000
- Bootstrap seed: 12345
- Bootstrap mode: probe resampling
- Fourier calibration:
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`
- Transport homogeneity scale:
  - median: 446.108 h^-1 Mpc
  - q16: 418.555 h^-1 Mpc
  - q84: 486.916 h^-1 Mpc
- Late spectral dimension:
  - median: 3.00976 approximately
- Late logarithmic slope:
  - median: 0.01162 approximately

## Fiducial density homogeneity, z=0

- Status: AUTHORITATIVE
- Final reconstruction directory:
  `muchouchuu_density_transport_ratio_bootstrap_1pct_final`
- Bootstrap realisations: 10000
- Density homogeneity scale:
  - median: 114.575 h^-1 Mpc
  - q16: 107.717 h^-1 Mpc
  - q84: 120.397 h^-1 Mpc
- Transport-to-density ratio:
  - median: 3.913
  - q16: 3.598
  - q84: 4.369

## Transport homogeneity by redshift

### z=0.49

- Status: AUTHORITATIVE
- Step6 directory:
  `growth_test/z049/step6_window7_s64_strictcentered_tmin3584_fixed`
- A_RMS: 10.438294184707798
- R_hom median: 449.784 h^-1 Mpc
- q16: 426.772
- q84: 481.480

### z=1.03

- Status: AUTHORITATIVE
- Step6 directory:
  `growth_test/z103/step6_window7_s64_strictcentered_tmin3584_fixed`
- A_RMS: 10.4894608905163
- R_hom median: 452.176 h^-1 Mpc
- q16: 424.028
- q84: 500.799

### z=2.03

- Status: AUTHORITATIVE
- Step6 directory:
  `growth_test/z203/step6_window7_s64_strictcentered_tmin3584_fixed`
- A_RMS: 10.598993479525339
- R_hom median: 453.702 h^-1 Mpc
- q16: 428.439
- q84: 488.538

## Density homogeneity by redshift

### z=0.49

- Status: AUTHORITATIVE
- Reconstruction directory:
  `growth_test/z049/muchouchuu_density_transport_ratio_bootstrap_1pct`
- R_hom median: 112.136 h^-1 Mpc
- q16: 105.163
- q84: 118.726

### z=1.03

- Status: AUTHORITATIVE
- Reconstruction directory:
  `growth_test/z103/muchouchuu_density_transport_ratio_bootstrap_1pct`
- R_hom median: 114.087 h^-1 Mpc
- q16: 106.910
- q84: 125.003

### z=2.03

- Status: AUTHORITATIVE
- Reconstruction directory:
  `growth_test/z203/muchouchuu_density_transport_ratio_bootstrap_1pct_10000`
- Bootstrap realisations: 10000
- R_hom median: 98.989 h^-1 Mpc
- q16: 94.012
- q84: 106.073
- q025: 89.897
- q975: 112.572

## Redshift evolution

- Status: AUTHORITATIVE
- Analysis script:
  `analyze_homogeneity_redshift_evolution.py`
- Output JSON:
  `homogeneity_redshift_evolution_bootstrap_summary.json`
- Output CSV:
  `homogeneity_redshift_evolution_bootstrap_summary.csv`

### Transport relative to z=0

Transport uses paired probe-bootstrap realisations.

- z=0.49:
  - median delta R: +3.705 h^-1 Mpc
  - 68%: [-21.641, 25.959]
  - 95%: [-82.978, 61.988]
  - P(delta R > 0): 0.56735

- z=1.03:
  - median delta R: +5.924 h^-1 Mpc
  - 68%: [-16.518, 29.142]
  - 95%: [-60.772, 80.661]
  - P(delta R > 0): 0.61745

- z=2.03:
  - median delta R: +5.975 h^-1 Mpc
  - 68%: [-20.411, 29.334]
  - 95%: [-80.184, 68.519]
  - P(delta R > 0): 0.60310

### Density relative to z=0

Density uses independent bootstrap distributions.

- z=0.49:
  - median delta R: -2.320 h^-1 Mpc
  - 68%: [-11.528, 7.060]
  - 95%: [-20.562, 16.127]
  - P(delta R < 0): 0.59806

- z=1.03:
  - median delta R: +0.196 h^-1 Mpc
  - 68%: [-9.508, 12.020]
  - 95%: [-17.697, 23.703]
  - P(delta R < 0): 0.49272

- z=2.03:
  - median delta R: -14.769 h^-1 Mpc
  - 68%: [-22.924, -5.627]
  - 95%: [-29.634, 3.629]
  - P(delta R < 0): 0.94352

## Current redshift-evolution figure

- Status: AUTHORITATIVE
- Script:
  `plot_homogeneity_redshift_evolution_bootstrap.py`
- PDF:
  `figure_homogeneity_redshift_evolution_bootstrap.pdf`
- PNG:
  `figure_homogeneity_redshift_evolution_bootstrap.png`

## Important methodological rules

- Transport redshift comparisons are paired only because the same
  64 probe ordering and bootstrap seed are used.
- Only rows finite at both redshifts are retained in paired transport
  differences.
- Density redshift comparisons are independent unless physical
  centre matching is explicitly established.
- Do not describe all pairwise density combinations as independent
  measurements.
- Do not use raw Step9 density crossings when the reconstructed,
  interpolated bootstrap product exists.
- The z=1.03 graph and downstream analysis use the corrected periodic
  coordinate binary.

## Halo-selection code provenance

- `select_muchouchuu_vpeak_v4.py`
  - Status: AUTHORITATIVE
  - Selector used for the current MuchoUchuu manuscript analysis.

- `select_muchouchuu_vpeak.py`
  - Status: SUPERSEDED
  - Retained for provenance and comparison.
  - Not used for the current manuscript analysis.

## Appendix A1: real-space diffusion containment scaling

- Status: AUTHORITATIVE
- Script:
  `make_appendix_figureA1_muchouchuu_diffusion_length_v3.py`
- Input:
  `muchouchuu_step7_length_calibration_extended_3000/muchouchuu_diffusion_length_calibration.csv`
- Measured quantity:
  `ell_f0.500_mpc_h`
- Interpretation:
  50% real-space containment scale; this is not the RMS diffusion length.
- Fit:
  `ell_0.5(t) = A_0.5 sqrt(t)`
- Fitted coefficient:
  `A_0.5 = 6.449187635001151 h^-1 Mpc step^-1/2`
- Fit interval:
  `t = 512--16384`
- Number of fitted points:
  `6`
- Fractional RMS residual:
  `0.03056717051228973`
- Maximum absolute fractional residual:
  `0.05245949843202924`
- Output:
  `appendix_figureA1_muchouchuu_diffusion_length_v3.pdf`
- Important:
  This containment-scale diagnostic does not replace or modify the
  AUTHORITATIVE Fourier RMS calibration
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`.

## Appendix F: redshift-dependent spectral-dimension trajectories

- Status: AUTHORITATIVE
- Detailed provenance:
  MANUSCRIPT_MAP.md -> Appendix F: redshift-dependent spectral-dimension trajectories
- Script registration:
  CODE_MANIFEST.tsv -> plot_z0_z049_z103_z203_ds_physical_pub.py
- Registry entry:
  RESULTS_REGISTRY.yaml -> appendix_F_trajectories

## z=2.03 canonical probe merge

- `merge_z203_64probes.py`
  - Status: AUTHORITATIVE
  - Input:
    `growth_test/z203/trace_batches/`
  - Output:
    `growth_test/z203/canonical_64x21/`
  - Merges eight validated 8-probe batches into the canonical
    64-probe global ordering.
  - Seeds:
    `12345, 41001, 41002, 41003, 41004, 41005, 41006, 41007`
  - Global probe indices:
    `0--63`
  - Principal summary:
    `growth_test/z203/canonical_64x21/z203_return_probability_heat_trace_s64.csv`
  - The AUTHORITATIVE z=2.03 Step-6 product
    `growth_test/z203/step6_window7_s64_strictcentered_tmin3584_fixed/`
    records this canonical summary and the canonical per-probe glob in
    `crossing_summary.json`.
  - This establishes the probe ordering used for the paired transport
    redshift comparison.

## Gaussian-test canonical probe merges

### Ph(k)-matched Gaussian realization

- `merge_grf_64probes.py`
  - Status: AUTHORITATIVE
  - Merges eight 8-probe batches into global probe indices `0--63`.
  - Output:
    `gaussian_pk_control_seed1001/grf_trace_merged_s64/`
  - Principal summary:
    `gaussian_pk_control_seed1001/grf_trace_merged_s64/return_probability_heat_trace_extended_s64.csv`
  - Downstream Step-6 product:
    `gaussian_pk_control_seed1001/grf_step6_window7_s64_strictcentered_tmin3584/`
  - Its `crossing_summary.json` records the merged summary as the input.

### CAMB Gaussian matter realization

- `merge_camb_64probes.py`
  - Status: AUTHORITATIVE
  - Merges eight 8-probe batches into global probe indices `0--63`.
  - Output:
    `gaussian_camb_z0_seed3001/camb_trace_merged_s64/`
  - Principal summary:
    `gaussian_camb_z0_seed3001/camb_trace_merged_s64/return_probability_heat_trace_extended_s64.csv`
  - Downstream Step-6 product:
    `gaussian_camb_z0_seed3001/camb_step6_window7_s64_strictcentered_tmin3584/`
  - Its `crossing_summary.json` records the merged summary as the input.

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
- Historical README command paths used `biguchuu_knn_bootstrap_homogeneity_all_w{w}`, but those directories are absent.
- Surviving LEGACY bootstrap products are `biguchuu_knn_bootstrap_homogeneity_w4`, `biguchuu_knn_bootstrap_homogeneity_w5`, and `biguchuu_knn_bootstrap_homogeneity_w6`.
- The historical README calibration path `biguchuu_knn_physical_length_all/physical_diffusion_length.csv` is absent because that calibration run is incomplete.
- The surviving completed calibration is `biguchuu_physical_diffusion_length/physical_diffusion_length.csv`.
- Change: historical approximate window-averaged 600 h^-1 Mpc summary replaced by the reproducible 64-probe seven-point control: central crossing approximately 643.16 and bootstrap median approximately 630.20 h^-1 Mpc (distinct estimators). Correct the former README claim of crossing fractions near 95% to the preserved legacy fractions above. No calculation was rerun.
- Documentation affected: PROJECT_LEDGER.md, RESULTS_REGISTRY.yaml, CODE_MANIFEST.tsv, README_REPRODUCE_ALL.md, and MANUSCRIPT_MAP.md. MANUSCRIPT_MAP.md now explicitly registers the BigUchuu finite-volume uses in Sects. 2.2, 3.5, and 6, Appendix B, and conclusion item 6. This provenance-map correction does not change any manuscript number, figure, table, or scientific conclusion. MuchoUchuu and redshift numerical results are unchanged.

