# Euclidean graph diffusion project ledger

## Rule

This file records the authoritative analysis products used in the
current manuscript.

Every product should be marked as one of:

- AUTHORITATIVE
- EXPERIMENTAL
- SUPERSEDED

Do not replace an authoritative product silently. When a result
changes, record the old value, new value, reason for change, and the
files/manuscript locations affected.

## Current manuscript

- Title: Euclidean graph diffusion as a probe of large-scale homogeneity in the cosmic web
- Status: current working draft
- Manuscript date: 2026-09-14

## Authoritative z=0 Figure 3 product

- Script:
  `make_figure3_muchouchuu_centered_updated_v4.py`
- Input directory:
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/`
- Output:
  `figure3_muchouchuu_centered_authoritative.pdf`
- Calibration:
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`
- Regenerated fiducial values:
  - first central persistent time: `2560`
  - bootstrap median crossing time: `1836.6072440054406`
  - bootstrap median transport homogeneity scale:
    `446.10841874518684 h^-1 Mpc`
  - bootstrap 68% interval:
    `[418.5554077950529, 486.91625778750154] h^-1 Mpc`
  - late-time plateau median:
    `3.009757842732382`

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

