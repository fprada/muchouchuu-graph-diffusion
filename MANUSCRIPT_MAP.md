# Manuscript provenance map

## Abstract
- Fiducial transport scale:
  PROJECT_LEDGER.md -> z=0 transport
- Fiducial density scale:
  PROJECT_LEDGER.md -> z=0 density
- Redshift evolution:
  homogeneity_redshift_evolution_bootstrap_summary.json

## Figure 3

- Status: AUTHORITATIVE
- Preferred manuscript output:
  `figure3_muchouchuu_centered_updated_v5.pdf`
- Preferred manuscript PNG:
  `figure3_muchouchuu_centered_updated_v5.png`
- Authoritative reproducible regeneration script:
  `make_figure3_muchouchuu_centered_updated_v4.py`
- Historical rendering provenance:
  `make_figure3_muchouchuu_centered_updated_v3.py` is the likely
  manuscript-era renderer based on filesystem timestamps, but the exact
  historical generating command was not recovered.
- Inputs:
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/spectral_dimension_sliding_bootstrap.csv`
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/crossing_summary.json`
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/late_time_plateau_summary.json`
- Calibration:
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`
- Fiducial transport-scale summary:
  `R_hom = 446.108 h^-1 Mpc`
  with 68% interval `[418.555, 486.916] h^-1 Mpc`
- Previous registered output:
  `figure3_muchouchuu_centered_authoritative.pdf`
- Change:
  manuscript rendering only; scientific values are unchanged.

## Redshift-evolution figure
- Script:
  plot_homogeneity_redshift_evolution_bootstrap.py
- Statistics:
  analyze_homogeneity_redshift_evolution.py
- Input:
  homogeneity_redshift_evolution_bootstrap_summary.csv
- Output:
  figure_homogeneity_redshift_evolution_bootstrap.pdf

## Sect. 4.3
- Transport differences:
  paired probe bootstrap
- Density differences:
  independent bootstrap distributions
- z=2.03 density bootstrap:
  10000 realizations

## Appendix F: redshift-dependent spectral-dimension trajectories

- Status: AUTHORITATIVE
- Script:
  `plot_z0_z049_z103_z203_ds_physical_pub.py`
- Output:
  `figure_z0_z049_z103_z203_ds_physical_scale_pub_v2.pdf`
- Redshifts:
  z = 0, 0.49, 1.03, 2.03
- Role:
  Full transport-trajectory diagnostic supporting Sect. 4.3
- Main statistical redshift figure remains:
  `figure_homogeneity_redshift_evolution_bootstrap.pdf`


## Appendix A1: real-space diffusion containment scaling

- Status: AUTHORITATIVE
- Script:
  `make_appendix_figureA1_muchouchuu_diffusion_length_v3.py`
- Input:
  `muchouchuu_step7_length_calibration_extended_3000/muchouchuu_diffusion_length_calibration.csv`
- Quantity:
  50% containment scale `ell_f0.500_mpc_h`
- Fit:
  `ell_0.5(t) = A_0.5 sqrt(t)`
- Fitted coefficient:
  `A_0.5 = 6.449187635001151 h^-1 Mpc step^-1/2`
- Fractional RMS residual:
  `3.06%`
- Output:
  `appendix_figureA1_muchouchuu_diffusion_length_v3.pdf`
- Scientific role:
  Independent real-space diagnostic of diffusive sqrt(t) scaling.
  It is distinct from the Fourier RMS calibration used to convert
  diffusion time to the manuscript physical transport scale.

## BigUchuu finite-volume control

- Status: AUTHORITATIVE
- Scientific role:
  finite-volume diagnostic only; BigUchuu does not define a second
  fiducial cosmological homogeneity measurement.

### Sect. 2.2 and Sect. 3.5

- BigUchuu box:
  `L = 4000 h^-1 Mpc`
- Graph vertices:
  `N = 32,000,000`
- Gaussian probes:
  `64`
- Graph:
  periodic symmetrised `k = 12` nearest-neighbour graph.
- Role:
  smaller-volume comparison used to test the finite-volume
  interpretation of the late-time trajectory.

### Appendix B.1: matched-range BigUchuu test

- Status: AUTHORITATIVE
- Product:
  `biguchuu_diffusion_reproduction_v2/biguchuu_step6_s64_finite_volume_tmax7168`
- Step-6 settings:
  64 probes, seven-point windows, 10000 probe-bootstrap realisations,
  seed 12345, tolerance 0.03, persistence 3, minimum local R2 = 0.98,
  plateau tmin = 3584.
- Crossing-success fraction:
  `0.5547`, quoted in the manuscript as `0.555`.
- Crossing-time quantiles:
  q16 = `2215.8173371082135`,
  median = `3152.4442966820093`,
  q84 = `3780.5566250620045`.
- Manuscript matched-range diagnostic:
  `R_hom,B = 581.1^{+55.3}_{-93.9} h^-1 Mpc`.
- Interpretation:
  this is a matched-range finite-volume diagnostic and is distinct
  from the full `tmax=32768` BigUchuu transport-control result.

### Sect. 6 and Appendix B.2: late-time finite-volume downturn

- Status: AUTHORITATIVE
- BigUchuu central Euclidean-band departure:
  `t = 8192`.
- Completed Fourier calibration:
  `R(8192) = 937.8870646045921 h^-1 Mpc`.
- Manuscript rounded value:
  `937 h^-1 Mpc`.
- Scientific role:
  comparison with the later MuchoUchuu downturn to test the
  periodic finite-volume interpretation.
- Important:
  this downturn scale is not the BigUchuu transport-homogeneity
  crossing scale.

### Conclusion item 6

- Uses the BigUchuu late-time downturn value
  `937 h^-1 Mpc`, compared with the MuchoUchuu downturn value
  `1884 h^-1 Mpc`.
- No manuscript numerical change is required from the
  BigUchuu provenance reconstruction.

