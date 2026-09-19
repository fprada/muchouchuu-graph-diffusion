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
- Script:
  `make_figure3_muchouchuu_centered_updated_v4.py`
- Inputs:
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/spectral_dimension_sliding_bootstrap.csv`
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/crossing_summary.json`
  `muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/late_time_plateau_summary.json`
- Calibration:
  `A_RMS = 10.4095576 h^-1 Mpc step^-1/2`
- Output:
  `figure3_muchouchuu_centered_authoritative.pdf`
- Fiducial transport-scale summary:
  `R_hom = 446.108 h^-1 Mpc`
  with 68% interval `[418.555, 486.916] h^-1 Mpc`

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
