# MuchoUchuu graph-diffusion manuscript reproducibility guide

This repository contains the curated code, small authoritative result
products, publication figures, and provenance records for

"Euclidean graph diffusion as a probe of large-scale homogeneity in the cosmic web."

## Authority

The authoritative status of scripts and products is defined by:

- PROJECT_LEDGER.md
- RESULTS_REGISTRY.yaml
- CODE_MANIFEST.tsv
- MANUSCRIPT_MAP.md

Do not infer authority from filenames, modification times, or version numbers.

## Repository scope

Git contains:

- authoritative and provenance-relevant analysis code;
- compact authoritative CSV/JSON/NPY result products;
- publication figures;
- the CAMB linear P(k) input table;
- provenance and manuscript mapping files.

Large MuchoUchuu catalogues, graph objects, raw probe batches, large
bootstrap/all-pair arrays, and other generated HPC products are not stored
in Git. See EXTERNAL_DATA_MANIFEST.tsv.

## Fiducial transport result

Authoritative Step-6 product:

    muchouchuu_step6_window7_s64_plateau_tmin3584_strictcentered_full/

Analysis:

    step6_muchouchuu_extended_centered.py

Fourier calibration:

    A_RMS = 10.4095576 h^-1 Mpc step^-1/2

Fiducial result:

    R_hom = 446.108 h^-1 Mpc
    68% interval = [418.555, 486.916] h^-1 Mpc

Publication Figure 3:

    make_figure3_muchouchuu_centered_updated_v4.py
    figure3_muchouchuu_centered_authoritative.pdf

## Density homogeneity

Authoritative density inference is reconstructed with:

    reconstruct_density_bootstrap_and_ratio.py

Manuscript inference uses the interpolated products:

    density_crossing_radii_interpolated.npy

Raw/discrete crossings are not the manuscript inference product.

Fiducial z=0 result:

    R_hom,density = 114.575 h^-1 Mpc
    68% interval = [107.717, 120.397] h^-1 Mpc

## Redshift evolution

Transport redshift comparisons use paired probe-bootstrap realizations only
where identical probe ordering/resampling has been established.

Density redshift comparisons use independent bootstrap distributions unless
physical centre matching is explicitly demonstrated.

Analysis:

    analyze_homogeneity_redshift_evolution.py

Figure:

    plot_homogeneity_redshift_evolution_bootstrap.py
    figure_homogeneity_redshift_evolution_bootstrap.pdf

Summary:

    homogeneity_redshift_evolution_bootstrap_summary.json
    homogeneity_redshift_evolution_bootstrap_summary.csv

## Canonical z=2.03 probe merge

Script:

    merge_z203_64probes.py

Seeds:

    12345, 41001, 41002, 41003, 41004, 41005, 41006, 41007

Global probe indices:

    0--63

This ordering supports the paired transport redshift comparison.

Gaussian-control canonical merges:

    merge_grf_64probes.py
    merge_camb_64probes.py

## Appendix A1

Script:

    make_appendix_figureA1_muchouchuu_diffusion_length_v3.py

Input:

    muchouchuu_step7_length_calibration_extended_3000/muchouchuu_diffusion_length_calibration.csv

This is the 50-percent real-space containment diagnostic

    ell_0.5(t) = A_0.5 sqrt(t)

and is distinct from the Fourier RMS calibration.

## CAMB input spectrum

Tracked input:

    PkTable.dat

SHA256:

    c55007cb7d4bd1345b587741647846e37218c5a744137b62655b168b121bb3c2

## External data

See EXTERNAL_DATA_MANIFEST.tsv for large simulation, graph, trace, bootstrap,
and control products intentionally retained outside Git.
