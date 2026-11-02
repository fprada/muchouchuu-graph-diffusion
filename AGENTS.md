# Euclidean graph diffusion project instructions

This repository supports the paper:

"Euclidean graph diffusion as a probe of large-scale homogeneity
in the cosmic web."

## Scientific provenance

Before stating which result, script, parameter set, figure, or product
is current, read in this order:

1. PROJECT_LEDGER.md
2. RESULTS_REGISTRY.yaml
3. CODE_MANIFEST.tsv
4. MANUSCRIPT_MAP.md when the task concerns the manuscript

Git and these provenance files are authoritative.

Never infer that a newer file or higher version number is authoritative.

Statuses are:

- AUTHORITATIVE
- EXPERIMENTAL
- SUPERSEDED
- LEGACY

Only AUTHORITATIVE products may be used for manuscript inference.

## Scientific rules

- Transport redshift comparisons may be paired only when identical
  probe resampling is established.
- Density redshift comparisons are independent unless physical centre
  matching is explicitly demonstrated.
- All-pair bootstrap combinations are not independent measurements.
- Reconstructed interpolated density products supersede raw Step-9
  crossing products for manuscript inference.

## Code provenance

- Preserve scientifically relevant old versions.
- Never silently replace authoritative code.
- Do not promote an EXPERIMENTAL result to AUTHORITATIVE without
  explicit provenance evidence.
- Recommend a Git commit whenever analysis code or authoritative
  results change.
- Recommend a Git tag for manuscript milestones.

When a numerical result changes, report:

- old value
- new value
- reason for change
- generating script
- input products
- output product
- manuscript sections, figures, or tables affected

## HPC rule

Never claim that an HPC command or calculation was executed unless
terminal output confirming the execution is available.

Do not launch very large MuchoUchuu or BigUchuu calculations unless
explicitly requested.

## End-of-session provenance report

For substantial analysis work, report proposed changes to:

1. PROJECT_LEDGER.md
2. RESULTS_REGISTRY.yaml
3. CODE_MANIFEST.tsv
4. MANUSCRIPT_MAP.md
5. files promoted to AUTHORITATIVE
6. files marked SUPERSEDED or LEGACY
7. manuscript consequences
8. exact git add and git commit commands
