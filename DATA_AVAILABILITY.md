# Data Availability

All data required to reproduce the data-derived manuscript figures and
statistical outputs are included in this repository.

Raw inputs are in `data/raw/` and include keypoint-MoSeq syllable usage tables,
per-frame MoSeq syllable assignments, supervised freezing predictions,
behavioral group metadata, behavioral flexibility scores, validation support
tables, and compressed MoSeq result objects used by the downstream dynamics
analyses.

The original supervised freezing predictions are provided as 98 per-animal CSVs
in `data/raw/freezing_predictions/`. The repository also includes
`data/raw/freezing_predictions_index.csv` and
`data/raw/freezing_predictions_light.csv.gz`, which are regenerated from the
per-animal CSVs by `scripts/derive_tables/freezing_predictions_light.py`.

Generated analysis-ready tables are in `data/processed/`. Plotted values for
manuscript figures are in `figure_source_data/`. Statistical model outputs are
in `statistics/`. Manuscript-facing Excel workbooks are:

- `report/raw_data.xlsx`
- `report/statistical_report.xlsx`

For public release, archive the repository on a persistent platform such as
Zenodo, Figshare, Dryad, or OSF and cite the archive DOI in the manuscript.
