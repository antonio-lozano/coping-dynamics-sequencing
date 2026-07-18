# Data Availability

All data required to reproduce the data-derived manuscript figures and
statistical outputs are included in this repository.

Raw inputs are under `data/raw/` and include keypoint-MoSeq syllable usage
tables, per-frame MoSeq syllable assignments, supervised freezing predictions,
behavioral group metadata, BFL scores, validation support tables, and compressed
MoSeq result objects used by downstream dynamics analyses.

The freezing predictions are provided as original per-animal CSVs in
`data/raw/freezing_predictions/`, together with:

- `data/raw/freezing_predictions_index.csv`
- `data/raw/freezing_predictions_light.csv.gz`

Generated analysis-ready tables are under `data/derived/`. Figure source-data
tables, statistical outputs, helper tables, and model artifacts are under
`results/`. Manuscript-facing workbooks are:

- `report/raw_data.xlsx`
- `report/statistical_report.xlsx`

For public release, archive the repository on a persistent platform such as
Zenodo, Figshare, Dryad, or OSF and cite the archive DOI in the manuscript.
