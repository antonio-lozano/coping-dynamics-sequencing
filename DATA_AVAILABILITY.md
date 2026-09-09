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

The upstream inputs that this repository does not ship are deposited on Figshare
as a companion dataset: the 98 raw behavioral videos, the DeepLabCut pose
estimates (raw and median-filtered) and network snapshot, the keypoint-MoSeq
model configuration and outputs (`results.h5`, `moseq_df.csv`, PCA), the
per-frame supervised freezing predictions, and the representative syllable clips
behind Supplementary Video 1.

Sanguino Gómez J, Güçlü U, Lozano A, Krugers HJ. Coping strategies dynamics and
resilience profiles after early-life stress revealed by behavioral sequencing.
figshare. Dataset. https://doi.org/10.6084/m9.figshare.33439885

The Figshare record is held privately by the journal until the manuscript is
accepted, so the DOI resolves only after publication. The dataset is released
under CC BY 4.0.
