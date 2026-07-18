# Data Dictionary

This file summarizes the tracked inputs and generated artifacts included in the
repository. Row counts can be verified with `scripts/check_reproducibility.py`
and checksums are recorded in `MANIFEST.csv`.

## Raw Inputs

| Path | Description |
| --- | --- |
| `data/raw/animal_groups.csv` | Animal recording names and Control/ELS group labels used for validation panels. |
| `data/raw/bfl_scores.xlsx` | Behavioral flexibility score table used for resilience grouping. |
| `data/raw/freezing_overlap_by_group.csv` | Per-animal overlap between supervised freezing calls and MoSeq freezing syllables. |
| `data/raw/freezing_predictions/` | Original per-animal supervised freezing prediction CSVs. |
| `data/raw/freezing_predictions_index.csv` | File-level index for the raw freezing prediction CSVs. |
| `data/raw/freezing_predictions_light.csv.gz` | Long-format compact companion table with animal, group, frame, and freezing columns. |
| `data/raw/moseq_syllables_per_frame.csv.gz` | Per-frame MoSeq syllable table used by validation and classifier workflows. |
| `data/raw/syllable_classification_metrics.csv` | Precision/recall summary for syllable-based behavior classes. |
| `data/raw/syllable_usage_per_timebin_30s.csv` | MoSeq syllable usage at 30 s resolution. |
| `data/raw/syllable_usage_per_timebin_250ms.csv` | MoSeq syllable usage at 250 ms resolution. |
| `data/raw/updated_results.pkl.gz` | Compressed MoSeq result object used for dynamics and distance analyses. |

## Derived Tables

| Path | Built by | Description |
| --- | --- | --- |
| `data/derived/cluster_frequency_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal total behavior-cluster frequency table. |
| `data/derived/cluster_timecourse_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal behavior-cluster time course. |
| `data/derived/s0s28_timecourse_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal S0/S28 freezing syllable time course. |
| `data/derived/supplementary_figure1_tracking_clusters.csv` | `scripts/derive_tables/tracking_exclusions.py` | Tracking-control table for Supplementary Figure 1. |
| `data/derived/tracking_exclusions_per_animal.csv` | `scripts/derive_tables/tracking_exclusions.py` | Tracking exclusion summary by animal. |

## Results And Reports

| Path | Description |
| --- | --- |
| `results/source_data/source_data_figure2.csv` through `source_data_figure6.csv` | Figure source-data CSVs exported by the figure scripts. |
| `results/statistics/` | Statistical model tables and manuscript Results text audit. |
| `results/intermediate/tables/` | Helper tables used for figure and supplementary analyses. |
| `results/intermediate/figures/` | Non-canonical figure renders retained for audit. |
| `results/models/behavior_classifier/fig7_behavior_xgb.joblib` | Tracked classifier model artifact associated with Figure 7. |
| `report/raw_data.xlsx` | Manuscript-facing raw-data workbook. |
| `report/statistical_report.xlsx` | Manuscript-facing statistical report workbook. |
