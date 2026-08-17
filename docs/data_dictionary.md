# Data Dictionary

This document summarizes the tracked inputs and regenerated products included
in the repository. Row counts and checksums are verified by
`scripts/check_reproducibility.py` and recorded in `MANIFEST.csv`.

## Raw Inputs

| Path | Description |
| --- | --- |
| `data/raw/animal_groups.csv` | Animal recording names and Control/ELS group labels used for validation panels. |
| `data/raw/behavioral_flexibility_scores.xlsx` | Behavioral flexibility score table used for resilience grouping. |
| `data/raw/freezing_overlap_by_group.csv` | Per-animal overlap between supervised freezing calls and MoSeq freezing syllables. |
| `data/raw/freezing_predictions/` | Original per-animal supervised freezing prediction CSVs. |
| `data/raw/freezing_predictions_index.csv` | File-level index for the raw freezing prediction CSVs. |
| `data/raw/freezing_predictions_light.csv.gz` | Long-format compact companion table with animal, group, frame, and freezing columns. |
| `data/raw/moseq_syllables_per_frame.csv.gz` | Per-frame MoSeq syllable table used by validation and classifier workflows. |
| `data/raw/simba_validation_manual_vs_automatic.csv` | Manual versus automatic SimBA freezing percentages used for the Figure 1 validation correlation. |
| `data/raw/syllable_classification_metrics.csv` | Precision/recall summary for syllable-based behavior classes. |
| `data/raw/syllable_usage_per_timebin_30s.csv` | MoSeq syllable usage at 30 s resolution. |
| `data/raw/syllable_usage_per_timebin_250ms.csv` | MoSeq syllable usage at 250 ms resolution. |
| `data/raw/updated_results.pkl.gz` | Compressed MoSeq result object used for dynamics and distance analyses. |

The legacy classifier figure PDF once kept at
`data/raw/legacy_figures/figure7_classifier_original.pdf` was lost before it
could be committed and is retired;
`figure_source_data/figure7_classifier_global_shap.csv` is the archival source
of record for its values (see `docs/figure_structure.md`).

## Processed Tables

| Path | Built by | Description |
| --- | --- | --- |
| `data/processed/cluster_frequency_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal total behavior-cluster frequency table. |
| `data/processed/cluster_timecourse_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal behavior-cluster time course. |
| `data/processed/s0s28_timecourse_per_animal.csv` | `scripts/derive_tables/cluster_tables.py` | Per-animal S0/S28 freezing syllable time course. |
| `data/processed/supplementary_figure1_tracking_clusters.csv` | `scripts/derive_tables/tracking_exclusions.py` | Tracking-control table for Supplementary Figure 1. |
| `data/processed/tracking_exclusions_per_animal.csv` | `scripts/derive_tables/tracking_exclusions.py` | Tracking exclusion summary by animal. |
| `data/processed/transition_metrics_per_animal.csv` | `scripts/generate_figures/figure_4_diversity_dynamics.py` | Per-animal transition metrics used for Figure 4 and Figure 6. |
| `data/processed/figure5_dynamics_scores.csv` | `scripts/generate_figures/figure_5_resilience_dynamics.py` | Per-animal dynamics scores and resilience grouping audit columns. |
| `data/processed/figure5_resilience_threshold_audit.csv` | `scripts/generate_figures/figure_5_resilience_dynamics.py` | Threshold audit for Figure 5 resilience grouping. |
| `data/processed/supplementary_figure1_time_summary.csv` | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` | Time-resolved tracking-control summary. |
| `data/processed/supplementary_figure3_distance_scores.csv` | `scripts/generate_figures/supplementary_figure_3_distances.py` | Per-animal scores for distance-metric controls. |
| `data/processed/supplementary_figure3_distance_summary.csv` | `scripts/generate_figures/supplementary_figure_3_distances.py` | Summary table for distance-metric controls. |
| `data/processed/supplementary_figure3_threshold_audit.csv` | `scripts/generate_figures/supplementary_figure_3_distances.py` | Threshold audit for distance-metric controls. |

## Figure Source Data

`figure_source_data/figure2.csv` through `figure_source_data/figure7.csv`
contain the plotted values and summary values exported by the figure scripts.
Detailed Figure 7 prediction, permutation, SHAP, and time-course tables use the
`figure7_*.csv` prefix in the same directory.
`figure7_classifier_global_shap.csv` supports the stacked global summary in
Figure 7B. `supplementary_figure4_shap_summary.csv` contains all 719 legacy
classifier parameters for each of the eight classifier outputs (5,752 rows);
Supplementary Figure 4 plots the top 10 parameters per output, flagged by
`plotted_top_10`. Each row carries the raw legacy `parameter` name and a
readable `parameter_label`.
These files are generated products for figure transparency; raw analysis inputs
remain in `data/raw/`.

## Statistical Outputs And Reports

| Path | Description |
| --- | --- |
| `statistics/` | Machine-readable statistical model outputs and formal contrast audit tables. |
| `statistics/manuscript_consistency_audit.csv` | Comparison of selected manuscript Results claims against the regenerated statistical CSVs; rebuilt by `scripts/audit_manuscript_results.py`. |
| `figures/` | Canonical manuscript figure exports in PDF, SVG, and PNG formats. |
| `report/raw_data.xlsx` | Manuscript-facing raw-data workbook. |
| `report/statistical_report.xlsx` | Manuscript-facing statistical report workbook. |
| `classifier/figure7_behavior_classifier.joblib` | Reusable classifier artifact for Figure 7A-C (33 centroid-kinematic features). |
| `classifier/legacy_shap/xgb_model.pkl` | Legacy classifier behind Supplementary Figure 4 (719 body-part features); with `label_encoder.pkl`. |

The canonical panel-to-file mapping is documented in
`docs/figure_structure.md`.
