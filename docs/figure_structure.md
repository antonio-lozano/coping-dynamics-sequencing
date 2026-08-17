# Canonical Figure Structure

This is the repository-wide panel map used by the figure generators, source-data
exports, Excel reports, and reproducibility checks.

## Main Figures

| Figure | Panels | Content | Primary source data |
| --- | --- | --- | --- |
| Figure 1 | A | SimBA manual-versus-automatic validation | `data/raw/simba_validation_manual_vs_automatic.csv` |
| Figure 2 | A-H | Ground-truth freezing and MoSeq validation | `data/raw/` and `data/processed/s0s28_timecourse_per_animal.csv` |
| Figure 3 | A-H | Behavior frequency and time courses | `data/processed/cluster_*` |
| Figure 4 | A-W | Diversity, usage, bouts, and transitions | `figure_source_data/figure4.csv` and per-animal processed tables |
| Figure 5 | A-J | Behavioral dynamics score and resilient ELS profiles | `figure_source_data/figure5.csv` and `data/processed/figure5_*` |
| Figure 6 | A-Z | Resilience-group diversity, bouts, and transitions | `figure_source_data/figure6.csv` and `statistics/fig6_*` |
| Figure 7 | A-C | Behavior-classifier accuracy, global SHAP, and confusion matrix | `figure_source_data/figure7_classifier_*` |
| Figure 7 | D-F | Cross-cohort/LOOCV resilience prediction and Shapley contributions | `figure_source_data/figure7_full_session_auc.csv` and `figure7_shapley_*` |
| Figure 7 | G-H | Predictor catalog performance | `figure_source_data/figure7_predictor_catalog.csv` |
| Figure 7 | I | Prediction onset and permutation reference | `figure_source_data/figure7_prediction_onset.csv` |
| Figure 7 | J | Predictor-family performance over time | `figure_source_data/figure7_predictor_timecourse.csv` |
| Figure 7 | K-N | Individual metric performance over time | `figure_source_data/figure7_individual_timecourse_auc.csv` |

## Supplementary Figures

| Figure | Panels | Content | Primary source data |
| --- | --- | --- | --- |
| Supplementary Figure 1 | A-D | Omitted/mixed behavior dynamics and total frequencies | `data/processed/supplementary_figure1_*` |
| Supplementary Figure 2 | assembled | Behavioral-dynamics workflow | Canonical tracked export |
| Supplementary Figure 3 | A-K | Alternative distance metrics and resilient-group overlap | `data/processed/supplementary_figure3_*` |
| Supplementary Figure 4 | A-H (published as Figure 7 panels D-M) | Class-specific behavior-classifier SHAP panels | `figure_source_data/supplementary_figure4_shap_summary.csv`; original at `data/raw/legacy_figures/figure7_classifier_original.pdf` |

### Supplementary Figure 4 comes from the legacy classifier

The published class-specific SHAP panels come from the **legacy** classifier,
not from the bundled one. Its feature space is pairwise DeepLabCut body-part
distances and velocities over a 14-marker skeleton (`nose`, `H1R`/`H2R`/`H1L`/
`H2L`, `B1R`/`B2R`/`B3R`/`B1L`/`B2L`/`B3L`, `tail`, `S1`/`S2`), expanded with
lags and 5-frame rolling mean/SD/sum into **719 parameters**.

`figure_source_data/supplementary_figure4_shap_summary.csv` holds all 719
parameters for each of the eight classes (5,752 rows). Each panel plots the top
10 for its class, flagged by `plotted_top_10`. Alongside the raw legacy
`parameter` name each row carries a readable `parameter_label`
(`dist_B2R_H2L_roll_std` -> `SD distance Right body 2 - Left head 2`); the
published panel labels were manually polished, so word order can differ
slightly and `parameter` remains the authoritative identifier.

The legacy analysis is archived under `classifier/legacy_shap/`:
`shap_values.npz` (54 MB - the SHAP values, the feature values behind the dot
colours, and the class order), `legacy_feature_ranks.csv` (the authoritative
per-class ranks), `xgb_model.pkl` and `label_encoder.pkl`. The values were
repacked out of a scikit-learn 1.2.1 joblib pickle by
`scripts/import_legacy_shap_values.py`, because a pickle is neither a safe nor a
durable archive format. Only the 179 MB balanced feature matrix, needed solely
to retrain the legacy model, is left outside the repository.

`scripts/generate_figures/supplementary_figure_4_classifier_shap.py` redraws the
eight panels from that archive with `shap.summary_plot`, as the legacy figure
script did, and needs nothing outside the repository. Both it and
`scripts/import_legacy_shap_summary.py` accept `--source` to read the legacy
keypoint-MoSeq results directory instead, which reproduces the tracked outputs
byte for byte.

Each panel's x-axis is limited to the central 99.5% of its own plotted values
(`XLIM_PERCENTILE`). Without this, a few extreme SHAP outliers stretch the axis
far past the bulk of the distribution and squash every beeswarm against zero -
`Groom` ran to 3.29 while its central mass sits inside +/-0.9. The clipped
points are hidden by the axes, not removed from the analysis, and remain in
`supplementary_figure4_shap_summary.csv`; they are 0.11-0.43% of plotted points
per panel (`Groom` highest at 114 of 26,380).

Behavior names use the short forms shared with every other figure in the set
(`Freeze`, `Sniff`, `Groom`, `Turn`, `Locomotion`, `Climb`, `Jump`,
`Unassigned`). The legacy classifier labels its classes with verb forms
(`Freezing`, `Grooming`, ...) and calls the unlabeled class `Unspecified`
(`No labeled` in its summary table); both are normalised on import.

The bundled `classifier/figure7_behavior_classifier.joblib` is a **different
model on a different feature set** (33 features derived from the five centroid
kinematic signals in `data/raw/moseq_syllables_per_frame.csv.gz`) and is used by
Figure 7A-C only. Running SHAP on it produces a re-analysis, not a reproduction
of the published panels.

## Workbook Assignment

`report/raw_data.xlsx` contains the plotted source values and animal-level
contributions. Figure 7 helper fields such as `panel` and `animal_label` are not
included, and dataset abbreviations are expanded to the full source names.

`report/statistical_report.xlsx` contains only the Figure 7 performance and
inference tables that add information beyond the raw workbook: detailed
held-out/LOOCV metrics (D-E), aggregate Shapley estimates (F), and permutation
results (I). Descriptive source tables for A-C, G-H, J, and K-N are not copied
into the statistical report.

Figure 7B contains all 20 features displayed in the published global SHAP
panel across all eight behavior classes. The reusable classifier artifact has
a separate 33-column engineering feature catalog; those columns are not mixed
into Figure 7B because they are not the feature labels shown in that panel.
