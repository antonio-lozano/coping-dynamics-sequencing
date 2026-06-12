# Figure 2 Code Comparison

This note compares the original keypoint-moseq analysis script with the repo
regenerator added for manuscript Figure 2.

| Panel | Original keypoint-moseq code | Repo regenerator |
| --- | --- | --- |
| D, syllable usage | `plot_syllable_distribution(moseq_df)` uses `moseq_df['syllable'].value_counts().sort_values(ascending=False)` and a cumulative `>=95%` cutoff. See `H:\antonio\keypoint_moseq_project\code\SURF_results_analysis_spyder_CAPITULO.py:736`. | `_load_syllable_usage_from_moseq_df()` now uses the same `value_counts().sort_values(ascending=False)` rule. See `scripts/generate_figures/figure_2_validation_keypoint_moseq.py:298`. |
| E/F, precision/recall | `plot_top_95()` first aggregates per-animal `freezing_overlap_per_syllable`, then sorts by `mean_all`, and keeps syllables until cumulative contribution reaches 95%. See `H:\antonio\keypoint_moseq_project\code\SURF_results_analysis_spyder_CAPITULO.py:609`. | `_top_metric_from_original_metrics()` mirrors that aggregation and top-95 selection for both precision and recall. See `scripts/generate_figures/figure_2_validation_keypoint_moseq.py:513`. |
| G, overlap | `selected_syllables = [0, 28, 40]`; overlap is `selected_syllable_frames intersection freezing_frames / total_freeze_frames`. See `H:\antonio\keypoint_moseq_project\code\SURF_results_analysis_spyder_CAPITULO.py:305`. | `OVERLAP_SYLLABLES = {0, 28, 40}` and overlap is computed with the same denominator. See `scripts/generate_figures/figure_2_validation_keypoint_moseq.py:41`. |
| G, animal order | Original loops through `moseq_df['name'].unique()`, then uses Python `sorted(..., key=label_number)`, which is stable for ties. See `H:\antonio\keypoint_moseq_project\code\SURF_results_analysis_spyder_CAPITULO.py:421`. | The repo loader now preserves `moseq_df` order using `groupby("name", sort=False)` before the stable sort. See `scripts/generate_figures/figure_2_validation_keypoint_moseq.py:275`. |
| Physical layout | Original standalone G was `figsize=(16, 6)` and D was `figsize=(20, 6)` before manual manuscript assembly. | The composite now uses an A4-like page and places G/H in a single bottom row instead of stretching G across two rows. See `scripts/generate_figures/figure_2_validation_keypoint_moseq.py:607`. |

Important data caveat: the available `moseq_df.csv` files do not exactly match
the exported `syllable_recall_precision_f1_usage.csv`. For example, the current
`2025_01_24-16_44_21\moseq_df.csv` panel-D order starts `0, 1, 6, 2, 3...`,
while the exported reference CSV starts `0, 1, 2, 3, 6...`. That means exact
visual reproduction of the submitted manuscript PDF requires either the exact
in-memory `moseq_df` used when the exported SVG/CSV panels were made, or direct
reuse of those saved panels.
