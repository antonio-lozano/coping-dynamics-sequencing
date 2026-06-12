# Manuscript Figure Replication

The trimmed manuscript archive is the visual reference:

`H:\Downloads\Trimmed_version-20260515T084756Z-3-001.zip`

Run this command to export the exact figure pack from that archive:

```powershell
python scripts\manuscript\export_trimmed_original_figures.py
```

Outputs are written to:

`results\manuscript_figures\final_from_trimmed`

The manuscript-ready copy for day-to-day use is:

`Figures`

That folder contains PDF and SVG versions of:

- `figure1`
- `figure2`
- `figure3`
- `figure4`
- `figure5`
- `figure6`
- `figure7`
- `supplementary_figure1`
- `supplementary_figure2`
- `supplementary_figure3`
- `supplementary_figure4`

`manifest.csv` records the page size and page count for each exported figure.

## Current Data Workbooks

The common manuscript workbooks are:

- `data\Raw_data.xlsx`
- `data\Statistical_report.xlsx`

They currently include:

- SimBA manual-vs-automatic validation and correlation.
- Freezing progression raw data and linear model checks.
- Figure 2 syllable/freezing validation raw data and stats.
- Figure 3 behavior-cluster time courses and frequency summaries.
- Supplementary Figure 1 Mix behaviors and Inaccurate tracking cluster time courses and stats.
- Archived 30 s syllable time-bin data and BFL scores.
- Consolidated archived statistical tables for frequency/diversity, transition metrics, bout duration, and resilience analyses.

Regenerate them with:

```powershell
python scripts\data_exports\build_freezing_data_workbooks.py
```

## Data-Driven Figure Scripts Updated So Far

- Figure 2 validation:
  `scripts\figure_2_validation_keypoint_moseq\generate_figure_2_validation.py`
- Figure 3 behavior clusters:
  `scripts\figure_3_behavior_clusters\generate_figure_3_behavior_clusters.py`
- Figure 4 diversity/dynamics:
  `scripts\figure_4_diversity_dynamics\generate_figure_4_diversity_dynamics.py`

Figure 4 is generated from the archived COping 250 ms syllable table and follows
the COping frequency, bout-duration, and transition-metric code paths. The
ethogram/barcode panels use the predominant behavior per 250 ms bin. The metric
panels use the full 250 ms table, including the blank/unassigned cluster exactly
as the original COping scripts do. See `docs\FIGURE4_COPING_PROVENANCE.md`.

The exact visual pack should be used when the goal is to match the trimmed
archive exactly. The data-driven scripts are being brought into alignment
figure by figure.
