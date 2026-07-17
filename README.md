# Coping Dynamics Sequencing

Reproducible manuscript figures for unsupervised behavioral motif discovery in
stress coping dynamics. This repository regenerates the data-derived manuscript
figures from keypoint-MoSeq syllable data and supervised freezing predictions.

## Setup

Using conda:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

Using pip:

```bash
pip install -r requirements.txt
```

All figure inputs are bundled under `data/source/` (see
`data/README.md`), so the data-derived figures regenerate out of the box.

## Reproducibility Check

Before rerunning analyses, verify that the repository has all required data,
figures, reports, intermediate CSVs, and checksums:

```bash
python scripts/check_reproducibility.py
```

## Running Figures

Run scripts from the repo root. Each script writes the clean manuscript figure
to `figures/` and supporting tables/intermediate renders to
`results/figure_data/`.

| Manuscript figure | Script |
| --- | --- |
| Figure 2 - Validation (keypoint-MoSeq) | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3 - Behavior clusters over time | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4 - Diversity dynamics | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5 - Resilience dynamics | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6 - Resilience diversity | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Supplementary Figure 1 - Tracking clusters | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` |
| Supplementary Figure 3 - Distance metrics | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Regenerate all data-derived figures:

```bash
python scripts/run_all_figures.py
```

Figure 1 is a hand-made schematic with no repo generator. Figure 7 is supplied
as a canonical assembled manuscript figure in `figures/`.

## Project Structure

```text
|-- data/
|   `-- source/             # Bundled figure inputs
|-- docs/                   # Figure provenance / replication notes
|-- figures/                # Canonical manuscript figures
|-- report/                 # Final styled statistical workbook
|-- results/
|   |-- figure_data/        # Supporting tables and intermediate renders
|   `-- statistical_reports/# Statistical CSVs and final Results text
|-- scripts/
|   |-- generate_figures/   # Data-derived figure generators
|   |-- derive_tables/      # Rebuild committed source/statistical tables
|   |-- build_statistical_report.py
|   `-- run_all_figures.py
|-- src/                    # Shared analysis, plotting, and config code
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Data And Reports

All required figure inputs are self-contained in `data/source/`. The large
per-frame syllable table and results pickle are gzip-compressed.

The final report workbook is `report/STATISTICAL_REPORT_FINAL.xlsx`. Rebuild it
with:

```bash
python scripts/build_statistical_report.py
```

Figure 4 uses the default MixedLM analysis implemented in the figure generator
and mirrored in the final statistical report.

Manuscript-facing workbook outputs are:

- `report/RAW_DATA.xlsx`
- `report/STATISTICAL_REPORT_FINAL.xlsx`

See `REPRODUCIBILITY.md`, `DATA_AVAILABILITY.md`, and `CODE_AVAILABILITY.md`
for publication and repository-release notes.

## License

See `LICENSE`.
