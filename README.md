# Coping Dynamics Sequencing

Reproducible manuscript figures for unsupervised behavioral motif discovery in
stress coping dynamics. This repository regenerates the data-derived manuscript
figures from keypoint-MoSeq syllable data and supervised freezing predictions.

## Setup

For an exact locked environment with `uv`:

```bash
uv sync
uv run python scripts/check_reproducibility.py
```

Using conda:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

Using pip:

```bash
python --version  # use Python 3.9, 3.10, or 3.11
pip install -r requirements.txt
```

All immutable figure inputs are bundled under `data/raw/`; generated
analysis-ready tables live under `data/derived/` (see `data/README.md`), so the
data-derived figures regenerate out of the box.

## Reproducibility Check

Before rerunning analyses, verify that the repository has all required data,
figures, reports, intermediate CSVs, and checksums:

```bash
python scripts/check_reproducibility.py
```

## Running Figures

Run scripts from the repo root. Each script writes the clean manuscript figure
to `figures/`. Source-data CSVs go to `results/source_data/`, statistical
outputs go to `results/statistics/`, and helper tables/renders go to
`results/intermediate/`.

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

Regenerate the full manuscript reproducibility package from tracked inputs:

```bash
python scripts/run_all.py
```

The full rebuild prints task-level progress by default; add `--verbose` to
stream each child script's model output.

Figure 1 is a hand-made schematic with no repo generator. Figure 7 is supplied
as a canonical assembled manuscript figure in `figures/`.

## Project Structure

```text
|-- data/
|   |-- raw/                # Immutable bundled inputs
|   `-- derived/            # Tables generated from data/raw
|-- docs/                   # Figure provenance / replication notes
|-- figures/                # Canonical manuscript figures
|-- report/                 # Styled statistical workbook
|-- results/
|   |-- source_data/        # Figure source-data CSVs
|   |-- statistics/         # Model/statistical CSVs and manuscript text audit
|   |-- intermediate/       # Helper tables and non-canonical renders
|   `-- models/             # Trained model artifacts
|-- scripts/
|   |-- generate_figures/   # Data-derived figure generators
|   |-- derive_tables/      # Rebuild committed source/statistical tables
|   |-- build_raw_data_workbook.py
|   |-- build_statistical_report.py
|   |-- run_all_figures.py
|   `-- run_all.py
|-- src/                    # Shared analysis, plotting, and config code
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Data And Reports

All required figure inputs are self-contained in `data/raw/`. The large
per-frame syllable table and results pickle are gzip-compressed. Generated
analysis-ready inputs are committed under `data/derived/` and can be rebuilt
with `scripts/derive_tables/`.

The statistical report workbook is `report/statistical_report.xlsx`. Rebuild it
with:

```bash
python scripts/build_statistical_report.py
```

Figure 4 uses the default MixedLM analysis implemented in the figure generator
and mirrored in the final statistical report.

Manuscript-facing workbook outputs are:

- `report/raw_data.xlsx`
- `report/statistical_report.xlsx`

See `REPRODUCIBILITY.md`, `DATA_AVAILABILITY.md`, and `CODE_AVAILABILITY.md`
for publication and repository-release notes.

## License

See `LICENSE`.
