# Coping Dynamics Sequencing

Reproducible manuscript figures for unsupervised behavioral motif discovery in
stress coping dynamics. This repository regenerates every published figure from
keypoint-MoSeq syllable data and supervised freezing predictions.

## Setup

### Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

### Using pip
```bash
pip install -r requirements.txt
```

All figure inputs are bundled under `data/source/` (see `data/README.md`), so the
figures regenerate out of the box with no external data.

## Running figures
Run scripts from the repo root. Each script writes the clean manuscript figure
(`figureN.pdf/.svg/.png`) to the top-level `figures/` directory, and supporting
tables / intermediate renders to `results/figure_data/`.

| Manuscript figure | Script |
| --- | --- |
| Figure 2 — Validation (keypoint-MoSeq) | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3 — Behavior clusters over time | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4 — Diversity dynamics | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5 — Resilience dynamics | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6 — Resilience diversity | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Supplementary Figure 3 — Distance metrics | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Regenerate all figures at once:

```bash
python scripts/run_all_figures.py
```

(Figure 1 is a hand-made schematic with no repo generator.)

## Project structure
```
|-- data/
|   |-- source/             # Bundled figure inputs (CSV/XLSX/SVG + compressed tables)
|   |-- Raw_data.xlsx       # Manuscript data workbooks
|   `-- Statistical_report.xlsx
|-- docs/                   # Figure provenance / replication notes
|-- figures/                # Canonical manuscript figures (figureN.*)
|-- results/figure_data/    # Supporting tables, audits, intermediate renders
|-- scripts/
|   |-- generate_figures/   # One script per manuscript figure (figure_2..6, supplementary_figure_3)
|   |-- data_exports/       # Manuscript data-workbook exporter
|   `-- run_all_figures.py  # Regenerate every figure
|-- src/
|   |-- config.py           # Centralized paths, source-data resolution, constants
|   `-- plotting.py         # Shared plotting utilities (chord diagram, etc.)
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Data
All required figure inputs are tracked under `data/source/` (the large per-frame
table and results pickle are stored gzip-compressed). Overrides are available via
`COPING_DYNAMICS_SOURCE_DIR`, and the original archives can be pointed at with
`COPING_DYNAMICS_DOWNLOADS`, `COPING_DATA_ZIP`, `COPING_DATA2_ZIP`. See
`data/README.md` for the full inventory.

## License
See `LICENSE`.
