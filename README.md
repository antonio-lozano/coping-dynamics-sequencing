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

All figure inputs are bundled under `dataset/source/` (see `dataset/README.md`), so the
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
| Figure 7 — Imported assembled PDF | `scripts/generate_figures/figure_7_import_pdf.py` |
| Supplementary Figure 1 — Tracking clusters | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` |
| Supplementary Figure 3 — Distance metrics | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Regenerate all figures at once:

```bash
python scripts/run_all_figures.py
```

(Figure 1 is a hand-made schematic with no repo generator.)

## Project structure
```
|-- dataset/
|   |-- source/             # Bundled figure inputs (CSV/XLSX/SVG + compressed tables)
|   `-- Raw_data.xlsx       # Manuscript raw-data workbook
|-- docs/                   # Figure provenance / replication notes
|-- figures/                # Canonical manuscript figures (figureN.*)
|-- report/                 # Final styled statistical workbook
|-- results/figure_data/    # Supporting tables, audits, intermediate renders
|-- results/statistical_reports/
|                           # Reproducible statistical CSVs and manuscript audit notes
|-- scripts/
|   |-- generate_figures/   # One script per manuscript figure (figure_2..7, supplementary figures)
|   |-- build_statistical_report.py
|   `-- run_all_figures.py  # Regenerate every figure
|-- src/
|   |-- config.py           # Centralized paths, source-data resolution, constants
|   `-- plotting.py         # Shared plotting utilities (chord diagram, etc.)
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Data
All required figure inputs are self-contained in `dataset/source/` (the large
per-frame syllable table and results pickle are stored gzip-compressed). The 
figures regenerate reproducibly without any external dependencies.

For reproducibility extensions or re-extraction from source archives, override paths via:
- `COPING_DYNAMICS_SOURCE_DIR` — extracted source directory
- `COPING_DYNAMICS_DOWNLOADS` — directory containing `Coping_data.zip`/`Coping_data2.zip`
- `COPING_DATA_ZIP`, `COPING_DATA2_ZIP` — specific archive paths

The final report workbook is `report/STATISTICAL_REPORT_FINAL.xlsx`. Rebuild it with:

```bash
python scripts/build_statistical_report.py
```

Figure 4 uses the final no-seed MixedLM analysis. The seeded analysis is recoverable
from tag `pre-noseed-seeded` or branch `backup-seeded-analysis`.

See `dataset/README.md` for the full inventory of bundled inputs.

## License
See `LICENSE`.
