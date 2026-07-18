# Repository Layout

This repository is organized as a self-contained manuscript reproducibility
package. Folder-level README files are intentionally avoided in artifact
directories so the root README and the documents in `docs/` remain the single
source of orientation.

| Path | Role | Tracked? | Rebuilt by |
| --- | --- | --- | --- |
| `data/raw/` | Immutable inputs required for the analyses | Yes | Primary source |
| `data/derived/` | Analysis-ready tables rebuilt from raw inputs | Yes | `scripts/derive_tables/` |
| `figures/` | Canonical manuscript figure exports | Yes | `scripts/run_all_figures.py` for data-derived figures |
| `report/` | Manuscript raw-data and statistical workbooks | Yes | `scripts/build_raw_data_workbook.py`, `scripts/build_statistical_report.py` |
| `results/source_data/` | Figure source-data CSVs | Yes | Figure scripts |
| `results/statistics/` | Statistical model outputs and Results text audit | Yes | Figure and derived-table scripts |
| `results/intermediate/` | Helper tables and non-canonical figure renders | Yes | Figure scripts |
| `results/models/` | Tracked model artifacts used by classifier entry points | Yes | `scripts/train_behavior_classifier.py` when retraining |
| `scripts/` | Rebuild and validation entry points | Yes | Maintained code |
| `src/` | Shared plotting, statistics, config, and classifier code | Yes | Maintained code |
| `.github/workflows/` | Lightweight continuous reproducibility checks | Yes | GitHub Actions |

## Path Contract

All default paths resolve relative to the repository root through
`src/config.py`. The pipeline does not depend on user-specific directories or
download folders. Optional environment variables can override paths for local
development, but a clean clone should run from tracked files alone.

## Canonical Outputs

The canonical manuscript figures are the PDF/SVG/PNG files in `figures/`.
Helper renders in `results/intermediate/figures/` are retained for auditability
but are not the manuscript-facing outputs. The two workbook outputs are:

- `report/raw_data.xlsx`
- `report/statistical_report.xlsx`

`MANIFEST.csv` records checksums and byte sizes for publication artifacts.
