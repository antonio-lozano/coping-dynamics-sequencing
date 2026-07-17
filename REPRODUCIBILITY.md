# Reproducibility Guide

This repository is organized as a self-contained manuscript-reproducibility
package. All data required for the data-derived figures are tracked under
`data/source/`; generated supporting tables and statistical outputs are tracked
under `results/`; manuscript-facing workbooks are tracked under `report/`.

## Quick Check

Run the lightweight repository check first:

```bash
python scripts/check_reproducibility.py
```

This verifies required files, expected figure exports, intermediate CSVs,
freezing-prediction file count, absence of common scratch artifacts, absence of
machine-local absolute paths in text files, and `MANIFEST.csv` checksums.

Use Python 3.9, 3.10, or 3.11 for the analysis environment. The conda
environment pins Python 3.9; the GitHub Actions workflow uses Python 3.11. For
an exact locked dependency resolution, use `uv sync` with the tracked
`uv.lock`.

## Rebuild Figures

Install dependencies, then run:

```bash
python scripts/run_all_figures.py
```

This regenerates the data-derived figures in `figures/` and writes supporting
tables to `results/figure_data/`. Figure 1 is a hand-made schematic and Figure
7 is a supplied assembled manuscript figure; both are tracked as canonical
exports in `figures/`.

## Rebuild Statistical Report

```bash
python scripts/build_statistical_report.py
```

The statistical report is `report/STATISTICAL_REPORT.xlsx`. The manuscript raw
data workbook is `report/RAW_DATA.xlsx`.

## Rebuild Derived Tables

Utility scripts in `scripts/derive_tables/` rebuild committed intermediate
tables from the bundled source data:

```bash
python scripts/derive_tables/cluster_tables.py
python scripts/derive_tables/tracking_exclusions.py
python scripts/derive_tables/fig6_resilience_stats.py
```

## Update Manifest

After intentionally changing tracked data, figures, reports, or results:

```bash
python scripts/update_manifest.py
python scripts/check_reproducibility.py
```

Commit the updated `MANIFEST.csv` with the changed artifacts.

## Expected Repository Controls

- All required input data are tracked under `data/source/`.
- Manuscript figures are tracked under `figures/`.
- Intermediate analysis tables are tracked under `results/figure_data/`.
- Statistical outputs and manuscript Results text audit are tracked under
  `results/statistical_reports/`.
- Manuscript raw-data and statistical workbooks are tracked under `report/`.
- `MANIFEST.csv` records byte sizes and SHA-256 hashes for publication
  artifacts.
- `.github/workflows/reproducibility.yml` runs the lightweight checks on GitHub.
