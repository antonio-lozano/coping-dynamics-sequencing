# Reproducibility Guide

This repository is organized as a self-contained manuscript-reproducibility
package. All data required for the data-derived figures are tracked under
`data/raw/`; generated analysis-ready tables are tracked under `data/derived/`;
source-data CSVs, statistics, intermediate helper files, and model artifacts are
tracked under `results/`; manuscript-facing workbooks are tracked under
`report/`.

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

This regenerates the data-derived figures in `figures/`, source-data CSVs in
`results/source_data/`, statistics in `results/statistics/`, and helper
tables/renders in `results/intermediate/`. Figure 1 is a hand-made schematic
and Figure 7 is a supplied assembled manuscript figure; both are tracked as
canonical exports in `figures/`.

To rebuild the full package from tracked inputs, including derived tables,
figures, workbooks, manifest, and checks:

```bash
python scripts/run_all.py
```

The full rebuild is concise by default. Add `--verbose` to stream the detailed
model output from each child script.

## Rebuild Statistical Report

```bash
python scripts/build_statistical_report.py
```

The statistical report is `report/statistical_report.xlsx`. The manuscript raw
data workbook is `report/raw_data.xlsx`.

## Rebuild Derived Tables

Utility scripts in `scripts/derive_tables/` rebuild committed intermediate
tables from the bundled source data:

```bash
python scripts/derive_tables/cluster_tables.py
python scripts/derive_tables/tracking_exclusions.py
python scripts/derive_tables/fig6_resilience_stats.py
```

The raw-data workbook can be rebuilt with:

```bash
python scripts/build_raw_data_workbook.py
```

## Update Manifest

After intentionally changing tracked data, figures, reports, or results:

```bash
python scripts/update_manifest.py
python scripts/check_reproducibility.py
```

Commit the updated `MANIFEST.csv` with the changed artifacts.

## Expected Repository Controls

- All required input data are tracked under `data/raw/`.
- Generated analysis-ready data are tracked under `data/derived/`.
- Manuscript figures are tracked under `figures/`.
- Figure source-data CSVs are tracked under `results/source_data/`.
- Statistical outputs and manuscript Results text audit are tracked under
  `results/statistics/`.
- Helper tables and non-canonical renders are tracked under
  `results/intermediate/`.
- Trained model artifacts are tracked under `results/models/`.
- Manuscript raw-data and statistical workbooks are tracked under `report/`.
- `MANIFEST.csv` records byte sizes and SHA-256 hashes for publication
  artifacts.
- `.github/workflows/reproducibility.yml` runs the lightweight checks on GitHub.
