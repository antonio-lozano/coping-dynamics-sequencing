# Scripts

Primary entry points:

- `run_all_figures.py` regenerates all data-derived manuscript figures.
- `run_all.py` rebuilds derived tables, figures, workbooks, manifest, and checks
  with concise task-level output by default; add `--verbose` for child-script
  output.
- `build_raw_data_workbook.py` rebuilds the manuscript raw-data workbook.
- `build_statistical_report.py` rebuilds the styled statistical workbook.
- `check_reproducibility.py` validates required files, checksums, and repo hygiene.
- `update_manifest.py` rewrites `MANIFEST.csv` after intentional artifact changes.
- `train_behavior_classifier.py` trains or applies the Fig. 7 behavior classifier.

Subdirectories:

- `generate_figures/` contains one script per data-derived manuscript figure.
- `derive_tables/` contains utilities for rebuilding committed derived/statistical
  tables from bundled raw inputs.
