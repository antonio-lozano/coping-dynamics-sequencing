# Code Availability

All custom code needed to regenerate the data-derived figures, tables, reports,
and statistical outputs is included in this repository.

Primary entry points:

- `scripts/run_all.py`
- `scripts/run_all_figures.py`
- `scripts/check_reproducibility.py`
- `scripts/update_manifest.py`
- `scripts/build_raw_data_workbook.py`
- `scripts/build_statistical_report.py`
- `scripts/train_behavior_classifier.py`

Figure scripts live in `scripts/generate_figures/`. Derived-table scripts live
in `scripts/derive_tables/`. Shared analysis, plotting, statistics, path
configuration, and classifier code live in `src/`.

For publication, cite the archived release DOI for this repository when
available.
