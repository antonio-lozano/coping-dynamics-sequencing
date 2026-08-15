# Legacy statistical report source

`reported_cells.jsonl` and `reported_layout.json` are the canonical, text-based
source for the historical statistical-report cells whose original model inputs
are not all available. They were imported from the supplied manuscript report
with `scripts/import_legacy_statistical_report.py`.

This source is intentionally separate from the live analysis. The repository
generates one canonical statistical output:

- `report/statistical_report.xlsx`: recomputed combined, Experiment 1, and
  Experiment 3 analyses where the tracked inputs contain experiment identity.

The source-cell archive preserves reported values, including legacy rounding
and known historical inconsistencies. It must not be interpreted as evidence
that report-only values were recalculated. The generated workbook includes a
manuscript comparison sheet so discrepancies remain visible.
