# Reproducibility

This repository is intended to run from a clean clone without external data
downloads. All default paths resolve inside the repository, and all
data-derived figures, statistical outputs, reports, and checks are rebuilt from
tracked inputs.

## Environment

The instructions below are identical on Windows and Linux.

Recommended:

```bash
uv sync --locked
```

`uv` installs the interpreter named in `.python-version` and resolves from
`uv.lock`, so the environment is the same on every machine. `--locked` fails
rather than silently re-resolving if the lock and `pyproject.toml` disagree.

Alternative environments:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

```bash
python --version  # Python 3.9, 3.10, or 3.11
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

Every command in the rest of this document is written as `python ...`. On the
`uv` route, prefix it with `uv run`, as in
`uv run python scripts/check_reproducibility.py`. On the conda and pip routes,
activate the environment first and run it as written.

## Validation

Run:

```bash
python scripts/check_reproducibility.py
```

The validation checks:

- required inputs, processed tables, source-data CSVs, statistics, figures,
  reports, scripts, and documentation files
- exactly 98 raw per-animal freezing prediction CSVs
- compact freezing prediction index and light table
- absence of retired artifact paths and duplicate figure-render folders
- absence of local absolute paths in text files
- absence of local tool traces in versioned text files
- `MANIFEST.csv` byte sizes and SHA-256 hashes

The same check runs in `.github/workflows/reproducibility.yml`.

## Full Rebuild

Run:

```bash
python scripts/run_all.py
```

This executes, in order:

1. Build compact freezing prediction companions in `data/raw/`.
2. Rebuild processed analysis tables in `data/processed/`.
3. Rebuild Figure 6 statistical tables in `statistics/`.
4. Re-render data-derived manuscript figures in `figures/`.
5. Rebuild `report/raw_data.xlsx`.
6. Rebuild `report/statistical_report.xlsx`.
7. Update `MANIFEST.csv`.
8. Run `scripts/check_reproducibility.py`.
9. Run `scripts/check_manuscript.py`.

The command is concise by default. Add `--verbose` to stream child-script
output. Add `--skip-figures` to rebuild tables, workbooks, the manifest, and
checks without re-rendering figures.

## Targeted Rebuilds

Figures only:

```bash
python scripts/run_all_figures.py
```

Processed tables:

```bash
python scripts/derive_tables/freezing_predictions_light.py
python scripts/derive_tables/cluster_tables.py
python scripts/derive_tables/tracking_exclusions.py
python scripts/derive_tables/fig6_resilience_stats.py
```

Reports:

```bash
python scripts/build_raw_data_workbook.py
python scripts/build_statistical_report.py
```

Manifest and checks:

```bash
python scripts/update_manifest.py
python scripts/check_reproducibility.py
python scripts/check_manuscript.py
```

## Byte-Stable Rebuilds

Rebuilding unchanged inputs reproduces every artifact byte for byte: the
repository `matplotlibrc` pins the SVG hash salt, `run_all_figures.py` sets
`SOURCE_DATE_EPOCH=0` to freeze the dates embedded in PDF and SVG output, the
report workbooks are written with fixed timestamps, and the compressed
freezing-prediction table is gzipped without an mtime header. When running a
single figure script directly rather than through `run_all_figures.py`, set
`SOURCE_DATE_EPOCH=0` in the environment to get the same bytes the runner
produces; run it from the repository root so `matplotlibrc` is picked up.

## Expected Clean State

A publication-ready state should satisfy:

- `git status --short` is clean after committed rebuilds.
- `python scripts/check_reproducibility.py` passes.
- `git diff --check` reports no whitespace errors.
- No default path points to a user-specific directory.
- `MANIFEST.csv` is committed with the artifacts it describes.
