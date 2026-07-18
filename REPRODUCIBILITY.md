# Reproducibility

This repository is intended to run from a clean clone without external data
downloads. All default paths resolve inside the repository, and all data-derived
figures and reports are rebuilt from tracked inputs.

## Environment

Recommended:

```bash
uv sync
```

Alternative environments:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

```bash
python --version  # Python 3.9, 3.10, or 3.11
pip install -r requirements.txt
```

## Minimum Validation

Run:

```bash
python scripts/check_reproducibility.py
```

The check verifies:

- required data, figures, reports, scripts, and documentation files
- exactly 98 raw per-animal freezing prediction CSVs
- presence of the compact freezing prediction index and light table
- absence of retired scratch/layout paths
- absence of common local absolute paths in text files
- `MANIFEST.csv` byte sizes and SHA-256 hashes

The same check runs in `.github/workflows/reproducibility.yml`.

## Full Rebuild

Run:

```bash
python scripts/run_all.py
```

This executes, in order:

1. Build compact freezing prediction companions in `data/raw/`.
2. Rebuild derived tables in `data/derived/`.
3. Rebuild Figure 6 statistical tables.
4. Re-render data-derived manuscript figures.
5. Rebuild `report/raw_data.xlsx`.
6. Rebuild `report/statistical_report.xlsx`.
7. Update `MANIFEST.csv`.
8. Run `scripts/check_reproducibility.py`.

The command is concise by default. Add `--verbose` to stream child-script
output. Add `--skip-figures` to rebuild tables, workbooks, manifest, and checks
without re-rendering figures.

## Targeted Rebuilds

Figures only:

```bash
python scripts/run_all_figures.py
```

Derived tables:

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

Manifest:

```bash
python scripts/update_manifest.py
python scripts/check_reproducibility.py
```

## Figure Notes

Figures 2-6, Supplementary Figure 1, and Supplementary Figure 3 are generated
from tracked data. Figure 1 is a hand-made schematic. Figure 7 is tracked as a
canonical assembled manuscript figure in `figures/`; the associated classifier
model is tracked in `results/models/behavior_classifier/`.

## Expected Repository State

A publication-ready state should satisfy:

- `git status --short` is clean after committed rebuilds.
- `python scripts/check_reproducibility.py` passes.
- `git diff --check` reports no whitespace errors.
- No default path points to a user-specific directory.
- `MANIFEST.csv` is committed with the artifacts it describes.
