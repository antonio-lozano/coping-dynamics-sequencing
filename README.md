# Coping Dynamics Sequencing

Reproducible manuscript package for behavioral motif and stress-coping dynamics
analyses from keypoint-MoSeq syllable data and supervised freezing predictions.
The repository is self-contained: all inputs needed for the data-derived
figures, statistics, workbooks, and audit tables are tracked in the repository.

## Quick Start

Use the locked `uv` environment when possible:

```bash
uv sync
uv run python scripts/check_reproducibility.py
```

Conda and pip entry points are also provided:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

```bash
python --version  # Python 3.9, 3.10, or 3.11
pip install -r requirements.txt
```

## Rebuild The Package

Run commands from the repository root.

```bash
python scripts/run_all.py
```

This rebuilds compact raw freezing-prediction companions, derived tables,
figures, workbooks, `MANIFEST.csv`, and the reproducibility checks. The command
prints concise task-level progress by default; add `--verbose` for detailed
child-script output. Use `--skip-figures` to rebuild tables, reports, manifest,
and checks without re-rendering the figure panels.

Figure-only rebuild:

```bash
python scripts/run_all_figures.py
```

Validation-only check:

```bash
python scripts/check_reproducibility.py
```

## Repository Contents

```text
data/raw/                 Immutable analysis inputs bundled with the repo
data/derived/             Analysis-ready tables rebuilt from data/raw
docs/                     Repository layout, data dictionary, and provenance notes
figures/                  Canonical manuscript figure exports
report/                   Manuscript raw-data and statistical workbooks
results/source_data/      Figure source-data CSVs
results/statistics/       Statistical CSV outputs and Results text audit
results/intermediate/     Helper tables and non-canonical figure renders
results/models/           Tracked model artifacts used by the classifier entry point
scripts/                  Rebuild, validation, figure, and derived-table scripts
src/                      Shared analysis, plotting, statistics, and config code
```

Additional documentation:

- `docs/REPOSITORY_LAYOUT.md` describes the directory contract.
- `docs/DATA_DICTIONARY.md` lists the tracked data and generated artifacts.
- `REPRODUCIBILITY.md` gives the exact rebuild and validation workflow.
- `DATA_AVAILABILITY.md` and `CODE_AVAILABILITY.md` provide manuscript-facing
  availability language.

## Data Model

Raw inputs are kept in `data/raw/`. Generated analysis-ready tables are kept in
`data/derived/`. Figure source-data tables, statistical outputs, helper tables,
and model artifacts are kept under `results/`.

The raw freezing predictions are included in two forms:

- `data/raw/freezing_predictions/`: original per-animal prediction CSVs.
- `data/raw/freezing_predictions_light.csv.gz`: compact long-format companion
  built from those per-animal files.
- `data/raw/freezing_predictions_index.csv`: file-level index with frame counts,
  freezing fractions, byte sizes, and SHA-256 checksums.

## Figure Scripts

| Manuscript figure | Script |
| --- | --- |
| Figure 2, validation | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3, behavior clusters over time | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4, diversity dynamics | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5, resilience dynamics | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6, resilience diversity | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Supplementary Figure 1, tracking clusters | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` |
| Supplementary Figure 3, distance metrics | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Figure 1 is a hand-made schematic. Figure 7 is supplied as a canonical assembled
manuscript figure in `figures/`; the associated classifier model is tracked in
`results/models/behavior_classifier/`.

## Quality Controls

- `MANIFEST.csv` records byte sizes and SHA-256 hashes for tracked publication
  artifacts.
- `.github/workflows/reproducibility.yml` compiles Python files and runs the
  reproducibility check on GitHub.
- `scripts/check_reproducibility.py` verifies required files, expected figure
  exports, freezing-prediction file count, artifact layout, absence of common
  scratch files, absence of machine-local absolute paths in text files, and
  manifest hashes.
- All configured default paths resolve inside the repository. External paths are
  only used if explicitly supplied through environment variables.

## Citation And License

Use `CITATION.cff` for software citation metadata. See `LICENSE` for licensing.
