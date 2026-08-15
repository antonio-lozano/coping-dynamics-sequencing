# Coping Dynamics Sequencing

Self-contained manuscript repository for the behavioral motif, freezing, and
stress-coping dynamics analyses. All data required to rebuild the data-derived
figures, statistical CSVs, Excel reports, and validation manifest are included
in the repository.

## Quick Start

Use the locked `uv` environment when possible:

```bash
uv sync
uv run python scripts/check_reproducibility.py
```

Alternative environments:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

```bash
python --version  # Python 3.9, 3.10, or 3.11
pip install -r requirements.txt
python scripts/check_reproducibility.py
```

## Rebuild Everything

Run commands from the repository root.

```bash
python scripts/run_all.py
```

This rebuilds the compact freezing-prediction tables, processed analysis
tables, data-derived figures, raw-data workbook, full and source-dataset
statistical report, artifact manifest, and
reproducibility checks. Add `--verbose` to stream each
child script's output. Add `--skip-figures` to rebuild tables, reports, the
manifest, and checks without re-rendering figures.

Targeted rebuilds:

```bash
python scripts/run_all_figures.py
python scripts/build_raw_data_workbook.py
python scripts/build_statistical_report.py
python scripts/audit_manuscript_results.py
python scripts/update_manifest.py
python scripts/check_reproducibility.py
```

## Repository Contract

```text
data/raw/             Immutable bundled inputs used by the analyses
data/processed/       Deterministic tables regenerated from data/raw
figure_source_data/   Plotted values and summary values behind Figures 2-6
statistics/           Machine-readable statistical model outputs
figures/              Canonical manuscript figure exports
report/               raw-data workbook and canonical statistical report
classifier/           Figure 7 behavior-classifier artifact
scripts/              Rebuild, report, figure, and validation entry points
src/                  Shared analysis, plotting, statistics, and classifier code
docs/                 Data dictionary and focused provenance notes
config/               Figure metadata used by the report package
```

`data/raw/` contains inputs. `data/processed/`, `figure_source_data/`,
`statistics/`, `figures/`, and `report/` are regenerated products. The term
`figure_source_data` is used only for the plotted data underlying manuscript
figures, not for raw experimental inputs.

The statistical builder creates one canonical output:

- `report/statistical_report.xlsx` recalculates the Combined full dataset,
  Sanguino-Gomez & Krugers, and Sanguino-Gomez et al. analyses where source
  dataset identity is available. It includes raw and adjusted p-values, direct
  resilience contrasts, per-time-bin posthoc tests, model/sample metadata,
  time-effect units, and the manuscript consistency audit.

The historical report is retained as transparent source-cell records under
`data/raw/manuscript_tables/statistical_report/`; it is provenance, not a
second final workbook. See `docs/workbook_match_audit.md` for details.

## Key Data

Raw inputs include:

- `data/raw/freezing_predictions/`: 98 original per-animal supervised freezing
  prediction CSVs.
- `data/raw/freezing_predictions_light.csv.gz`: compact long-format companion
  generated from the per-animal prediction CSVs.
- `data/raw/freezing_predictions_index.csv`: file-level index with frame counts,
  freezing fractions, byte sizes, and SHA-256 checksums.
- `data/raw/moseq_syllables_per_frame.csv.gz`: per-frame MoSeq syllable table.
- `data/raw/simba_validation_manual_vs_automatic.csv`: manual versus automatic
  SimBA freezing validation values.
- `data/raw/syllable_usage_per_timebin_30s.csv` and
  `data/raw/syllable_usage_per_timebin_250ms.csv`: time-bin syllable usage.
- `data/raw/behavioral_flexibility_scores.xlsx`: behavioral flexibility score
  table used for resilience grouping.

## Figure Scripts

| Output | Script |
| --- | --- |
| Figure 1 | Canonical tracked export (hand-assembled schematic) |
| Figure 2 | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3 | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4 | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5 | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6 | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Figure 7 | Canonical tracked export with associated classifier artifact |
| Supplementary Figure 1 | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` |
| Supplementary Figure 2 | Canonical tracked export (analysis workflow schematic) |
| Supplementary Figure 3 | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Figures 1 and 7 and Supplementary Figure 2 are tracked as canonical assembled
manuscript figures. Figure 7 has an associated classifier artifact in
`classifier/figure7_behavior_classifier.joblib`.

## Quality Controls

- `MANIFEST.csv` records byte sizes and SHA-256 hashes for tracked publication
  artifacts.
- `statistics/manuscript_consistency_audit.csv` compares selected manuscript
  Results claims against regenerated statistical CSVs and is rebuilt by
  `scripts/audit_manuscript_results.py`.
- `scripts/check_reproducibility.py` verifies required files, figure exports,
  98 raw freezing prediction CSVs, manifest hashes, retired-path absence,
  absence of local absolute paths in text files, and absence of local tool
  traces.
- `.github/workflows/reproducibility.yml` compiles all Python files and runs the
  repository validation on GitHub.
- All default paths resolve inside the repository through `src/config.py`.

## Additional Documentation

- `REPRODUCIBILITY.md`: exact rebuild and validation workflow.
- `DATA_AVAILABILITY.md`: manuscript-facing data availability language.
- `CODE_AVAILABILITY.md`: manuscript-facing code availability language.
- `docs/data_dictionary.md`: file-level description of inputs and outputs.
- `docs/figure4_coping_provenance.md`: Figure 4 analysis provenance.
- `docs/behavior_classifier.md`: Figure 7 classifier usage.

Use `CITATION.cff` for software citation metadata and `LICENSE` for licensing.
