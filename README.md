<p align="center">
  <img src="docs/media/logo.png" alt="Coping Dynamics Sequencing" width="700">
</p>

# Coping Dynamics Sequencing

[![Reproducibility checks](https://github.com/antonio-lozano/coping-dynamics-sequencing/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/antonio-lozano/coping-dynamics-sequencing/actions/workflows/reproducibility.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)
[![Preprint](https://img.shields.io/badge/preprint-bioRxiv-b31b1b.svg)](https://doi.org/10.1101/2025.09.01.673507)

[`docs/web/coping-dynamics-sequencing.pdf`](docs/web/coping-dynamics-sequencing.pdf)
is the study in five minutes: the behavioural repertoire, the paradigm, the
findings, and the two coping profiles. It is printed from `docs/web/index.html`
by `scripts/build_page_pdf.py`, whose media come from
`scripts/build_page_media.py`.

**Preprint** &mdash; Sanguino-Gomez J, G&uuml;&ccedil;l&uuml; U, Krugers HJ, Lozano A.
*Coping strategies dynamics and resilience profiles after early life stress revealed by behavioral sequencing.*
bioRxiv, 2025. [doi:10.1101/2025.09.01.673507](https://doi.org/10.1101/2025.09.01.673507)

Self-contained manuscript repository for the behavioral motif, freezing, and
stress-coping dynamics analyses. All data required to rebuild the data-derived
figures, statistical CSVs, Excel reports, and validation manifest are included
in the repository.

<p align="center">
  <img src="docs/media/syllable_atlas.gif" alt="Representative pose-overlaid occurrences, one per behavioral cluster" width="870">
</p>
<p align="center"><sub>One representative keypoint-MoSeq syllable per behavioral
cluster, from the same verified sources as
<a href="supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4">Supplementary
Video 1 (grid view)</a>.</sub></p>

## Quick Start

The same commands work on Windows and Linux. Use the locked `uv` environment
when possible: it installs the Python recorded in `.python-version` itself, and
resolves from `uv.lock`, so every machine gets identical versions.

```bash
uv sync --locked
uv run python scripts/check_reproducibility.py
```

Alternative environments:

```bash
conda env create -f environment.yml
conda activate coping-dynamics
python scripts/check_reproducibility.py
```

```bash
python --version  # Python 3.10 or 3.11
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_reproducibility.py
```

## Rebuild Everything

Run commands from the repository root. The `uv run` prefix shown below is only
needed for the `uv` route; with conda or pip, activate the environment first and
call `python` directly.

```bash
uv run python scripts/run_all.py
```

This rebuilds the compact freezing-prediction tables, processed analysis
tables, data-derived figures, raw-data workbook, full and source-dataset
statistical report, artifact manifest, and
reproducibility checks. Add `--verbose` to stream each
child script's output. Add `--skip-figures` to rebuild tables, reports, the
manifest, and checks without re-rendering figures.

Targeted rebuilds:

```bash
uv run python scripts/run_all_figures.py
uv run python scripts/build_raw_data_workbook.py
uv run python scripts/build_statistical_report.py
uv run python scripts/update_manifest.py
uv run python scripts/check_reproducibility.py
uv run python scripts/check_manuscript.py
```

## Repository Contract

```text
data/raw/             Immutable bundled inputs used by the analyses
data/processed/       Deterministic tables regenerated from data/raw
figure_source_data/   Plotted values and summary values behind Figures 2-7
statistics/           Machine-readable statistical model outputs
figures/              Manuscript figure exports
report/               Raw-data workbook and statistical report
classifier/           Classifier artifact used in Figure 7A-C and Supplementary Figure 4
supplementary_media/  Submission-ready audiovisual supplement and source index
scripts/              Rebuild, report, figure, and validation entry points
scripts/migrations/   One-time imports from the legacy tree, kept for provenance
coping_dynamics/      Shared analysis, plotting, statistics, and classifier code (installed package)
docs/                 Data dictionary and focused provenance notes
config/               Figure metadata used by the report package
```

Two kinds of scripts are not part of a rebuild. `scripts/migrations/` holds the
one-time utilities that imported the legacy SHAP analysis; they need `--source`
pointing at the legacy keypoint-MoSeq tree, which is not part of this
repository. `scripts/update_bundled_manuscripts.py` rewrote the Results
statistics of the two manuscript copies in `report/` for one specific model
change and goes stale at the next edit; see its docstring.
`scripts/check_manuscript.py`, not that script, is what proves the copies are
current.

`data/raw/` contains inputs. `data/processed/`, `figure_source_data/`,
`statistics/`, `figures/`, and `report/` are regenerated products. The term
`figure_source_data` is used only for the plotted data underlying manuscript
figures, not for raw experimental inputs.

The statistical builder creates one output:

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

<p align="center">
  <img src="docs/media/figure_tour.gif" alt="Slideshow of the main data figures" width="560">
</p>
<p align="center"><sub>The main data figures, each rebuilt byte-identically by
<code>scripts/run_all.py</code>.</sub></p>

| Output | Script |
| --- | --- |
| Figure 1 | Tracked export, hand-assembled schematic |
| Figure 2 | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3 | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4 | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5 | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6 | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Figure 7 | `scripts/generate_figures/figure_7_resilience_prediction.py` |
| Supplementary Figure 1 | `scripts/generate_figures/supplementary_figure_1_tracking_clusters.py` |
| Supplementary Figure 2 | Tracked export, analysis workflow schematic |
| Supplementary Figure 3 | `scripts/generate_figures/supplementary_figure_3_distances.py` |
| Supplementary Figure 4 | `scripts/generate_figures/supplementary_figure_4_classifier_shap.py --source DIR` |
| Supplementary Figure 5 | Tracked export, convex-hull climbing validation |

Three figures were assembled by hand and are tracked as files rather than
rebuilt: Figure 1, Supplementary Figure 2 and Supplementary Figure 5.

Supplementary Figure 4 draws its panels from the archived classifier in
`classifier/legacy_shap/`, not from
`classifier/figure7_behavior_classifier.joblib`. See `docs/figure_structure.md`.

## Supplementary Media

`supplementary_media/` holds Supplementary Video 1 in two versions. The atlas
shows representative pose-overlaid syllables and skeleton trajectories. The grid
version plays all 25 syllables at once, ordered and coloured by behavioral
cluster. `report/SIGuide.docx` has the editable legend, and
`Supplementary_Video_1_source_index.csv` the source hashes both are checked
against.

Both are rebuilt from the archived keypoint-MoSeq clips:

```bash
python scripts/generate_supplementary_video_1.py   --clip-dir PATH/TO/video_clips   --skeleton-gif PATH/TO/skeleton_trajectories.gif
python scripts/generate_supplementary_video_1_grid.py   --clip-dir PATH/TO/video_clips
```

See `docs/supplementary_media.md`.

## Quality Controls

- `MANIFEST.csv` records sizes and SHA-256 hashes for tracked publication
  artifacts.
- `scripts/check_manuscript.py` checks every statistic in the manuscript
  against `statistics/`, `report/statistical_report.xlsx` and the significance
  markers on the figures. It fails if they disagree, or if the manuscript
  quotes a statistic it does not cover.
- `scripts/check_reproducibility.py` verifies required files, figure exports,
  the 98 freezing prediction CSVs and manifest hashes, and checks that retired
  paths, local absolute paths and local tool traces are absent.
- `tests/` pins the metric implementations in `coping_dynamics/statistics.py`
  to golden values, checks the implementations agree with each other, runs the
  layout checker, and rebuilds twice to confirm determinism. Run
  `uv run pytest -m "not slow"` to skip the slow rebuild.
- `.github/workflows/reproducibility.yml` runs three jobs: source guards
  (pre-commit, `uv lock --check`, citation metadata), locked-environment checks
  on Ubuntu and Windows, and a full rebuild with determinism tests, which is
  informational only.
- `.pre-commit-config.yaml` catches oversized files, conflict markers, malformed
  YAML, TOML and JSON, whitespace, and ruff lint and formatting. Enable it with
  `uvx pre-commit install`. It never touches generated artifacts.
- All default paths resolve inside the repository through
  `coping_dynamics/config.py`.

## Additional Documentation

- `REPRODUCIBILITY.md`: exact rebuild and validation workflow.
- `DATA_AVAILABILITY.md`: manuscript-facing data availability language.
- `CODE_AVAILABILITY.md`: manuscript-facing code availability language.
- `docs/data_dictionary.md`: file-level description of inputs and outputs.
- `docs/figure4_coping_provenance.md`: Figure 4 analysis provenance.
- `docs/figure_structure.md`: main and supplementary panel mapping.
- `docs/behavior_classifier.md`: classifier and Supplementary Figure 4 usage.
- `docs/manuscript_statistics_validation/`: independent validation of the
  statistics reported in the manuscript (Wald-identity checks plus mixed-model
  re-fits), with the HTML report and the scripts that rebuild it under
  `scripts/analysis/`.

## Citation and licensing

Cite the preprint.

```bibtex
@article{SanguinoGomez2025coping,
  title   = {Coping strategies dynamics and resilience profiles after early life stress revealed by behavioral sequencing},
  author  = {Sanguino-Gomez, Jeniffer and G{\"u}{\c{c}}l{\"u}, Umut and Krugers, Harm J. and Lozano, Antonio},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.09.01.673507},
  url     = {https://doi.org/10.1101/2025.09.01.673507}
}
```

Code is released under the MIT License (`LICENSE`). The bundled data —
`data/`, `figure_source_data/`, `statistics/`, `report/` and
`supplementary_media/` — are released under Creative Commons Attribution 4.0
International (`LICENSE-DATA`).
