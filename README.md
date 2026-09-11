<p align="center">
  <img src="docs/media/logo.png" alt="Coping Dynamics Sequencing" width="700">
</p>

# Coping Dynamics Sequencing

[![Checks](https://github.com/antonio-lozano/coping-dynamics-sequencing/actions/workflows/reproducibility.yml/badge.svg?branch=main)](https://github.com/antonio-lozano/coping-dynamics-sequencing/actions/workflows/reproducibility.yml)
[![Python 3.10–3.11](https://img.shields.io/badge/Python-3.10%E2%80%933.11-3776AB)](docs/installation.md)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC_BY_4.0-green)](LICENSE-DATA)

[Documentation](docs/index.md) · [Quickstart](#start-here) · [Features](#what-you-can-do) · [Repository map](docs/repository-structure.md) · [Changelog](CHANGELOG.md) · [Cite](#citation-and-licensing)

<p align="center">
  <a href="supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4"><img src="docs/media/syllable_atlas.gif" alt="Animated atlas of representative pose-overlaid behavioral motifs" width="870"></a>
</p>
<p align="center"><sub>Explore the behavioral repertoire. <a href="supplementary_media/Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4">Watch the full atlas video</a>.</sub></p>

**From behavioral sequences to inspectable analyses.** Explore the study, run a real-data example, or use the companion desktop classifier.

[First recording](docs/first-recording.md) · [Installation](docs/installation.md) · [Visual app](apps/behavior-dlc-classifier/GUIDE.md#the-desktop-interface) · [Research website](https://umguec.github.io/coping-dynamics-sequencing/) · [Study data](DATA_AVAILABILITY.md)

Research code supporting **“Coping strategies dynamics and resilience profiles after early-life stress revealed by behavioral sequencing.”**

## What you can do

| Workflow | What you get | Start |
| --- | --- | --- |
| Classify a recording | Per-frame labels, counts and a recorded run configuration | [Real-data quickstart](docs/first-recording.md) |
| Inspect coping dynamics | Figure gallery and an animal-level time-course explorer | [Research website](https://umguec.github.io/coping-dynamics-sequencing/) |
| Analyze tracked video | Desktop interface, behavior summaries and annotated video | [App guide](apps/behavior-dlc-classifier/GUIDE.md) |
| Reproduce downstream analyses | Isolated figures, tables, workbooks and comparison report | [Reproduction tutorial](docs/tutorials.md) |
| Build on the methods | Shared metrics and classifier Python APIs | [API reference](docs/api.md) |

## Behavior Studio

<a href="docs/assets/behavior-studio-results.png"><img src="docs/assets/behavior-studio-results.png" alt="Behavior Studio browsing a completed recording analysis" width="1000"></a>

[View full-resolution screenshot](docs/assets/behavior-studio-results.png)

A four-tab desktop workspace for raw video or existing tracking, model settings,
live progress, saved results and freezing-model refinement.
[Explore the app](apps/behavior-dlc-classifier/README.md) · [Desktop walkthrough](apps/behavior-dlc-classifier/GUIDE.md#the-desktop-interface)

## Start here

Try one complete recording from the bundled study data. No GPU, video or DeepLabCut installation is needed. Use a source checkout: a wheel does **not** contain the study data, figure scripts or trained models.

```bash
git clone --branch main https://github.com/antonio-lozano/coping-dynamics-sequencing.git
cd coping-dynamics-sequencing
uv sync --locked
uv run python scripts/check_reproducibility.py
uv run python docs/examples/first_recording.py --output ../coping-first-recording
```

The example writes `predictions.csv`, `label_counts.csv` and `run.json` into a new directory outside the checkout. An existing destination is refused. See the [first-recording walkthrough](docs/first-recording.md) to inspect the outputs.

Using an installed wheel instead? Supply explicit `behavior-classifier predict --input /path/to/moseq.csv --model /path/to/figure7_behavior_classifier.joblib --output /path/to/new/predictions.csv` paths to your own compatible assets and destination. The [first-recording example](docs/first-recording.md) requires a checkout; the [classifier tutorial](docs/tutorials.md#4-apply-the-root-behavior-classifier-to-bundled-moseq-data) explains saved-model inference.

| Your goal | Guide |
| --- | --- |
| Try one complete recording without changing study outputs | [First-recording walkthrough](docs/first-recording.md) |
| Install the locked study environment | [Installation](docs/installation.md) |
| Verify inputs, rebuild figures or use a classifier | [Step-by-step tutorials](docs/tutorials.md) |
| Understand what reproduction does and does not establish | [Reproducibility](REPRODUCIBILITY.md) |
| Use the shared Python functions | [API guide](docs/api.md) |
| Change code without obscuring scientific differences | [Contributing](CONTRIBUTING.md) |

## Follow the data

```text
Bundled per-frame tables → features and sequence metrics → figures + workbooks
                         ↘ saved-model inference → frame labels + probabilities
Video + matching DLC tracks → companion app → summaries + annotated video
```

The root environment does not install or fit keypoint-MoSeq; it consumes bundled sequence tables. For new videos, use the [companion installation and hardware guide](apps/behavior-dlc-classifier/GUIDE.md) in its separate environment.

These are distinct workflows. The manuscript rebuild does not execute raw-video tracking or retrain the historical classifiers. [Choose a workflow](docs/tutorials.md) before selecting an environment.

## Study data

Raw recordings, pose estimates, keypoint-MoSeq outputs and supplementary media are available on [Figshare](https://doi.org/10.6084/m9.figshare.33439885). Download just the archives you need:

```bash
uv run python -m coping_dynamics.datasets --list
uv run python -m coping_dynamics.datasets freezing_predictions --output ../coping-data --extract
```

The bundled tables are sufficient for the first-recording example and downstream figure workflows. Use the [dataset guide](docs/datasets.md) for Figshare availability, archive contents and app input paths. Large recordings and app weights require authorized access during the embargo.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Bundled source inputs and derived compact companions; preserve provenance |
| `data/processed/` | Analysis-ready generated tables |
| `figure_source_data/` | Plotted values underlying manuscript panels |
| `statistics/` | Machine-readable statistical outputs |
| `figures/` | Generated and assembled manuscript figures |
| `report/` | Study workbooks and supplementary reports |
| `coping_dynamics/` | Shared Python analysis and classifier code |
| `scripts/` | Checkout-based rebuild and validation commands |
| `scripts/migrations/` | Historical imports, not routine rebuild steps |
| `apps/behavior-dlc-classifier/` | Independently packaged desktop and video application |
| `docs/` | Installation, tutorials, API and provenance guides |
| `tests/` | Root unit, repository-contract and selected determinism checks |

## Development checks

```bash
uv run pytest -m "not slow" -q
uv run ruff check coping_dynamics scripts tests behavior_classifier.py
```

The current slow determinism tests rebuild in isolated temporary copies and verify that source files stay unchanged. Direct figure/report generator commands still write in place; use the [isolated reproduction workflow](REPRODUCIBILITY.md) for complete runs.

## Contributors

Developed by Jeniffer Sanguino-Gómez, [Antonio Lozano](https://github.com/antonio-lozano), and [Umut Güçlü](https://github.com/umguec). See [all code contributors](https://github.com/antonio-lozano/coping-dynamics-sequencing/graphs/contributors).

## Citation and licensing

If you use this software or study data, please cite the preprint:

**Sanguino-Gómez J, Güçlü U, Krugers H, Lozano A (2025).** Coping strategies dynamics and resilience profiles after early life stress revealed by behavioral sequencing. *bioRxiv*. [doi:10.1101/2025.09.01.673507](https://doi.org/10.1101/2025.09.01.673507).

```bibtex
@article{SanguinoGomez2025Coping,
  author = {Sanguino-G{\'o}mez, Jeniffer and G{\"u}{\c c}l{\"u}, Umut and Krugers, Harm and Lozano, Antonio},
  title = {Coping strategies dynamics and resilience profiles after early life stress revealed by behavioral sequencing},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.09.01.673507},
  url = {https://www.biorxiv.org/content/early/2025/09/04/2025.09.01.673507}
}
```

GitHub's **Cite this repository** button uses the same preprint via [CITATION.cff](CITATION.cff). Code is [MIT licensed](LICENSE); study data and media are [CC BY 4.0](LICENSE-DATA). Please retain author credits when adapting the software or reusing media.

## Model and example assets

Behavior Studio model weights and example recordings are distributed separately, not in Git history. During the data embargo, obtain the authorized `behavior-studio-assets.zip` from the authors and install it with:

```bash
python scripts/install_app_assets.py /path/to/behavior-studio-assets.zip
```

The installer verifies every file against the committed SHA-256 manifest before installation and never overwrites differing files. Alternatively, select your own compatible models and recordings in the app. The study dataset is identified by [Figshare DOI](https://doi.org/10.6084/m9.figshare.33439885); access follows its publication schedule.
