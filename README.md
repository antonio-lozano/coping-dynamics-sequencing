# Coping Dynamics Sequencing

Unsupervised behavioral motif discovery and explainability for stress coping dynamics. This repository contains figure-generation scripts, preprocessing utilities, and model training/inference workflows.

## Setup

### Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate coping-dynamics
```

### Using pip
```bash
pip install -r requirements.txt
```

Place data under `data/` following `data/README.md` (or set `COPING_DYNAMICS_DATA` to your data path).

## Running figures
Run scripts from the repo root. Each script writes the clean manuscript figure
(`figureN.pdf/.svg/.png`) to the top-level `figures/` directory, and supporting
tables / intermediate renders to `figures/data/`.

| Manuscript figure | Script |
| --- | --- |
| Figure 2 — Validation (keypoint-MoSeq) | `scripts/generate_figures/figure_2_validation.py` |
| Figure 3 — Behavior clusters over time | `scripts/generate_figures/figure_3_behavior_clusters.py` |
| Figure 4 — Diversity dynamics | `scripts/generate_figures/figure_4_diversity_dynamics.py` |
| Figure 5 — Resilience dynamics | `scripts/generate_figures/figure_5_resilience_dynamics.py` |
| Figure 6 — Resilience diversity | `scripts/generate_figures/figure_6_resilience_diversity.py` |
| Supplementary Figure 3 — Distance metrics | `scripts/generate_figures/supplementary_figure_3_distances.py` |

Regenerate all figures at once:

```bash
python scripts/run_all_figures.py
```

(Figure 1 is a hand-made schematic with no repo generator.)

## Model training and inference
Use the XGBoost workflow to train on labeled DLC data and predict behavior on new DLC datasets.

See `docs/MODEL_INFERENCE.md` for full commands and output layout.

Main scripts:
- `scripts/analysis/train_behavior_xgb.py`
- `scripts/analysis/predict_behavior_xgb.py`
- `scripts/analysis/visualize_behavior_predictions.py`

## GUI viewers
Desktop viewers live under `gui/`.

- DearPyGui mapping explorer:
  `python gui/mapping_explorer_dpg/app.py`
- Minimal PyQtGraph comparison viewer:
  `python gui/app_visual.py`

The PyQtGraph viewer loads left/right videos explicitly, plus pose files (`.csv` or `.h5`) and optional feature tables (`.csv` or `.parquet`).

## Project structure
```
|-- data/                   # Raw + processed data (not in git)
|-- docs/                   # Reproduction and workflow docs
|-- figures/                # Canonical manuscript figures (figureN.*) + data/ provenance
|-- results/                # Model training + prediction outputs, QC
|-- scripts/
|   |-- analysis/           # Analysis and model train/predict/QC scripts
|   |-- data_exports/       # Data workbook exporters
|   |-- generate_figures/   # One script per manuscript figure (figure_2..6, supplementary_figure_3)
|   |-- preprocessing/      # Data preprocessing utilities
|   `-- run_all_figures.py  # Regenerate every figure
|-- src/
|   |-- config.py           # Centralized paths and constants
|   |-- ml/                 # Reusable pose features + XGBoost helpers
|   |-- plotting.py         # Plotting utilities
|   |-- stats_utils.py      # Statistical testing helpers
|   `-- transition_utils.py # Transition/BFL utilities
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Data
Data are not stored in the repo. See `data/README.md` for expected layout.

## License
See `LICENSE`.
