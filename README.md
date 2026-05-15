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
Run scripts from repo root; outputs are written to `results/` as publication-ready PNG (600 dpi) and PDF files:

```bash
python scripts/generate_figures/figure_3_freezing.py
python scripts/generate_figures/figure_4_validation.py
python scripts/generate_figures/figure_5_clusters.py
python scripts/generate_figures/figure_6_diversity.py
python scripts/generate_figures/figure_7_dynamics.py
python scripts/generate_figures/figure_8_resilience.py
python scripts/generate_figures/figure_9_shap.py
```

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
|-- figures/                # Legacy figures directory
|-- results/                # Publication-ready figure outputs
|-- scripts/
|   |-- analysis/           # Analysis and model train/predict/QC scripts
|   |-- generate_figures/   # Figure generation scripts (figure_3 to figure_9)
|   `-- preprocessing/      # Data preprocessing utilities
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
