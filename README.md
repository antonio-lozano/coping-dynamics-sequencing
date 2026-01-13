# Coping Dynamics Sequencing

Unsupervised behavioral motif discovery and explainability for stress coping dynamics. This repository contains the figure-generation scripts, preprocessing utilities, and configuration needed to reproduce the analyses.

## Setup
1) Install conda, then create the environment:
```
conda env create -f environment.yml
conda activate coping-dynamics
```
2) Place data under `data/` following `data/README.md` (or set `COPING_DYNAMICS_DATA` to your data path).

## Running figures
Run scripts from the repo root; outputs are written to `figures/`:
```
python scripts/generate_figures/figure_3_freezing.py
python scripts/generate_figures/figure_4_validation.py
python scripts/generate_figures/figure_5_clusters.py
python scripts/generate_figures/figure_6_diversity.py
python scripts/generate_figures/figure_7_dynamics.py
python scripts/generate_figures/figure_8_resilience.py
python scripts/generate_figures/figure_9_shap.py
```

## Data
Data are not stored in the repo. See `data/README.md` for expected layout and download instructions.

## License
See LICENSE.
