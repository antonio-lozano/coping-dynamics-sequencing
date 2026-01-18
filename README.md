# Coping Dynamics Sequencing

Unsupervised behavioral motif discovery and explainability for stress coping dynamics. This repository contains the figure-generation scripts, preprocessing utilities, and configuration needed to reproduce the analyses.

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
Run scripts from the repo root; outputs are written to `results/` as publication-ready PNG (600 dpi) and PDF files:
```bash
python scripts/generate_figures/figure_3_freezing.py
python scripts/generate_figures/figure_4_validation.py
python scripts/generate_figures/figure_5_clusters.py
python scripts/generate_figures/figure_6_diversity.py
python scripts/generate_figures/figure_7_dynamics.py
python scripts/generate_figures/figure_8_resilience.py
python scripts/generate_figures/figure_9_shap.py
```

## Project Structure
```
├── data/               # Raw and processed data (not in git)
├── docs/               # Documentation (data preparation, reproduction)
├── figures/            # Legacy figures directory
├── results/            # Publication-ready figure outputs (PNG, PDF)
├── scripts/
│   ├── analysis/       # Analysis scripts (resilience_analysis.py)
│   ├── generate_figures/  # Figure generation scripts (figure_3 to figure_9)
│   └── preprocessing/  # Data preprocessing utilities
├── src/
│   ├── config.py       # Centralized paths, constants, and color palette
│   ├── io_utils.py     # I/O helpers
│   ├── metrics.py      # Metric computation utilities
│   ├── plotting.py     # Plotting utilities
│   └── stats_utils.py  # Statistical testing utilities
├── environment.yml     # Conda environment specification
├── requirements.txt    # pip requirements (alternative to conda)
└── README.md
```

## Data
Data are not stored in the repo. See `data/README.md` for expected layout and download instructions.

## License
See LICENSE.
