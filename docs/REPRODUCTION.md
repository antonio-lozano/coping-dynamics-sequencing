# Reproduction Guide

1) `conda env create -f environment.yml`
2) `conda activate coping-dynamics`
3) Place data under `data/` (or set `COPING_DYNAMICS_DATA`).
4) Run figure scripts from repo root:
   - `python scripts/generate_figures/figure_3_freezing.py`
   - `python scripts/generate_figures/figure_4_validation.py`
   - `python scripts/generate_figures/figure_5_clusters.py`
   - `python scripts/generate_figures/figure_6_diversity.py`
   - `python scripts/generate_figures/figure_7_dynamics.py`
   - `python scripts/generate_figures/figure_8_resilience.py`
   - `python scripts/generate_figures/figure_9_shap.py`
5) Outputs are saved to `results/`.

For model training/inference on new DLC datasets, see `docs/MODEL_INFERENCE.md`.
