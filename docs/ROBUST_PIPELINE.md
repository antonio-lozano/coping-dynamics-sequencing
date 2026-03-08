# Robust XGBoost Behavior Prediction Pipeline

This document explains the **robust pretraining → inference → visualization** pipeline for predicting mouse behavior from DeepLabCut (DLC) pose tracking data using XGBoost.

---

## Overview

The pipeline has three stages:

```
┌──────────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐
│  1. PRETRAIN (JEN data)  │ ──▶ │  2. PREDICT (new data)   │ ──▶ │  3. VISUALIZE / LOAD    │
│  pretrain_xgb_robust.py  │     │  predict_behavior_xgb.py │     │  Python / pandas        │
└──────────────────────────┘     └──────────────────────────┘     └─────────────────────────┘
```

**Goal**: Train a behavior classifier on JEN's labeled DLC data (which has MoSeq syllable labels), then deploy it on new DLC recordings (e.g., AYA's data) that have no behavior labels.

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Only common bodyparts** | JEN has 14 bodyparts, AYA may have fewer. Train only on bodyparts present in both; missing ones → NaN → 0. Use `--exclude-bodyparts` to drop non-shared bodyparts (e.g., tail). |
| **Pose augmentation** | Mild rotation (±15°), translation (±5%), scale jitter (±10%) applied to raw x/y before feature extraction. Simulates camera/rig differences between labs. |
| **Extended features** | Temporal context (lags, deltas, acceleration), postural shape (body area, elongation, symmetry, curvature), frequency-domain (FFT energy). Makes the model richer without changing `src/`. |
| **Global normalization** | Median/IQR computed over *all* JEN training frames, saved as `normalization_stats.json`, applied *identically* at inference. Ensures train/test scales match. |
| **No `src/` changes** | All new code lives in `scripts/analysis/`. The core library is untouched. |

---

## Inputs and outputs

### Training inputs

| Input | Description | Example path |
|-------|-------------|-------------|
| `--dlc-dir` | Directory of JEN DLC CSV files | `equipo_project_data/CSVs_all/` |
| `--results-clusters-pkl` | MoSeq syllable labels (pickle) | `equipo_project_data/new_results_clusters.pkl` |
| `--index-csv` | Recording → group mapping | `equipo_project/index.csv` |

Each DLC CSV has columns like `nose_x, nose_y, H1R_x, H1R_y, ...` (14 bodyparts × 2 coordinates).

### Training outputs (saved in `--out-dir / --model-version`)

| File | Description |
|------|-------------|
| `model.json` | XGBoost model (JSON format) |
| `model.pkl` | XGBoost model (pickle fallback) |
| `metadata.json` | Feature names, class names, normalization config, augmentation params, training provenance |
| `normalization_stats.json` | Per-feature `{median, q25, q75, iqr, scale_used}` from training data |
| `cv_metrics.json` | Cross-validation accuracy, F1, confusion matrices |
| `cv_confusion_matrix_oof.png` | Visual confusion matrix |
| `cv_classification_report_oof.csv` | Per-class precision/recall/F1 |

### Inference inputs

| Input | Description |
|-------|-------------|
| `--model-dir` | Path to trained model directory (contains `metadata.json`, `model.json`, `normalization_stats.json`) |
| `--input-dir` | Directory of new DLC CSV files to classify |

### Inference outputs (saved in `--out-dir / --dataset-id / model-version`)

| File | Description |
|------|-------------|
| `predictions_frame.parquet` | Per-frame predictions: recording, frame, predicted class, confidence, per-class probabilities |
| `predictions_frame.csv` | Same as above in CSV format |
| `predictions_summary.csv` | Per-recording summary: frame count, mean/median confidence, % low confidence, % per behavior class |
| `computed_features_raw.parquet` | Per-frame raw computed features before normalization/alignment |
| `computed_features_raw.csv` | CSV version of raw computed features |
| `computed_features_model_aligned.parquet` | Per-frame model-ready features after normalization + schema alignment |
| `computed_features_model_aligned.csv` | CSV version of model-ready aligned features |

---

## Where is everything?

### Source data

| What | Path | Contents |
|------|------|----------|
| **JEN DLC CSVs** (training) | `C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\CSVs_all\` | 98 DLC CSV files (14 bodyparts × x/y/likelihood, ~11,250 frames each) |
| **MoSeq syllable labels** | `C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\new_results_clusters.pkl` | Pickle dict mapping recording names → syllable arrays |
| **Recording metadata** | `C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\index.csv` | Recording name → group (Control / ELS) |
| **AYA DLC CSVs** (mapped) | `data/to_predict/matched_dlc_aya_to_jen/` | 9 AYA recordings mapped to JEN bodypart schema (7,200 frames each) |
| **AYA DLC CSVs** (subset) | `data/to_predict/matched_dlc_aya_to_jen_d1t3_d2t1/` | 3 AYA recordings (Day1-Trial3, Day2-Trial1 subset) |

### Trained models

All models live under `results/model_training/models/xgb_behavior/`:

| Model version | Description | OOF Accuracy | Macro F1 |
|---------------|-------------|--------------|----------|
| `robust_light_no_tail_v2/` | **Recommended.** 10 JEN recordings, 2 augmentations, tail excluded, 500 trees, global norm | 89.8% | 0.903 |
| `robust_light_no_tail_v1/` | Same but `--quick` (10 trees) — for smoke-testing only | 64.8% | 0.642 |
| `2026-02-10_final_gpu_full_no_tail_v1/` | Full 98 recordings, no tail, per-recording norm (legacy) | — | — |
| `2026-02-10_final_gpu_3min_norm_v1/` | All bodyparts, per-recording norm (legacy) | — | — |

Each model directory contains:

```
robust_light_no_tail_v2/
├── model.json                              # XGBoost model
├── model.pkl                               # Pickle fallback
├── metadata.json                           # Feature names, classes, normalization config
├── normalization_stats.json                # Global median/IQR per feature (710 features)
├── cv_metrics.json                         # Cross-validation metrics
├── cv_confusion_matrix_oof.png             # Confusion matrix plot
├── cv_confusion_matrix_oof_counts.csv      # Confusion matrix (counts)
├── cv_confusion_matrix_oof_normalized.csv  # Confusion matrix (row-normalized)
└── cv_classification_report_oof.csv        # Per-class precision/recall/F1
```

### Predictions

All predictions live under `results/predictions/`:

| Dataset | Model used | Path |
|---------|------------|------|
| **AYA (robust v2)** | `robust_light_no_tail_v2` | `results/predictions/aya_robust_v2/robust_light_no_tail_v2/` |
| **AYA (robust v1 quick)** | `robust_light_no_tail_v1` | `results/predictions/aya_robust/robust_light_no_tail_v1/` |
| **AYA (legacy, no tail)** | various legacy models | `results/predictions/matched_dlc_aya_to_jen_no_tail/` etc. |
| **Test set** | various | `results/predictions/test*/` |

Each prediction directory contains:

```
aya_robust_v2/robust_light_no_tail_v2/
├── predictions_frame.parquet               # Per-frame: recording, frame, pred_class, confidence, probabilities
├── predictions_frame.csv                   # Same in CSV
├── predictions_summary.csv                 # Per-recording summary stats
├── computed_features_raw.parquet           # Raw kinematic + extended features (before normalization)
├── computed_features_raw.csv               # Same in CSV
├── computed_features_model_aligned.parquet  # Normalized + schema-aligned features fed to model
└── computed_features_model_aligned.csv     # Same in CSV
```

### Latest AYA results at a glance

Model `robust_light_no_tail_v2` on 9 AYA recordings (64,800 frames):

| Metric | Value |
|--------|-------|
| Mean confidence | 0.582 |
| Median confidence | 0.583 |

| Behavior | % of frames |
|----------|-------------|
| Freezing | 58.6% |
| Unassigned | 27.6% |
| Turn | 13.8% |

---

## Quick start

### 1. Pretrain on JEN data (light run with 10 recordings)

```bash
python scripts/analysis/pretrain_xgb_robust.py \
  --max-recordings 10 \
  --n-augmentations 2 \
  --exclude-bodyparts "tail" \
  --model-version robust_light_no_tail_v2 \
  --max-samples-per-class 5000
```

**Quick smoke test** (tiny model, no augmentation):

```bash
python scripts/analysis/pretrain_xgb_robust.py \
  --max-recordings 10 \
  --exclude-bodyparts "tail" \
  --model-version robust_light_no_tail_v1 \
  --quick
```

**Full training** (all 98 JEN recordings, GPU):

```bash
python scripts/analysis/pretrain_xgb_robust.py \
  --n-augmentations 3 \
  --exclude-bodyparts "tail" \
  --model-version robust_full_v1 \
  --device cuda \
  --tree-method hist
```

### 2. Predict on new (AYA) data

```bash
python scripts/analysis/predict_behavior_xgb.py \
  --model-dir results/model_training/models/xgb_behavior/robust_light_no_tail_v2 \
  --input-dir data/to_predict/matched_dlc_aya_to_jen \
  --dataset-id aya_robust_v2 \
  --glob "*.csv"
```

### 3. Visualize predictions

```bash
python scripts/analysis/visualize_behavior_predictions.py \
  --predictions-dir results/predictions/aya_robust_v2/robust_light_no_tail_v2
```

---

## How to load and explore the outputs in Python

### Load frame-level predictions

```python
import pandas as pd

# Load predictions (parquet is faster; CSV also available)
pred = pd.read_parquet("results/predictions/aya_robust_v2/robust_light_no_tail_v2/predictions_frame.parquet")

print(pred.columns.tolist())
# ['recording', 'frame', 'pred_class', 'pred_confidence',
#  'p_Climbing', 'p_Freezing', 'p_Grooming', 'p_Jump',
#  'p_Locomotion', 'p_Sniffing', 'p_Turn', 'p_Unassigned']

print(pred.head())
```

### Load computed feature tables

```python
import pandas as pd

raw_feat = pd.read_parquet(
    "results/predictions/aya_robust_v2/robust_light_no_tail_v2/computed_features_raw.parquet"
)
aligned_feat = pd.read_parquet(
    "results/predictions/aya_robust_v2/robust_light_no_tail_v2/computed_features_model_aligned.parquet"
)

print(raw_feat.shape, aligned_feat.shape)
print(raw_feat.columns[:8].tolist())
```

### Load per-recording summary

```python
summary = pd.read_csv("results/predictions/aya_robust_v2/robust_light_no_tail_v2/predictions_summary.csv")
print(summary)
#                            recording  n_frames  mean_confidence  ...  %_Freezing  %_Locomotion
# 0  Day1-Trial1-Mouse3DLC_...            18000           0.45   ...        32.1          18.5
```

### Plot behavior composition per recording

```python
import matplotlib.pyplot as plt

behavior_cols = [c for c in summary.columns if c.startswith("%_")]
summary.set_index("recording")[behavior_cols].plot(
    kind="barh", stacked=True, figsize=(12, 6), colormap="Set2"
)
plt.xlabel("% of frames")
plt.title("Predicted behavior composition per recording")
plt.tight_layout()
plt.savefig("behavior_composition.png", dpi=150)
plt.show()
```

### Plot behavior over time for a single recording

```python
import matplotlib.pyplot as plt
import numpy as np

rec_name = pred["recording"].unique()[0]
rec = pred[pred["recording"] == rec_name].copy()

# Map behaviors to numeric codes for visualization  
behaviors = sorted(rec["pred_class"].unique())
behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
rec["behavior_idx"] = rec["pred_class"].map(behavior_to_idx)

fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

# Ethogram (color strip)
cmap = plt.cm.get_cmap("tab10", len(behaviors))
for i, b in enumerate(behaviors):
    mask = rec["pred_class"] == b
    axes[0].fill_between(rec["frame"], 0, 1, where=mask, color=cmap(i), label=b, alpha=0.8)
axes[0].set_ylabel("Behavior")
axes[0].set_yticks([])
axes[0].legend(loc="upper right", fontsize=7, ncol=4)

# Confidence trace
axes[1].plot(rec["frame"], rec["pred_confidence"], linewidth=0.5, color="steelblue")
axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="50% threshold")
axes[1].set_ylabel("Confidence")
axes[1].set_xlabel("Frame")
axes[1].legend()

plt.suptitle(f"Behavior predictions: {rec_name}", fontsize=11)
plt.tight_layout()
plt.savefig("ethogram_example.png", dpi=150)
plt.show()
```

### Plot confidence distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=pred, x="pred_confidence", hue="pred_class", 
             bins=50, stat="density", common_norm=False, ax=ax)
ax.axvline(0.5, color="red", linestyle="--", label="50% threshold")
ax.set_title("Prediction confidence by behavior class")
ax.legend()
plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=150)
plt.show()
```

### Load normalization stats (for debugging / QC)

```python
import json

with open("results/model_training/models/xgb_behavior/robust_light_no_tail_v2/normalization_stats.json") as f:
    norm_stats = json.load(f)

# Show stats for a few features
for feat in ["Summed Velocity", "Angular Velocity", "Body Area"]:
    if feat in norm_stats:
        print(f"{feat}: median={norm_stats[feat]['median']:.2f}, IQR={norm_stats[feat]['iqr']:.2f}")
```

### Load model metadata (check what the model was trained on)

```python
import json

with open("results/model_training/models/xgb_behavior/robust_light_no_tail_v2/metadata.json") as f:
    meta = json.load(f)

print("Classes:", meta["class_names"])
print("Features:", len(meta["feature_names"]))
print("Normalization:", meta["feature_normalization"]["method"])
print("Augmentation:", meta["augmentation"])
print("Excluded bodyparts:", meta["excluded_bodyparts"])
print("Training recordings:", meta["training_sources"]["n_recordings_original"])
```

### Compare model CV performance

```python
import json

with open("results/model_training/models/xgb_behavior/robust_light_no_tail_v2/cv_metrics.json") as f:
    cv = json.load(f)

print(f"Overall accuracy:   {cv['overall_accuracy_oof']:.3f}")
print(f"Balanced accuracy:  {cv['balanced_accuracy_oof']:.3f}")
print(f"Macro F1:           {cv['macro_f1_oof']:.3f}")
print(f"Log loss:           {cv['log_loss_oof']:.3f}")

# Per-class breakdown
for cls, prf in cv["per_class_prf_oof"].items():
    print(f"  {cls:15s}  P={prf['precision']:.2f}  R={prf['recall']:.2f}  F1={prf['f1']:.2f}  n={prf['support']}")
```

---

## File reference

| Script | Purpose |
|--------|---------|
| `scripts/analysis/pretrain_xgb_robust.py` | Train with augmentation + extended features + global normalization |
| `scripts/analysis/predict_behavior_xgb.py` | Run inference on new DLC data (auto-detects normalization method) |
| `scripts/analysis/visualize_behavior_predictions.py` | Generate QC figures and skeleton-overlay videos |
| `scripts/analysis/extended_features.py` | Helper module: augmentation, extended features, global normalization |
| `scripts/analysis/smoke_test_pretrain_robust.py` | End-to-end smoke test with synthetic data |

---

## Behavior classes

| Code | Behavior | Description |
|------|----------|-------------|
| 1 | Freezing | Immobility |
| 2 | Sniffing | Exploratory sniffing |
| 3 | Grooming | Self-grooming |
| 4 | Turn | Body turns |
| 5 | Locomotion | Walking / running |
| 6 | Climbing | Climbing walls |
| 7 | Jump | Jumping |
| — | Unassigned | Syllables not mapped to a behavior |

---

## Feature groups

The model uses ~200–500 features depending on configuration:

| Group | Examples | Count (approx.) |
|-------|----------|-----------------|
| **Pairwise distances** | `Nose - Tail Distance`, `Head Right 1 - Left Body 2 Distance` | ~78–91 |
| **Bodypart velocities** | `Nose Velocity`, `Tail Velocity` | ~13–14 |
| **Angular velocity** | `Angular Velocity`, `SD Angular Velocity` | 2 |
| **Summed velocity** | `Summed Velocity` | 1 |
| **Center body** | `Center Body X`, `Center Body Y`, `Nose - Center Body Distance` | ~15 |
| **Rolling stats** | `Mean Nose Velocity`, `SD Nose - Tail Distance` | 2× above |
| **Tail variability** | `Tail Variability`, `Right Body 3 - Tail Variability` | ~4 |
| **Temporal (extended)** | `Summed Velocity Lag1`, `Delta Nose Velocity`, `Accel Angular Velocity` | ~50+ |
| **Postural (extended)** | `Body Area`, `Elongation Ratio`, `Symmetry Body1`, `Body Curvature` | ~8 |
| **Frequency (extended)** | `Summed Velocity FFT Low Energy`, `Angular Velocity Dominant Freq` | ~6 |
| **Rolling stats of extended** | `Mean Body Area`, `SD Elongation Ratio`, ... | 2× extended |

---

## Normalization

**Why median/IQR (robust) instead of standard mean/std?**  
DLC tracking produces outlier spikes (dropped keypoints, identity swaps). Mean and std are sensitive to these; median and IQR are not.

**Why clipping to ±4?**  
Features with near-zero IQR (e.g., a nearly-constant distance) would produce extreme normalized values that dominate XGBoost tree splits. Clipping prevents this.

**Why global instead of per-recording?**  
Per-recording normalization recomputes statistics from each recording independently. For short clips or clips dominated by one behavior, this shifts the feature distribution unpredictably. Global normalization uses the *same* statistics (from training data) everywhere, ensuring consistency.

**Stats are saved** in `normalization_stats.json` alongside the model and loaded automatically at inference.
