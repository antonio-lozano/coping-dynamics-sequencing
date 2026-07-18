# Fig. 7 Behavior Classifier

This classifier uses the hand-curated Fig. 7 syllable annotations to train a
per-frame behavior classifier from the bundled MoSeq table:

```text
Freeze: 0, 28
Sniff: 18, 20
Groom: 24
Turn: 1, 3, 5, 6, 10, 15, 26, 27
Locomotion: 11, 12, 14, 16, 19, 21, 25
Climb: 111
Jump: 23, 29, 30, 34
Unassigned: all other syllables
```

The feature set mirrors the Fig. 7 XGBoost feature extraction in `src/behavior_classifier/features.py`:
centroid position, heading, angular velocity, velocity, absolute angular velocity,
rolling movement statistics, and short lags within each recording. When using
DLC tracks directly, the feature table also includes the SHAP-style movement
features noted in the project manifest: body tilt variability, angular velocity,
and global turning speed.

## Install

```powershell
uv pip install -r requirements.txt
```

## Quick Smoke Test

```powershell
python scripts\train_behavior_classifier.py train --skip-cv --max-frames 5000
```

## Full Training

```powershell
python scripts\train_behavior_classifier.py train
```

Outputs:

- `results/behavior_classifier/fig7_behavior_xgb.joblib`
- `results/intermediate/tables/fig7_behavior_classifier_cv_metrics.csv`
- `results/intermediate/tables/fig7_behavior_classifier_confusion_matrix.csv`

## Train From DLC Features

Use this when you have filtered DLC tracks and matching frame-level labels. Label
CSVs must contain `frame_index` plus either `syllable` or `behavior_label`.

```powershell
python scripts\train_behavior_classifier.py train-from-dlc `
  --dlc-dir path\to\filtered_dlc_tracks `
  --labels-dir path\to\frame_labels `
  --model results\behavior_classifier\fig7_behavior_xgb.joblib
```

## Predict

```powershell
python scripts\train_behavior_classifier.py predict `
  --model results\behavior_classifier\fig7_behavior_xgb.joblib `
  --output results\intermediate\tables\fig7_behavior_classifier_predictions.csv
```

## Raw Video To Predictions

This path mirrors the freezing-classifier project: run DLC, convert filtered DLC
tracks into the Fig. 7/SHAP-style pose summaries, then predict behaviors.

```powershell
python scripts\train_behavior_classifier.py run-from-raw `
  --video path\to\video.mp4 `
  --dlc-config path\to\dlc_config.yaml `
  --model results\behavior_classifier\fig7_behavior_xgb.joblib `
  --output-root results\behavior_classifier\example_video
```

If DLC has already produced a filtered CSV/H5 for the video:

```powershell
python scripts\train_behavior_classifier.py run-from-dlc `
  --dlc-file path\to\filtered_dlc_tracks.csv `
  --model results\behavior_classifier\fig7_behavior_xgb.joblib `
  --output-dir results\behavior_classifier\example_video\behavior_predictions
```
