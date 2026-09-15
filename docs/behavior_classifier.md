# Experimental behavior classifier and archived model provenance

The reusable classifier predicts the same behavior labels as the paper, but it is **not** the historical model behind Figure 7A–C or Supplementary Figure 4. The reusable model has 33 centroid-derived features; archived SHAP uses 719 features. Figure 7A/C replay stored historical values and Figure 7B uses an extracted table. Regeneration is not retraining or independent cross-validation.

The label mapping is:

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

The bundled classifier artifact is:

```text
classifier/figure7_behavior_classifier.joblib
```

The feature extraction code is in `coping_dynamics/behavior_classifier/features.py`.

## Train Or Evaluate From The Bundled MoSeq Table

```bash
python behavior_classifier.py train --model classifier/outputs/experimental.joblib
```

Quick smoke run:

```bash
python behavior_classifier.py train --skip-cv --max-frames 5000 --model classifier/outputs/experimental-small.joblib
```

Training defaults now write under ignored `classifier/outputs/`; prediction continues to load the archived model by default. Pass a distinct destination for each experiment. Historical artifact names (not new training defaults) are:

- `classifier/figure7_behavior_classifier.joblib`
- `classifier/figure7_behavior_classifier_cv_metrics.csv`
- `classifier/figure7_behavior_classifier_confusion_matrix.csv`

## Train From DLC Tracks

Use this when filtered DLC tracks and matching frame-level labels are available.
Label CSVs must contain `frame_index` plus either `syllable` or
`behavior_label`.

```bash
python behavior_classifier.py train-from-dlc \
  --dlc-dir path/to/filtered_dlc_tracks \
  --labels-dir path/to/frame_labels \
  --model classifier/figure7_behavior_classifier.joblib
```

## Predict From A MoSeq Table

```bash
python behavior_classifier.py predict \
  --model classifier/figure7_behavior_classifier.joblib \
  --output classifier/outputs/figure7_behavior_classifier_predictions.csv
```

## Predict From DLC Tracks

From raw video, run DLC first and then apply the classifier:

```bash
python behavior_classifier.py run-from-raw \
  --video path/to/video.mp4 \
  --dlc-config path/to/dlc_config.yaml \
  --model classifier/figure7_behavior_classifier.joblib \
  --output-root classifier/outputs/raw_video_runs
```

If filtered DLC CSV/H5 files already exist:

```bash
python behavior_classifier.py run-from-dlc \
  --dlc-file path/to/filtered_dlc_tracks.csv \
  --model classifier/figure7_behavior_classifier.joblib \
  --output-dir classifier/outputs/behavior_predictions
```

## Training change and interpretation

Temporal features are now computed on complete recording sequences before label filtering or random frame subsampling. The old order manufactured adjacency between nonadjacent frames. This correction changes experimental retraining only; bundled model bytes and reported classifier scores remain unchanged. Root cross-validation still splits frames, not animals, and is not independent-animal validation. See [reproduction guidance](../REPRODUCIBILITY.md).

## Companion archived-model compatibility

The separate companion's 719-feature behavior model now uses the centered-only
coordinate frame evidenced by its archived feature rows. See the
[coordinate correction and real-data check](classifier-coordinate-correction.md).
This does not turn the root 33-feature model into the historical classifier.
