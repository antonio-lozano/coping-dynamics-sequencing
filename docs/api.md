# Python API guide

The shared package is `coping_dynamics`. Manuscript figure/report orchestration remains in checkout-based `scripts/`; those scripts are not installed by a wheel. This page describes selected existing functions, not a stability guarantee for every internal helper.

## DeepLabCut API boundary

`coping_dynamics` does not expose a DeepLabCut pose-estimation API. Use the
[companion tracking CLI](../apps/behavior-dlc-classifier/GUIDE.md#tracking-new-videos)
for the bundled network, and [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/intro.html)
for its own Python API. Root MoSeq inference is not a replacement for pose tracking.

## Sequence metrics

Implementations: [statistics.py](../coping_dynamics/statistics.py). Supply an ordered sequence from one animal/recording at a defined temporal resolution. Do not concatenate different animals or silently remove unmapped bins: doing so changes temporal metrics.

```python
from coping_dynamics.statistics import (
    compute_diversity_metrics,
    compute_bout_duration,
)

sequence = ["Freezing", "Freezing", "Locomotion", "Locomotion"]
metrics = compute_diversity_metrics(sequence)
bouts = compute_bout_duration(sequence)
print(metrics)
print(bouts)
```

For this illustrative two-state sequence, Shannon entropy is `ln(2)`, Simpson diversity is `0.5`, and each bout lasts `0.5` seconds under the function's fixed 250-ms-bin convention. This is an API illustration, not study evidence.

| Function | Return and conventions |
| --- | --- |
| `compute_diversity_metrics(sequence)` | Dictionary: `shannon_entropy_index` (natural logarithms), `simpson_index`, `evenness_index`, `cumulative_usage_index` |
| `compute_bout_duration(sequence)` | DataFrame of individual contiguous bouts: `cluster`, `bout_duration_seconds`; each element represents 0.25 seconds, not a 25-fps video frame |
| `lempel_ziv_complexity(sequence)` | Integer from this repository's token-parsing implementation; do not substitute a similarly named library metric without checking conventions |
| `recurrence_rate(sequence)` | Recurrence statistic using the implemented equality matrix convention |
| `determinism(sequence, min_length=2)` | Recurrence diagonal-line statistic using the specified minimum length |
| `markov_entropy(sequence, smoothing_factor=0.01)` | Occupancy-weighted transition entropy in bits with additive smoothing |
| `transition_sequence_metrics(sequence)` | Dictionary of Lempel–Ziv complexity, recurrence rate, determinism and Markov entropy |

Missing/degenerate cases are method-specific. Diversity evenness and cumulative usage can be undefined for degenerate inputs. An empty bout sequence returns an empty DataFrame. Do not globally replace undefined statistics with zero. Confirm current edge-case behavior against [metric tests](../tests/test_statistics_metrics.py) before applying these functions to a different dataset.

`cohens_d(a, b)` accepts pandas Series, coerces numeric values and drops invalid/missing observations separately in each group. It returns the first-minus-second mean difference divided by the pooled sample standard deviation (`ddof=1`), or NaN when insufficient/zero-variance data prevent estimation.

Model-fitting helpers in `statistics.py` embody study-specific assumptions. Review their formulas, clustering/group construction and fallback behavior with the original manuscript before reusing them; a fitted model object is not evidence that the scientific model matches the paper.

## Behavior classifier

Implementations: [pipeline](../coping_dynamics/behavior_classifier/pipeline.py) and [feature extraction](../coping_dynamics/behavior_classifier/features.py).

```python
from pathlib import Path
from coping_dynamics.behavior_classifier import (
    load_bundle,
    load_moseq_table,
    predict_behaviors,
)

root = Path("/path/to/your/checkout")
bundle = load_bundle(root / "classifier/figure7_behavior_classifier.joblib")
frames = load_moseq_table(root / "data/raw/moseq_syllables_per_frame.csv.gz")
predictions = predict_behaviors(frames, bundle)
print(predictions.head())
```

Replace the example root with your checkout location. Use only trusted model artifacts: joblib deserialization is not a safe format for untrusted downloads.

- `load_moseq_table(path)` loads the per-frame table used by this classifier.
- `load_bundle(path)` returns a `BehaviorClassifierBundle`; it raises `TypeError` for an incompatible deserialized object type.
- `predict_behaviors(df, bundle, id_columns=("name", "frame_index"))` returns sorted identifier columns plus `predicted_behavior`; supported models also provide `predicted_probability` and per-class `prob_*` columns.
- `build_frame_features(df, group_col="name", frame_col="frame_index", windows=(5, 15, 30), lags=(1, 5, 15))` returns the feature DataFrame and ordered feature-name list. Required base columns include centroid coordinates, heading, angular velocity and pixel velocity. Missing required columns raise `ValueError`; the trained bundle can require additional optional columns.

Feature extraction sorts by recording/frame, computes within-recording rolling and lagged features, and fills non-finite feature values with zero. Consequently, missing frames or filtering the table before extraction can change temporal features. Verify units, feature schema and preprocessing before applying a model to DLC-derived inputs.

Training functions are available but are not a drop-in independent validation protocol. The root implementation uses frame-level stratified CV, and its sampling/preprocessing behavior needs explicit scientific review. Do not overwrite a publication model or claim animal-level generalization from frame-level validation.

For commands that preserve the saved model, see [inference tutorial](tutorials.md). The [companion application's guide](../apps/behavior-dlc-classifier/GUIDE.md) covers its separate API/environment; archived Supplementary Figure 4 explanations do not belong to the current root model.


## Errors and recovery

Use the exception and the workflow together to diagnose a failure. These are
current diagnostic messages, not a versioned error-code API. The first-recording
and reproduction CLIs exit nonzero on failure; no stable numeric error-code enum
is defined. Keep the complete traceback and candidate revision in a support report.

| Symptom / exception | Meaning | Recovery |
| --- | --- | --- |
| `Choose a new output directory outside the source checkout` (first-recording CLI) | Output already exists or resolves inside the checkout | Choose a new, nonexistent sibling directory. Preserve earlier results. |
| `RuntimeError: Prediction identifiers do not match the complete recording` | Returned identifiers differ from the complete input recording | Retain full recording rows and their identifiers; inspect missing/duplicate frames and preprocessing before retrying. |
| `RuntimeError: Missing predicted labels` | At least one prediction is missing | Check the input schema and compatible model; retain the failing inputs/logs. Do not fill labels to make validation pass. |
| `FileNotFoundError` while loading a table or model | The supplied path cannot be opened | Check the absolute asset path and checkout revision. Wheels do not include study assets. |
| `TypeError: Unexpected classifier bundle type: ...` | The deserialized object is not a root `BehaviorClassifierBundle` | For the bundled tutorial use `classifier/figure7_behavior_classifier.joblib`; companion freezing models are not interchangeable. Only deserialize trusted artifacts. |
| `ModuleNotFoundError` mentioning classifier dependencies or joblib | Required dependencies are absent from the active environment | From the root checkout run `uv sync --locked`, then invoke commands with `uv run`. Do not upgrade the root environment to install DLC. |
| `ValueError: Missing required columns: ...` | Frame features lack required input columns | Compare the reported fields with the documented input schema and source table. Do not rename unrelated coordinates merely to satisfy the check. |
| `ValueError: Output must be outside the source checkout` / `FileExistsError` (reproduction API) | The isolated destination is unsafe or already exists | Choose a new path outside the checkout. The CLI reports the equivalent constraint as an argument error. |
| `ValueError: Unsafe checkout path: ...` / `FileNotFoundError` (reproduction copy) | A required source path is unsafe or missing | Check the named source path against the candidate checkout. Do not disable isolation checks or regenerate the reference manifest. |
| `RuntimeError: No converged finite MixedLM fit: ...` / `Converged MixedLM likelihood is worse than another attempt` | The fitting attempts did not satisfy the implemented convergence/likelihood checks | Retain warnings, sample counts, formula and environment. Investigate the fit; do not substitute zeros or change scientific tolerances to obtain a pass. |

The dependency exception may also suggest `uv pip install -r requirements.txt`;
for the locked source checkout, prefer `uv sync --locked` followed by `uv run`.

For raw-video, tracking and GUI diagnostics use the separate
[companion guide](../apps/behavior-dlc-classifier/GUIDE.md). For environment
problems see [installation troubleshooting](installation.md#troubleshooting).
