# Your first recording

Run saved-model inference on one complete, real recording from the bundled MoSeq
table. No GPU, raw video or DeepLabCut installation is needed. This uses the root
study environment, not the separate desktop-app environment.

## 1. Install and check the source

Follow [installation](installation.md) on the `main` branch. From
the checkout root, record `git rev-parse HEAD` and run:

```bash
uv sync --locked
uv run python scripts/check_reproducibility.py
```

## 2. Run the example

Choose a new directory outside the checkout:

```bash
uv run python docs/examples/first_recording.py --output ../coping-first-recording
```

You may see an XGBoost warning about historical model serialization while loading
the archived model. The warning is retained; it is not by itself an execution
failure or evidence of historical reproducibility. Do not resave or retrain the
archive to silence it. Verify completion and the outputs below.

The example selects the first recording by name and retains every frame, so
rolling and lagged features are not changed by truncation. It uses the archived
model without retraining, verifies frame identifiers and checks for missing labels.
It refuses an existing destination or a destination inside the source checkout.

## 3. Inspect your outputs

| File | Contents |
| --- | --- |
| `predictions.csv` | Frame identifiers, predicted labels and model outputs |
| `label_counts.csv` | Number of frames assigned to each predicted behavior |
| `run.json` | Recording name, frame count, input/model SHA-256 and runtime |

The macOS arm64 locked-environment run on 9 September 2026 processed **11,250
frames** from `Animal 11_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered`.
Your output should retain that recording's complete frame count; compare hashes
in `run.json` before comparing results. A warning about historical XGBoost model
serialization is retained, not suppressed.

These predictions demonstrate software execution, **not independent accuracy**.
Do not interpret the label counts as validation against human annotations or as a
new manuscript result. Missing training provenance and other scientific limits
remain in the [reproduction report](../REPRODUCIBILITY.md).

This walkthrough consumes existing MoSeq features. It does not run DeepLabCut,
create a 14-marker pose table, verify raw-video tracking or test the desktop UI.
For that separate environment and requirements, see
[tracking installation](installation.md#tracking-installation-readiness-versus-completion).

## Next steps

- [Rebuild downstream figures in isolation](../REPRODUCIBILITY.md).
- [Use video and matching tracking files in the visual app](../apps/behavior-dlc-classifier/GUIDE.md).
- [Explore the archived summaries in the research website](https://umguec.github.io/coping-dynamics-sequencing/).
