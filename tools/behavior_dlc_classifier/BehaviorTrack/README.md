# BehaviorTrack

BehaviorTrack is the stepped interface for the bundled mouse-behavior classifier. It follows the BarnesTrack layout but replaces maze analysis with pose features and frame-by-frame behavior classification.

## Start it

On Windows, double-click `launch_behaviortrack.bat`. On macOS or Linux, run:

```bash
bash launch_behaviortrack.sh
```

The Windows launcher tests candidate Python environments by constructing a hidden Tk window. On this portable machine it selects `C:\Users\jenif\anaconda3\python.exe`; the `uv` Python can import `tkinter` but lacks Tcl's runtime files and cannot open a GUI. To set up a dedicated ordinary-analysis environment manually, use a Python distribution that includes Tcl/Tk, then install the tool dependencies:

```bash
cd tools/behavior_dlc_classifier
uv sync --extra ml --extra video
```

DeepLabCut is deliberately separate because its TensorFlow/PyTorch stack is much heavier. Step 2 explains what is missing and accepts a virtual-environment folder, Python interpreter, or conda environment name. The supported bundled-environment setup is:

```bash
cd tools/behavior_dlc_classifier
uv venv .venv-dlc --python 3.10
uv pip install --python .venv-dlc -r requirements-dlc.txt
```

## Workflow

1. Select the external video folder and external results folder, parse animal/session names, and save `sessions.csv`.
2. Track only missing DeepLabCut stages. Existing analysis, filtered tracking, and tracked-video outputs are reused independently.
3. Reproduce the archived 719-feature schema and use one frozen XGBoost model for all seven behaviors.
4. Draw ethograms and build `behavior_summary.xlsx` with the final tracking-QC-aware call beside the untouched model argmax.
5. Create blinded annotation templates and evaluate all seven classes against independent manual target labels.

Videos are referenced in place; they are never copied into BehaviorTrack's `Data` folder. Step 1 stores the selected external video and results paths for the project.

When `sessions.csv` contains a session type, outputs use that session as the first level. If it is blank, that level is omitted:

```text
Results/
  DLC/<optional-session>/Filtered_CSV/
  DLC/<optional-session>/Tracked_Videos/
  Features/<optional-session>/
  Behaviors/<optional-session>/Predictions/
  Behaviors/<optional-session>/Summaries/
  Behaviors/<optional-session>/Provenance/
  Figures/<optional-session>/
  Annotated_Videos/<optional-session>/
  Validation/Manual_Labels/
  Validation/Reports/
```

## Models and interpretation

- The bundled DLC project selects iteration 1, snapshot 100000 (ResNet-50, 14 body parts).
- One archived classifier predicts Jump, Climbing, Locomotion, Turn, Grooming, Sniffing, and Freezing. There is no separate freezing override, percentile rule, or canonical smoothing step.
- The 719 columns are now reconstructed against the archived training matrix: subtract the per-frame pose centroid, but do not rotate or body-length-scale coordinates. Model and encoder hashes, feature schema, input scale, and policy are written beside every prediction.
- Historical source accuracy (`0.627`) used pooled frame folds, not animal/session-held-out folds. It is evidence about the old source rows, not validation on a new rig.
- Step 5 keeps annotation blinded to model calls and reports per-class precision, recall, F1, confusion, and recording-level performance. It only calls a report independent when the user confirms label/model independence; incomplete class or animal coverage remains explicit.

## Verify without opening windows

From the portable repository root:

```bash
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_setup.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_track.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_features.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_results.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_model.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_app.py --self-test
.venv/Scripts/python.exe tools/behavior_dlc_classifier/BehaviorTrack/src/bh_app.py --ui-smoke-test
```

The tracking self-test may pass with a warning until `.venv-dlc` is created. It verifies the project, selected iteration, and snapshot without running tracking.
