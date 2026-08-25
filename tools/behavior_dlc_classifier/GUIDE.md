# Behavior DLC Classifier

Classify mouse behavior from DeepLabCut tracking, then summarize and annotate it.

The tool takes raw behavior videos, runs DeepLabCut pose tracking, and turns the
tracked coordinates into per-frame behavior calls. It produces two kinds of output:

- **Freezing** — a binary freezing/moving call per frame, from a random forest.
  This is the validated half of the tool.
- **Seven behaviors** — Freezing, Sniffing, Grooming, Turn, Locomotion, Climbing
  and Jump, from a gradient boosted classifier. Frames that fall outside those
  seven, or whose tracking is too poor to judge, are recorded as `Unassigned`
  and left blank in the ethograms. This half is exploratory; read
  [what the seven-behavior labels are worth](#what-the-seven-behavior-labels-are-worth)
  before using it for anything quantitative.

Results are written as annotated videos, ethograms, per-frame CSVs and Excel
workbooks.

This is a standalone analysis tool. It is not part of any manuscript
reproduction pipeline, it runs in its own Python environment, and it can be
pointed at your own videos and your own trained networks.

**In a hurry?** [Install](#installation), then
[run the demo](#quick-start-the-bundled-demo). **Reporting results in a paper?**
Read [how the classifier decides](#how-the-freezing-classifier-decides) and
[what the scores mean](#about-the-bundled-freezing-model) first — freezing
scores are easy to overstate, and the two sections explain how.

| | |
| --- | --- |
| [What is bundled](#what-is-bundled) | Models, demo data and the tracking network that ship with the tool |
| [Installation](#installation) | uv or pip; DeepLabCut gets a second uv environment |
| [Quick start](#quick-start-the-bundled-demo) | One command, no tracking needed |
| [The desktop interface](#the-desktop-interface) | The no-command-line route |
| [Tracking new videos](#tracking-new-videos) | Running DeepLabCut on your own recordings |
| [How the classifier decides](#how-the-freezing-classifier-decides) | Why a frame is judged from the frames around it |
| [About the bundled model](#about-the-bundled-freezing-model) | Cross-validated scores, and what they do and do not support |
| [The seven-behavior labels](#what-the-seven-behavior-labels-are-worth) | What that half of the tool can and cannot support |
| [Relation to the paper](#relation-to-the-paper) | Where this tool matches the manuscript and where it does not |
| [Training on your own data](#improving-the-model-on-your-own-data) | Labelling, retraining, and reading the scores honestly |
| [Command line reference](#command-line-reference) | Every subcommand |
| [Layout](#layout) | Where everything lives |
| [Troubleshooting](#notes-and-troubleshooting) | Common problems |

The model changed on 2026-08-15 and results before and after that date are not
comparable. [CHANGELOG.md](CHANGELOG.md) has the before-and-after and the
numbers behind it.

---

## What is bundled

Everything needed for a complete run ships with the tool:

| Item | Path | Size |
| --- | --- | --- |
| Freezing model (fallback) | `models/freezing_model.sav` | 16 MB |
| Seven-behavior model | `models/behavior/xgb_model.pkl` (+ `.json`) | 3 MB |
| DeepLabCut networks | `models/dlc_project/Freezing_07-2020-Sanguino-Lozano-2020-09-29/` | 190 MB |
| Demo videos and tracking | `data/raw/original_videos/` | 50 MB |
| Manual freezing labels | `data/raw/manual_labels/` | 3 files |

The bundled DeepLabCut project holds two ResNet-50 snapshots of the same
network, so a fresh checkout can track new videos without downloading or
training anything first.

| Iteration | Snapshot | Test error | Used for |
| --- | --- | --- | --- |
| 1 (default) | `snapshot-100000` | 4.64 px | the published freezing and behaviour analyses |
| 0 | `snapshot-500000` | 6.38 px | the demo tracking files bundled here |

Iteration 1 was refined from iteration 0 on 2,704 labelled frames from 100
videos, 368 frames and 14 videos more than iteration 0, and those additions are
the resized 25 fps recordings. It tracks held-out frames closer and, more
importantly, it is the network every published prediction came from: its
tracking files carry the scorer name
`DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000`. New tracking therefore
matches the manuscript by default.

The demo tracking files shipped in `data/raw/` are named `..._500000` and came
from iteration 0. Re-tracking those same videos now produces `..._100000` files
instead; both work with everything downstream, but do not pool the two. Set
`iteration: 0` in the project `config.yaml` to reproduce the demo tracking
exactly.

**One thing is not bundled: the freezing model predictions are made with.**
`models/freezing_model_full.sav` is about 1 GB — far past what a git host
accepts — so it is published as a release asset instead. Download it into
`models/` and both front doors pick it up automatically; nothing to configure.
Until it is there, the committed 16 MB model is used instead, so a fresh clone
still runs end to end. Which one ran is printed in the log as the first
`[model]` line, and the two do not give the same numbers, so do not pool
results from both.

| | full | fallback |
| --- | --- | --- |
| File | `freezing_model_full.sav` | `freezing_model.sav` |
| Size | ~1 GB | 16 MB |
| Features | 4046 (`legacy4k`) | 61 (`stillness`) |
| Trees | 700 | 400 |
| Stored threshold | 0.40 | 0.25 |
| Leave-one-video-out MCC | 0.452 ± 0.177 | 0.400 ± 0.150 |
| ROC AUC | 0.908 ± 0.068 | 0.870 ± 0.070 |

Both MCC figures come from training that feature set on the three
human-labelled demo videos and holding each out in turn, which is the only
comparison between them that means anything. The full model's own recorded
"validation F1 0.803" is **not** such a figure and should not be quoted: about
95% of its training labels were an earlier model's output thresholded at 0.125
rather than human scoring, and the three human-labelled videos sit inside its
training split. The larger feature set is better on average but worse on the
animal that froze least, and both were selected on the same three videos, so
treat either MCC as an upper bound for a new setup.

You can substitute your own DeepLabCut project, your own freezing model, or your
own behavior model at any point, from either the interface or the command line.

---

## Installation

The tool installs and runs identically on Windows and Linux. Give it its own
environment; do not install it alongside other analyses, because it needs newer
pandas and XGBoost than many pipelines pin.

In this repository specifically, the tool's environment is deliberately separate
from the root manuscript environment and must stay that way: the root pins
`xgboost >=1.7,<3` while the tool needs `>=2,<4`, and XGBoost 1.7 returns
transposed probability arrays that would silently corrupt the tool's behavior
predictions. The root pre-commit hooks, ruff config, and artifact manifest all
exclude `tools/` for the same reason — the tool manages itself. Its test suite
runs in CI in its own locked environment (see
`.github/workflows/reproducibility.yml`, job `behavior-dlc-classifier tests`).

### With uv

[uv](https://docs.astral.sh/uv/) is the shortest route. It downloads a suitable
Python itself, so nothing has to be installed first, and it resolves from the
committed `uv.lock`, so everyone gets the same versions.

```bash
uv sync --extra video --extra ml --extra dev
uv run pytest -q
```

There is no environment to activate: every command below is prefixed with
`uv run`.

### With pip

```bash
python -m venv .venv
python -m pip install -e ".[video,ml,dev]"
```

Activate `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on
Linux first, then drop the `uv run` prefix from the commands below.

Python 3.9 through 3.12 are supported, and the tests are run on both ends of
that range. The extras are:

- `video` — OpenCV, needed for annotated videos and for manual labeling
- `ml` — scikit-learn, needed to load the freezing models
- `dev` — pytest

DeepLabCut is not one of them. It needs a Python and a NumPy years behind the
rest of the stack, so it gets its own environment — also made with uv, in one
command. See [Tracking new videos](#tracking-new-videos).

---

## Quick start: the bundled demo

The demo runs without DeepLabCut, because tracking for the demo videos is
already included. From the tool folder:

```bash
uv run freezing-dlc run-auto --project-root .
```

This reads the six demo videos, reuses their tracking files, classifies freezing
and writes:

- `output/predictions/` — per-frame freezing probability and call
- `output/features/` — the feature table the classifier used
- `output/summaries/freezing_summary.xlsx` — totals and per-time-bin summaries

Add `--annotate` to also write videos with the freezing call burned in.

---

## The desktop interface

For the full workflow, including tracking, the seven-behavior analysis and the
ethograms, use the interface:

```bash
uv run freezing-dlc-gui
```

There is also a launcher that needs no command line. Double-click
`launch_freezing_gui.bat` on Windows, or run `bash launch_freezing_gui.sh` on
Linux. Both use uv when it is present and fall back to the active Python
environment when it is not.

Choose a folder of videos and press **Run full analysis**. The freezing model,
behavior model and DeepLabCut project fields are pre-filled with the bundled
ones; change them to use your own.

The run creates a dated package next to your videos, for example
`20260815_MyExperiment_analysis/`:

```
01_original_videos/     source videos, plus CLAHE_preprocessed_videos/
02_tracked_videos/      tracked videos, filtered and unfiltered
03_DLC_analysis/        tracking coordinates, filtered and unfiltered
04_features/            feature tables and per-frame predictions
05_annotated_videos/    freezing and seven-behavior overlays
06_results/             Excel workbooks
07_ethogram/            freezing and seven-behavior ethograms
package_summary.txt     what each folder contains and which settings were used
```

### Settings worth knowing

- **Threshold** — leave blank to use the value stored inside the model file,
  which is the threshold that scored best when the model was trained. Type a
  number only to override it.
- **Box side (cm) and (px)** — the calibration used to convert locomotion from
  pixels to centimetres. Measure the arena in one frame and enter both numbers.
- **CLAHE contrast preprocessing** — enhances contrast before tracking. Enabled
  by default because the bundled network was trained on contrast-enhanced video.
- **DeepLabCut environment** — leave blank to use the bundled `.venv-dlc`. See below.

---

## Tracking new videos

Classifying needs nothing but the tool. Tracking needs DeepLabCut, which needs
its own environment — two commands with uv:

```bash
uv venv .venv-dlc --python 3.10
uv pip install --python .venv-dlc -r requirements-dlc.txt
```

Then leave the interface's **DeepLabCut environment** field blank. The tool
looks for `.venv-dlc` beside itself and runs DeepLabCut there.

**Why a second environment.** The bundled network is a TensorFlow-era ResNet-50
snapshot, so it needs DeepLabCut 2.x and TensorFlow 2.10, which caps Python at
3.10 and NumPy at 1.x. The tool itself runs on 3.11 with NumPy 2. The two cannot
share one environment, but they can both be uv environments.

`requirements-dlc.txt` is a full lockfile taken from a working install, so it
resolves to the same versions every time. It names `tensorpack` and `tf-slim`
explicitly: DeepLabCut 2.3 imports both but omits them from its wheel metadata,
so an install without them fails at `import deeplabcut`. On Windows the pinned
TensorFlow 2.10 is also the last release with native GPU support.

`torch` comes in as a hard dependency of DeepLabCut 2.3 and is never used here.
Adding `--extra-index-url https://download.pytorch.org/whl/cpu
--index-strategy unsafe-best-match` to the install fetches the CPU build and
saves a few gigabytes.

**If you would rather use conda,** `environment-dlc.yml` still works:

```bash
conda env create -f environment-dlc.yml
```

Then type `dlc-freezing` into the **DeepLabCut environment** field. Any conda
environment name works there, and conda does not have to be on PATH — the tool
reads `CONDA_EXE` and checks the standard Miniforge, Mambaforge, Miniconda and
Anaconda locations on both operating systems.

**What the field accepts,** in the order the tool tries them: blank for the
bundled `.venv-dlc`, a path to any virtual environment folder, a path straight
to a Python interpreter, or a conda environment name. If DeepLabCut can be
imported in the tool's own environment, that is used and the field is ignored.

**Project paths are handled for you.** DeepLabCut project files record an
absolute path from the machine that trained them. The tool writes a corrected
copy, `dlc_config_resolved.yaml`, next to your `config.yaml` and hands
DeepLabCut that, leaving your own file untouched. It has to sit in the project
folder rather than in a working directory, because DeepLabCut resets
`project_path` to whichever folder it read the config from and would otherwise
look for `dlc-models` in the wrong place. Moving or renaming the tool folder
needs no manual edit.

From the command line, tracking and classification in one step:

```bash
uv run freezing-dlc run-from-raw \
    --root "path/to/folder" \
    --model models/freezing_model.sav \
    --dlc-config models/dlc_project/Freezing_07-2020-Sanguino-Lozano-2020-09-29/config.yaml \
    --annotate
```

The folder must contain an `original_videos/` subfolder. Forward slashes work on
Windows too; drop the backslashes and put it on one line for PowerShell.

---

## How the freezing classifier decides

**It answers one frame at a time, but it looks at the frames around each one.**
A tracking file with 7,516 rows produces 7,516 calls; the output is per frame
from end to end. What is not per frame is the *evidence*.

A single frame cannot tell freezing from movement. A walking mouse has near-zero
velocity between steps, at the apex of a turn, or whenever the tracker happens to
sit steady for a moment. On the demo videos, the fraction of frames the observer
marked as **moving** that are nonetheless as still as the median **freezing**
frame is:

| Video | Moving frames indistinguishable from freezing | Separability from 1 frame | From 25 frames |
| --- | --- | --- | --- |
| Trial 14 mouse 13 | 5.1% | 0.862 | 0.920 |
| Trial 9 mouse 4 | 18.8% | 0.704 | 0.762 |
| Trial 9 mouse 3 | 34.9% | 0.627 | 0.767 |

(Separability is the ROC AUC of velocity alone.) On Trial 9 mouse 3 a third of
the moving frames are, on their own, identical to freezing. Widening the view
from one frame to twenty-five — one second — recovers most of the difference
using the very same velocity signal, with nothing new measured.

That is also how the labels were made. Scoring presses a key at the start of a
bout and again at the end, so the truth recorded for a frame reflects the bout it
belongs to, and the field's definition of freezing is explicitly a duration:
immobility sustained for around a second or more. Features describing single
frames were being asked to reproduce a durational judgement from evidence that
contained no duration.

So each frame carries 61 numbers summarising a window centred on it, at seven
widths from 0.2 s to 20 s — for example `still2_frac61` is the fraction of the
61 frames around this one that fell below the recording's motion floor. One row
in, one row out; a wider view behind each answer.

> **This makes the classifier offline, not real-time.** The windows are centred,
> so calling frame *t* uses frames after *t*. That is correct for analysing a
> recording, which is what this tool does. It rules out closed-loop use, such as
> triggering a stimulus the moment an animal freezes. That would need
> trailing-only windows and would cost some accuracy.

---

## About the bundled freezing model

Holding each of the three labeled demo videos out in turn and fitting on the
other two gives:

| Held-out video | Manual freezing | F1 | F1 if every frame were called freezing | MCC | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Trial 14 mouse 13 | 85.7% | 0.948 | 0.923 | 0.57 | 0.95 |
| Trial 9 mouse 4 | 48.4% | 0.704 | 0.652 | 0.33 | 0.84 |
| Trial 9 mouse 3 | 9.0% | 0.359 | 0.165 | 0.31 | 0.82 |
| **mean** | | **0.67 ± 0.30** | | **0.40 ± 0.15** | **0.87 ± 0.07** |

Reproduce it with:

```bash
uv run freezing-dlc cross-validate \
    --dlc-dir data/raw/original_videos \
    --labels-dir data/raw/manual_labels \
    --label-lag-frames 2
```

**Read the F1 column against the one beside it, always.** F1 rises with how much
the animal froze, so on Trial 14 it starts at 0.923 before the model does
anything at all. The Matthews correlation and the ROC AUC do not move with the
base rate, which is why they are the ones to quote.

### What this replaced, and why it matters for interpreting older results

The model shipped before this one scored MCC 0.21 ± 0.14 by the same procedure,
with an F1 within two points of the call-everything classifier on all three
videos. It carried a headline of "F1 0.804", which should not be quoted: that
figure was read off the same frames that chose its threshold and stopped its
training, and the forest behind it had been fitted on a single video.

Two changes account for most of the difference, and both are worth knowing about
if you are scoring freezing by any other means:

- **Freezing is a duration, not a frame.** The old features described single
  frames, with rolling means over three and five frames as their only memory. The
  demo videos have median bouts of 12, 26 and 66 frames, so that window was far
  too short to represent the behaviour being scored.
- **Choosing a threshold by F1 drives it towards calling everything freezing.**
  F1 climbs with the positive rate, so on an animal that freezes most of the time
  it is maximised near the bottom of the range; the old model settled at 0.10.
  Choosing by Matthews correlation instead, changing nothing else, moved
  leave-one-video-out agreement from 0.21 to 0.30.

The remaining gain comes from the motion floor. Across the six bundled
recordings the median frame-to-frame displacement spans two orders of magnitude,
0.012 to 1.4 pixels, while the animals are all about 50 pixels long: that is
tracking jitter, not behaviour. Measuring stillness against each recording's own
floor is what lets a threshold learned on one animal mean anything on another.

**One caveat on these numbers.** The feature set was chosen by comparing
candidates on these same three videos. Each fold is clean, but the design was
not selected blind to them, so 0.40 is optimistic as a prediction for a new
laboratory's data. It is reported here because the improvement holds on every
video and every random seed tried, which selection noise on three videos does
not usually produce. Treat it as an upper bound and cross-validate on your own
labels.

### The full model

The 1 GB archived model is a different kind of object from the one that ships,
and the difference is not mainly its size.

It is a 700-tree forest over 4046 features, fitted on 422,346 frames from 101
videos, and its recorded F1 of 0.803 *was* measured on 21 videos it had not been
fitted on. That part is sound. The problem is where its labels came from:

| Source | Labels | Frames |
| --- | --- | --- |
| `CSVs_all_filtered` | `Freezing_predictions_light` — a thresholded classifier score | 400,000 (94.7%) |
| `London` | manual scoring — the three demo videos | 22,346 (5.3%) |

`Freezing_predictions_light` is worth describing exactly, because the name is
misleading in both halves. "Light" does **not** mean a lighter model: the files
there are the first and last column of a much larger per-frame feature table
(1754 columns, 98 recordings, ~34 GB), sliced out by a three-line script. In
that sense they are simply a trimmed copy of the original files.

The label column they carry is `Freezing_Jen_0-125_threshold`, and it is
reproduced exactly by `Probability_Freezing >= 0.125`, where
`Probability_Freezing` is a continuous per-frame score sitting in the same
table. Checked frame by frame, the two agree with **zero mismatches**. A human's
frame-by-frame scoring cannot equal a threshold rule to the frame, and the
zero-mismatch result also rules out a human having reviewed and corrected the
output afterwards — any correction would show up as a disagreement. The bout
structure says the same thing independently: one recording contains 145 freezing
bouts of which 29 are a *single frame* long and 63% are shorter than half a
second, which is not something an observer pressing a key produces, and not
freezing as the behaviour is defined.

The human contribution here is the cutoff. Someone chose 0.125 rather than the
0.5 used by the neighbouring `Freezing` column — a human-tuned threshold on a
model's probability, not human scoring of frames.

So about 95% of what it learned, and almost all of what it was validated
against, is a classifier's output rather than a human's. Its 0.803 largely
measures how faithfully it reproduces that earlier score. Meanwhile the only
human-labelled recordings in the set are the three demo videos, and those sit
inside its *training* split, so scoring it on them measures memory.

None of this makes it useless — a model distilled from a large pseudo-labelled
corpus can still be good — but it does mean **0.803 and the 0.40 above are not
comparable numbers**, and neither is evidence that the larger model is better.

### Do the 4046 features actually help?

Trained on the same human labels and scored the same way, held out one video at
a time:

| Feature set | MCC | ROC AUC | Worst video (MCC) | Model size |
| --- | --- | --- | --- | --- |
| `stillness`, 61 features | 0.404 ± 0.146 | 0.870 ± 0.071 | **0.311** | 16 MB |
| `legacy4k`, 4046 features | **0.452 ± 0.177** | **0.908 ± 0.068** | 0.252 | ~1 GB |

The large set does carry more signal, and the evidence for that is the AUC: it
improves on **every one of the three videos** (0.951→0.979, 0.819→0.844,
0.841→0.901), which is a consistent direction rather than a lucky mean.

It does not straightforwardly follow that you should use it. The MCC gap of
0.048 sits well inside a between-video spread of 0.15 to 0.18 on three videos,
so on this evidence it is not a reliable difference. The large set is also
*worse* on the animal that froze least — the case that already fails hardest —
and it spreads wider across animals. The better ranking is not reaching the
calls, which is the same operating-point problem described above, now showing up
as a threshold that transfers poorly to a rare-freezing animal.

So: the small set is the default because it is far cheaper, more consistent
between animals, and better where performance is weakest. If you have many
labelled videos of your own, `--feature-mode legacy4k` is worth cross-validating
on them, because the AUC result suggests the ceiling is higher.

Reproduce with:

```bash
uv run freezing-dlc cross-validate --feature-mode legacy4k \
    --dlc-dir data/raw/original_videos \
    --labels-dir data/raw/manual_labels --label-lag-frames 2
```

The archived model itself is published as a release asset rather than committed.

> Release download link: add it here once the release is published.

Training on your own labeled data usually beats both.

---

## What the seven-behavior labels are worth

Everything above concerns the legacy binary-freezing workflow. The seven-behavior
classifier is a different model: it learns hand-curated MoSeq behavior labels
from engineered DeepLabCut keypoint features. It is a supervised DLC
approximation of that taxonomy, not a rerun of MoSeq on a new video.

**It has never been validated on an animal it did not train on.** The reported
cross-validated accuracy is 0.627 across eight classes, but the folds were drawn
over pooled frames rather than over animals. Neighbouring frames are nearly
identical and land on opposite sides of the split, so the same animal — often
the same second of video — appears in both training and test. The real figure
for a new animal is lower by an unknown margin. The manuscript repository
carries the same warning against its own version of this model.

**Grooming is effectively absent.** The training set holds 131 Grooming frames
against roughly 4,000 for every other class, and the model does not predict it
on any bundled recording. A recording with no Grooming in the output is not
evidence that the animal did not groom.

**Freezing was over-called on the three old demo videos.** The unified model
called it on 31%, 80% and 95% of frames where a human scored 9%, 48% and 86%.
Those videos are also far outside the archived camera scale, so this is a domain
failure warning, not a calibrated estimate of target performance.

**`Unassigned` is not an uncertainty flag.** At training it covered MoSeq
syllables outside the seven curated clusters; at inference it marks frames where
too few body parts were tracked well enough to judge. A confident wrong answer
is reported as a behavior, not as `Unassigned`. Those frames are left blank in
the ethograms.

BehaviorTrack keeps the labels mechanically auditable:

- The number beside a behavior is that behavior's own probability. Where
  tracking failed it is blank.
- One model decides all seven behaviors, including Freezing. Earlier versions
  layered hand-tuned rules on top — a jump rule, a climbing rule, and a
  fallback that called anything left over Locomotion. The rules used
  per-recording percentiles, so they labelled a fixed fraction of every video as
  Jump whether or not a jump occurred. Those rules, the separate freezing
  override, and canonical smoothing are absent from BehaviorTrack.

The tool now refuses tracking files whose body parts differ from the bundled
network's, rather than filling the missing ones with zeros and predicting anyway.

---

## Relation to the paper

This tool is a companion to the manuscript, not a reproduction of it. The paper
repository regenerates the figures; this repository applies the same ideas to new
video.

What is deliberately shared:

- **The behavior taxonomy.** The same seven behaviors, from the same hand-curated
  grouping of MoSeq syllables, plus the same eighth `Unassigned` class for
  everything outside them.
- **The time base.** 25 frames per second and 30-second bins, matching the
  manuscript's analysis settings.

What is not shared, and should not be presented as if it were:

- **The classifier itself is a different model.** The paper's Figure 7 model is
  built from MoSeq-derived kinematics; this one is built from DeepLabCut keypoint
  geometry, and it was trained by a separate script on separate recordings.
  Per-behavior numbers from this tool are not expected to match the paper's and
  do not reproduce them.
- **Freezing.** The manuscript's freezing ground truth comes from a SimBA
  classifier thresholded at 0.125. The model here is a different classifier
  trained on manual scoring of the demo videos. The two agree on what freezing
  is; they are not the same instrument.

Because the model uses raw pixel distances and velocities, camera scale is part
of its domain. BehaviorTrack compares median nose-tail length with the archived
training rows and writes the warning into provenance, but only manual target
labels can establish that a new camera works.

### The coordinate frame

The archived SHAP input matrix makes this testable rather than inferred. For each
frame, training subtracted the mean x/y position of all 14 body parts. It retained
the nose-tail orientation angle and raw pixel scale: it did not rotate the animal
onto a common heading and did not normalize body length. BehaviorTrack now uses
that exact transform. Lagged and five-frame rolling columns are reconstructed
from the same centered coordinates, while pairwise distances, angles, velocities,
accelerations, and PCA-derived body geometry retain their archived definitions.

The central 90% of archived nose-tail lengths is about 162--333 px (median
243 px). A recording outside that range is not rescaled silently; it is marked
out of domain so it can be reshot, calibrated, or validated explicitly.

---

## Improving the model on your own data

The interface has **Label videos and create refined model**. Select a folder of
videos, then for each video press `f` at the start of a freezing bout and `f`
again at the end, and `s` to save. When you stop, a new model is trained from
your labels and loaded into the model field automatically.

To retrain from the command line, using the bundled demo labels as the example:

```bash
uv run python scripts/train_model.py
```

With no arguments this retrains the default model from the demo videos and their
manual labels, writing `output/retrained_model.sav`, then scores it. Pass
`--dlc-dir`, `--labels-dir` and `--out-model` to train on your own data. Run it
with `--help` for the rest.

Three feature modes are available through `--feature-mode`:

| Mode | Features | What it describes |
| --- | --- | --- |
| `stillness` (default) | 61 | Stillness over windows from 0.2 s to 20 s, relative to each recording's motion floor |
| `compact` | 8 | Single frames, plus 3 and 5 frame rolling means. Kept so older models still load |
| `legacy4k` | 4046 | Per-bodypart kinematics with wide rolling windows. What the archived 1 GB model uses |

Labels are matched to tracking files by animal name, so
`Trial_9_mouse3_labels.csv` pairs with
`Trial     9_mouse3DLC_resnet50_...filtered.csv`.

`--label-lag-frames` compensates for the delay between a behavior starting and
the observer pressing the key. The bundled models use 2 frames, about 80 ms at
25 fps.

### Reading the score that training prints

Training splits your videos three ways and keeps whole recordings together,
because frames from one video are near duplicates of each other and splitting
inside a video inflates every score. One part fits the forest, one chooses the
decision threshold and stops training, and one is touched only at the end. The
figure reported as `held_out_videos` comes from that last part and is the only
one that describes a new animal. The `selection_metrics` in the model file are
the higher, optimistic numbers from the part that chose the threshold.

Alongside precision, recall and F1, training reports the fraction of frames the
animal actually froze and what a classifier calling every frame freezing would
score. If your F1 is not clearly above that, the model has not learned anything,
however high the F1 looks.

Once the split has done its measuring, the model that gets **saved** is refitted
on every labeled frame, because holding two thirds of the data back costs real
accuracy: on the demo videos, fitting on two recordings rather than one lifts
agreement on a held-out third from 0.11 to 0.63 for the animal least like the
others. The reported metrics therefore describe a model trained on *less* data
than the one you end up with, which makes them conservative. `--no-refit-on-all`
turns this off if you want the saved model to correspond exactly to the numbers.

With fewer than three labeled videos nothing can be held back. Training says so
and marks the result `selection_set_optimistic`; do not quote that number.

`eval-model` reports `is_out_of_sample`. It is `false` when you score a model on
videos it was fitted on, which measures memory rather than skill. After a refit
that covers every labeled video, so the honest figure comes from
`cross-validate`.

With only a handful of videos, any single split is mostly luck. `cross-validate`
holds each video out in turn, so every recording contributes, and it reports the
spread as well as the mean:

```bash
uv run freezing-dlc cross-validate \
    --dlc-dir data/raw/original_videos --labels-dir data/raw/manual_labels
```

It prints F1 next to the F1 of a classifier that calls every frame freezing, so
you can see immediately how much of the score is the model and how much is the
base rate, plus MCC and ROC AUC, which do not move with the base rate.

Pass `--min-samples-leaf` the same value you trained with, so the figure
describes the model you actually ship.

### Why the threshold is not chosen by F1

`--threshold-criterion` defaults to `mcc`. F1 counts correctly-called freezing
frames but never counts correctly-called moving frames, so on an animal that
freezes most of the time it keeps rising as the cutoff falls, and is maximised
close to calling everything freezing. Matthews correlation and balanced accuracy
both charge for that. `f1` remains available for comparison with older results.

A consequence worth expecting: because the threshold no longer chases F1, the
reported F1 can sit slightly *below* the call-everything baseline while the model
is much better by every base-rate-robust measure. On Trial 14 the bundled model
scores F1 0.916 against a 0.923 baseline, with balanced accuracy 0.82 where the
baseline scores 0.50.

---

## Command line reference

| Command | Purpose |
| --- | --- |
| `run` | Classify freezing from tracking files that already exist |
| `run-from-raw` | Track videos with DeepLabCut, then classify |
| `run-auto` | Same, using the bundled models and the `data/raw` layout |
| `train-model` | Train a freezing model from tracking files and labels |
| `cross-validate` | Hold each video out in turn; the score to quote in a paper |
| `eval-model` | Score an existing model against labels |
| `label-video` | Manually score freezing on one video |

Run any of them with `--help` for the full options. Every command that predicts
accepts `--threshold`; omit it to use the value stored in the model.

---

## Layout

```
src/freezing_dlc/
    features.py         turns tracking coordinates into the numbers a model sees
    train.py            training, leave-one-video-out scoring, evaluation
    predict.py          applies a model to a feature table
    pipeline.py         tracking files -> predictions -> summary workbook
    friendly_pipeline.py  the full workflow the desktop interface runs
    dlc_stage.py        finding videos and running DeepLabCut
    behavior.py         the seven-behavior classifier
    gui.py, cli.py      the two front doors
    annotate.py, summarize.py, label_video.py, model.py, config.py

models/
    freezing_model_full.sav  the model predictions use (~1 GB, downloaded)
    freezing_model.sav  committed fallback if the above is absent (16 MB)
    behavior/           the seven-behavior model
    dlc_project/        the DeepLabCut networks: iteration 1 (default), iteration 0
    archive/            retired models, ignored by version control

data/raw/               demo videos, their tracking, and manual labels
output/                 results of local runs, ignored by version control
scripts/                train_model.py, check_portability.py, make_side_by_side.py
tests/                  48 tests, no model or data files needed
requirements-dlc.txt    locked DeepLabCut environment, installed with uv
environment-dlc.yml     the same environment for conda users
uv.lock                 resolved dependency versions, shared by everyone
GUIDE.md                this file
CHANGELOG.md            what changed and when, with the numbers behind it
```

---

## Notes and troubleshooting

**scikit-learn version warnings.** The freezing models were saved with
scikit-learn 1.7, so newer releases warn once per tree when loading them. The
loader suppresses that warning, because both shipped models were checked against
1.6.1, 1.7.2 and 1.8.0 on Python 3.9, 3.10 and 3.11 and every probability
matched to the bit. The `ml` extra keeps the version inside that tested range.

**Results differ from an older run of this tool.** Expected, and there are two
separate reasons. The threshold now defaults to the value stored in the model
rather than a fixed number. And the shipped model was replaced: the previous
one, an eight-feature model whose threshold was chosen by F1, called noticeably
more frames freezing, particularly on animals that freeze rarely. Do not pool
old and new numbers in one analysis. To reproduce old output exactly, use the
archived model, which still loads:

```bash
uv run freezing-dlc run --root <folder> \
    --model models/archive/freezing_model_compact_8feature.sav --threshold 0.4
```

**A model refuses to run on my tracking.** Models record the names of the
features they expect, and the tool checks them rather than filling in whatever is
missing with zeros — a forest fed zeros still returns confident-looking
percentages, so a silent mismatch would produce a full set of plausible and
meaningless results. If you see this, your body parts differ from the ones the
model was trained on. Retrain on your own tracking.

**DeepLabCut cannot be found.** Create its environment with
`uv venv .venv-dlc --python 3.10` then
`uv pip install --python .venv-dlc -r requirements-dlc.txt`, and leave the
interface's **DeepLabCut environment** field blank. If you use conda instead,
put the environment name in that field; conda does not need to be on PATH.

**`No module named 'tensorpack'` or `'tf_slim'` when tracking.** The DeepLabCut
environment was built without them. They are imported by DeepLabCut 2.3 but
missing from its wheel metadata, so `pip install deeplabcut` alone does not
pull them in. Installing from `requirements-dlc.txt` does.

**The interface will not start on Linux.** It needs Tkinter, which several
distributions ship separately as `python3-tk`. Running through `uv` avoids this,
because the Python that uv installs already includes it.

**Videos are not written.** Annotation and manual labeling need the `video`
extra.

**Dependency versions.** The tool keeps its own environment, and `uv.lock`
records exactly what that resolves to. The seven-behavior model ships as both a
pickle and a portable `xgb_model.json`, and the loader falls back to the JSON
automatically, so it still loads under XGBoost versions that cannot read the
pickle. XGBoost 2.0 is the floor: 1.7 accepts the JSON but returns a transposed
probability array, which is why the pin stops there rather than going lower.
