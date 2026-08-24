# Changelog

## 2026-08-24 -- The tracking network the paper used is now the one that ships

The bundle carried iteration 0 of the DeepLabCut project (`snapshot-500000`),
while every published freezing and behaviour prediction came from iteration 1
(`snapshot-100000`): the refined network trained on 2,704 labelled frames from
100 videos, 368 frames and 14 videos more than iteration 0, all of them resized
25 fps recordings. Held-out error is 4.64 px against iteration 0's 6.38 px.

- **Iteration 1 added and made the default.** `models/dlc_project/.../dlc-models/iteration-1/`
  holds the snapshot, its `test/pose_cfg.yaml` and a trimmed training config;
  the project `config.yaml` now reads `iteration: 1`. Tracking a new video
  produces `DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000` files, the same
  scorer name as every prediction in the manuscript pipeline.
- **Iteration 0 kept.** It is still in `dlc-models/iteration-0/` and still the
  network behind the demo tracking files in `data/raw/`, which are named
  `..._500000`. Set `iteration: 0` to reproduce those exactly. Predictions from
  the two networks should not be pooled.
- **No machine-specific paths travelled with it.** The copied configs carry
  relative dataset paths, as iteration 0's already did, so the bundle stays
  portable and passes `scripts/check_portability.py`.

## 2026-08-17 -- A bug-hunt pass over every module, held down by 18 new tests

The whole tool was reviewed for correctness, duplication and speed. Every fix
below has a test or was verified by running the affected path; the suite grew
from 59 to 77 tests and still runs in about two seconds.

- **Unchecking "Run DeepLabCut" always failed.** The run collected tracking
  from a working folder that had just been created empty, so the documented
  reuse of existing DLC outputs could never happen. It now looks where the
  outputs actually are: beside your videos. Covered by an end-to-end test that
  runs the whole pipeline on a synthetic video with pre-existing tracking.
- **Manual labeler.** Pressing `S` with Caps Lock on silently stepped a frame
  instead of saving, and `Q` stepped backwards instead of quitting -- the key
  codes for capital S/Q collide with Linux arrow codes once truncated to a
  byte, which also left the arrow keys dead on Windows. Keys are now read
  untruncated, case-folded, with the real arrow codes for Windows, Linux and
  macOS. A video whose container promises more frames than it can decode no
  longer crashes the session and discards unsaved labels; the labeler pauses on
  the last readable frame instead. Clicking the timeline now only jumps -- it
  used to also insert a freezing toggle wherever you clicked, corrupting labels
  one toggle per scrub. Playback stopped seeking the codec on every frame, and
  the freezing summary no longer rebuilds a per-frame array each repaint, so
  long videos play smoothly. The mouse is now listed on the controls card.
- **Training.** Combining `--max-train-frames` with `--label-lag-frames`
  silently paired each frame's features with the label of the next *sampled*
  row, which can be dozens of frames away; the cap now waits until after the
  lag. `scripts/train_model.py` uses exactly that combination by default, so
  its results on datasets above the cap were affected. A DLC CSV with a
  trailing blank line crashed the dataset build with a bare IndexError; a fit
  split holding only one class now says so instead of failing inside
  scikit-learn. `eval-model` reports MCC and balanced accuracy next to F1, and
  says `null` rather than claiming out-of-sample when an older model file
  never recorded which videos it was fitted on.
- **Wrong-video pairing.** A tracking file for `Trial 1` could be paired with
  `Trial 1 (2).mp4` when both exist, and square brackets in a video name broke
  tracking-file discovery entirely (they read as glob syntax). Exact stems now
  win, globs are escaped, and a skipped-DLC run warns when a video has no
  tracking instead of silently leaving it out of the results.
- **Results workbook.** Sheet names carry the real time-bin length; a 60 s run
  no longer labels its sheets "30stimebin". The default 30 s run keeps the
  lab's historical sheet names exactly, which a test now pins.
- **Interface.** Numeric fields report errors by name ("Frames per second must
  be a number; it is 'abc'.") instead of a bare Python message; the progress
  bar survives an analysis and a refinement running at once; the form scrolls
  under macOS trackpads, whose wheel deltas are smaller than one Windows notch.
- **Fewer moving parts.** The translucent-box painter existed in three copies,
  two already diverged, and each blended the entire canvas to draw one small
  box; one shared helper now blends only the covered pixels. The evaluator's
  private copies of model loading and feature alignment were replaced with the
  shared ones. Locomotor activity reuses tracking already in memory instead of
  parsing every CSV a second time. Every video writer is checked after opening
  and released on error, so a failed run cannot leave a locked, unplayable
  file behind.

## 2026-08-17 -- DeepLabCut runs under uv, and the full model is the default

Two changes, plus one bug that had made raw-video tracking impossible.

- **DeepLabCut no longer needs conda.** It gets a second uv environment:

  ```bash
  uv venv .venv-dlc --python 3.10
  uv pip install --python .venv-dlc -r requirements-dlc.txt
  ```

  Leave the interface's **DeepLabCut environment** field blank and the tool
  finds `.venv-dlc` beside itself. The field now also accepts a path to any
  virtual environment or interpreter; a bare name is still looked up as a conda
  environment, and `environment-dlc.yml` is unchanged, so conda users lose
  nothing. The `dlc_conda_env` setting is renamed `dlc_env`.

  `requirements-dlc.txt` is a lockfile taken from a verified install. It names
  `tensorpack` and `tf-slim` explicitly, which DeepLabCut 2.3 imports but omits
  from its wheel metadata; without them `import deeplabcut` fails. Conda-forge
  happened to supply them, which is why the conda route never hit this.

- **Tracking raw videos was broken, in both routes.** The corrected project
  config was written into the run's working folder, but DeepLabCut's own
  `read_config` overwrites `project_path` with whatever folder it read the
  config from, so it then looked for `dlc-models` inside the working folder and
  reported the network as missing. The corrected copy is now written next to
  the project's own `config.yaml` as `dlc_config_resolved.yaml`, which is the
  folder DeepLabCut needs it in. Your `config.yaml` is still never edited. This
  affected every attempt to track new videos; classifying already-tracked files
  was never affected.

- **Predictions now use the full 4046-feature model.** `freezing_model_full.sav`
  moved out of `models/archive/` to `models/` and is what the GUI and
  `run-auto` load when it is present. It is ~1 GB and cannot be committed, so it
  is downloaded separately; the committed 16 MB `freezing_model.sav` is used
  when it is absent, so a fresh clone still runs. **The two do not agree**, so
  do not pool results across them — the first `[model]` line in the log now
  names the file that ran.

  On leave-one-video-out over the three human-labelled videos the larger feature
  set scores MCC 0.452 ± 0.177 against 0.400 ± 0.150, and AUC 0.908 against
  0.870, but it is worse on the animal that froze least. The full model's own
  recorded "validation F1 0.803" remains not comparable to either and should
  not be quoted; `models/archive/ARCHIVE_NOTE.txt` explains why at length.

## 2026-08-16 -- seven-behavior labels corrected

The seven-behavior half of the tool was wrong in ways that made its output look
more confident than it was. **Seven-behavior results produced before this date
should be discarded**, not compared. Freezing output is unaffected: none of the
freezing code was touched.

Five changes, in the order they matter.

- **The model was being shown the wrong coordinates.** It had been trained on
  poses centred on the animal and rotated onto a common heading, and was being
  given raw image coordinates instead. Two of its features were constants during
  training and could never take that value at inference. The transform is now
  applied before predicting, which changes about a quarter of all frames.
- **Coordinates are rescaled to the training body size.** The training camera saw
  a mouse about 244 pixels long; the demo camera sees the same animal at 54.
  Without this, the corrected model called almost every frame Freezing or
  `Unassigned` and never once called Locomotion. On a camera matching the
  original setup the factor is 1.
- **The confidence belonged to a different behavior.** The percentage printed in
  the CSVs and burned into the annotated videos was the model's highest
  probability, while the label beside it could be another behavior entirely — on
  one recording that was true of 87% of frames, including a `Locomotion` frame
  labelled "97.9% confidence" where the 97.9% was `Unassigned`'s. The number now
  always describes the behavior it sits next to, and is blank where tracking
  failed.
- **The hand-tuned rules are gone.** A layer of heuristics sat on top of the
  model deciding Jump, Climbing, Turn, Grooming and Sniffing, with anything left
  over defaulting to Locomotion. Their thresholds were per-recording percentiles,
  so the jump rule labelled a fixed ~0.8% of every recording as Jump whether or
  not a jump happened, and a synthetic animal moving at constant speed came out
  100% Jump. Only two overrides remain: the validated freezing model decides
  Freezing, and untrackable frames become `Unassigned`.
- **Foreign tracking files are rejected.** A file with different body-part names
  used to have its missing features filled with zeros and still got predictions
  back at 66% confidence, with 99.5% of the feature values fabricated. It now
  raises and names the body parts it expected.

`Unspecified` is renamed `Unassigned` throughout, matching the manuscript's term
for the same class. Column names in `behavior_FINAL_results.xlsx` change with it.
[GUIDE.md](GUIDE.md) gained two sections on what these labels do and do not
support, and on where the tool matches the paper.

## 2026-08-15 -- freezing model retrained, evaluation made honest

The shipped freezing model was replaced and the code around it was corrected.
**Freezing percentages produced before this date are not comparable with ones
produced after it.** Do not pool them. The old model still loads, from
`models/archive/freezing_model_compact_8feature.sav`, if you need to reproduce
earlier output.

### Before

```
  6 demo videos
       |
       v
  +-------------+   tracking CSVs (14 body parts: x, y, likelihood)
  | DeepLabCut  |-------------------+
  +-------------+                   |
                                    |
        PREDICT <-------------------+-------------------> TRAIN
        -------                                           -----
                                             tracking CSVs + manual labels
  features.py                                             |
    freezing_feature_set()                                v
    8 numbers, EACH FRAME ALONE                 2-way split, by video
    velocity, 3- and 5-frame means                fit    = 1 video
       |                                          select = 2 videos
       |  (!) a frame on its own cannot                   |
       |      show "sustained" immobility,                v
       |      which is what freezing IS.        threshold picked to max F1
       |      Model ROC AUC held out: 0.76                |
       v                                                  |  (!) F1 rises with
  predict_freezing()                                      |      the freezing
    reindex(fill_value=0)                                 |      rate, so the
       |                                                  |      cutoff sinks
       |  (!) unknown features silently                   |      to 0.10 and
       |      become zeros; the forest                    |      nearly every
       |      still returns confident                     |      frame is
       |      percentages from nothing                    |      called frozen
       v                                                  v
  predictions + summary workbook              threshold, trees AND the
       ^                                      reported score all come from
       |                                      the SAME 2 videos
  pipeline.py                                           |
    _find_original_video()  ---.                        v
  friendly_pipeline.py         |            freezing_model_compact.sav
    _find_matching_video()  ---'              58 MB, 760k nodes, depth 30
       (!) one job, two copies,               fitted on 1 of 3 videos
           drifted apart: this one            headline "F1 0.804"
           globs *.mp4 and fails on
           Linux for .MP4 folders     ...but held out properly:
                                        MCC 0.21    AUC 0.76
                                        F1 within 2 points of "call
                                        every frame freezing" on all
                                        3 videos -> no better than
                                        doing nothing
```

### After

```
  6 demo videos
       |
       v
  +-------------+   tracking CSVs (14 body parts: x, y, likelihood)
  | DeepLabCut  |-------------------+
  +-------------+                   |
                                    |
        PREDICT <-------------------+-------------------> TRAIN
        -------                                           -----
                                             tracking CSVs + manual labels
  features.py                                             |
    stillness_feature_set()                               v
    61 numbers, each summarising a               3-way split, by video
    WINDOW CENTRED ON THE FRAME                    fit    = 1 video
      7 widths: 0.2 s .. 20 s                      select = 1 video
      stillness measured against                   report = 1 video (untouched)
      each recording's own motion                           |
      floor -> transfers between                            v
      recordings (median velocity            threshold picked to max MCC
      varies 100x across the 6                                |
      demo videos at equal body                               |  (+) MCC pays
      size: that is tracking jitter)                          |      for both
       |                                                      |      classes, so
       |  1 row in, 1 row out. Offline                        |      it will not
       |  only: centred windows read                          |      buy recall
       |  frames after t.                                     |      with false
       v                                                      |      positives
  predict_freezing()                                          v
    checks feature names, RAISES                    score read ONLY from
    on mismatch                                     the report videos
       |                                                      |
       |  (+) wrong tracking now stops                        v
       |      the run instead of                    refit on ALL labelled
       |      producing plausible                   frames once measuring
       |      nonsense                              is done
       v                                                      |
  predictions + summary workbook                              v
       ^                                        models/freezing_model.sav
       |                                          16 MB, 200k nodes, depth 17
  pipeline.py           ----.                     fitted on 3 of 3 videos
  friendly_pipeline.py  ----+--> dlc_stage.py     threshold 0.25
                                 find_source_video()
                                 one definition,        held out properly:
                                 case-insensitive,        MCC 0.40   AUC 0.87
                                 sorted, falls back       beats "call every
                                 past the extension       frame freezing" on
                                                          every video

  freezing-dlc cross-validate  <- holds each video out in turn; prints F1
                                  beside the do-nothing F1, plus MCC and AUC
```

### What changed, and what it was worth

| Change | Effect |
| --- | --- |
| Features describe windows of 0.2-20 s, not single frames | Largest single gain; held-out ROC AUC 0.76 -> 0.87 |
| Stillness measured against each recording's motion floor | Makes a threshold learned on one animal mean something on another |
| Threshold chosen by MCC, not F1 | MCC 0.21 -> 0.30 on its own, changing nothing else |
| Saved model refitted on every labelled video | Held-out MCC on trial14mouse13: 0.11 -> 0.63 |
| Leaves must cover >= 20 frames | 58 MB -> 16 MB, and less memorisation of video identity |
| Reported score comes from videos nothing selected on | The old "F1 0.804" was measured where the threshold was picked |

Bottom line, leave-one-video-out over the three labelled demo videos:

```
              MCC              ROC AUC          worst video (MCC)
  before   0.21 +/- 0.14     0.76 +/- 0.12           0.07
  after    0.40 +/- 0.15     0.87 +/- 0.07           0.31
```

Reproduce with `freezing-dlc cross-validate --dlc-dir data/raw/original_videos
--labels-dir data/raw/manual_labels --label-lag-frames 2`.

**Caveat.** The feature set was chosen by comparing candidates on these same
three videos. Each fold is clean, but the design was not selected blind to them,
so 0.40 is optimistic for a new laboratory's data. Cross-validate on your own
labels before quoting a number.

### What the archived 1 GB model turned out to be

Opening it settled a question the documentation had been guessing at. It is a
700-tree forest over 4046 features, 12.6 million nodes, fitted on 422,346 frames
from 101 videos with a proper 80/21 split by video -- so its recorded F1 of 0.803
was measured on videos it had not been fitted on. Earlier notes in this
repository said otherwise; they were wrong.

The real limitation is its labels. About 95% of its training frames came from a
directory named `Freezing_predictions_light`; only the three manually labelled
demo videos (5%) are human.

The name misleads twice over. "Light" is not a lighter model -- those files are
the first and last column of a 1754-column per-frame feature table, sliced out
by a three-line script, so they really are just a trimmed copy of the original
files. But the label column they carry, `Freezing_Jen_0-125_threshold`, is
reproduced exactly by `Probability_Freezing >= 0.125` against the continuous
score in the same table: zero mismatches, frame by frame. Human scoring cannot
equal a threshold rule to the frame, and zero mismatches also rules out a human
having corrected it afterwards. The bout structure agrees -- 145 bouts in one
recording, 29 of them a single frame long, 63% shorter than half a second, which
is neither what an observer produces nor what freezing is. What a human did
choose is the cutoff: 0.125 rather than the 0.5 used by the adjacent `Freezing`
column.

So its validation videos are almost all machine-labelled too, and 0.803 mostly
measures agreement with that earlier score. The three human-labelled videos sit
inside its training split, so scoring it on those measures memory.

Consequence: **0.803 and the 0.40 above are not comparable**, and neither is
evidence that more features are better. The guide no longer tells users to
prefer the full model for real analyses.

Its feature set was then tested properly, on the same human labels and the same
leave-one-video-out procedure as the shipped one:

```
                        MCC            ROC AUC      worst video   size
  stillness  (61)   0.40 +/- 0.15   0.87 +/- 0.07      0.31       16 MB
  legacy4k (4046)   0.45 +/- 0.18   0.91 +/- 0.07      0.25       ~1 GB
```

The large set does rank frames better, and consistently: AUC improves on all
three videos. But the MCC gap of 0.048 is well inside a between-video spread of
0.15-0.18 on three videos, it is worse on the animal that froze least, and it
spreads wider across animals. The small set stays the default; the large one is
worth cross-validating on any lab that has many labelled videos of its own.

### Also fixed

- `filter(like="_x")` matched `paw_x1_y` as well as `paw_x1_x`, which would pair
  one body part's x with another's y for such names. Now matched by suffix.
- Tracking files were globbed unsorted, so summary rows came out in whichever
  order the filesystem gave; and video lookup was case-sensitive, failing on
  Linux for `.MP4`. Both are now sorted and case-insensitive.
- Two training files reducing to the same recording key silently dropped one,
  filesystem order deciding which. Now refused with both names reported.
- MCC and balanced accuracy had two implementations; now one, pinned to
  scikit-learn by test.
- `simple_velocity_feature()` was reachable only from its own test. Removed.
- `.claude/`, `.vscode/` and `.idea/` are ignored, so machine-local settings do
  not travel with the tool.

Tests: 34 -> 48. Verified on Windows and on Linux (WSL2), with cross-validation
output identical to the digit on both.
