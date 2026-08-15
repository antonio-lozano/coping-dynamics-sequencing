# Plan: Integrating the DLC Behavior GUI into coping-dynamics-sequencing

**Status:** PLAN ONLY — do not implement yet. This documents the target design for
folding the `JSG2026_behavior_classifier` end-to-end GUI/inference tool into this repo.
The immediate active workstream is *transferring the classifier to new video setups*
(different camera distance, different DLC markers); this integration follows that.

Author context: written 2026-06-13 after a runnability/compatibility audit of both repos.

---

## 1. Why integrate

Today the two repos are the two halves of one project:

- **coping-dynamics-sequencing** = *producer*: keypoint-MoSeq syllables → 7-behavior
  taxonomy → trains & explains the XGBoost classifier (`figure_9_shap.py`,
  `train_behavior_xgb.py`, `src/ml/`).
- **JSG2026_behavior_classifier** (`freezing_dlc`) = *consumer*: a packaged GUI/CLI that
  takes raw videos → DeepLabCut → features → applies the same XGBoost model → ethograms,
  annotated videos, Excel summaries.

The connecting artifact is `models/behavior/xgb_model.pkl` + `label_encoder.pkl`, which
originates from this repo's SHAP results. The consumer currently **re-implements** the
producer's feature logic and taxonomy by hand, which is the main risk we are removing.

## 2. Current state & problems (audit findings, 2026-06-13)

| Area | Finding |
|---|---|
| Feature builder | JSG `behavior.py:build_behavior_feature_set` (719 feats) is a hand-replica of this repo's `src/ml/` feature code. Silent drift = silently wrong predictions. |
| Taxonomy duplicated | `BEHAVIOR_MAPPING` (here, `src/config.py`) vs `BEHAVIOR_ORDER` / `BEHAVIOR_BODYPARTS` / `BEHAVIOR_COLORS` (JSG `behavior.py`). Two sources of truth. |
| Model carries 8 classes | The shipped model predicts the 7 behaviors **plus `Unspecified`** over 719 features. Any consumer must respect this. |
| Env conflict | This repo pins `xgboost>=1.5,<3.0`, `numpy<2`, `pandas<3`. JSG needs `xgboost>=3.2`. **No single pinned env satisfies both.** The pickle happens to load across versions today, but that is luck, not contract. |
| Broken launcher | JSG `launch_freezing_gui.bat` + `.vscode/tasks.json` hardcode `C:/Users/Lozano/...`, conda envs `sacroiliitis-ai` / `DLC-GPU`, and `H:/antonio/...`. None exist on the current machine. |
| Model format | Loaded via `joblib.load` on a pickle (version-fragile) rather than xgboost's native `model.json` (version-robust). This repo already emits `model.json` in `train_behavior_xgb.py`. |
| Packaging | JSG = setuptools + pip/conda story, no uv. This repo = conda `environment.yml`. Neither is a uv project. |
| Verified working | uv venv + `uv pip install -e ".[ml,video,dev]"` → 34/34 tests pass, model loads & predicts, GUI launches and runs clean. So the code is sound; only the *plumbing* is the problem. |

## 3. Target architecture

One shared core, two thin app layers:

```
coping_dynamics/
  behavior_core/                 # SINGLE SOURCE OF TRUTH (new)
    taxonomy.py                  # 7+1 labels, codes, colors, BEHAVIOR_BODYPARTS, FPS
    features.py                  # the ONE feature builder (719-feat scheme) used by BOTH
    model_io.py                  # load model.json (native) + metadata + feature schema
    schema.py                    # versioned feature-spec + invariance flags (see transfer work)
  training/                      # = today's src/ml + scripts/analysis (producer)
  app/                           # = vendored freezing_dlc GUI/CLI/pipeline (consumer)
    gui.py, cli.py, friendly_pipeline.py, dlc_stage.py, label_video.py, ...
```

Both `training/` and `app/` import `behavior_core`. Training writes a versioned model
artifact (`model.json` + `metadata.json`) that `app/` loads — no hand-copied features, no
bare pickles.

## 4. Phased plan (each phase independently shippable)

**Phase 0 — Environment unification (prerequisite).**
- Convert this repo to a uv project (`pyproject.toml` with `[tool.uv]`, lockfile);
  define extras: `[train]`, `[inference]`, `[gui]`, `[dlc]`.
- Resolve the xgboost pin conflict by moving to a single modern xgboost and **loading
  models via native `Booster.load_model(model.json)`** instead of joblib pickle. Re-save
  the current shipped model to `model.json` and validate predictions are identical.
- Replace the conda/`Lozano` launcher with `uv run coping-behavior-gui`.
- Exit criteria: one env runs both `figure_9_shap.py` and the GUI; predictions byte-stable.

**Phase 1 — Extract `behavior_core` (the high-leverage fix).**
- Move the 719-feature builder + taxonomy + bodyparts into `behavior_core`.
- Point both this repo's training/figures AND JSG's `behavior.py` at it.
- Add a golden-file regression test: same DLC CSV → identical feature matrix and identical
  predictions from old path vs new core. This is the test that prevents drift forever.
- Exit criteria: JSG `build_behavior_feature_set` is deleted and re-exported from core; all
  34 JSG tests + this repo's smoke tests pass against the shared core.

**Phase 2 — Vendor the app layer.**
- Bring `freezing_dlc` in as `coping_dynamics/app/` behind the `[gui]` extra.
- Keep the freezing `.sav` path separate from the 7-behavior path (two models, one GUI).
- Exit criteria: `uv run coping-behavior-gui` launches from this repo; raw-video→ethogram
  works when DLC is available, behavior-only works on pre-tracked CSVs.

**Phase 3 — Model registry & metadata.**
- Standardize `metadata.json`: feature-schema hash, label order, training lib versions,
  keypoint set/skeleton name, normalization spec, training cohort.
- GUI surfaces model version + warns loudly on feature-schema / keypoint-set mismatch.
- Exit criteria: loading a model with a mismatched keypoint set or feature hash fails fast
  with a human-readable message (directly supports the transfer work below).

**Phase 4 — Hook in the new-video transfer workflow.**
- The transfer pipeline (separate workstream) plugs in here: keypoint remap/normalization
  layer + optional re-label/fine-tune step + re-fit keypoint-MoSeq path. See
  `docs/` transfer design (to be written) and the literature review.

## 5. Non-goals / explicit "do not do yet"
- Do **not** merge the repos or move files yet.
- Do **not** change the shipped model or feature math until Phase 1 golden tests exist.
- Do **not** delete `JSG2026_behavior_classifier` until `app/` is proven in-repo.

## 6. Open questions
- Monorepo (`coping-dynamics[gui]`) vs. separate deploy repo depending on a published
  `behavior_core` package? (Bench users may want the GUI without the research stack.)
- Keep tkinter GUI, or converge on the existing `gui/` viewers (DearPyGui / PyQtGraph)?
- Where do trained models live for sharing — in-repo `results/model_training/` vs. an
  external model registry / release artifacts?
