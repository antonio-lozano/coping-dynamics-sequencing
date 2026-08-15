# Transferring the behavior classifier to new video setups — literature & attack paths

**Purpose:** survey how the state of the art transfers a DLC-keypoint-based supervised
behavior classifier to *new* mouse videos that differ in **camera distance** and **DLC
marker set**, and enumerate **all viable attack paths** with effort/payoff and what this
repo already implements.

**Context:** upcoming videos are still mice but with a different camera distance and a
different DLC marker set. Written 2026-06-13. Companion to `GUI_INTEGRATION_PLAN.md`.

---

## 1. The problem = two stacked domain shifts

| Shift | Cause | Effect on current model | Hard or soft? |
|---|---|---|---|
| **Scale / zoom** | Different camera distance | Pixel coords, pixel distances, pixel velocities change meaning | Soft (fixable by normalization) |
| **Schema mismatch** | Different DLC markers | The 719-feature vector cannot be reconstructed from different keypoints | **Hard** (must remap or re-track) |
| (Implicit) appearance | Lighting, arena, angle | Distribution drift in features | Soft |

A classifier on **raw** pixel features fails under both. The literature's answer is:
make features **invariant**, **reconcile the keypoints**, and **adapt with a little
target-domain signal**.

> ⚠️ **Two classifier lineages exist in this repo — don't confuse them.**
> - **Legacy SHAP model** (`models/behavior/xgb_model.pkl`, 719 features *including raw
>   `bp_x/bp_y` coords*, **no normalization**) — this is what the JSG GUI currently ships.
>   It is **not** transfer-ready: raw coordinates break under a different camera distance.
> - **Robust pipeline** (`pretrain_xgb_robust.py` + `extended_features.py`, body-length
>   normalized, pose-augmented, global median/IQR norm, common-bodypart only) — already
>   validated cross-dataset on AYA (`robust_light_no_tail_v2`, 89.8% OOF acc). **This is
>   the correct foundation to extend for the new videos.**

---

## 2. State of the art, by layer

### Layer A — Pose / keypoint (fixes "different markers")
- **Foundation pose models w/ standardized keypoints.** [SuperAnimal-TopViewMouse](https://www.nature.com/articles/s41467-024-48792-2)
  (Ye et al., *Nat Commun* 2024; [arXiv 2203.07436](https://arxiv.org/pdf/2203.07436)):
  27 standardized top-view mouse keypoints, **zero-shot across labs**, plus an **optimal
  keypoint-matching algorithm to align out-of-distribution datasets**. Re-tracking both
  cohorts with one shared skeleton dissolves the mismatch.
- **Keypoint remapping / correspondence.** Map the new skeleton onto the trained schema,
  NaN ambiguous/missing markers. *This repo already does it* in `map_aya_to_jen.py`.
- **Common-subset modeling.** Train on the intersection of shared keypoints. *Already
  partially done* via `UNMAPPED_BODYPARTS` / `--exclude-bodyparts`.

### Layer B — Features (fixes "camera distance" + robustness)
- **Metric normalization (px→mm).** [SimBA](https://www.nature.com/articles/s41593-024-01649-9)
  (Goodwin et al., *Nat Neurosci* 2024): user draws a known arena dimension; all distances
  → mm, velocities → mm/s. *The JSG GUI exposes `box_side_cm`/`box_side_pixels` but the
  behavior path doesn't use them.*
- **Egocentric alignment + scale invariance.** [keypoint-MoSeq](https://www.nature.com/articles/s41592-024-02318-2)
  (Weinreb et al., *Nat Methods* 2024): egocentric alignment along the **tail–nose axis**;
  generalizes across overhead/bottom-up, 2D/3D, mice & rats. General pose work
  ([ContrastivePose](https://www.sciencedirect.com/science/article/abs/pii/S0010482523008818),
  2023): recenter on centroid, rescale by body length to unit, rotate to body axis. *This
  repo's `extended_features.preprocess_coordinates` already does centroid centering +
  body-length normalization (`ref pair = nose,S1`).*
- **Robust statistical normalization.** Global median/IQR with clipping (robust to DLC
  spikes). *Already implemented* (`normalization_stats.json`).
- **Learned / self-supervised features.** [TREBA/task programming](https://arxiv.org/abs/2104.02710),
  [B-SOiD](https://www.nature.com/articles/s41467-021-25420-x) embeddings as transferable inputs.

### Layer C — Classifier (adapts the decision boundary)
- **Data-efficient active learning.** [A-SOiD](https://www.nature.com/articles/s41592-024-02200-1)
  (Tillmann et al., *Nat Methods* 2024): working classifier with **~85% less labeled data**, GUI-driven.
- **Limited-data / annotator transfer benchmark.** [CalMS21/MABe](https://arxiv.org/abs/2104.02710)
  (Sun et al., NeurIPS 2021): explicit tasks for limited-data new behaviors & annotator style transfer.
- **Self-/semi-supervised + domain adaptation.** [Selfee](https://elifesciences.org/articles/76218)
  (zero-shot mouse→rat); [unsupervised domain adaptation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10603736/)
  aligns source/target feature distributions.

### Layer D — Unsupervised re-discovery (MoSeq-native)
- Re-fit **keypoint-MoSeq** on the new videos, then map new syllables → 7 behaviors
  (`map_syllables_to_clusters.py`). Sidesteps both shifts; only the labeling scheme transfers.

---

## 3. All potential attack paths

Status legend: ✅ already in repo · 🟡 partially present · ⬜ not yet.

### A. Keypoint / skeleton reconciliation (for different markers)

| # | Path | Addresses | Effort | Payoff | Status | Ref |
|---|---|---|---|---|---|---|
| A1 | **Re-track both cohorts with SuperAnimal-TopViewMouse** (one shared 27-kp skeleton) | schema | High | High (removes mismatch at the source) | ⬜ | SuperAnimal |
| A2 | **Remap new markers → JEN 14-kp schema**, NaN the rest | schema | Low | Med (limited by anatomical overlap) | ✅ `map_aya_to_jen.py` | — |
| A3 | **Common-subset model**: train only on keypoints shared by both setups | schema | Low | Med (lower ceiling, robust) | 🟡 `--exclude-bodyparts`, `UNMAPPED_BODYPARTS` | — |
| A4 | **Re-label new videos with the original 14-kp skeleton & train a new DLC model** | schema | High | High (native features, no remap) | ⬜ | DLC |
| A5 | **Impute missing markers** from present ones (learned regression) | schema | Med | Med (risk of fabricated geometry) | ⬜ | — |

### B. Feature invariance (for camera distance / scale)

| # | Path | Addresses | Effort | Payoff | Status | Ref |
|---|---|---|---|---|---|---|
| B1 | **Egocentric alignment** (recenter centroid + rotate to tail–nose axis) | scale+pos+rot | Low | High | 🟡 centroid centering done; rotation TBD | keypoint-MoSeq |
| B2 | **Body-length scale normalization** (divide by a reference length) | scale | Low | High | ✅ `preprocess_coordinates` | ContrastivePose |
| B3 | **Metric px→mm calibration** (known arena dim) | scale | Low | High | 🟡 GUI fields exist, unused by behavior path | SimBA |
| B4 | **Drop raw-coordinate features**; keep distances-ratios + angles + angular velocity only | scale+pos | Low | High (biggest fix for legacy model) | 🟡 robust path reduces reliance | — |
| B5 | **Global median/IQR normalization** saved & reapplied at inference | drift | Low | Med | ✅ `normalization_stats.json` | — |
| B6 | **Pose augmentation** (rotation/translation/scale jitter) to simulate rig differences | all soft | Low | Med-High | ✅ in `pretrain_xgb_robust.py` | — |
| B7 | **Self-supervised / learned features** (TREBA, B-SOiD, contrastive) | all | High | Med-High | ⬜ | TREBA / B-SOiD |

### C. Classifier adaptation (decision boundary)

| # | Path | Needs new labels? | Effort | Payoff | Status | Ref |
|---|---|---|---|---|---|---|
| C1 | **Zero-shot apply** (invariant features only) — baseline | No | Low | Low-Med | 🟡 (robust model on AYA) | — |
| C2 | **Retrain on invariant features** (old labels, new feature space) | No | Med | High | 🟡 robust pipeline | — |
| C3 | **Fine-tune / recalibrate on a small labeled new-setup set** | Yes (few) | Med | High | ⬜ | A-SOiD |
| C4 | **Active learning** to choose which new clips to label | Yes (few) | Med | High (label-efficient) | ⬜ | A-SOiD |
| C5 | **Unsupervised domain adaptation** (align source/target feature dists, e.g. CORAL) | No | Med-High | Med | ⬜ | UDA |
| C6 | **Self-training / pseudo-labeling** on confident new-domain frames | No | Med | Med | ⬜ | Selfee |
| C7 | **Per-setup probability calibration + threshold re-tuning** | A few | Low | Med | ⬜ | — |
| C8 | **Mixture-of-experts / per-setup heads** sharing a backbone | Yes | High | Med | ⬜ | — |

### D. Unsupervised re-discovery (bypass classifier transfer)

| # | Path | Effort | Payoff | Status | Ref |
|---|---|---|---|---|---|
| D1 | **Re-fit keypoint-MoSeq on new videos → map syllables to 7 behaviors** | Med | High (native to this project) | 🟡 mapping code exists | keypoint-MoSeq |
| D2 | **B-SOiD / VAME unsupervised clustering on new videos → map clusters** | Med | Med | ⬜ | B-SOiD |

### E. Validation / QC (how to trust transfer without full labels)

| # | Path | Status |
|---|---|---|
| E1 | Hand-label a small hold-out from the new setup; report P/R/F1 | ⬜ |
| E2 | Sanity-check behavior proportions & transition structure vs. known biology | 🟡 figures 5–8 do this for JEN/AYA |
| E3 | Feature-shift diagnostics between old & new domains | ✅ `debug_feature_shift.py`, `feature_shift_jen_vs_aya.csv` |
| E4 | Confidence calibration / %-low-confidence monitoring | ✅ in `predictions_summary.csv` |

---

## 4. Recommended attack sequence (cheapest-highest-payoff first)

1. **B-layer first (invariance).** Finish egocentric alignment (add rotation, B1) and wire
   metric calibration (B3) into the robust feature path; ensure no raw-coordinate features
   leak (B4). This is the single biggest lever and mostly already scaffolded.
2. **A-layer (keypoints).** Decide remap (A2/A3) vs. standardized re-track (A1). If overlap
   with the 14-kp schema is poor, A1 (SuperAnimal) becomes the better long-term choice.
3. **Retrain on invariant features (C2)** with pose augmentation (B6) → get a zero-shot
   baseline on the new setup (C1).
4. **Measure on a small hand-labeled hold-out (E1).** If short, **active-learn + fine-tune
   (C3/C4)**. This is the accepted remedy when zero-shot underperforms.
5. **Keep D1 (keypoint-MoSeq re-fit) as a parallel validation / fallback.**

**Decision driver = how different the new keypoints are, and how many new labels exist.**
- Good keypoint overlap + some labels → A2/A3 + B + C2/C3.
- Poor overlap → A1 (standardized re-track) + B + C2/C3.
- No labels at all → lean on B + C1/C5/C6 and D1, with E2/E3 for trust.

---

## 5. Open questions to resolve with the new-video context
- Does the new DLC skeleton share any of the 14 JEN bodyparts? How many?
- Camera-distance ratio vs. the JEN rig (e.g., 2× zoom-out, full arena vs. crop)?
- Any hand-labeled behavior on the new videos, or must transfer be zero-shot?
- Is a known arena dimension available for px→mm calibration (B3)?
- Same frame rate (25 fps) and view (top-down)?

## 6. Sources
SuperAnimal — Nat Commun 2024: https://www.nature.com/articles/s41467-024-48792-2 ·
SimBA — Nat Neurosci 2024: https://www.nature.com/articles/s41593-024-01649-9 ·
keypoint-MoSeq — Nat Methods 2024: https://www.nature.com/articles/s41592-024-02318-2 ·
A-SOiD — Nat Methods 2024: https://www.nature.com/articles/s41592-024-02200-1 ·
CalMS21/MABe — NeurIPS 2021: https://arxiv.org/abs/2104.02710 ·
B-SOiD — Nat Commun 2021: https://www.nature.com/articles/s41467-021-25420-x ·
Selfee — eLife 2022: https://elifesciences.org/articles/76218 ·
UDA for animal activity recognition: https://pmc.ncbi.nlm.nih.gov/articles/PMC10603736/ ·
ContrastivePose — 2023: https://www.sciencedirect.com/science/article/abs/pii/S0010482523008818
