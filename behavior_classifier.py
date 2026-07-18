#!/usr/bin/env python
"""
behavior_classifier.py — single entry point for the reusable behavior classifier.

This is the file to grow the model in. It wraps the package in
`src/behavior_classifier/` (feature extraction, training, CLI) and exposes one
place to (re)build, evaluate, and apply a per-frame behavior classifier that
annotates the seven ethological clusters — freeze, sniff, groom, turn,
locomotion, climb, jump — directly from pose data, without re-running keypoint
MoSeq.

The Fig. 7 model (XGBoost on egocentric pose features) is the starting point.
Extend the TODO sections below to iterate on the production model.

Cluster ↔ syllable map (hand-curated, see docs/behavior_classifier.md):
    Freeze: 0, 28        Sniff: 18, 20       Groom: 24
    Turn: 1, 3, 5, 6, 10, 15, 26, 27
    Locomotion: 11, 12, 14, 16, 19, 21, 25
    Climb: 111           Jump: 23, 29, 30, 34
    Unassigned: all other syllables

Usage
-----
    # Train / evaluate on the bundled MoSeq feature table (Fig. 7 reproduction)
    python behavior_classifier.py train

    # Train directly from DeepLabCut tracks
    python behavior_classifier.py train-dlc --help

Outputs (CV metrics, confusion matrix, predictions) are written under
results/intermediate/tables/ by default; models are written under
results/models/ by default. See `src/behavior_classifier/cli.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.behavior_classifier.cli import main

# ---------------------------------------------------------------------------
# TODO — future model build (kept here so the classifier has one home):
#   [ ] swap/compare estimators (XGBoost -> gradient boosting / temporal model)
#   [ ] hyperparameter search + held-out test split
#   [ ] persist the fitted model (e.g. models/behavior_classifier.joblib)
#   [ ] add an `apply` command: load a saved model + new DLC csv -> annotations
#   [ ] calibrate per-cluster thresholds; report per-cluster precision/recall
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
