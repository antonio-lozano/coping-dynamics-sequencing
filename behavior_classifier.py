#!/usr/bin/env python
"""Single command-line entry point for the Figure 7 behavior classifier.

The implementation lives in `coping_dynamics/behavior_classifier/` and supports training,
cross-validation, prediction from MoSeq tables, and prediction from DLC-derived
pose tracks. The bundled classifier artifact is
`classifier/figure7_behavior_classifier.joblib`.

Usage
-----
    python behavior_classifier.py train
    python behavior_classifier.py train-from-dlc --help
    python behavior_classifier.py predict
"""

from __future__ import annotations

from coping_dynamics.behavior_classifier.cli import main

if __name__ == "__main__":
    main()
