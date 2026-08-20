#!/usr/bin/env python
"""Train a freezing model and score it against the labels, on any platform.

With no arguments this retrains the shipped model from the bundled demo videos
and their manual labels, writing ``output/retrained_model.sav``. Point the
options somewhere else to train on your own data.

    python scripts/train_model.py
    python scripts/train_model.py --dlc-dir my/tracking --labels-dir my/labels \
        --out-model my/model.sav --feature-mode legacy4k

The scoring pass at the end runs over every labelled video, including the ones
just fitted on, so it reports ``is_out_of_sample: false``. Use ``cross-validate``
for a figure that describes a new animal.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dlc-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "original_videos")
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "manual_labels")
    parser.add_argument("--out-model", type=Path, default=PROJECT_ROOT / "output" / "retrained_model.sav")
    parser.add_argument("--feature-cache-dir", type=Path, default=PROJECT_ROOT / "output" / "feature_cache")
    parser.add_argument("--feature-mode", default="stillness", choices=["stillness", "compact", "legacy4k"])
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-train-frames", type=int, default=400000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--label-lag-frames",
        type=int,
        default=2,
        help="Compensates for the delay between a behavior starting and the observer "
        "pressing the key. The bundled models use 2 frames, about 80 ms at 25 fps.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Train without the scoring pass.")
    return parser.parse_args()


def run(step: str, argv: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    # The child must find the package on a fresh checkout with nothing
    # installed, exactly as the desktop launchers arrange it.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")) if p
    )
    result = subprocess.run([sys.executable, "-m", "freezing_dlc.cli", *argv], env=env)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()

    for directory in (args.out_model.parent, args.feature_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run(
        f"Training from '{args.dlc_dir}' and '{args.labels_dir}'",
        [
            "train-model",
            "--dlc-dir", str(args.dlc_dir),
            "--labels-dir", str(args.labels_dir),
            "--out-model", str(args.out_model),
            "--feature-mode", args.feature_mode,
            "--split-mode", "group",
            "--n-estimators", str(args.n_estimators),
            "--step-estimators", "100",
            "--stabilize",
            "--stability-patience", "4",
            "--stability-min-delta", "0.0005",
            "--min-estimators-for-stability", "200",
            "--max-train-frames", str(args.max_train_frames),
            "--feature-cache-dir", str(args.feature_cache_dir),
            "--n-jobs", str(args.n_jobs),
            "--test-size", str(args.test_size),
            "--random-state", str(args.random_state),
            "--label-lag-frames", str(args.label_lag_frames),
        ],
    )

    if not args.skip_eval:
        run(
            "Evaluating on the full labeled set",
            [
                "eval-model",
                "--model", str(args.out_model),
                "--dlc-dir", str(args.dlc_dir),
                "--labels-dir", str(args.labels_dir),
                "--feature-cache-dir", str(args.feature_cache_dir),
            ],
        )

    print(f"\nDone. Model saved at '{args.out_model}'")


if __name__ == "__main__":
    main()
