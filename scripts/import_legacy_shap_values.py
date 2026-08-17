"""Repack the legacy class-specific SHAP values into a tracked archive.

Supplementary Figure 4 is drawn from the legacy classifier's SHAP values, which
live outside this repository as a joblib pickle written under scikit-learn
1.2.1.  Pickles are not a durable archive format - they already warn on newer
scikit-learn and can stop loading altogether - so this one-time utility converts
them into a single compressed ``.npz`` holding plain arrays::

    <source>/shap_values.pkl   (8 classes x frames x 719)
    <source>/shap_sample.csv   (feature values behind the dot colours)
    <source>/label_encoder.pkl (class order)

        -> classifier/legacy_shap/shap_values.npz

Once written, the figure regenerates from the repository alone and no longer
needs ``--source``.  Values are preserved exactly (float32, verified on write).

Run:
    python scripts/import_legacy_shap_values.py --source DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "classifier" / "legacy_shap" / "shap_values.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Legacy keypoint-MoSeq code/shapley/results directory",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Legacy SHAP source not found: {args.source}")

    shap_values = joblib.load(args.source / "shap_values.pkl")
    encoder = joblib.load(args.source / "label_encoder.pkl")
    sample = pd.read_csv(args.source / "shap_sample.csv")

    classes = [str(value) for value in encoder.classes_]
    if len(shap_values) != len(classes):
        raise AssertionError("SHAP value list does not match the label encoder")

    # (classes, frames, features); keep the library's native float32 so the
    # magnitudes reduce exactly as they did in the legacy analysis.
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in shap_values])
    features = np.asarray(sample.columns, dtype=object)
    values = sample.to_numpy(dtype=np.float32)
    if stacked.shape[1:] != values.shape:
        raise AssertionError(
            f"SHAP array {stacked.shape[1:]} does not match the sample {values.shape}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        shap_values=stacked,
        feature_values=values,
        feature_names=features,
        class_names=np.asarray(classes, dtype=object),
    )

    with np.load(args.output, allow_pickle=True) as archive:
        if not np.array_equal(archive["shap_values"], stacked):
            raise AssertionError("SHAP values did not round-trip through the archive")
        if not np.array_equal(archive["feature_values"], values):
            raise AssertionError("Feature values did not round-trip through the archive")

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"Wrote {args.output.relative_to(REPO)} ({size_mb:.1f} MB)")
    print(f"  {stacked.shape[0]} classes x {stacked.shape[1]} frames x {stacked.shape[2]} parameters")


if __name__ == "__main__":
    main()
