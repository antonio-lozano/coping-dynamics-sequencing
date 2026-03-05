#!/usr/bin/env python
"""Apply a saved XGBoost behavior model to new DLC data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import PREDICTIONS_DIR, TO_PREDICT_DIR  # noqa: E402
from src.ml.behavior_xgb import (  # noqa: E402
    align_features_for_inference,
    load_model_artifacts,
    predict_probabilities,
)
from src.ml.pose_features import (  # noqa: E402
    compute_kinematic_features,
    load_dlc_csv,
    robust_normalize_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to trained model version directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing new DLC CSV files.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset identifier used in output path. Defaults to input-dir folder name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Base predictions directory.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*DLC*.csv",
        help="Glob pattern for input CSVs.",
    )
    parser.add_argument(
        "--exclude-bodyparts",
        type=str,
        default=None,
        help="Optional comma-separated bodyparts to exclude. Defaults to metadata.excluded_bodyparts.",
    )
    return parser.parse_args()


def _parse_bodyparts(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _make_summary(recording: str, pred_df: pd.DataFrame, class_names: list[str]) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "recording": recording,
        "n_frames": int(len(pred_df)),
        "mean_confidence": float(pred_df["pred_confidence"].mean()),
        "median_confidence": float(pred_df["pred_confidence"].median()),
        "%_low_confidence(<0.5)": float((pred_df["pred_confidence"] < 0.5).mean() * 100.0),
    }
    for cls in class_names:
        row[f"%_{cls}"] = float((pred_df["pred_class"] == cls).mean() * 100.0)
    return row


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("Predicting behavior classes with saved XGBoost model")
    print("=" * 72)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    model, metadata, load_source = load_model_artifacts(args.model_dir)
    class_names = list(metadata.get("class_names", []))
    feature_names = list(metadata.get("feature_names", []))
    if not class_names or not feature_names:
        raise ValueError("Metadata must include class_names and feature_names.")

    print(f"Loaded model from {args.model_dir} (source={load_source}).")
    print(f"Expected features: {len(feature_names)}; classes: {class_names}")
    norm_cfg = metadata.get("feature_normalization", {})
    norm_enabled = bool(norm_cfg.get("applied", False))
    norm_method = str(norm_cfg.get("method", "none"))
    norm_clip = float(norm_cfg.get("clip_value", 4.0) or 4.0)
    if norm_enabled:
        print(f"Feature normalization enabled from metadata: method={norm_method}, clip={norm_clip}")

    meta_excluded = {str(x).strip().lower() for x in metadata.get("excluded_bodyparts", [])}
    arg_excluded = _parse_bodyparts(args.exclude_bodyparts)
    excluded_bodyparts = arg_excluded if args.exclude_bodyparts is not None else meta_excluded
    if excluded_bodyparts:
        print(f"Excluding bodyparts at inference: {sorted(excluded_bodyparts)}")

    input_csvs = sorted(args.input_dir.glob(args.glob))
    if not input_csvs:
        raise FileNotFoundError(
            f"No DLC CSV files found in {args.input_dir} matching pattern '{args.glob}'."
        )

    model_version = args.model_dir.name
    dataset_id = args.dataset_id or args.input_dir.name
    output_dir = args.out_dir / dataset_id / model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset id: {dataset_id}")

    frame_rows = []
    summary_rows = []
    failed = []

    for csv_path in input_csvs:
        try:
            pose_df = load_dlc_csv(csv_path)
            feat_df = compute_kinematic_features(
                pose_df,
                excluded_bodyparts=excluded_bodyparts,
            )
            if norm_enabled and norm_method == "robust_iqr_per_recording":
                feat_df, _ = robust_normalize_features(
                    feat_df,
                    feature_cols=list(feat_df.columns),
                    clip_value=norm_clip,
                )
            aligned = align_features_for_inference(feat_df, feature_names=feature_names)
            pred_df = predict_probabilities(model, aligned, class_names=class_names)
            pred_df.insert(0, "frame", range(len(pred_df)))
            pred_df.insert(0, "recording", csv_path.stem)
            frame_rows.append(pred_df)
            summary_rows.append(_make_summary(csv_path.stem, pred_df, class_names))
            print(f"  Processed {csv_path.name}: {len(pred_df)} frames")
        except Exception as exc:
            failed.append((csv_path.name, str(exc)))
            print(f"  Skip {csv_path.name}: {exc}")
            continue

    if not frame_rows:
        fail_msg = "\n".join([f"  - {name}: {msg}" for name, msg in failed])
        raise RuntimeError(f"No valid CSVs processed.\nFailures:\n{fail_msg}")

    frame_df = pd.concat(frame_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("recording").reset_index(drop=True)

    frame_parquet = output_dir / "predictions_frame.parquet"
    frame_csv = output_dir / "predictions_frame.csv"
    summary_csv = output_dir / "predictions_summary.csv"

    frame_df.to_parquet(frame_parquet, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved prediction outputs:")
    print(f"  - {frame_parquet}")
    print(f"  - {frame_csv}")
    print(f"  - {summary_csv}")

    if failed:
        print("\nCSV files skipped due to read/parse errors:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
