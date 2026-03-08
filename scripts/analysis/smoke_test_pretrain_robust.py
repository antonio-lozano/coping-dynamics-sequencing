#!/usr/bin/env python
"""Smoke test for the robust pretraining + inference pipeline.

Validates:
1. ``pretrain_xgb_robust.py`` trains a model with augmentation, extended
   features, and global normalization.
2. ``normalization_stats.json`` is saved alongside the model.
3. ``predict_behavior_xgb.py`` loads the saved normalization stats and
   produces predictions.
4. Extended features are present in the model schema.
5. Schema stability: shuffled/extra columns are handled correctly.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.ml.behavior_xgb import align_features_for_inference  # noqa: E402
from src.ml.pose_features import compute_kinematic_features, load_dlc_csv  # noqa: E402

# Ensure scripts/analysis is importable for extended_features
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from extended_features import (  # noqa: E402
    augment_pose,
    compute_extended_features,
    load_normalization_stats,
    apply_global_normalizer,
    has_extended_features,
    preprocess_coordinates,
)


def _write_dlc_csv(path: Path, n_frames: int, seed: int) -> None:
    """Write a synthetic DLC CSV with realistic random-walk tracks."""
    rng = np.random.default_rng(seed)
    # Include nose and S1 so coordinate preprocessing can compute reference distance
    bodyparts = ["nose", "tail", "B1R", "B1L", "S1"]
    coords = ["x", "y", "likelihood"]
    cols = pd.MultiIndex.from_product([bodyparts, coords])
    data = np.zeros((n_frames, len(cols)), dtype=float)

    for b_idx, bp in enumerate(bodyparts):
        x = np.cumsum(rng.normal(0, 0.2, size=n_frames)) + (b_idx + 1) * 5
        y = np.cumsum(rng.normal(0, 0.2, size=n_frames)) + (b_idx + 1) * 3
        likelihood = np.clip(rng.normal(0.95, 0.01, size=n_frames), 0.0, 1.0)
        data[:, b_idx * 3 + 0] = x
        data[:, b_idx * 3 + 1] = y
        data[:, b_idx * 3 + 2] = likelihood

    df = pd.DataFrame(data, columns=cols)
    df.index.name = "frame"
    df.to_csv(path)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="robust_xgb_smoke_") as tmp:
        tmp_path = Path(tmp)
        train_dlc = tmp_path / "train_dlc"
        predict_dlc = tmp_path / "predict_dlc"
        models_dir = tmp_path / "models"
        preds_dir = tmp_path / "predictions"
        train_dlc.mkdir(parents=True, exist_ok=True)
        predict_dlc.mkdir(parents=True, exist_ok=True)

        # ---- Synthetic training data ----
        rec_a = "Animal1DLC_session1"
        rec_b = "Animal2DLC_session1"
        _write_dlc_csv(train_dlc / f"{rec_a}.csv", n_frames=200, seed=1)
        _write_dlc_csv(train_dlc / f"{rec_b}.csv", n_frames=200, seed=2)

        index_csv = tmp_path / "index.csv"
        pd.DataFrame({"name": [rec_a, rec_b], "group": ["Control", "ELS"]}).to_csv(
            index_csv, index=False,
        )

        labels_a = np.tile(np.array([1, 2, 3, 4, 5, 6, 7]), 30)[:200]
        labels_b = np.tile(np.array([7, 6, 5, 4, 3, 2, 1]), 30)[:200]
        results = {rec_a: {"syllable": labels_a}, rec_b: {"syllable": labels_b}}
        results_pkl = tmp_path / "new_results_clusters.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(results, f)

        # ---- 1. Unit tests for helpers ----
        print("Unit tests ...")

        # Coordinate preprocessing
        raw_csv = train_dlc / f"{rec_a}.csv"
        raw_pose = load_dlc_csv(raw_csv)
        pp_pose, ref_dist = preprocess_coordinates(raw_pose, ref_bodyparts=("nose", "S1"))
        assert pp_pose.shape == raw_pose.shape, "Coordinate preprocessing changed shape!"
        assert ref_dist > 0, "Reference distance should be positive!"
        # Centroid should be ~0 after centering
        x_cols = [c for c in pp_pose.columns if c.endswith("_x")]
        y_cols = [c for c in pp_pose.columns if c.endswith("_y")]
        cx = pp_pose[x_cols].mean(axis=1)
        cy = pp_pose[y_cols].mean(axis=1)
        assert np.abs(cx.mean()) < 0.01, f"Centroid X should be ~0 after centering, got {cx.mean():.4f}"
        assert np.abs(cy.mean()) < 0.01, f"Centroid Y should be ~0 after centering, got {cy.mean():.4f}"
        print(f"  Coordinate preprocessing: ref_dist={ref_dist:.3f}, centroid ~0 ✓")

        # Augment pose
        aug_pose = augment_pose(raw_pose, seed=99)
        assert aug_pose.shape == raw_pose.shape, "Augmentation changed shape!"
        # Values should differ (augmentation is not identity)
        assert not np.allclose(
            raw_pose.to_numpy(dtype=float, na_value=0.0),
            aug_pose.to_numpy(dtype=float, na_value=0.0),
        ), "Augmentation did not change any values!"

        # Extended features
        base_feat = compute_kinematic_features(raw_pose)
        ext_feat = compute_extended_features(raw_pose, base_feat)
        assert len(ext_feat) == len(base_feat), "Extended features row count mismatch!"
        assert ext_feat.shape[1] > 0, "No extended features computed!"
        # Check some expected columns exist
        for marker in ["Lag1", "Delta ", "Body Area", "Body Curvature"]:
            found = any(marker in c for c in ext_feat.columns)
            assert found, f"Expected extended feature containing '{marker}' not found!"

        # has_extended_features detection
        combined_cols = list(base_feat.columns) + list(ext_feat.columns)
        assert has_extended_features(combined_cols), "has_extended_features should return True"
        assert not has_extended_features(list(base_feat.columns)), (
            "has_extended_features should return False for base-only features"
        )

        print("  Unit tests passed.")

        # ---- 2. Train with pretrain_xgb_robust.py ----
        print("Training robust model (--quick) ...")
        model_version = "robust_smoke_v1"
        train_cmd = [
            sys.executable,
            "scripts/analysis/pretrain_xgb_robust.py",
            "--dlc-dir", str(train_dlc),
            "--results-clusters-pkl", str(results_pkl),
            "--index-csv", str(index_csv),
            "--model-version", model_version,
            "--out-dir", str(models_dir),
            "--n-augmentations", "1",
            "--quick",
        ]
        train_result = _run(train_cmd, cwd=repo_root)
        if train_result.returncode != 0:
            print("STDOUT:", train_result.stdout[-2000:])
            print("STDERR:", train_result.stderr[-2000:])
            raise RuntimeError("Robust training smoke test failed.")
        print("  Training succeeded.")

        # ---- 3. Check artifacts ----
        model_dir = models_dir / model_version
        required = ["model.json", "model.pkl", "metadata.json",
                     "cv_metrics.json", "normalization_stats.json"]
        for artifact in required:
            path = model_dir / artifact
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")
        print("  All artifacts present.")

        # Verify metadata
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["feature_normalization"]["method"] == "global_robust_iqr", (
            f"Expected global_robust_iqr, got {meta['feature_normalization']['method']}"
        )
        assert meta["extended_features"] is True, "Expected extended_features=True"
        assert meta["augmentation"]["enabled"] is False, (
            "Expected augmentation disabled in --quick mode"
        )
        # Verify coordinate preprocessing metadata
        assert meta.get("coordinate_preprocessing", {}).get("enabled", False), (
            "Expected coordinate_preprocessing.enabled=True in metadata"
        )

        # Verify normalization stats
        norm_stats = load_normalization_stats(model_dir / "normalization_stats.json")
        feature_names = meta["feature_names"]
        assert len(norm_stats) > 0, "Empty normalization stats!"
        # Every feature should have stats
        for fname in feature_names:
            assert fname in norm_stats, f"Feature '{fname}' missing from normalization stats!"
        print(f"  Normalization stats: {len(norm_stats)} features.")
        print(f"  Model features: {len(feature_names)}")

        # Extended features should be in the model schema
        assert has_extended_features(feature_names), (
            "Model feature list should contain extended features!"
        )
        print("  Extended features confirmed in model schema.")

        # ---- 4. Predict with the robust model ----
        print("Running inference ...")
        rec_pred = "Animal3DLC_session1"
        _write_dlc_csv(predict_dlc / f"{rec_pred}.csv", n_frames=150, seed=3)

        predict_cmd = [
            sys.executable,
            "scripts/analysis/predict_behavior_xgb.py",
            "--model-dir", str(model_dir),
            "--input-dir", str(predict_dlc),
            "--dataset-id", "robust_test_set",
            "--out-dir", str(preds_dir),
            "--glob", "*.csv",
        ]
        predict_result = _run(predict_cmd, cwd=repo_root)
        if predict_result.returncode != 0:
            print("STDOUT:", predict_result.stdout[-2000:])
            print("STDERR:", predict_result.stderr[-2000:])
            raise RuntimeError("Prediction with robust model failed.")

        pred_dir = preds_dir / "robust_test_set" / model_version
        for out_file in ["predictions_frame.parquet", "predictions_frame.csv",
                         "predictions_summary.csv"]:
            if not (pred_dir / out_file).exists():
                raise FileNotFoundError(f"Missing prediction output: {pred_dir / out_file}")
        print("  Prediction outputs present.")

        # Verify predictions are reasonable
        pred_frame = pd.read_csv(pred_dir / "predictions_frame.csv")
        assert len(pred_frame) == 150, f"Expected 150 frames, got {len(pred_frame)}"
        assert "pred_class" in pred_frame.columns
        assert "pred_confidence" in pred_frame.columns
        print(f"  Predictions: {len(pred_frame)} frames, "
              f"mean confidence = {pred_frame['pred_confidence'].mean():.3f}")

        # ---- 5. Schema stability check ----
        print("Schema stability check ...")
        raw_pose2 = load_dlc_csv(predict_dlc / f"{rec_pred}.csv")
        base_feats2 = compute_kinematic_features(raw_pose2)
        ext_feats2 = compute_extended_features(raw_pose2, base_feats2)
        combined2 = pd.concat([base_feats2, ext_feats2], axis=1)
        # Apply global normalization
        normed2 = apply_global_normalizer(combined2, norm_stats)
        # Shuffle + add extra column
        shuffled = normed2.sample(frac=1.0, axis=1, random_state=42).copy()
        shuffled["totally_new_feature"] = 999.0
        aligned = align_features_for_inference(shuffled, feature_names=feature_names)
        assert list(aligned.columns) == feature_names, "Aligned columns don't match!"
        assert "totally_new_feature" not in aligned.columns
        print("  Schema stability check passed.")

        print("\n" + "=" * 50)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 50)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
