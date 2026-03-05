#!/usr/bin/env python
"""Synthetic smoke test for train/predict/visualize XGBoost workflow."""
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


def _write_dlc_csv(path: Path, n_frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
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
    with tempfile.TemporaryDirectory(prefix="xgb_workflow_smoke_") as tmp:
        tmp_path = Path(tmp)
        train_dlc = tmp_path / "train_dlc"
        predict_dlc = tmp_path / "predict_dlc"
        models_dir = tmp_path / "models"
        preds_dir = tmp_path / "predictions"
        train_dlc.mkdir(parents=True, exist_ok=True)
        predict_dlc.mkdir(parents=True, exist_ok=True)

        rec_train_a = "Animal1DLC_session1"
        rec_train_b = "Animal2DLC_session1"
        _write_dlc_csv(train_dlc / f"{rec_train_a}.csv", n_frames=180, seed=1)
        _write_dlc_csv(train_dlc / f"{rec_train_b}.csv", n_frames=180, seed=2)

        index_csv = tmp_path / "index.csv"
        pd.DataFrame(
            {
                "name": [rec_train_a, rec_train_b],
                "group": ["Control", "ELS"],
            }
        ).to_csv(index_csv, index=False)

        labels_a = np.tile(np.array([1, 2, 3, 4, 5, 6, 7]), 26)[:180]
        labels_b = np.tile(np.array([7, 6, 5, 4, 3, 2, 1]), 26)[:180]
        results = {
            rec_train_a: {"syllable": labels_a},
            rec_train_b: {"syllable": labels_b},
        }
        results_pkl = tmp_path / "new_results_clusters.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(results, f)

        model_version = "smoke_v1"
        train_cmd = [
            sys.executable,
            "scripts/analysis/train_behavior_xgb.py",
            "--dlc-dir",
            str(train_dlc),
            "--results-clusters-pkl",
            str(results_pkl),
            "--index-csv",
            str(index_csv),
            "--model-version",
            model_version,
            "--out-dir",
            str(models_dir),
            "--quick",
        ]
        train_result = _run(train_cmd, cwd=repo_root)
        if train_result.returncode != 0:
            print(train_result.stdout)
            print(train_result.stderr)
            raise RuntimeError("Training smoke test failed.")

        model_dir = models_dir / model_version
        required_artifacts = ["model.json", "model.pkl", "metadata.json", "cv_metrics.json"]
        for artifact in required_artifacts:
            if not (model_dir / artifact).exists():
                raise FileNotFoundError(f"Missing artifact: {model_dir / artifact}")

        # Prepare prediction data: one valid file, one malformed file.
        rec_pred = "Animal3DLC_session1"
        _write_dlc_csv(predict_dlc / f"{rec_pred}.csv", n_frames=150, seed=3)
        (predict_dlc / "bad_DLC_file.csv").write_text("this,is,not,a,valid,dlc,csv\n1,2,3,4,5,6,7\n")

        predict_cmd = [
            sys.executable,
            "scripts/analysis/predict_behavior_xgb.py",
            "--model-dir",
            str(model_dir),
            "--input-dir",
            str(predict_dlc),
            "--dataset-id",
            "test_set",
            "--out-dir",
            str(preds_dir),
            "--glob",
            "*.csv",
        ]
        predict_result = _run(predict_cmd, cwd=repo_root)
        if predict_result.returncode != 0:
            print(predict_result.stdout)
            print(predict_result.stderr)
            raise RuntimeError("Prediction smoke test failed.")

        pred_dir = preds_dir / "test_set" / model_version
        frame_parquet = pred_dir / "predictions_frame.parquet"
        frame_csv = pred_dir / "predictions_frame.csv"
        summary_csv = pred_dir / "predictions_summary.csv"
        for path in (frame_parquet, frame_csv, summary_csv):
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction output: {path}")

        # Schema stability check: shuffled columns + extra column are aligned to metadata schema.
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        feature_names = metadata["feature_names"]
        raw_pose = load_dlc_csv(predict_dlc / f"{rec_pred}.csv")
        feats = compute_kinematic_features(raw_pose)
        shuffled = feats.sample(frac=1.0, axis=1, random_state=42).copy()
        shuffled["new_unseen_feature"] = 123.0
        aligned = align_features_for_inference(shuffled, feature_names=feature_names)
        if list(aligned.columns) != feature_names:
            raise AssertionError("Aligned features do not match training schema.")

        # Missing-model failure mode check.
        missing_model_cmd = [
            sys.executable,
            "scripts/analysis/predict_behavior_xgb.py",
            "--model-dir",
            str(tmp_path / "missing_model_dir"),
            "--input-dir",
            str(predict_dlc),
            "--dataset-id",
            "fail_set",
            "--out-dir",
            str(preds_dir),
        ]
        missing_result = _run(missing_model_cmd, cwd=repo_root)
        if missing_result.returncode == 0:
            raise AssertionError("Expected missing model-dir command to fail.")

        # Empty-input failure mode check.
        empty_input = tmp_path / "empty_input"
        empty_input.mkdir(parents=True, exist_ok=True)
        empty_cmd = [
            sys.executable,
            "scripts/analysis/predict_behavior_xgb.py",
            "--model-dir",
            str(model_dir),
            "--input-dir",
            str(empty_input),
            "--dataset-id",
            "empty_set",
            "--out-dir",
            str(preds_dir),
        ]
        empty_result = _run(empty_cmd, cwd=repo_root)
        if empty_result.returncode == 0:
            raise AssertionError("Expected empty input-dir command to fail.")

        viz_cmd = [
            sys.executable,
            "scripts/analysis/visualize_behavior_predictions.py",
            "--predictions-dir",
            str(pred_dir),
        ]
        viz_result = _run(viz_cmd, cwd=repo_root)
        if viz_result.returncode != 0:
            print(viz_result.stdout)
            print(viz_result.stderr)
            raise RuntimeError("Visualization smoke test failed.")

        qc_png = pred_dir / "figures" / "prediction_qc.png"
        qc_pdf = pred_dir / "figures" / "prediction_qc.pdf"
        if not qc_png.exists() or not qc_pdf.exists():
            raise FileNotFoundError("QC figure outputs were not generated.")

        print("Smoke test passed.")
        print(f"Model artifacts: {model_dir}")
        print(f"Prediction outputs: {pred_dir}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
