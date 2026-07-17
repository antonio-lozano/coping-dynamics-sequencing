"""End-to-end raw video/DLC/prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dlc_features import dlc_tracks_to_pose_table
from .dlc_stage import run_deeplabcut_to_filtered
from .pipeline import load_bundle, predict_behaviors


def _output_stem(path: Path) -> str:
    stem = path.stem
    return stem.replace("DLC", "_DLC").split("_DLC")[0].rstrip("_")


def predict_from_dlc_file(
    *,
    dlc_file: Path,
    model_path: Path,
    output_dir: Path,
    fps: float = 25.0,
    name: str | None = None,
) -> tuple[Path, Path]:
    bundle = load_bundle(model_path)
    pose = dlc_tracks_to_pose_table(dlc_file, name=name or _output_stem(dlc_file), fps=fps)
    pred = predict_behaviors(pose, bundle)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = name or _output_stem(dlc_file)
    features_path = output_dir / f"{stem}_fig7_features.csv"
    pred_path = output_dir / f"{stem}_behavior_predictions.csv"
    pose.to_csv(features_path, index=False)
    pred.to_csv(pred_path, index=False)
    return features_path, pred_path


def run_from_raw_video(
    *,
    video: Path,
    dlc_config: Path,
    model_path: Path,
    output_root: Path,
    fps: float = 25.0,
    videotype: str | None = None,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    filtered_dir = output_root / "DLC_filtered"
    prediction_dir = output_root / "behavior_predictions"
    dlc_files = run_deeplabcut_to_filtered(
        config_path=dlc_config,
        videos=[video],
        filtered_dir=filtered_dir,
        videotype=videotype,
        reuse_existing=reuse_existing,
    )

    rows = []
    for dlc_file in dlc_files:
        features_path, pred_path = predict_from_dlc_file(
            dlc_file=dlc_file,
            model_path=model_path,
            output_dir=prediction_dir,
            fps=fps,
            name=video.stem,
        )
        rows.append(
            {
                "video": str(video),
                "dlc_file": str(dlc_file),
                "features_csv": str(features_path),
                "predictions_csv": str(pred_path),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = output_root / "behavior_prediction_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary
