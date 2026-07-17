"""Training helpers for DLC-derived SHAP-style behavior features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .annotations import behavior_lookup
from .dlc_features import dlc_tracks_to_pose_table
from .pipeline import cross_validate, save_bundle, train_behavior_classifier


def _normal_stem(path: Path) -> str:
    stem = path.stem
    if "DLC" in stem:
        stem = stem.split("DLC")[0].rstrip("_")
    return stem


def _read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    if "frame_index" not in labels.columns:
        first = labels.columns[0]
        labels = labels.rename(columns={first: "frame_index"})
    if "behavior_label" not in labels.columns:
        if "syllable" not in labels.columns:
            raise ValueError(
                f"{path} must contain either 'behavior_label' or 'syllable'."
            )
        lookup = behavior_lookup()
        labels["behavior_label"] = (
            pd.to_numeric(labels["syllable"], errors="coerce")
            .astype("Int64")
            .map(lookup)
            .fillna("Unassigned")
        )
    return labels[["frame_index", "behavior_label"]].copy()


def _find_label_file(labels_dir: Path, stem: str) -> Path | None:
    candidates = [
        labels_dir / f"{stem}.csv",
        labels_dir / f"{stem}_labels.csv",
        labels_dir / f"{stem}_behavior_labels.csv",
        labels_dir / f"{stem}_moseq.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(labels_dir.glob(f"{stem}*.csv"))
    return matches[0] if matches else None


def build_training_table_from_dlc(
    *,
    dlc_dir: Path,
    labels_dir: Path,
    fps: float = 25.0,
) -> pd.DataFrame:
    rows = []
    dlc_files = sorted(dlc_dir.glob("*filtered.csv")) + sorted(dlc_dir.glob("*filtered.h5"))
    if not dlc_files:
        raise FileNotFoundError(f"No '*filtered.csv' or '*filtered.h5' files found in {dlc_dir}")

    for dlc_file in dlc_files:
        stem = _normal_stem(dlc_file)
        label_file = _find_label_file(labels_dir, stem)
        if label_file is None:
            continue
        pose = dlc_tracks_to_pose_table(dlc_file, name=stem, fps=fps)
        labels = _read_labels(label_file)
        merged = pose.merge(labels, on="frame_index", how="inner")
        if not merged.empty:
            rows.append(merged)

    if not rows:
        raise FileNotFoundError(
            f"No DLC files in {dlc_dir} could be matched to label CSVs in {labels_dir}."
        )
    return pd.concat(rows, ignore_index=True)


def train_from_dlc(
    *,
    dlc_dir: Path,
    labels_dir: Path,
    out_model: Path,
    metrics_path: Path,
    confusion_path: Path,
    training_table_path: Path,
    fps: float = 25.0,
    max_frames: int | None = None,
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.2,
    random_state: int = 42,
    skip_cv: bool = False,
) -> dict[str, str]:
    table = build_training_table_from_dlc(dlc_dir=dlc_dir, labels_dir=labels_dir, fps=fps)
    training_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(training_table_path, index=False)

    model_kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }
    if not skip_cv:
        metrics, confusion = cross_validate(
            table,
            include_unassigned=True,
            balanced_weights=True,
            random_state=random_state,
            max_frames=max_frames,
            model_kwargs=model_kwargs,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        confusion_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        confusion.to_csv(confusion_path, index=False)

    bundle = train_behavior_classifier(
        table,
        include_unassigned=True,
        balanced_weights=True,
        random_state=random_state,
        max_frames=max_frames,
        model_kwargs=model_kwargs,
    )
    save_bundle(bundle, out_model)
    return {
        "training_table": str(training_table_path),
        "model": str(out_model),
        "metrics": str(metrics_path),
        "confusion": str(confusion_path),
    }
