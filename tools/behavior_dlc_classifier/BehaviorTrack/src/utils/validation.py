"""Independent target-domain validation for the seven-behavior classifier.

Prediction files are never copied into annotation templates: a human annotator
receives frame/time only, so the model call cannot anchor the label. Validation
then joins manual labels to frozen model output and reports every class and
recording rather than treating pooled frames as independent replicates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BEHAVIORS = ("Jump", "Climbing", "Locomotion", "Turn", "Grooming", "Sniffing", "Freezing")
PREDICTION_CLASSES = (*BEHAVIORS, "Unassigned")


def recording_stem(path: Path) -> str:
    stem = Path(path).stem
    for suffix in ("_manual_labels", "_labels", "_behaviors"):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
    return stem


def create_blinded_template(prediction_csv: Path, out_csv: Path, fps: float) -> Path:
    """Write frame/time and an empty truth column; never expose predictions."""
    source = pd.read_csv(prediction_csv)
    frame = (
        pd.to_numeric(source["frame"], errors="raise").astype(int)
        if "frame" in source
        else pd.Series(np.arange(len(source)), name="frame")
    )
    template = pd.DataFrame(
        {"frame": frame.to_numpy(), "time_sec": frame.to_numpy(dtype=float) / float(fps), "true_behavior": ""}
    )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_csv, index=False)
    return out_csv


def _truth_column(frame: pd.DataFrame) -> str:
    for name in ("true_behavior", "ground_truth", "manual_behavior", "label"):
        if name in frame.columns:
            return name
    raise ValueError(
        "Manual labels need a truth column named true_behavior, ground_truth, manual_behavior, or label."
    )


def _normalise_label(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unlabelled", "unlabeled", "skip"}:
        return None
    matches = {name.lower(): name for name in PREDICTION_CLASSES}
    return matches.get(text.lower(), text)


def _manual_frames(path: Path) -> pd.DataFrame:
    """Read per-frame labels or expand inclusive start/end-frame intervals."""
    labels = pd.read_csv(path)
    truth = _truth_column(labels)
    if "frame" in labels.columns:
        out = labels[["frame", truth]].rename(columns={truth: "true_behavior"})
    elif {"start_frame", "end_frame"}.issubset(labels.columns):
        rows: list[dict[str, object]] = []
        for start, end, behavior in labels[["start_frame", "end_frame", truth]].itertuples(index=False, name=None):
            start, end = int(start), int(end)
            if end < start:
                raise ValueError(f"Invalid interval {start}..{end} in {path.name}")
            rows.extend({"frame": value, "true_behavior": behavior} for value in range(start, end + 1))
        out = pd.DataFrame(rows, columns=["frame", "true_behavior"])
    else:
        raise ValueError(f"{path.name}: labels need frame, or start_frame and end_frame columns")
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int)
    out["true_behavior"] = out["true_behavior"].map(_normalise_label)
    out = out.dropna(subset=["true_behavior"])
    if out["frame"].duplicated().any():
        raise ValueError(f"{path.name}: a frame has more than one manual label")
    unknown = sorted(set(out["true_behavior"]) - set(BEHAVIORS))
    if unknown:
        raise ValueError(f"{path.name}: unknown manual behavior(s): {', '.join(unknown)}")
    return out


def _prediction_frames(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if "behavior" not in predictions:
        raise ValueError(f"{path.name}: prediction file has no behavior column")
    frame = (
        pd.to_numeric(predictions["frame"], errors="raise").astype(int)
        if "frame" in predictions
        else pd.Series(np.arange(len(predictions)))
    )
    out = pd.DataFrame({"frame": frame, "predicted_behavior": predictions["behavior"].map(_normalise_label)})
    unknown = sorted(set(out["predicted_behavior"].dropna()) - set(PREDICTION_CLASSES))
    if unknown:
        raise ValueError(f"{path.name}: unknown predicted behavior(s): {', '.join(unknown)}")
    return out


def match_files(predictions_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, str]]:
    predictions = {recording_stem(path): path for path in sorted(Path(predictions_dir).rglob("*_behaviors.csv"))}
    labels = {
        recording_stem(path): path
        for path in sorted(Path(labels_dir).rglob("*.csv"))
        if not path.name.startswith("_")
    }
    return [(predictions[key], labels[key], key) for key in sorted(set(predictions) & set(labels))]


def confusion_table(joined: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        pd.Categorical(joined["true_behavior"], categories=BEHAVIORS),
        pd.Categorical(joined["predicted_behavior"], categories=PREDICTION_CLASSES),
        dropna=False,
    ).rename_axis(index="true_behavior", columns="predicted_behavior")


def metric_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    truth = joined["true_behavior"].to_numpy(dtype=str)
    predicted = joined["predicted_behavior"].to_numpy(dtype=str)
    for behavior in BEHAVIORS:
        tp = int(np.sum((truth == behavior) & (predicted == behavior)))
        fp = int(np.sum((truth != behavior) & (predicted == behavior)))
        fn = int(np.sum((truth == behavior) & (predicted != behavior)))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if support and precision + recall
            else (0.0 if support else float("nan"))
        )
        rows.append({"behavior": behavior, "support_frames": support, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def _summary(joined: pd.DataFrame) -> dict[str, float | int]:
    metrics = metric_table(joined)
    supported = metrics[metrics["support_frames"] > 0]
    return {
        "labeled_frames": int(len(joined)),
        "accuracy": float(np.mean(joined["true_behavior"] == joined["predicted_behavior"])),
        "macro_f1_supported_classes": float(supported["f1"].mean()),
        "balanced_accuracy_supported_classes": float(supported["recall"].mean()),
        "supported_behaviors": int(len(supported)),
    }


@dataclass(frozen=True)
class ValidationResult:
    report_json: Path
    confusion_csv: Path
    metrics_csv: Path
    by_recording_csv: Path
    joined_csv: Path
    matched_recordings: int
    labeled_frames: int
    claim_status: str
    warnings: tuple[str, ...]


def validate_predictions(
    predictions_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    sessions: Iterable[object] = (),
    independent_holdout_confirmed: bool = False,
) -> ValidationResult:
    """Evaluate frozen predictions against independent manual target labels."""
    pairs = match_files(predictions_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError("No manual-label CSV names matched *_behaviors.csv prediction names")
    session_map = {str(getattr(row, "stem", "")): row for row in sessions}
    parts = []
    for prediction_path, label_path, stem in pairs:
        merged = _manual_frames(label_path).merge(
            _prediction_frames(prediction_path), on="frame", how="inner", validate="one_to_one"
        )
        if merged.empty:
            raise ValueError(f"{stem}: manual labels and predictions have no frames in common")
        row = session_map.get(stem)
        merged.insert(0, "recording", stem)
        merged.insert(1, "animal_id", str(getattr(row, "animal_id", "") or ""))
        merged.insert(2, "session", str(getattr(row, "session_type", "") or ""))
        parts.append(merged)
    joined = pd.concat(parts, ignore_index=True)
    confusion = confusion_table(joined)
    metrics = metric_table(joined)
    by_recording = pd.DataFrame(
        [{"recording": name, **_summary(group)} for name, group in joined.groupby("recording", sort=True)]
    )
    overall = _summary(joined)
    animals = sorted(value for value in joined["animal_id"].unique() if value)
    sessions_seen = sorted(value for value in joined["session"].unique() if value)
    missing_classes = metrics.loc[metrics["support_frames"] == 0, "behavior"].tolist()
    warnings: list[str] = []
    if not independent_holdout_confirmed:
        warnings.append("Annotation/model independence was not confirmed; this is an agreement audit, not an independent validation claim.")
    if missing_classes:
        warnings.append("No manual examples for: " + ", ".join(missing_classes))
    if len(pairs) < 2:
        warnings.append("Only one recording was evaluated; recording-to-recording generalisation is unknown.")
    if len(animals) < 2:
        warnings.append("Fewer than two named animals were evaluated; animal-level generalisation is unknown.")
    if independent_holdout_confirmed and len(pairs) >= 2 and len(animals) >= 2 and not missing_classes:
        claim_status = "independent target holdout evaluated for all seven behaviors"
    elif independent_holdout_confirmed:
        claim_status = "independent target holdout evaluated with coverage limitations"
    else:
        claim_status = "agreement audit only; independence not confirmed"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_csv = out_dir / "matched_labeled_frames.csv"
    confusion_csv = out_dir / "confusion_matrix.csv"
    metrics_csv = out_dir / "per_behavior_metrics.csv"
    by_recording_csv = out_dir / "by_recording_metrics.csv"
    report_json = out_dir / "validation_report.json"
    joined.to_csv(joined_csv, index=False)
    confusion.to_csv(confusion_csv)
    metrics.to_csv(metrics_csv, index=False)
    by_recording.to_csv(by_recording_csv, index=False)
    per_behavior = metrics.astype(object).where(pd.notna(metrics), None).to_dict(orient="records")
    payload = {
        "claim_status": claim_status,
        "independent_holdout_confirmed": bool(independent_holdout_confirmed),
        "model_was_refit_during_validation": False,
        "unit_of_independence": "recording/animal; frames are not treated as independent folds",
        "matched_recordings": len(pairs),
        "animals": animals,
        "sessions": sessions_seen,
        "overall": overall,
        "per_behavior": per_behavior,
        "warnings": warnings,
        "files": {
            "matched_frames": str(joined_csv),
            "confusion_matrix": str(confusion_csv),
            "per_behavior_metrics": str(metrics_csv),
            "by_recording_metrics": str(by_recording_csv),
        },
    }
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return ValidationResult(
        report_json, confusion_csv, metrics_csv, by_recording_csv, joined_csv,
        len(pairs), len(joined), claim_status, tuple(warnings)
    )
