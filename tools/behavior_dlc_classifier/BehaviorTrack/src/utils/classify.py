"""Features and behavior classification, headless.

This is the step that replaces BarnesTrack's maze metrics: a tracking file goes
in, and per-frame behavior calls, bouts and per-class totals come out. The GUI
window is a thin shell over ``classify_one`` so the same work can run from a
terminal, from a test, or from the launcher without three copies of the logic.

The model contract fixes the parts that must not drift between training and
inference:

* one 719-feature model decides all seven behaviors, including Freezing;
* no percentile rules, separate freezing override, or canonical smoothing are
  allowed; frames whose tracking is too poor to judge become ``Unassigned``;
* the reported confidence is the probability of the behavior actually named, not
  the model's highest probability, which are different numbers on most frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

LogFn = Callable[[str], None]

#: Ethogram rows, top first. Imported from the engine at call time so the GUI and
#: the annotated videos cannot disagree about the class list.
UNASSIGNED = "Unassigned"


@dataclass(frozen=True)
class ClassifySettings:
    behavior_model: Path
    behavior_label_encoder: Path
    fps: float = 25.0
    bin_sec: int = 30
    behavior_window: int = 5


@dataclass
class ClassifyResult:
    stem: str
    frames: int
    labels: np.ndarray
    confidence: np.ndarray
    totals: pd.DataFrame
    bouts: pd.DataFrame
    bins: pd.DataFrame
    freezing_percent: float
    predictions_csv: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _engine():
    """Import the engine lazily so ``--help`` works without scientific packages."""
    from freezing_dlc import behavior as behavior_mod
    from freezing_dlc import features as features_mod

    return features_mod, behavior_mod


def bouts_from_labels(labels: Sequence[str], fps: float) -> pd.DataFrame:
    """Contiguous runs of one label, as a table of bouts."""
    labels = np.asarray(labels, dtype=object)
    if labels.size == 0:
        return pd.DataFrame(columns=["behavior", "start_frame", "end_frame", "frames", "start_sec", "duration_sec"])
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [labels.size]))
    return pd.DataFrame(
        {
            "behavior": labels[starts],
            "start_frame": starts,
            "end_frame": ends - 1,
            "frames": ends - starts,
            "start_sec": starts / float(fps),
            "duration_sec": (ends - starts) / float(fps),
        }
    )


def totals_from_labels(labels: Sequence[str], fps: float, classes: Sequence[str]) -> pd.DataFrame:
    """Seconds and percent of the recording spent in each class.

    Every class is listed even when it never occurred, so tables from different
    recordings stack without an outer join inventing NaNs.
    """
    labels = np.asarray(labels, dtype=str)
    total = labels.size
    counts = pd.Series(labels).value_counts()
    rows = []
    for name in classes:
        n = int(counts.get(name, 0))
        rows.append(
            {
                "behavior": name,
                "frames": n,
                "seconds": n / float(fps),
                "percent": (100.0 * n / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def bins_from_labels(labels: Sequence[str], fps: float, bin_sec: int, classes: Sequence[str]) -> pd.DataFrame:
    """Per-class percent within fixed time bins.

    The final bin is kept even when short, and its ``bin_seconds`` column says
    how long it actually was, so a partial bin is visible rather than being
    silently compared against full ones.
    """
    labels = np.asarray(labels, dtype=str)
    if labels.size == 0:
        return pd.DataFrame(columns=["bin", "bin_start_sec", "bin_seconds", *classes])
    per_bin = max(1, int(round(float(bin_sec) * float(fps))))
    rows = []
    for index, start in enumerate(range(0, labels.size, per_bin)):
        chunk = labels[start : start + per_bin]
        row = {
            "bin": index,
            "bin_start_sec": start / float(fps),
            "bin_seconds": chunk.size / float(fps),
        }
        counts = pd.Series(chunk).value_counts()
        for name in classes:
            row[name] = 100.0 * int(counts.get(name, 0)) / chunk.size
        rows.append(row)
    return pd.DataFrame(rows)


def classify_one(
    tracking_csv: Path,
    settings: ClassifySettings,
    log: LogFn = print,
    *,
    out_dir: Path | None = None,
    feature_dir: Path | None = None,
    summary_dir: Path | None = None,
    provenance_dir: Path | None = None,
) -> ClassifyResult:
    """Compute features for one tracking file and classify every frame."""
    features_mod, behavior_mod = _engine()
    from .model_audit import audit_input_domain, audit_model, write_provenance
    tracking_csv = Path(tracking_csv)
    stem = tracking_csv.stem.split("DLC")[0] or tracking_csv.stem
    warnings: list[str] = []
    if int(settings.behavior_window) != 5:
        raise ValueError(
            "The archived model was trained with a 5-frame lag/rolling window. "
            "Changing that number changes the feature definitions and requires retraining."
        )
    if not np.isclose(float(settings.fps), 25.0):
        raise ValueError(
            "The archived model was trained at 25 fps. Resample the tracking/video to 25 fps "
            "or train and validate a model for the new time base."
        )

    log(f"[tracking] {tracking_csv.name}")
    df_flat = features_mod.dlc_filtered_csv_to_flat_df(tracking_csv)
    log(f"[features] {len(df_flat)} frames, {len(df_flat.columns)} tracking columns")

    # --- one model for all seven behaviors ------------------------------- #
    behavior_model = behavior_mod.load_behavior_model(
        settings.behavior_model, settings.behavior_label_encoder
    )
    model_contract = audit_model(
        settings.behavior_model, settings.behavior_label_encoder, behavior_model
    )
    if not model_contract.contract_ok:
        raise ValueError(
            "The selected behavior model failed the archived 719-feature contract; "
            "classification stopped before labels were produced."
        )
    log(
        f"[model] {Path(settings.behavior_model).name} "
        f"sha256={model_contract.model_sha256[:12]}... "
        f"features={model_contract.feature_count}"
    )
    for message in model_contract.warnings:
        warnings.append(message)
        log(f"[validation] {message}")

    domain = audit_input_domain(df_flat, list(behavior_mod.BEHAVIOR_BODYPARTS))
    log(
        f"[domain] median nose-tail={domain.median_nose_tail_px:.1f}px "
        f"(training={domain.training_median_nose_tail_px:.1f}px; "
        f"ratio={domain.scale_ratio_to_training:.2f})"
    )
    if domain.warning:
        warnings.append(domain.warning)
        log(f"[validation] {domain.warning}")

    log("[features] training-faithful 719 features; transform=centroid subtraction only")
    behavior_feats = behavior_mod.build_behavior_feature_set(
        df_flat,
        behavior_model.feature_names,
        fps=settings.fps,
        window=settings.behavior_window,
    )
    raw_labels, pred_codes, prob_df = behavior_mod.predict_behaviors(behavior_feats, behavior_model)
    final_labels, label_source, metrics = behavior_mod.refine_behavior_labels(
        raw_labels, prob_df, df_flat, None, None
    )
    confidence = metrics["confidence"].to_numpy(dtype=float)
    log(
        f"[behavior] {len(final_labels)} frames; one XGBoost model for all seven behaviors; "
        "behavior heuristics=none; smoothing=none"
    )

    classes = [*behavior_mod.BEHAVIOR_DISPLAY_ORDER, behavior_mod.UNASSIGNED]
    totals = totals_from_labels(final_labels, settings.fps, classes)
    bouts = bouts_from_labels(final_labels, settings.fps)
    bins = bins_from_labels(final_labels, settings.fps, settings.bin_sec, classes)

    # Bout summaries are descriptive views of the unchanged per-frame model calls.
    short = int((bouts.frames < settings.fps).sum())
    log(
        f"[behavior] canonical bouts={len(bouts)}, median={bouts.frames.median():.0f} "
        f"frame(s), under 1s={short}"
    )
    freezing_percent = float(
        totals.loc[totals["behavior"] == "Freezing", "percent"].iloc[0]
    )

    predictions_csv = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        feature_dir = Path(feature_dir) if feature_dir is not None else out_dir
        summary_dir = Path(summary_dir) if summary_dir is not None else out_dir
        provenance_dir = Path(provenance_dir) if provenance_dir is not None else out_dir
        feature_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        provenance_dir.mkdir(parents=True, exist_ok=True)
        predictions_csv = out_dir / f"{stem}_behaviors.csv"
        behavior_feats.to_csv(feature_dir / f"{stem}_features_719.csv", index=False)
        behavior_mod.save_behavior_predictions(
            predictions_csv,
            np.asarray(final_labels, dtype=object),
            pred_codes,
            prob_df,
            raw_labels=np.asarray(raw_labels, dtype=object),
            label_source=label_source,
            metrics_df=metrics,
            confidence=confidence,
        )
        totals.to_csv(summary_dir / f"{stem}_behavior_totals.csv", index=False)
        bouts.to_csv(summary_dir / f"{stem}_behavior_bouts.csv", index=False)
        bins.to_csv(summary_dir / f"{stem}_behavior_bins.csv", index=False)
        write_provenance(
            provenance_dir / f"{stem}_classification_provenance.json",
            model_contract,
            domain,
        )
        log(
            f"[write] {predictions_csv.name} (+ 719 features, totals, bouts, bins, provenance)"
        )

    return ClassifyResult(
        stem=stem,
        frames=int(len(final_labels)),
        labels=np.asarray(final_labels, dtype=object),
        confidence=confidence,
        totals=totals,
        bouts=bouts,
        bins=bins,
        freezing_percent=freezing_percent,
        predictions_csv=predictions_csv,
        warnings=warnings,
    )
