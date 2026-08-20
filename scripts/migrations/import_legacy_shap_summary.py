"""Import the legacy class-specific SHAP analysis behind Supplementary Figure 4.

This is a one-time migration utility.  Supplementary Figure 4 was produced by the
legacy behavior classifier, whose feature space is pairwise DeepLabCut body-part
distances and velocities.  That analysis lives outside this repository, in the
``code/shapley/results`` directory of the keypoint-MoSeq working tree, which
must be supplied with ``--source``::

    <source>/xgb_model.pkl
    <source>/label_encoder.pkl
    <source>/shap_values.pkl                              (8 classes x frames x 719)
    <source>/shap_sample.csv                              (feature names)
    <source>/features_defining_each_behavior_from_shap.csv (authoritative ranks)

The bundled ``classifier/figure7_behavior_classifier.joblib`` is a DIFFERENT
model on a DIFFERENT feature set and must not be used for these panels.

This script writes one self-contained summary table so the repository does not
have to carry the 179 MB feature matrix or the version-fragile SHAP pickles.
Ranks are taken from the legacy CSV; the distribution statistics are computed
from ``shap_values.pkl``.

Run:
    python scripts/import_legacy_shap_summary.py --source DIR
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figure_source_data" / "supplementary_figure4_shap_summary.csv"
LEGACY_DIR = REPO / "classifier" / "legacy_shap"
SHAP_ARCHIVE = LEGACY_DIR / "shap_values.npz"
RANKS_CSV = LEGACY_DIR / "legacy_feature_ranks.csv"

# The legacy classifier labels its classes with verb forms ("Freezing"); the
# repository uses the short behavior names shown in the published panels, so
# they are normalised here.  The unlabeled class is called "Unspecified" by the
# model and "No labeled" by the legacy table, and "Unassigned" in the figures.
CLASS_LABELS = {
    "Freezing": "Freeze",
    "Sniffing": "Sniff",
    "Grooming": "Groom",
    "Climbing": "Climb",
    "Unspecified": "Unassigned",
    "No labeled": "Unassigned",
}
BEHAVIOR_ORDER = (
    "Freeze",
    "Sniff",
    "Groom",
    "Turn",
    "Locomotion",
    "Climb",
    "Jump",
    "Unassigned",
)

# DeepLabCut marker -> descriptive keypoint, from the legacy analysis script.
# S1 and S2 are both labelled "Center body", as in the published figure, so the
# label is not unique on its own; ``parameter`` remains the unique identifier.
KEYPOINTS = {
    "nose": "Nose",
    "H1R": "Right head 1",
    "H2R": "Right head 2",
    "H1L": "Left head 1",
    "H2L": "Left head 2",
    "B1R": "Right body 1",
    "B2R": "Right body 2",
    "B3R": "Right body 3",
    "B1L": "Left body 1",
    "B2L": "Left body 2",
    "B3L": "Left body 3",
    "tail": "Tail",
    "S1": "Center body",
    "S2": "Center body",
}
AGGREGATES = {"roll_std": "SD", "roll_mean": "Mean", "roll_sum": "Summed"}
SCALARS = {
    "angular_velocity": "Angular velocity",
    "global_body_angular_velocity": "Global body angular velocity",
    "global_body_angle": "Global body angle",
    "orientation_angle": "Orientation angle",
    "centroid_velocity": "Centroid velocity",
    "centroid_x": "Centroid x",
    "centroid_y": "Centroid y",
}
_MARKER = "|".join(sorted(KEYPOINTS, key=len, reverse=True))
_AGG = "|".join(AGGREGATES)


def parameter_label(parameter: str) -> str:
    """Readable label for a legacy feature name.

    The published panels were manually polished, so word order can differ
    slightly; ``parameter`` remains the authoritative identifier.
    """
    name, lag = parameter, ""
    match = re.match(r"^(?P<base>.+?)_t-(?P<lag>\d+)$", name)
    if match:
        name, lag = match.group("base"), f" (lag {match.group('lag')})"

    match = re.match(rf"^dist_(?P<a>{_MARKER})_(?P<b>{_MARKER})(?:_(?P<agg>{_AGG}))?$", name)
    if match:
        pair = f"{KEYPOINTS[match.group('a')]} - {KEYPOINTS[match.group('b')]}"
        agg = match.group("agg")
        return (f"{AGGREGATES[agg]} distance {pair}" if agg else f"Distance {pair}") + lag

    match = re.match(rf"^(?P<bp>{_MARKER})_velocity(?:_(?P<agg>{_AGG}))?$", name)
    if match:
        body = KEYPOINTS[match.group("bp")]
        agg = match.group("agg")
        if agg:
            return f"{AGGREGATES[agg]} {body[:1].lower() + body[1:]} velocity" + lag
        return f"{body} velocity" + lag

    match = re.match(rf"^(?P<bp>{_MARKER})_(?P<axis>[xy])(?:_(?P<agg>{_AGG}))?$", name)
    if match:
        body = KEYPOINTS[match.group("bp")]
        axis = match.group("axis")
        agg = match.group("agg")
        base = f"{body} {axis}"
        return (f"{AGGREGATES[agg]} {base[:1].lower() + base[1:]}" if agg else base) + lag

    match = re.match(rf"^(?P<base>.+?)_(?P<agg>{_AGG})$", name)
    if match and match.group("base") in SCALARS:
        base = SCALARS[match.group("base")]
        return f"{AGGREGATES[match.group('agg')]} {base[:1].lower() + base[1:]}" + lag
    if name in SCALARS:
        return SCALARS[name] + lag

    match = re.match(r"^centroid_displacement_(?P<window>\d+)$", name)
    if match:
        return f"Centroid displacement ({match.group('window')} frames)" + lag
    match = re.match(r"^orientation_change_(?P<window>\d+)$", name)
    if match:
        return f"Orientation change ({match.group('window')} frames)" + lag

    return parameter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Legacy keypoint-MoSeq results directory; defaults to the tracked archive",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args()


def load_inputs(source: Path | None) -> tuple[list[str], np.ndarray, list[str], pd.DataFrame]:
    """Class names, SHAP array, parameter names and the authoritative ranks."""
    if source is not None:
        shap_values = joblib.load(source / "shap_values.pkl")
        encoder = joblib.load(source / "label_encoder.pkl")
        classes = [str(value) for value in encoder.classes_]
        stacked = np.stack([np.asarray(v) for v in shap_values])
        parameters = list(pd.read_csv(source / "shap_sample.csv", nrows=0).columns)
        ranks = pd.read_csv(source / "features_defining_each_behavior_from_shap.csv")
        return classes, stacked, parameters, ranks

    if not SHAP_ARCHIVE.exists() or not RANKS_CSV.exists():
        raise SystemExit(
            f"Missing {SHAP_ARCHIVE.relative_to(REPO)} or {RANKS_CSV.relative_to(REPO)}; "
            "pass --source to rebuild them from the legacy results directory."
        )
    with np.load(SHAP_ARCHIVE, allow_pickle=True) as archive:
        classes = [str(name) for name in archive["class_names"]]
        stacked = archive["shap_values"]
        parameters = [str(name) for name in archive["feature_names"]]
    return classes, stacked, parameters, pd.read_csv(RANKS_CSV)


def build_summary(source: Path | None) -> pd.DataFrame:
    classes, shap_values, parameters, legacy = load_inputs(source)
    legacy["behavior"] = legacy["behavior"].replace(CLASS_LABELS)

    if len(shap_values) != len(classes):
        raise AssertionError("SHAP value list does not match the class names")

    rows: list[dict[str, object]] = []
    for index, legacy_class in enumerate(classes):
        behavior = CLASS_LABELS.get(str(legacy_class), str(legacy_class))
        # Keep the library's native dtype and reduce over the whole array at
        # once: the legacy table accumulated the magnitudes in float32 with
        # ``mean(axis=0)``, and either casting up or reducing column-by-column
        # changes the summation order enough to perturb them by ~1e-6.
        values = np.asarray(shap_values[index])
        if values.shape[1] != len(parameters):
            raise AssertionError(f"{behavior}: SHAP columns do not match the feature list")
        n_frames = values.shape[0]
        mean_absolute = np.abs(values).mean(axis=0)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        sem = sd / np.sqrt(n_frames)
        for column, parameter in enumerate(parameters):
            rows.append(
                {
                    "behavior": behavior,
                    "parameter": parameter,
                    "parameter_label": parameter_label(parameter),
                    "mean_absolute_shap": float(mean_absolute[column]),
                    "mean_shap": float(mean[column]),
                    "sd_shap": float(sd[column]),
                    "sem_shap": float(sem[column]),
                    "minimum_shap": float(minimum[column]),
                    "maximum_shap": float(maximum[column]),
                    "n_frames": int(n_frames),
                }
            )

    summary = pd.DataFrame(rows).merge(
        legacy[["behavior", "parameter", "rank"]].rename(columns={"parameter": "parameter"})
        if "parameter" in legacy.columns
        else legacy[["behavior", "feature", "rank"]].rename(columns={"feature": "parameter"}),
        on=["behavior", "parameter"],
        how="left",
    )
    if summary["rank"].isna().any():
        raise AssertionError("Some parameters have no legacy rank")

    # Cross-check the recomputed magnitudes against the legacy table.
    reference = (
        legacy.rename(columns={"feature": "parameter", "mean_abs_shap": "legacy_mean_abs"})
        [["behavior", "parameter", "legacy_mean_abs"]]
    )
    check = summary.merge(reference, on=["behavior", "parameter"])
    drift = (check["mean_absolute_shap"] - check["legacy_mean_abs"]).abs().max()
    if drift > 1e-9:
        raise AssertionError(f"Recomputed SHAP magnitudes drifted from the legacy table ({drift})")

    summary["rank"] = summary["rank"].astype(int)
    summary["plotted_top_10"] = summary["rank"] <= 10
    summary["__order"] = summary["behavior"].map(
        {behavior: index for index, behavior in enumerate(BEHAVIOR_ORDER)}
    )
    if summary["__order"].isna().any():
        raise AssertionError("Unmapped behavior label")
    return (
        summary.sort_values(["__order", "rank"], kind="stable")
        .drop(columns="__order")
        .reset_index(drop=True)[
            [
                "behavior",
                "rank",
                "parameter",
                "parameter_label",
                "mean_absolute_shap",
                "mean_shap",
                "sd_shap",
                "sem_shap",
                "minimum_shap",
                "maximum_shap",
                "n_frames",
                "plotted_top_10",
            ]
        ]
    )


def main() -> None:
    args = parse_args()
    if args.source is not None and not args.source.exists():
        raise SystemExit(f"Legacy SHAP source not found: {args.source}")
    summary = build_summary(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {args.output.relative_to(REPO)} ({len(summary)} rows)")
    print(f"  {summary['behavior'].nunique()} behaviors x {summary['parameter'].nunique()} parameters")


if __name__ == "__main__":
    main()
