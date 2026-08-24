#!/usr/bin/env python
"""Assemble Supplementary Figure 4 from the legacy class-specific SHAP analysis.

The published panels come from the legacy behavior classifier (719 pairwise
DeepLabCut body-part features), not from the bundled reusable classifier.  Their
SHAP values are tracked at ``classifier/legacy_shap/shap_values.npz``, so the
figure regenerates from the repository alone::

    python scripts/generate_figures/supplementary_figure_4_classifier_shap.py

``--source`` is only needed to rebuild the panels straight from the legacy
keypoint-MoSeq working tree; see ``scripts/import_legacy_shap_values.py``.

Each panel is a ``shap.summary_plot`` beeswarm of the top 10 parameters for one
behavior, matching the legacy figure script.  The per-parameter values behind
the panels are tracked in
``figure_source_data/supplementary_figure4_shap_summary.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.migrations.import_legacy_shap_summary import (  # noqa: E402
    BEHAVIOR_ORDER,
    CLASS_LABELS,
    parameter_label,
)

OUTPUT_STEM = REPO / "figures" / "supplementary_figure4"
SHAP_ARCHIVE = REPO / "classifier" / "legacy_shap" / "shap_values.npz"
TOP_FEATURES = 10
PANEL_LETTERS = "ABCDEFGH"
# Beeswarm marker area in points^2; summary_plot defaults to 16, which is far
# larger than the published panels.
DOT_SIZE = 3.0
# A few classes (notably Groom) have extreme SHAP outliers that stretch the axis
# far past the bulk of the distribution and squash every beeswarm against zero.
# Limit each panel to this central percentile of its plotted values; points
# beyond it are clipped by the axes rather than dropped from the data.
XLIM_PERCENTILE = 99.5

# Feature-value palette sampled from the archived published rendering during
# migration, listed from low to high. Note this is the reverse of the SHAP default: low values are red and
# high values are blue, through a warm off-white midpoint.
LEGACY_COLORS = [
    "#D8474E",
    "#DE696D",
    "#E58C8A",
    "#EBAEA9",
    "#F1D0C7",
    "#F9F1E6",
    "#E1E1DF",
    "#C9D1D7",
    "#AFC0CF",
    "#98B0C8",
    "#7FA2C2",
]
LEGACY_CMAP = LinearSegmentedColormap.from_list("legacy_feature_value", LEGACY_COLORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Legacy keypoint-MoSeq results directory; defaults to the tracked archive",
    )
    return parser.parse_args()


def load_archive() -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Read the tracked repository archive."""
    with np.load(SHAP_ARCHIVE, allow_pickle=True) as archive:
        classes = [str(name) for name in archive["class_names"]]
        shap_values = archive["shap_values"]
        sample = pd.DataFrame(
            archive["feature_values"],
            columns=[str(name) for name in archive["feature_names"]],
        )
    return classes, shap_values, sample


def load_source(source: Path) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Read the legacy keypoint-MoSeq working tree directly."""
    shap_values = joblib.load(source / "shap_values.pkl")
    encoder = joblib.load(source / "label_encoder.pkl")
    sample = pd.read_csv(source / "shap_sample.csv")
    classes = [str(value) for value in encoder.classes_]
    return classes, np.stack([np.asarray(v) for v in shap_values]), sample


def load_legacy(source: Path | None) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if source is not None:
        classes, shap_values, sample = load_source(source)
    elif SHAP_ARCHIVE.exists():
        classes, shap_values, sample = load_archive()
    else:
        raise SystemExit(
            f"Missing {SHAP_ARCHIVE.relative_to(REPO)}; pass --source to rebuild it "
            "from the legacy keypoint-MoSeq results directory."
        )

    sample.columns = [parameter_label(column) for column in sample.columns]
    by_behavior = {
        CLASS_LABELS.get(name, name): shap_values[index] for index, name in enumerate(classes)
    }
    missing = [behavior for behavior in BEHAVIOR_ORDER if behavior not in by_behavior]
    if missing:
        raise AssertionError(f"Legacy SHAP values are missing behaviors: {missing}")
    return by_behavior, sample


def build_figure(by_behavior: dict[str, np.ndarray], sample: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    # summary_plot jitters beeswarm dots through the global numpy RNG; seeding
    # here pins the dot positions so rebuilding the figure reproduces it.
    np.random.seed(0)
    figure = plt.figure(figsize=(8.27, 11.69), dpi=300)
    for index, behavior in enumerate(BEHAVIOR_ORDER):
        ax = figure.add_subplot(4, 2, index + 1)
        plt.sca(ax)
        # summary_plot draws its own oversized colorbar and rescales the axes;
        # suppress both and add a compact colorbar per panel instead.
        shap.summary_plot(
            by_behavior[behavior],
            sample,
            max_display=TOP_FEATURES,
            show=False,
            plot_size=None,
            color_bar=False,
            cmap=LEGACY_CMAP,
        )
        for collection in ax.collections:
            collection.set_sizes([DOT_SIZE])
        # Rank by mean |SHAP| exactly as summary_plot does, so the limit is
        # computed over the same top features the panel draws. Labels cannot be
        # used for this: "Center body" is not unique (S1 and S2 share it).
        values = by_behavior[behavior]
        top = np.argsort(-np.abs(values).mean(axis=0), kind="stable")[:TOP_FEATURES]
        limit = float(np.percentile(np.abs(values[:, top]), XLIM_PERCENTILE))
        ax.set_xlim(-limit * 1.08, limit * 1.08)
        ax.set_title(f"SHAP: {behavior}", fontsize=8, pad=4)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=6)
        ax.tick_params(axis="x", labelsize=5.5, length=2)
        ax.tick_params(axis="y", labelsize=5.0, length=0)
        ax.text(
            -0.72,
            1.06,
            PANEL_LETTERS[index],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
        )

        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=LEGACY_CMAP),
            ax=ax,
            fraction=0.022,
            pad=0.015,
            aspect=26,
        )
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"], fontsize=5)
        colorbar.ax.tick_params(length=0)
        colorbar.set_label("Feature value", fontsize=5.5, labelpad=2)
        colorbar.outline.set_visible(False)

    figure.subplots_adjust(
        # Page margins follow the rest of the figure set (Figure 7 uses
        # top=0.978, bottom=0.052); the right edge is pulled further in so the
        # per-panel colorbar and its label do not crowd the trim.
        left=0.215,
        right=0.935,
        top=0.978,
        bottom=0.052,
        wspace=0.80,
        hspace=0.34,
    )
    return figure


def main() -> None:
    args = parse_args()
    if args.source is not None and not args.source.exists():
        raise SystemExit(f"Legacy SHAP source not found: {args.source}")
    by_behavior, sample = load_legacy(args.source)
    figure = build_figure(by_behavior, sample)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        path = OUTPUT_STEM.with_suffix(f".{extension}")
        figure.savefig(path, dpi=400 if extension == "png" else None, facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT_STEM.name}.pdf/.png/.svg")


if __name__ == "__main__":
    main()
