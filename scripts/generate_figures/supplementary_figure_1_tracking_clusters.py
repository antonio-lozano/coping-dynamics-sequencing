# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""
Regenerate Supplementary Figure 1 on a full A4 page.

The figure shows the two supplementary syllable clusters that were omitted from
the main Figure 3 behavior panels: mixed behaviors and inaccurate tracking.
Values are animal-level percentages per 30 s bin from the bundled source CSV.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, SUPPLEMENTARY_TRACKING_CSV


SOURCE_CSV = SUPPLEMENTARY_TRACKING_CSV

COLORS = {"Control": "#F9C74F", "ELS": "#C37BA0"}
AXIS_COLOR = "#4D4D4D"
SHADE_COLOR = "#F5F5F5"
A4_PORTRAIT = (8.27, 11.69)
BIN_SECONDS = 30
FREQUENCY_YMAX = 150

PANEL_SPECS = [
    {
        "cluster": "Mix behaviors",
        "title": "Mixed behaviors",
        "letter": "A",
        "ylim": 14,
        "yticks": np.arange(0, 14.1, 2),
        "legend_loc": "upper right",
    },
    {
        "cluster": "Inaccurate tracking",
        "title": "Inaccurate tracking",
        "letter": "C",
        "ylim": 30,
        "yticks": np.arange(0, 30.1, 5),
        "legend_loc": "upper right",
    },
]


def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return 0.0
    return values.std(ddof=1) / math.sqrt(len(values))


def load_supplementary_time_data() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE_CSV}")

    long = pd.read_csv(SOURCE_CSV)
    required = {"animal_id", "group", "project", "experiment", "cluster", "time_bin", "percentage", "seconds", "time_min"}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Missing columns in {SOURCE_CSV.name}: {sorted(missing)}")
    long["percentage"] = pd.to_numeric(long["percentage"], errors="coerce")
    long["time_min"] = pd.to_numeric(long["time_min"], errors="coerce")
    long = long[long["cluster"].isin([spec["cluster"] for spec in PANEL_SPECS])]
    long = long[long["group"].isin(["Control", "ELS"])]
    return long


def summarize_time(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["cluster", "group", "time_min"], sort=False)["percentage"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )


def total_frequency(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.assign(frequency_seconds=long["percentage"] / 100.0 * BIN_SECONDS)
        .groupby(["animal_id", "group", "cluster"], as_index=False)["frequency_seconds"]
        .sum()
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=AXIS_COLOR, width=0.7, length=3, labelsize=7)
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    ax.title.set_color(AXIS_COLOR)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.17, y: float = 1.16) -> plt.Text:
    return ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=AXIS_COLOR,
    )


def add_shock_shading(ax: plt.Axes) -> None:
    for start in [3.5, 5.0, 6.5]:
        ax.axvspan(start, start + 0.5, color=SHADE_COLOR, zorder=0)


def plot_time_panel(ax: plt.Axes, summary: pd.DataFrame, spec: dict[str, object]) -> plt.Text:
    add_shock_shading(ax)
    panel = summary[summary["cluster"] == spec["cluster"]]

    for group in ["Control", "ELS"]:
        gdata = panel[panel["group"] == group].sort_values("time_min")
        x = gdata["time_min"].to_numpy(dtype=float)
        mean = gdata["mean"].to_numpy(dtype=float)
        err = gdata["sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[group],
            marker="o",
            markersize=2.3,
            linewidth=1.4,
            label=group,
        )
        ax.fill_between(x, mean - err, mean + err, color=COLORS[group], alpha=0.22, linewidth=0)

    ax.set_title(str(spec["title"]), fontsize=7, pad=5, y=1.03)
    # Data runs 0.5-7.5 min; start at zero and pad past 7.5 so the first and
    # last markers are drawn whole.
    ax.set_xlim(0.0, 7.65)
    ax.set_ylim(0, float(spec["ylim"]))
    # Ticks on whole minutes 1-7 only.
    ax.set_xticks(np.arange(1, 8))
    ax.set_yticks(spec["yticks"])
    ax.set_xlabel("Time (minutes)", fontsize=6.5, labelpad=2)
    ax.set_ylabel("% of time in cluster", fontsize=7, labelpad=3)
    ax.legend(
        frameon=False,
        loc=str(spec["legend_loc"]),
        fontsize=6.5,
        ncol=2,
        handlelength=1.2,
        columnspacing=0.5,
        handletextpad=0.35,
    )
    style_axis(ax)
    # After style_axis: the axis line runs on to 7.5 so the final data point
    # sits over the spine rather than past its end.
    ax.spines["bottom"].set_bounds(0.0, 7.5)
    return panel_letter(ax, str(spec["letter"]))


def plot_frequency_panel(
    ax: plt.Axes,
    frequency: pd.DataFrame,
    spec: dict[str, object],
    letter: str,
) -> plt.Text:
    panel = frequency[frequency["cluster"] == spec["cluster"]]
    groups = ["Control", "ELS"]
    values_by_group = [
        panel.loc[panel["group"] == group, "frequency_seconds"].dropna().to_numpy()
        for group in groups
    ]
    rng = np.random.default_rng(42 + ord(letter))
    for index, (group, values) in enumerate(zip(groups, values_by_group)):
        color = COLORS[group]
        boxplot = ax.boxplot(
            values,
            positions=[index],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.15},
            whiskerprops={"color": color, "linewidth": 1.15},
            capprops={"color": color, "linewidth": 1.15},
            medianprops={"color": color, "linewidth": 1.35},
        )
        boxplot["boxes"][0].set_alpha(0.15)
        jitter = rng.normal(index, 0.022, len(values))
        ax.scatter(jitter, values, s=8, color=color, alpha=0.78, linewidth=0, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.55, 1.55)
    # Both panels share one 0-150 s scale so B and D can be read against each
    # other; the largest animal total is 144.1 s, so nothing is clipped.
    maximum = max(float(values.max()) for values in values_by_group if len(values))
    if maximum > FREQUENCY_YMAX:
        raise AssertionError(
            f"{spec['title']}: total frequency {maximum:.1f} s exceeds the {FREQUENCY_YMAX} s axis"
        )
    ax.set_ylim(0, FREQUENCY_YMAX)
    ax.set_yticks(np.arange(0, FREQUENCY_YMAX + 1, 25))
    ax.set_title(str(spec["title"]), fontsize=7, pad=5, y=1.03)
    ax.set_ylabel("Total frequency (s)", fontsize=6.5, labelpad=2)
    style_axis(ax)
    # Stop the left axis at the last tick so it ends on 150, not above it.
    ax.spines["left"].set_bounds(0, FREQUENCY_YMAX)
    # Bounding the bottom spine to the two category positions left only a stub
    # between the ticks. Run it across the panel from the y axis, matching the
    # baseline the other categorical panels in the figure set draw.
    ax.spines["bottom"].set_bounds(*ax.get_xlim())
    return panel_letter(ax, letter, x=-0.28)


def align_letters_to_titles(
    fig: plt.Figure,
    letter_artists: list[tuple[plt.Axes, plt.Text]],
) -> None:
    """Put each panel letter on the title's line, in the y-axis title column.

    ``align_panel_letters`` already anchors the letters to the left edge of each
    panel's y-axis title, which is the column we want them in; it only places
    them vertically relative to the top of the axes.  This re-does the vertical
    placement so every letter sits on the same horizontal line as its own panel
    title, centred on the title text rather than floating above or below it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    for ax, text in letter_artists:
        if text is None:
            continue
        # Horizontal: left edge of the y-axis title (falling back to the tick
        # labels when the panel has no y-label), matching align_panel_letters.
        label_box = ax.yaxis.label.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        if label_box.width <= 0:
            boxes = [
                t.get_window_extent(renderer=renderer)
                for t in ax.get_yticklabels()
                if t.get_text()
            ]
            x_px = min((b.x0 for b in boxes), default=axes_box.x0)
        else:
            x_px = label_box.x0
        # Vertical: the vertical centre of the panel title, so the letter reads
        # on the same line as the title.
        title_box = ax.title.get_window_extent(renderer=renderer)
        x_fig, y_fig = fig_inv.transform((x_px, (title_box.y0 + title_box.y1) / 2.0))
        text.set_transform(fig.transFigure)
        text.set_position((x_fig, y_fig))
        text.set_ha("left")
        text.set_va("center")


def export_source_tables(summary: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["cluster", "group", "time_min"]).to_csv(
        PROCESSED_DATA_DIR / "supplementary_figure1_time_summary.csv",
        index=False,
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    long = load_supplementary_time_data()
    summary = summarize_time(long)
    frequency = total_frequency(long)

    fig = plt.figure(figsize=A4_PORTRAIT, dpi=300, facecolor="white")
    # Keep the manuscript page A4, but preserve the compact panel proportions
    # used by Figure 3B-H instead of stretching the two panels over the page.
    gs = fig.add_gridspec(
        nrows=4,
        ncols=14,
        left=0.075,
        right=0.955,
        top=0.955,
        bottom=0.24,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.56,
        wspace=0.28,
    )
    panel_grid = gs[0, :].subgridspec(
        1,
        4,
        width_ratios=[1.7, 0.7, 1.7, 0.7],
        wspace=0.45,
    )
    axes = [fig.add_subplot(panel_grid[0, index]) for index in range(4)]
    letter_artists = [
        (axes[0], plot_time_panel(axes[0], summary, PANEL_SPECS[0])),
        (axes[1], plot_frequency_panel(axes[1], frequency, PANEL_SPECS[0], "B")),
        (axes[2], plot_time_panel(axes[2], summary, PANEL_SPECS[1])),
        (axes[3], plot_frequency_panel(axes[3], frequency, PANEL_SPECS[1], "D")),
    ]
    align_letters_to_titles(fig, letter_artists)

    pdf_path = FIGURES_DIR / "supplementary_figure1.pdf"
    svg_path = FIGURES_DIR / "supplementary_figure1.svg"
    png_path = FIGURES_DIR / "supplementary_figure1.png"
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)

    export_source_tables(summary)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {png_path}")
    print(f"Animals used: {long['animal_id'].nunique()} ({long.groupby('group')['animal_id'].nunique().to_dict()})")


if __name__ == "__main__":
    main()
