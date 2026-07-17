# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""
Regenerate Supplementary Figure 1 on a full A4 page.

The figure shows the two supplementary syllable clusters that were omitted from
the main Figure 3 behavior panels: mixed behaviors and inaccurate tracking.
Values are animal-level percentages per 30 s bin from the manuscript workbook.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURE_DATA_DIR, FIGURES_DIR, DATA_DIR


RAW_WORKBOOK = DATA_DIR / "Raw_data.xlsx"

COLORS = {"Control": "#F9C74F", "ELS": "#C37BA0"}
AXIS_COLOR = "#4D4D4D"
SHADE_COLOR = "#F5F5F5"
A4_PORTRAIT = (8.27, 11.69)

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
        "letter": "B",
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
    if not RAW_WORKBOOK.exists():
        raise FileNotFoundError(f"Missing manuscript workbook: {RAW_WORKBOOK}")

    wide = pd.read_excel(RAW_WORKBOOK, sheet_name="Supp_cluster_time", header=2)
    wide = wide.dropna(subset=["animal_id", "group", "cluster"]).copy()
    time_cols = [col for col in wide.columns if str(col).startswith("t_")]
    if not time_cols:
        raise ValueError("No t_* time-bin columns found in Supp_cluster_time.")

    long = wide.melt(
        id_vars=["animal_id", "group", "project", "experiment", "cluster"],
        value_vars=time_cols,
        var_name="time_bin",
        value_name="percentage",
    )
    long["seconds"] = long["time_bin"].str.extract(r"(\d+)").astype(int)
    long["time_min"] = long["seconds"] / 60.0
    long["percentage"] = pd.to_numeric(long["percentage"], errors="coerce")
    long = long[long["cluster"].isin([spec["cluster"] for spec in PANEL_SPECS])]
    long = long[long["group"].isin(["Control", "ELS"])]
    return long


def summarize_time(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["cluster", "group", "time_min"], sort=False)["percentage"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
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


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.17, y: float = 1.16) -> None:
    ax.text(
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


def plot_time_panel(ax: plt.Axes, summary: pd.DataFrame, spec: dict[str, object]) -> None:
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
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, float(spec["ylim"]))
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
    panel_letter(ax, str(spec["letter"]))


def export_source_tables(long: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    long.sort_values(["cluster", "group", "animal_id", "time_min"]).to_csv(
        FIGURE_DATA_DIR / "source_data_supplementary_figure1.csv",
        index=False,
    )
    summary.sort_values(["cluster", "group", "time_min"]).to_csv(
        FIGURE_DATA_DIR / "supplementary_figure1_time_summary.csv",
        index=False,
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    long = load_supplementary_time_data()
    summary = summarize_time(long)

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
    axes = [
        fig.add_subplot(gs[0, 0:4]),
        fig.add_subplot(gs[0, 5:9]),
    ]

    for ax, spec in zip(axes, PANEL_SPECS):
        plot_time_panel(ax, summary, spec)

    pdf_path = FIGURES_DIR / "supplementary_figure1.pdf"
    svg_path = FIGURES_DIR / "supplementary_figure1.svg"
    png_path = FIGURES_DIR / "supplementary_figure1.png"
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)

    export_source_tables(long, summary)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {png_path}")
    print(f"Animals used: {long['animal_id'].nunique()} ({long.groupby('group')['animal_id'].nunique().to_dict()})")


if __name__ == "__main__":
    main()
