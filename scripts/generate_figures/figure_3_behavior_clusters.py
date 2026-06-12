"""
Regenerate manuscript Figure 3 from the archived MoSeq cluster time-bin data.

The figure uses the 82-animal Control/ELS cohort in
Syllable_per_timebin_final(30s).csv and the hand-curated syllable-to-behavior
cluster mapping used elsewhere in this repository.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_FIGURES_DIR = REPO_ROOT / "figures"
OUTPUT_DIR = LEGACY_FIGURES_DIR / "data"
ARCHIVE_PATH = Path(r"H:\Downloads\Coping_data2.zip")
ARCHIVE_MEMBER = "COping/Syllable_per_timebin_final(30s).csv"

COLORS = {"Control": "#F9C74F", "ELS": "#C37BA0"}
AXIS_COLOR = "#4D4D4D"
SHADE_COLOR = "#F5F5F5"
FPS_BIN_SECONDS = 30

CLUSTER_MAP = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}

PANEL_ORDER = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
Y_LIMITS = {
    "Freeze": 60,
    "Sniff": 20,
    "Groom": 0.40,
    "Turn": 60,
    "Locomotion": 12,
    "Climb": 12,
    "Jump": 4,
}

A4_PORTRAIT = (8.27, 11.69)


def load_timebin_data() -> pd.DataFrame:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Missing source archive: {ARCHIVE_PATH}. Expected member: {ARCHIVE_MEMBER}"
        )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        with archive.open(ARCHIVE_MEMBER) as handle:
            df = pd.read_csv(handle)
    df = df.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    df = df[df["group"].isin(["Control", "ELS"])].copy()
    df["Syllable"] = df["Syllable"].astype(int)
    df["time_bin"] = pd.to_numeric(df["time_bin"])
    return df


def complete_cluster_rows(df: pd.DataFrame, syllables: list[int]) -> pd.DataFrame:
    base_cols = ["Animal", "group", "time_bin", "Experiment"]
    base = df[base_cols].drop_duplicates()
    syllable_grid = pd.DataFrame({"Syllable": syllables})
    grid = base.assign(_key=1).merge(syllable_grid.assign(_key=1), on="_key").drop(columns="_key")
    out = grid.merge(df, on=base_cols + ["Syllable"], how="left")
    out["Percentage"] = out["Percentage"].fillna(0)
    return out


def build_cluster_data(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cluster, syllables in CLUSTER_MAP.items():
        cluster_df = complete_cluster_rows(df, syllables)
        cluster_df = (
            cluster_df.groupby(["Animal", "group", "Experiment", "time_bin"], as_index=False)[
                "Percentage"
            ]
            .sum()
            .assign(cluster=cluster)
        )
        frames.append(cluster_df)
    combined = pd.concat(frames, ignore_index=True)
    combined["time_min"] = (combined["time_bin"] + FPS_BIN_SECONDS) / 60.0
    return combined


def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return 0.0
    return values.std(ddof=1) / math.sqrt(len(values))


def summarize_time(cluster_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cluster_df.groupby(["cluster", "group", "time_bin", "time_min"])["Percentage"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )


def summarize_total_seconds(cluster_df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        cluster_df.assign(seconds=cluster_df["Percentage"] / 100.0 * FPS_BIN_SECONDS)
        .groupby(["Animal", "group", "cluster"], as_index=False)["seconds"]
        .sum()
    )
    return (
        totals.groupby(["group", "cluster"])["seconds"]
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


def plot_bar_panel(ax: plt.Axes, total_summary: pd.DataFrame) -> None:
    x = np.arange(len(PANEL_ORDER))
    width = 0.34
    for offset, group in [(-width / 2, "Control"), (width / 2, "ELS")]:
        data = total_summary[total_summary["group"] == group].set_index("cluster").reindex(PANEL_ORDER)
        ax.bar(
            x + offset,
            data["mean"],
            yerr=data["sem"],
            width=width,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            error_kw={"elinewidth": 0.8, "ecolor": AXIS_COLOR},
            label=group,
        )
    ax.set_ylabel("Frequency (s)", fontsize=8, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_ORDER, fontsize=5.6)
    ax.set_ylim(0, 300)
    ax.set_yticks(np.arange(0, 301, 50))
    ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.6)
    for cluster in ["Freeze", "Sniff", "Turn"]:
        idx = PANEL_ORDER.index(cluster)
        data = total_summary[total_summary["cluster"] == cluster]
        y = float((data["mean"] + data["sem"]).max()) + 8
        ax.plot([idx - width / 2, idx - width / 2, idx + width / 2, idx + width / 2], [y, y + 5, y + 5, y],
                color=AXIS_COLOR, linewidth=0.8)
        ax.text(idx, y + 7, "*", ha="center", va="bottom", color=AXIS_COLOR, fontsize=9, fontweight="bold")
    style_axis(ax)
    panel_letter(ax, "A", x=-0.10, y=1.10)


def plot_time_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    cluster: str,
    letter: str,
    star: bool = False,
    legend_loc: str = "upper right",
) -> None:
    add_shock_shading(ax)
    data = summary[summary["cluster"] == cluster]
    for group in ["Control", "ELS"]:
        gdata = data[data["group"] == group].sort_values("time_min")
        x = gdata["time_min"].to_numpy(dtype=float)
        mean = gdata["mean"].to_numpy(dtype=float)
        err = gdata["sem"].to_numpy(dtype=float)
        ax.plot(x, mean, color=COLORS[group], marker="o", markersize=2.3, linewidth=1.4, label=group)
        ax.fill_between(x, mean - err, mean + err, color=COLORS[group], alpha=0.22, linewidth=0)
    ymax = Y_LIMITS[cluster]
    ax.set_title(cluster, fontsize=7, pad=5, y=1.03)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=6.5, labelpad=2)
    ax.set_ylabel("% of time in cluster", fontsize=7, labelpad=3)
    if cluster == "Groom":
        ax.set_yticks(np.arange(0, 0.401, 0.05))
    elif cluster == "Jump":
        ax.set_yticks(np.arange(0, 4.1, 0.5))
    elif cluster in {"Locomotion", "Climb"}:
        ax.set_yticks(np.arange(0, 13, 2))
    elif cluster == "Sniff":
        ax.set_yticks(np.arange(0, 20.1, 2.5))
    else:
        ax.set_yticks(np.arange(0, ymax + 1, 10))
    ax.legend(frameon=False, loc=legend_loc, fontsize=6.5, ncol=2, handlelength=1.2, columnspacing=0.5)
    if star:
        ax.text(4.0, ymax * 0.96, "*", ha="center", va="center", color=AXIS_COLOR, fontsize=11, fontweight="bold")
    style_axis(ax)
    panel_letter(ax, letter)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_timebin_data()
    cluster_df = build_cluster_data(raw)
    time_summary = summarize_time(cluster_df)
    total_summary = summarize_total_seconds(cluster_df)

    fig = plt.figure(figsize=A4_PORTRAIT, dpi=300, facecolor="white")
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
    ax_a = fig.add_subplot(gs[0, 0:9])
    axes = {
        "Freeze": fig.add_subplot(gs[0, 10:14]),
        "Sniff": fig.add_subplot(gs[1, 0:4]),
        "Groom": fig.add_subplot(gs[1, 5:9]),
        "Turn": fig.add_subplot(gs[1, 10:14]),
        "Locomotion": fig.add_subplot(gs[2, 0:4]),
        "Climb": fig.add_subplot(gs[2, 5:9]),
        "Jump": fig.add_subplot(gs[2, 10:14]),
    }

    plot_bar_panel(ax_a, total_summary)
    for cluster, letter, star, legend_loc in [
        ("Freeze", "B", True, "lower right"),
        ("Sniff", "C", False, "upper right"),
        ("Groom", "D", True, "upper right"),
        ("Turn", "E", True, "lower left"),
        ("Locomotion", "F", True, "lower right"),
        ("Climb", "G", False, "upper right"),
        ("Jump", "H", False, "upper right"),
    ]:
        plot_time_panel(axes[cluster], time_summary, cluster, letter, star=star, legend_loc=legend_loc)

    pdf_path = OUTPUT_DIR / "figure_3_behavior_clusters.pdf"
    svg_path = OUTPUT_DIR / "figure_3_behavior_clusters.svg"
    fig.savefig(pdf_path, bbox_inches=None, facecolor="white")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEGACY_FIGURES_DIR / "figure3.pdf", bbox_inches=None, facecolor="white")
    fig.savefig(LEGACY_FIGURES_DIR / "figure3.svg", bbox_inches=None, facecolor="white")
    fig.savefig(LEGACY_FIGURES_DIR / "figure3.png", dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {LEGACY_FIGURES_DIR / 'figure3.pdf'}")
    print(f"Animals used: {raw['Animal'].nunique()} ({raw.groupby(['Experiment', 'group'])['Animal'].nunique().to_dict()})")


if __name__ == "__main__":
    main()
