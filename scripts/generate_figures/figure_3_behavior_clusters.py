# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gómez and Antonio Lozano
"""
Regenerate manuscript Figure 3 from the archived MoSeq cluster time-bin data.

The figure uses the 82-animal Control/ELS cohort in
Syllable_per_timebin_final(30s).csv and the hand-curated syllable-to-behavior
cluster mapping used elsewhere in this repository.
"""

from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    COPING_DATA2_ZIP,
    COPING2_MEMBER_TIMEBIN_30S as ARCHIVE_MEMBER,
    FIGURE_DATA_DIR as OUTPUT_DIR,
    SYLLABLE_TIMEBIN_30S,
)
from src.statistics import fit_mixed_models

LEGACY_FIGURES_DIR = REPO_ROOT / "figures"

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
    if SYLLABLE_TIMEBIN_30S.exists():
        df = pd.read_csv(SYLLABLE_TIMEBIN_30S)
    elif COPING_DATA2_ZIP.exists():
        with zipfile.ZipFile(COPING_DATA2_ZIP) as archive:
            with archive.open(ARCHIVE_MEMBER) as handle:
                df = pd.read_csv(handle)
    else:
        raise FileNotFoundError(
            f"Missing source data. Expected {SYLLABLE_TIMEBIN_30S} or member "
            f"{ARCHIVE_MEMBER} inside {COPING_DATA2_ZIP}."
        )
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


def align_panel_letters_to_ylabels(fig: plt.Figure, letter_artists: list[tuple[plt.Axes, plt.Text]]) -> None:
    """Align panel letters to each panel's y-axis title column."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    for ax, text in letter_artists:
        ylabel_box = ax.yaxis.label.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        x_fig = fig_inv.transform((ylabel_box.x0, ylabel_box.y0))[0]
        y_fig = fig_inv.transform((axes_box.x0, axes_box.y1))[1] + 0.012
        text.set_transform(fig.transFigure)
        text.set_position((x_fig, y_fig))


def add_shock_shading(ax: plt.Axes) -> None:
    for start in [3.5, 5.0, 6.5]:
        ax.axvspan(start, start + 0.5, color=SHADE_COLOR, zorder=0)


def plot_bar_panel(ax: plt.Axes, total_summary: pd.DataFrame) -> plt.Text:
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
    ax.set_ylabel("Frequency (s)", fontsize=8, labelpad=12)
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
    return panel_letter(ax, "A", x=-0.10, y=1.10)


def plot_time_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    cluster: str,
    letter: str,
    star: bool = False,
    legend_loc: str = "upper right",
) -> plt.Text:
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
    ax.set_title(cluster, fontsize=7, pad=5, y=1.075)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=6.5, labelpad=2)
    ax.set_ylabel("% of time in cluster", fontsize=7, labelpad=7)
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
        ax.text(
            4.0,
            0.992,
            "*",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=AXIS_COLOR,
            fontsize=11,
            fontweight="bold",
        )
    style_axis(ax)
    return panel_letter(ax, letter)


def compute_and_export_source_data(cluster_df: pd.DataFrame, time_summary: pd.DataFrame, total_summary: pd.DataFrame, output_dir: Path) -> None:
    """Compute Figure 3 source statistics and export to CSV."""
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    rows = []

    # Extract per-cluster, per-group summary statistics from total_summary
    for cluster in PANEL_ORDER:
        for group in ["Control", "ELS"]:
            data = total_summary[(total_summary["cluster"] == cluster) & (total_summary["group"] == group)]
            if not data.empty:
                row = data.iloc[0]
                rows.append({
                    "metric": f"{cluster} {group} frequency",
                    "cluster": cluster,
                    "group": group,
                    "mean_seconds": row["mean"],
                    "sem_seconds": row["sem"],
                    "n_animals": row["n"],
                })

    # Extract timecourse summaries (mean, SEM per time bin and group)
    for cluster in PANEL_ORDER:
        cluster_time = time_summary[time_summary["cluster"] == cluster]
        if cluster_time.empty:
            continue
        for group in ["Control", "ELS"]:
            group_time = cluster_time[cluster_time["group"] == group]
            if not group_time.empty:
                rows.append({
                    "metric": f"{cluster} {group} timecourse mean",
                    "cluster": cluster,
                    "group": group,
                    "mean_pct": group_time["mean"].mean(),
                    "sem_pct": group_time["sem"].mean(),
                    "n_timebins": len(group_time),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        output_csv = output_dir / "source_data_figure3.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    else:
        print("Warning: No source data generated for Figure 3")

    # --- GEE Negative Binomial for Fig 3A cluster frequency ---
    freq_csv = REPO_ROOT / "data" / "source" / "cluster_frequency_per_animal.csv"
    gee_rows: list[dict] = []
    if freq_csv.exists():
        freq_df = pd.read_csv(freq_csv)
        freq_df["group"] = pd.Categorical(freq_df["group"], categories=["Control", "ELS"])
        for cluster in PANEL_ORDER:
            sub = freq_df[freq_df["cluster"] == cluster].dropna(subset=["frequency_seconds"])
            if sub.empty or sub["frequency_seconds"].std() == 0:
                continue
            try:
                model = smf.gee(
                    "frequency_seconds ~ C(group, Treatment('Control')) + experiment",
                    data=sub,
                    groups=sub["animal_id"],
                    family=sm.families.NegativeBinomial(alpha=1.0),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
                ci = model.conf_int()
                for param in model.params.index:
                    gee_rows.append({
                        "figure": "3A",
                        "cluster": cluster,
                        "parameter": param,
                        "beta": model.params[param],
                        "se": model.bse[param],
                        "z": model.tvalues[param],
                        "p_value": model.pvalues[param],
                        "ci_low": ci.loc[param, 0],
                        "ci_high": ci.loc[param, 1],
                    })
            except Exception as e:
                print(f"GEE NegBin failed for {cluster}: {e}")
    else:
        print(f"Warning: {freq_csv} not found — skipping GEE for Fig 3A")

    if gee_rows:
        gee_path = output_dir / "stats_figure3A_GEE.csv"
        pd.DataFrame(gee_rows).to_csv(gee_path, index=False)
        print(f"Saved: {gee_path}")

    # --- MixedLM for overtime panels (Fig 3B-H, 30s timebins) ---
    mm_frames: list[pd.DataFrame] = []
    for cluster in PANEL_ORDER:
        sub = cluster_df[cluster_df["cluster"] == cluster].copy()
        sub = sub.rename(columns={
            "Animal": "animal_id",
            "Experiment": "experiment",
            "time_bin": "time_bin_numeric",
        })
        try:
            results = fit_mixed_models(sub, "Percentage")
            results.insert(0, "cluster", cluster)
            mm_frames.append(results)
        except Exception as e:
            print(f"MixedLM failed for {cluster}: {e}")

    if mm_frames:
        mm_path = output_dir / "stats_figure3_overtime.csv"
        pd.concat(mm_frames, ignore_index=True).to_csv(mm_path, index=False)
        print(f"Saved: {mm_path}")


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

    letter_artists = [(ax_a, plot_bar_panel(ax_a, total_summary))]
    for cluster, letter, star, legend_loc in [
        ("Freeze", "B", True, "lower right"),
        ("Sniff", "C", True, "upper right"),
        ("Groom", "D", False, "upper right"),
        ("Turn", "E", True, "lower left"),
        ("Locomotion", "F", True, "upper right"),
        ("Climb", "G", False, "upper right"),
        ("Jump", "H", False, "upper right"),
    ]:
        letter_artist = plot_time_panel(
            axes[cluster],
            time_summary,
            cluster,
            letter,
            star=star,
            legend_loc=legend_loc,
        )
        letter_artists.append((axes[cluster], letter_artist))

    align_panel_letters_to_ylabels(fig, letter_artists)

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

    # Compute and export source data
    print("Computing Figure 3 source statistics...")
    compute_and_export_source_data(cluster_df, time_summary, total_summary, OUTPUT_DIR)


if __name__ == "__main__":
    main()
