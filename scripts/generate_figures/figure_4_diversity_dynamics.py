# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""
Regenerate manuscript Figure 4 from the bundled MoSeq analysis inputs.

This script follows the manuscript analysis structure for frequency metrics,
bout duration, representative ethograms/barcodes, and transition metrics, but
assembles them into the manuscript Figure 4 layout.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import entropy

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting import plot_chord_diagram
from src.config import (
    CLUSTER_JSON,
    FIGURES_DIR,
    FIGURE_SOURCE_DATA_DIR,
    PROCESSED_DATA_DIR,
    STATISTICS_DIR,
    SYLLABLE_TIMEBIN_250MS,
)
from src.statistics import compute_diversity_metrics, compute_bout_duration

plt.rcParams["axes.grid"] = False

FIGURE_OUTPUT_DIR = FIGURES_DIR
SOURCE_OUTPUT_DIR = FIGURE_SOURCE_DATA_DIR
PROCESSED_OUTPUT_DIR = PROCESSED_DATA_DIR
STATISTICS_OUTPUT_DIR = STATISTICS_DIR

AXIS = "#4D4D4D"
CONTROL = "#F9C74F"
ELS = "#C37BA0"
PALETTE = {"Control": CONTROL, "ELS": ELS}
EVENT_STARTS = [3.0, 4.5, 6.0]
EVENT_DURATION_MIN = 30 / 60
SHOCK_DURATION_MIN = 2 / 60
EVENT_SPANS = [(start, start + EVENT_DURATION_MIN) for start in EVENT_STARTS]
REPRESENTATIVE_ANIMALS = {"Control": "129.4", "ELS": "88.3"}

CLUSTER_MAP = {
    "Freezing": [0, 28],
    "Sniffing": [18, 20],
    "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111],
    "Jump": [23, 29, 30, 34],
}
DISPLAY_ORDER = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
DISPLAY_LABELS = {
    "Freezing": "Freeze",
    "Sniffing": "Sniff",
    "Grooming": "Groom",
    "Turn": "Turn",
    "Locomotion": "Locomotion",
    "Climbing": "Climb",
    "Jump": "Jump",
}
COLORS = {
    "Freezing": "#C37BA0",
    "Sniffing": "#5B9AAA",
    "Grooming": "#8EC6DE",
    "Turn": "#B7DB45",
    "Locomotion": "#F1C232",
    "Climbing": "#F4A259",
    "Jump": "#E45756",
}
BOX_LABEL_X = -0.52
BOX_TAG_X = -0.58


def read_timebin_data() -> pd.DataFrame:
    if not SYLLABLE_TIMEBIN_250MS.exists():
        raise FileNotFoundError(f"Missing bundled raw data: {SYLLABLE_TIMEBIN_250MS}")
    return pd.read_csv(SYLLABLE_TIMEBIN_250MS)


def syllable_to_cluster() -> dict[int, str]:
    cluster_map = CLUSTER_MAP
    if CLUSTER_JSON.exists():
        with CLUSTER_JSON.open("r", encoding="utf-8") as handle:
            cluster_map = json.load(handle)
    out = {}
    for cluster, syllables in cluster_map.items():
        for syllable in syllables:
            out[int(syllable)] = cluster
    return out


def load_sequences() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]], pd.DataFrame]:
    raw = read_timebin_data()
    raw = raw.rename(columns={"Time Bin": "time_bin", "Condition": "group"})
    raw = raw[raw["group"].isin(["Control", "ELS"])].copy()
    raw["Animal"] = raw["Animal"].astype(str)
    raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
    raw["cluster"] = raw["Syllable"].map(syllable_to_cluster()).fillna("")
    raw["_row_order"] = np.arange(len(raw))

    full_sequences = {
        str(animal): group["cluster"].tolist()
        for animal, group in raw.groupby("Animal", sort=False)
    }

    # The 250 ms file can contain several syllables within one bin. The
    # ethogram/barcode display used the predominant behavior per bin. The metric
    # scripts, however, run on the full mapped table above.
    idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
    pred = raw.loc[idx].sort_values(["Animal", "time_bin"]).reset_index(drop=True)
    pred_sequences = {
        str(animal): group["cluster"].tolist()
        for animal, group in pred.groupby("Animal", sort=False)
    }
    meta = pred[["Animal", "group", "Experiment"]].drop_duplicates().reset_index(drop=True)
    return pred, pred_sequences, full_sequences, meta


def compute_frequency_metrics(sequences: dict[str, list[str]], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    usage_rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        vals, counts = np.unique(seq, return_counts=True)
        counts_map = dict(zip(vals, counts))
        order = sorted(counts_map)
        p = np.array([counts_map.get(cluster, 0) for cluster in order], dtype=float)
        p = p / p.sum()
        nonzero = p[p > 0]
        shannon = entropy(nonzero)
        evenness = shannon / np.log(len(nonzero)) if len(nonzero) else np.nan
        simpson = 1 - np.sum(p**2)
        named_p = np.array([counts_map.get(cluster, 0) for cluster in order if str(cluster).strip() != ""], dtype=float)
        named_p = named_p / counts.sum()
        sorted_p = np.sort(named_p)[::-1]
        cum = np.cumsum(sorted_p)
        baseline = (len(cum) + 1) / (2 * len(cum))
        cui = (cum.mean() - baseline) / (1 - baseline)
        rows.append(
            {
                "Animal": animal,
                "group": info["group"],
                "Experiment": info["Experiment"],
                "simpson": simpson,
                "shannon": shannon,
                "evenness": evenness,
                "cui": cui,
            }
        )
        usage_rows.append({"Animal": animal, "group": info["group"], "Experiment": info["Experiment"], **dict(zip(order, p))})
    return pd.DataFrame(rows), pd.DataFrame(usage_rows)


def bout_table(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        prev = seq[0]
        length = 1
        for cluster in seq[1:] + ["__END__"]:
            if cluster == prev:
                length += 1
                continue
            rows.append(
                {
                    "Animal": animal,
                    "group": info["group"],
                    "Experiment": info["Experiment"],
                    "cluster": prev,
                    "bout_duration": length * 0.25,
                }
            )
            prev = cluster
            length = 1
    return pd.DataFrame(rows)


def transition_matrix(seq: list[str]) -> np.ndarray:
    idx = {cluster: i for i, cluster in enumerate(DISPLAY_ORDER)}
    mat = np.zeros((len(DISPLAY_ORDER), len(DISPLAY_ORDER)), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    return mat


def transition_matrix_flow(seq: list[str]) -> np.ndarray:
    filtered = [seq[0]]
    for cluster in seq[1:]:
        if cluster != filtered[-1]:
            filtered.append(cluster)
    return transition_matrix(filtered)


def lz_complexity(seq: list[str]) -> int:
    token_map = {value: i for i, value in enumerate(pd.unique(pd.Series(seq)))}
    tokens = [token_map[x] for x in seq]
    n, i, c, k = len(tokens), 0, 1, 1
    while True:
        if i + k > n:
            break
        sub = tokens[i : i + k]
        found = any(tokens[j : j + k] == sub for j in range(i))
        if found:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break
    return c


def transition_metrics(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        rows.append(
            {
                "Animal": animal,
                "group": info["group"],
                "Experiment": info["Experiment"],
                "lz": lz_complexity(seq),
                "recurrence": recurrence_rate(seq),
                "determinism": determinism(seq),
                "markov": markov_entropy(seq, smoothing_factor=0.01),
            }
        )
    return pd.DataFrame(rows)


def _bout_lengths(seq: list[str]) -> list[int]:
    lengths = []
    prev = seq[0]
    length = 1
    for cluster in seq[1:] + ["__END__"]:
        if cluster == prev:
            length += 1
        else:
            lengths.append(length)
            prev = cluster
            length = 1
    return lengths


def recurrence_rate(seq: list[str]) -> float:
    _, counts = np.unique(seq, return_counts=True)
    n = len(seq)
    return float(np.sum(counts * counts) / (n * n))


def determinism(seq: list[str], min_length: int = 2) -> float:
    arr = np.asarray(seq)
    n = len(arr)
    total = 0
    diag_sum = 0
    for offset in range(-n + 1, n):
        diag = arr[: n - abs(offset)] == arr[abs(offset) :] if offset >= 0 else arr[-offset:] == arr[: n + offset]
        if offset == 0:
            total += int(diag.sum()) - n
        else:
            total += int(diag.sum())
        run = 0
        for value in diag:
            if value:
                run += 1
            else:
                if run >= min_length:
                    diag_sum += run
                run = 0
        if run >= min_length:
            diag_sum += run
    return float(diag_sum / total) if total > 0 else 0.0


def markov_entropy(seq: list[str], smoothing_factor: float = 0.01) -> float:
    states = list(seq)
    unique_states = list(pd.unique(pd.Series(states)))
    if len(unique_states) == 1:
        return 0.0
    idx = {state: i for i, state in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    for a, b in zip(states, states[1:]):
        counts[idx[a], idx[b]] += 1
    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)
    value_counts = pd.Series(states).value_counts()
    stationary = np.array([value_counts.get(state, 0) for state in unique_states], dtype=float) / len(states)
    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])
    return float(np.sum(stationary * inner))


def representative(usage: pd.DataFrame, group: str) -> str:
    if group in REPRESENTATIVE_ANIMALS:
        return REPRESENTATIVE_ANIMALS[group]
    sub = usage[usage["group"] == group].copy()
    cols = [col for col in sub.columns if col not in {"Animal", "group", "Experiment"}]
    mean = sub[cols].mean().to_numpy()
    dist = np.linalg.norm(sub[cols].to_numpy() - mean[None, :], axis=1)
    return str(sub.iloc[int(np.argmin(dist))]["Animal"])


def style_axis(ax: plt.Axes, labelsize: float = 5.5) -> None:
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.45)
    ax.tick_params(axis="both", colors=AXIS, labelsize=labelsize, width=0.45, length=2)
    ax.yaxis.set_tick_params(left=True)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def tag(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.12) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, fontweight="bold", color=AXIS, clip_on=False)


def add_events(ax: plt.Axes, y: float = 1.030, height: float = 0.074) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for start, end in EVENT_SPANS:
        ax.add_patch(
            Rectangle(
                (start, y),
                end - start,
                height,
                transform=trans,
                facecolor="#E0E0E0",
                edgecolor="none",
                clip_on=False,
                zorder=5,
            )
        )
        ax.add_patch(
            Rectangle(
                (end - SHOCK_DURATION_MIN, y),
                SHOCK_DURATION_MIN,
                height,
                transform=trans,
                facecolor="#F1C232",
                edgecolor="none",
                clip_on=False,
                zorder=6,
            )
        )


def behavior_legend(ax: plt.Axes, y_anchor: float = -0.28) -> None:
    handles = [plt.Line2D([0], [0], color=COLORS[c], linewidth=2.3, label=DISPLAY_LABELS[c]) for c in DISPLAY_ORDER]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=4.8,
        ncol=7,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.45,
        borderaxespad=0.1,
    )


def plot_ethogram(ax: plt.Axes, pred: pd.DataFrame, animal: str, letter: str) -> None:
    sub = pred[pred["Animal"].astype(str) == str(animal)].sort_values("time_bin")
    for yi, cluster in enumerate(DISPLAY_ORDER):
        x = sub.loc[sub["cluster"] == cluster, "time_bin"].to_numpy() / 60.0
        ax.vlines(x, yi - 0.31, yi + 0.31, color=COLORS[cluster], linewidth=0.12, alpha=0.82)
    add_events(ax)
    ax.set_yticks(np.arange(len(DISPLAY_ORDER)))
    ax.set_yticklabels([DISPLAY_LABELS[c] for c in DISPLAY_ORDER], fontsize=4.8)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=6.3, labelpad=0, loc="right")
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, length=0, pad=4)
    ax.tick_params(axis="x", labelsize=6.3, pad=0, length=1.5, width=0.35)
    ax.xaxis.set_label_coords(1.0, -0.120)
    behavior_legend(ax)
    tag(ax, letter, x=-0.12, y=1.16)


def plot_barcode(ax: plt.Axes, pred: pd.DataFrame, animal: str, letter: str) -> None:
    sub = pred[pred["Animal"].astype(str) == str(animal)].sort_values("time_bin")
    rgb = np.array([mcolors.to_rgb(COLORS.get(c, "#FFFFFF")) for c in sub["cluster"]])[None, :, :]
    ax.imshow(rgb, aspect="auto", extent=[0, 7.5, 0.00, 0.66])
    add_events(ax, y=0.735, height=0.084)
    ax.set_yticks([])
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=6.3, labelpad=0, loc="right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.35)
    ax.tick_params(axis="x", colors=AXIS, labelsize=6.3, width=0.35, length=1.5, pad=0)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.xaxis.label.set_color(AXIS)
    ax.xaxis.set_label_coords(1.0, -0.120)
    behavior_legend(ax)
    tag(ax, letter, x=-0.020, y=0.80)


def box_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    letter: str,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | None = None,
    yfmt: str | None = None,
    star: bool = False,
    y_jitter: bool = True,
) -> None:
    positions = [0, 1]
    all_values = df[y].dropna()
    if ylim:
        ax.set_ylim(*ylim)
    for i, group in enumerate(["Control", "ELS"]):
        values = df[df["group"] == group][y].dropna()
        color = PALETTE[group]
        ax.boxplot(values, positions=[i], widths=0.48, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": (*mcolors.to_rgb(color), 0.15), "edgecolor": color, "linewidth": 0.55},
                   whiskerprops={"color": color, "linewidth": 0.55},
                   capprops={"color": color, "linewidth": 0.55},
                   medianprops={"color": color, "linewidth": 0.70})
        rng = np.random.default_rng(42 + i)
        jitter = rng.normal(i, 0.024, len(values))
        display_values = values.to_numpy(dtype=float)
        if y_jitter and len(values):
            ymin, ymax = ax.get_ylim()
            yr = (ymax - ymin) if ylim else max(float(all_values.max() - all_values.min()), 1e-6)
            display_values = display_values + rng.normal(0, yr * 0.006, len(values))
            if ylim:
                display_values = np.clip(display_values, *ax.get_ylim())
        ax.scatter(jitter, display_values, color=color, s=4.0, alpha=0.76, edgecolors="none", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Control", "ELS"], fontsize=7.2)
    ax.set_ylabel(ylabel, fontsize=5.4, labelpad=2)
    ax.yaxis.set_label_coords(BOX_LABEL_X, 0.5)
    if star:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([0.0, 0.0, 1.0, 1.0], [1.005, 1.025, 1.025, 1.005], color=AXIS, linewidth=0.45, transform=trans, clip_on=False)
        ax.text(0.5, 1.018, "*", transform=trans, ha="center", va="bottom", fontsize=7.0, color=AXIS, clip_on=False)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yfmt is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    style_axis(ax, labelsize=5.0)
    tag(ax, letter, x=BOX_TAG_X, y=1.14)


def plot_cumulative(ax: plt.Axes, usage: pd.DataFrame, letter: str) -> None:
    named_usage = usage.copy()
    total_including_blank = named_usage[[col for col in named_usage.columns if col not in {"Animal", "group", "Experiment"}]].sum(axis=1)
    cols = [col for col in DISPLAY_ORDER if col in named_usage.columns]
    rank_order = named_usage[cols].sum().sort_values(ascending=False).index.tolist()
    x = np.arange(len(rank_order))
    for group in ["Control", "ELS"]:
        sub = named_usage[named_usage["group"] == group][rank_order]
        mean = sub.mean().cumsum()
        sem = sub.sem().fillna(0).cumsum()
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            color=PALETTE[group],
            marker="o",
            markersize=2.2,
            linewidth=0.75,
            elinewidth=0.55,
            capsize=1.4,
            label=group,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS.get(c, c) for c in rank_order], rotation=0, fontsize=5.2)
    ax.set_ylabel("Cumulative Usage", fontsize=5.8)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.set_ylim(0.4, 0.92)
    ax.set_yticks(np.arange(0.4, 1.0, 0.1))
    handles = [
        plt.Line2D([0], [0], color=PALETTE[group], marker="o", markersize=2.0, linewidth=0.75, label=group)
        for group in ["Control", "ELS"]
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.4, loc="lower right", ncol=2, columnspacing=0.9, handletextpad=0.35)
    style_axis(ax, labelsize=5.4)
    ax.spines[["top", "right"]].set_visible(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.45)
    tag(ax, letter, x=-0.11, y=1.00)


def plot_group_cumulative(ax: plt.Axes, usage: pd.DataFrame, group: str, letter: str) -> None:
    cols = [col for col in DISPLAY_ORDER if col in usage.columns]
    row = usage[usage["group"] == group][cols].mean().sort_values(ascending=False).cumsum()
    x = np.arange(len(row))
    ax.plot(x, row, color=PALETTE[group], marker="o", markersize=2, linewidth=0.65)
    ax.fill_between(x, 0, row, color=PALETTE[group], alpha=0.20)
    ax.set_title(group, fontsize=6, color=AXIS, pad=2)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS.get(s, s)[:3] for s in row.index], fontsize=4.8)
    ax.set_ylabel("Cumulative Usage", fontsize=5.4)
    style_axis(ax, labelsize=5.2)
    tag(ax, letter, x=-0.16)


def plot_chord(ax: plt.Axes, sequences: dict[str, list[str]], meta: pd.DataFrame, group: str, letter: str) -> None:
    mats = []
    group_map = meta.set_index("Animal")["group"].to_dict()
    for animal, seq in sequences.items():
        if group_map.get(str(animal)) != group:
            continue
        mats.append(transition_matrix_flow(seq))
    mat = np.mean(mats, axis=0)
    labels = [DISPLAY_LABELS[c] for c in DISPLAY_ORDER]
    color_map = {DISPLAY_LABELS[c]: COLORS[c] for c in DISPLAY_ORDER}
    plot_chord_diagram(
        mat,
        labels,
        color_map,
        title=group,
        ax=ax,
        r=1.48,
        gap=0.0,
        arc_width=0.064,
        label_fontsize=8,
        title_fontsize=9,
        title_pad=12,
        limit_pad=1.08,
        label_radius_factor=1.2,
        flip_labels=False,
        edge_lw_scale=7,
        edge_lw_offset=0.18,
        edge_alpha=0.90,
        edge_alpha_min=None,
        delta_angle=0.15,
    )
    tag(ax, letter, x=0.02, y=0.98)


def equalize_boxplot_heights(axes: list[plt.Axes], width_scale: float = 0.78) -> None:
    heights = [ax.get_position().height for ax in axes]
    if not heights:
        return
    target = min(heights)
    for ax in axes:
        pos = ax.get_position()
        y_center = pos.y0 + pos.height / 2
        x_center = pos.x0 + pos.width / 2
        target_width = pos.width * width_scale
        ax.set_position([x_center - target_width / 2, y_center - target / 2, target_width, target])


def shift_axes(axes: list[plt.Axes], dx: float = 0.0, dy: float = 0.0) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])


def expand_axes_left(axes: list[plt.Axes], amount: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - amount, pos.y0, pos.width + amount, pos.height])


def export_source_data(metrics: pd.DataFrame, bouts: pd.DataFrame, transitions: pd.DataFrame) -> None:
    """Export Figure 4 source data: diversity metrics, bout durations, and transitions."""
    import statsmodels.formula.api as smf

    rows = []

    # Descriptive stats: diversity metrics
    for metric_name in ["simpson", "shannon", "evenness", "cui"]:
        if metric_name in metrics.columns:
            for group in ["Control", "ELS"]:
                data = metrics[metrics["group"] == group][metric_name]
                if len(data) > 0:
                    rows.append({
                        "metric_category": "diversity",
                        "metric": metric_name,
                        "group": group,
                        "mean": data.mean(),
                        "std": data.std(),
                        "sem": data.sem(),
                        "n": len(data),
                    })

    # Descriptive stats: bout durations (per cluster)
    for cluster in ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]:
        cluster_bouts = bouts[bouts["cluster"] == cluster]
        if not cluster_bouts.empty:
            for group in ["Control", "ELS"]:
                group_bouts = cluster_bouts[cluster_bouts["group"] == group]["bout_duration"]
                if len(group_bouts) > 0:
                    rows.append({
                        "metric_category": "bout_duration",
                        "metric": cluster,
                        "group": group,
                        "mean": group_bouts.mean(),
                        "std": group_bouts.std(),
                        "sem": group_bouts.sem(),
                        "n": len(group_bouts),
                    })

    # Descriptive stats: transition metrics
    for metric_name in ["lz", "recurrence", "determinism", "markov"]:
        if metric_name in transitions.columns:
            for group in ["Control", "ELS"]:
                data = transitions[transitions["group"] == group][metric_name]
                if len(data) > 0:
                    rows.append({
                        "metric_category": "transition",
                        "metric": metric_name,
                        "group": group,
                        "mean": data.mean(),
                        "std": data.std(),
                        "sem": data.sem(),
                        "n": len(data),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        SOURCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_csv = SOURCE_OUTPUT_DIR / "figure4.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(PROCESSED_OUTPUT_DIR / "transition_metrics_per_animal.csv", index=False)

    # The Condition models for Figure 4 are no longer fitted here. They were
    # fitted a second time in scripts/build_statistical_report.py, and because
    # these MixedLM fits sit on the random-effect variance boundary the two runs
    # converged to different standard errors, so the repository reported two
    # answers for one model. The builder fits once and writes
    # statistics/stats_figure4_*.csv; this script only draws.


def main() -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred, pred_sequences, full_sequences, meta = load_sequences()
    metrics, usage = compute_frequency_metrics(full_sequences, meta)
    bouts = bout_table(full_sequences, meta)
    transitions = transition_metrics(full_sequences, meta)

    ctrl_rep = representative(usage, "Control")
    els_rep = representative(usage, "ELS")

    fig = plt.figure(figsize=(8.22, 11.28), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        5,
        14,
        left=0.075,
        right=0.955,
        top=0.955,
        bottom=0.045,
        hspace=0.50,
        wspace=0.42,
        height_ratios=[0.92, 0.92, 2.18, 1.34, 2.18],
    )

    axA = fig.add_subplot(gs[0, 0:6])
    axB = fig.add_subplot(gs[0, 8:14])
    axC = fig.add_subplot(gs[1, 0:6])
    axD = fig.add_subplot(gs[1, 8:14])
    plot_ethogram(axA, pred, ctrl_rep, "A")
    plot_barcode(axB, pred, ctrl_rep, "B")
    plot_ethogram(axC, pred, els_rep, "C")
    plot_barcode(axD, pred, els_rep, "D")
    shift_axes([axA, axC], dx=0.020)
    expand_axes_left([axB, axD], amount=0.065)

    freq_grid = gs[2, 0:4].subgridspec(2, 2, hspace=0.22, wspace=0.52)
    box_axes: list[plt.Axes] = []
    axE = fig.add_subplot(freq_grid[0, 0]); box_axes.append(axE)
    axF = fig.add_subplot(freq_grid[0, 1]); box_axes.append(axF)
    axG = fig.add_subplot(freq_grid[1, 0]); box_axes.append(axG)
    axH = fig.add_subplot(freq_grid[1, 1]); box_axes.append(axH)
    box_scatter(axE, metrics, "simpson", "Simpson index", "E", (0.58, 0.80), yticks=[0.60, 0.65, 0.70, 0.75, 0.80], yfmt="%.2f", star=False)
    box_scatter(axF, metrics, "shannon", "Shannon entropy index", "F", (1.20, 1.65), yticks=np.arange(1.20, 1.66, 0.05).round(2).tolist(), yfmt="%.2f")
    box_scatter(axG, metrics, "evenness", "Evenness index", "G", (0.58, 0.85), yticks=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85], yfmt="%.2f")
    box_scatter(axH, metrics, "cui", "Cumulative usage index", "H", (-0.25, 0.60), yticks=np.arange(-0.2, 0.61, 0.1).round(1).tolist(), yfmt="%.1f", star=True)
    axI = fig.add_subplot(gs[2, 5:10])
    plot_cumulative(axI, usage, "I")
    shift_axes([axI], dx=0.016)
    side_grid = gs[2, 11:14].subgridspec(2, 1, hspace=0.48)
    plot_group_cumulative(fig.add_subplot(side_grid[0]), usage, "Control", "")
    plot_group_cumulative(fig.add_subplot(side_grid[1]), usage, "ELS", "")

    bout_grid = gs[3, :].subgridspec(1, 8, wspace=0.48)
    overall = bouts.groupby(["Animal", "group"])["bout_duration"].mean().reset_index()
    cluster_means = (
        bouts[bouts["cluster"].isin(DISPLAY_ORDER)]
        .groupby(["Animal", "group", "cluster"], as_index=False)["bout_duration"]
        .mean()
    )
    for ax_i, (label, data, ylim, yticks, yfmt, star) in enumerate(
        [
            ("Overall", overall, (0, 2.0), [0, 0.5, 1.0, 1.5, 2.0], "%.1f", False),
            ("Freeze", cluster_means[cluster_means["cluster"] == "Freezing"], (0, 2.0), [0, 0.5, 1.0, 1.5, 2.0], "%.1f", True),
            ("Sniff", cluster_means[cluster_means["cluster"] == "Sniffing"], (0, 4.0), [0, 1, 2, 3, 4], "%.0f", True),
            ("Groom", cluster_means[cluster_means["cluster"] == "Grooming"], (0, 0.6), [0, 0.2, 0.4, 0.6], "%.1f", False),
            ("Turn", cluster_means[cluster_means["cluster"] == "Turn"], (0, 2.5), [0, 0.5, 1.0, 1.5, 2.0, 2.5], "%.1f", True),
            ("Locomotion", cluster_means[cluster_means["cluster"] == "Locomotion"], (0, 0.85), [0, 0.2, 0.4, 0.6, 0.8], "%.1f", False),
            ("Climb", cluster_means[cluster_means["cluster"] == "Climbing"], (0, 2.0), [0, 0.5, 1.0, 1.5, 2.0], "%.1f", False),
            ("Jump", cluster_means[cluster_means["cluster"] == "Jump"], (0, 1.25), [0, 0.25, 0.50, 0.75, 1.00, 1.25], "%.2f", False),
        ]
    ):
        ax = fig.add_subplot(bout_grid[ax_i])
        box_axes.append(ax)
        box_scatter(ax, data.rename(columns={"bout_duration": "value"}), "value", "Mean Bout Duration (s)", chr(ord("J") + ax_i), ylim, yticks=yticks, yfmt=yfmt, star=star)
        ax.set_title(label, fontsize=6.4, color=AXIS, pad=10)

    axR = fig.add_subplot(gs[4, 0:5])
    axS = fig.add_subplot(gs[4, 5:10])
    plot_chord(axR, full_sequences, meta, "Control", "R")
    plot_chord(axS, full_sequences, meta, "ELS", "S")
    shift_axes([axR], dx=-0.060)
    shift_axes([axS], dx=-0.085)
    metric_grid = gs[4, 10:14].subgridspec(2, 2, hspace=0.42, wspace=0.58)
    axT = fig.add_subplot(metric_grid[0, 0]); box_axes.append(axT)
    axU = fig.add_subplot(metric_grid[0, 1]); box_axes.append(axU)
    axV = fig.add_subplot(metric_grid[1, 0]); box_axes.append(axV)
    axW = fig.add_subplot(metric_grid[1, 1]); box_axes.append(axW)
    box_scatter(axT, transitions, "lz", "Lempel-Ziv complexity", "T", (100, 260), yticks=list(range(100, 261, 20)), yfmt="%.0f")
    box_scatter(axU, transitions, "recurrence", "Recurrence Rate", "U", (0.24, 0.43), yticks=[0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.430], yfmt="%.3f", star=False)
    box_scatter(axV, transitions, "determinism", "Determinism", "V", (0.72, 0.90), yticks=np.arange(0.700, 0.901, 0.025).round(3).tolist(), yfmt="%.3f", star=True)
    box_scatter(axW, transitions, "markov", "Markov Entropy", "W", (0.7, 1.5), yticks=np.arange(0.7, 1.51, 0.1).round(1).tolist(), yfmt="%.1f", star=False)

    equalize_boxplot_heights(box_axes)

    pdf = FIGURE_OUTPUT_DIR / "figure4.pdf"
    svg = FIGURE_OUTPUT_DIR / "figure4.svg"
    png = FIGURE_OUTPUT_DIR / "figure4.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {svg}")
    print(f"Saved {png}")
    print(f"Representative Control: {ctrl_rep}; ELS: {els_rep}")

    # Export source data
    print("Computing Figure 4 source statistics...")
    export_source_data(metrics, bouts, transitions)


if __name__ == "__main__":
    main()
