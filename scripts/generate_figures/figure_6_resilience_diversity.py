# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Regenerate manuscript Figure 6, the resilience version of Figure 4."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import entropy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plotting import plot_chord_diagram
from src.config import (
    CLUSTER_JSON,
    RESULTS_INTERMEDIATE_FIGURES_DIR,
    RESULTS_SOURCE_DATA_DIR,
    SYLLABLE_TIMEBIN_250MS,
)
from src.statistics import compute_diversity_metrics

LEGACY_FIGURES_DIR = REPO_ROOT / "figures"
FIGURE_OUTPUT_DIR = RESULTS_INTERMEDIATE_FIGURES_DIR
SOURCE_OUTPUT_DIR = RESULTS_SOURCE_DATA_DIR

AXIS = "#4D4D4D"
# Match the physical boxplot panel dimensions used in Figure 4.
BOXPLOT_WIDTH = 0.72
FIGURE4_BOXPLOT_AXIS_WIDTH_IN = 0.6052
FIGURE4_BOXPLOT_AXIS_HEIGHT_IN = 0.8760
FIGURE4_CUMULATIVE_AXIS_WIDTH_IN = 2.483
FIGURE4_CUMULATIVE_X0 = 0.4121
FIGURE4_CUMULATIVE_SIDE_X0 = 0.7814
FIGURE4_CUMULATIVE_SIDE_WIDTH_IN = 1.427
FIGURE4_CUMULATIVE_SIDE_RIGHT = 0.9550
SIDE_CUMULATIVE_WIDTH_SCALE = 0.90
LEFT_METRIC_TAG_X = -0.52
FREQUENCY_YLABEL_X = -0.38
FREQUENCY_TAG_X = -0.42
K_TAG_X = -0.122
K_TAG_Y = 1.034
TOP_ETHOGRAM_SPACING_TRIM = 0.015
FREQUENCY_METRIC_AXIS_WIDTH_IN = 0.72
BOUT_METRIC_AXIS_WIDTH_IN = 0.62
BOUT_YLABEL_X = -0.30
BOUT_TAG_X = -0.34
BOUT_TAG_Y = 1.045
TRANSITION_YLABEL_X = -0.42
TRANSITION_TAG_X = -0.42
TRANSITION_TAG_Y = 1.16
METRIC_BLOCK_DY = -0.022
METRIC_ROW_EXTRA_GAP = 0.020
BOUT_BLOCK_DY = -0.020
SIGNIFICANCE_Y_BASE = 0.875   # inside the plot, below the top y-axis edge
SIGNIFICANCE_Y_HEIGHT = 0.030 # bracket arm height in axes fraction
SIGNIFICANCE_Y_STEP = 0.000   # step between non-overlapping brackets (same level)
SIGNIFICANCE_STAR_PAD = 0.006 # gap between bracket and asterisk in axes fraction
SIGNIFICANCE_JOIN_GAP = 0.10  # horizontal split between adjacent brackets
CONTROL = "#F9C74F"
ELS = "#C37BA0"
RESILIENT = "#90BE6D"
PALETTE = {"Control": CONTROL, "ELS": ELS, "ELS resilient": RESILIENT}
GROUP_ORDER = ["Control", "ELS", "ELS resilient"]
EVENT_STARTS = [3.0, 4.5, 6.0]
EVENT_DURATION_MIN = 30 / 60
SHOCK_DURATION_MIN = 2 / 60
EVENT_SPANS = [(start, start + EVENT_DURATION_MIN) for start in EVENT_STARTS]

ELS_RESILIENT = {
    "2.4",
    "15.4",
    "26.4",
    "43.3",
    "43.5",
    "49.6",
    "123.3",
    "134.2",
    "148.4",
    "150.3",
    "150.5",
    "159.4",
}
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


def read_timebin_data() -> pd.DataFrame:
    if not SYLLABLE_TIMEBIN_250MS.exists():
        raise FileNotFoundError(f"Missing bundled raw data: {SYLLABLE_TIMEBIN_250MS}")
    return pd.read_csv(SYLLABLE_TIMEBIN_250MS)


def group_ext(animal: str, group: str) -> str:
    animal = str(animal)
    if group == "ELS" and animal in ELS_RESILIENT:
        return "ELS resilient"
    return group


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
    raw["group_ext"] = raw.apply(lambda r: group_ext(r["Animal"], r["group"]), axis=1)

    full_sequences = {
        str(animal): group.sort_values("time_bin")["cluster"].tolist()
        for animal, group in raw.groupby("Animal", sort=False)
    }
    idx = raw.groupby(["Animal", "time_bin"])["Percentage"].idxmax()
    pred = raw.loc[idx].sort_values(["Animal", "time_bin"]).reset_index(drop=True)
    pred_sequences = {
        str(animal): group["cluster"].tolist()
        for animal, group in pred.groupby("Animal", sort=False)
    }
    meta = pred[["Animal", "group", "group_ext", "Experiment"]].drop_duplicates().reset_index(drop=True)
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
                "group_ext": info["group_ext"],
                "Experiment": info["Experiment"],
                "simpson": simpson,
                "shannon": shannon,
                "evenness": evenness,
                "cui": cui,
            }
        )
        usage_rows.append({"Animal": animal, "group_ext": info["group_ext"], "Experiment": info["Experiment"], **dict(zip(order, p))})
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
            rows.append({"Animal": animal, "group_ext": info["group_ext"], "Experiment": info["Experiment"], "cluster": prev, "bout_duration": length * 0.25})
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
    unique_states = list(pd.unique(pd.Series(seq)))
    if len(unique_states) == 1:
        return 0.0
    idx = {state: i for i, state in enumerate(unique_states)}
    counts = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[idx[a], idx[b]] += 1
    counts += smoothing_factor
    probs = counts / counts.sum(axis=1, keepdims=True)
    value_counts = pd.Series(seq).value_counts()
    stationary = np.array([value_counts.get(state, 0) for state in unique_states], dtype=float) / len(seq)
    inner = np.array([-np.sum(row[row > 0] * np.log2(row[row > 0])) for row in probs])
    return float(np.sum(stationary * inner))


def transition_metrics(sequences: dict[str, list[str]], meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_map = meta.set_index("Animal").to_dict("index")
    for animal, seq in sequences.items():
        info = meta_map[animal]
        rows.append({"Animal": animal, "group_ext": info["group_ext"], "Experiment": info["Experiment"], "lz": lz_complexity(seq), "recurrence": recurrence_rate(seq), "determinism": determinism(seq), "markov": markov_entropy(seq)})
    return pd.DataFrame(rows)


def representative(usage: pd.DataFrame, group: str) -> str:
    if group in REPRESENTATIVE_ANIMALS:
        return REPRESENTATIVE_ANIMALS[group]
    sub = usage[usage["group_ext"] == group].copy()
    cols = [col for col in sub.columns if col not in {"Animal", "group_ext", "Experiment"}]
    mean = sub[cols].mean().to_numpy()
    dist = np.linalg.norm(sub[cols].to_numpy() - mean[None, :], axis=1)
    return str(sub.iloc[int(np.argmin(dist))]["Animal"])


def style_axis(ax: plt.Axes, labelsize: float = 5.2) -> None:
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.42)
    ax.tick_params(axis="both", colors=AXIS, labelsize=labelsize, width=0.42, length=1.8)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def tag(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.12) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, fontweight="bold", color=AXIS, clip_on=False)


def add_events(ax: plt.Axes, y: float = 1.030, height: float = 0.068) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for start, end in EVENT_SPANS:
        ax.add_patch(Rectangle((start, y), end - start, height, transform=trans, facecolor="#E0E0E0", edgecolor="none", clip_on=False, zorder=5))
        ax.add_patch(Rectangle((end - SHOCK_DURATION_MIN, y), SHOCK_DURATION_MIN, height, transform=trans, facecolor="#F1C232", edgecolor="none", clip_on=False, zorder=6))


def behavior_legend(ax: plt.Axes, y_anchor: float = -0.30) -> None:
    handles = [plt.Line2D([0], [0], color=COLORS[c], linewidth=2.0, label=DISPLAY_LABELS[c]) for c in DISPLAY_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y_anchor), frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=4.2, ncol=7, handlelength=1.8, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.1)


def plot_ethogram(ax: plt.Axes, pred: pd.DataFrame, animal: str, letter: str, show_legend: bool = False) -> None:
    sub = pred[pred["Animal"].astype(str) == str(animal)].sort_values("time_bin")
    for yi, cluster in enumerate(DISPLAY_ORDER):
        x = sub.loc[sub["cluster"] == cluster, "time_bin"].to_numpy() / 60.0
        ax.vlines(x, yi - 0.31, yi + 0.31, color=COLORS[cluster], linewidth=0.12, alpha=0.82)
    add_events(ax)
    ax.set_yticks(np.arange(len(DISPLAY_ORDER)))
    ax.set_yticklabels([DISPLAY_LABELS[c] for c in DISPLAY_ORDER], fontsize=4.4)
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=5.7, labelpad=0, loc="right")
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, length=0, pad=3)
    ax.tick_params(axis="x", labelsize=5.7, pad=0, length=1.3, width=0.35)
    ax.xaxis.set_label_coords(1.0, -0.14)
    if show_legend:
        behavior_legend(ax)
    tag(ax, letter, x=-0.10, y=1.13)


def plot_barcode(ax: plt.Axes, pred: pd.DataFrame, animal: str, letter: str, show_legend: bool = False) -> None:
    sub = pred[pred["Animal"].astype(str) == str(animal)].sort_values("time_bin")
    rgb = np.array([mcolors.to_rgb(COLORS.get(c, "#FFFFFF")) for c in sub["cluster"]])[None, :, :]
    ax.imshow(rgb, aspect="auto", extent=[0, 7.5, 0.00, 0.66])
    add_events(ax, y=0.735, height=0.080)
    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    ax.set_yticks([])
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, 8))
    ax.set_xlabel("Time (minutes)", fontsize=5.7, labelpad=0, loc="right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.35)
    ax.tick_params(axis="x", colors=AXIS, labelsize=5.7, width=0.35, length=1.3, pad=0)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.gridline.set_visible(False)
    ax.xaxis.label.set_color(AXIS)
    ax.xaxis.set_label_coords(1.0, -0.14)
    if show_legend:
        behavior_legend(ax)
    tag(ax, letter, x=-0.020, y=0.80)


def sig_bracket(ax: plt.Axes, x1: float, x2: float, level: int, n_levels: int, y_base: float = SIGNIFICANCE_Y_BASE) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if y_base < 0:
        y_top = y_base - SIGNIFICANCE_Y_STEP * (n_levels - level - 1)
        y_bot = y_top - SIGNIFICANCE_Y_HEIGHT
        star_y = y_bot - SIGNIFICANCE_STAR_PAD
        star_va = "top"
    else:
        y_bot = y_base + SIGNIFICANCE_Y_STEP * (n_levels - level - 1)
        y_top = y_bot + SIGNIFICANCE_Y_HEIGHT
        star_y = y_top + SIGNIFICANCE_STAR_PAD
        star_va = "bottom"
    ax.plot(
        [x1, x1, x2, x2],
        [y_bot, y_top, y_top, y_bot],
        transform=trans,
        color=AXIS,
        linewidth=0.48,
        clip_on=False,
        zorder=4,
    )
    ax.text(
        (x1 + x2) / 2,
        star_y,
        "*",
        transform=trans,
        ha="center",
        va=star_va,
        fontsize=5.9,
        color=AXIS,
        clip_on=False,
        zorder=5,
    )


def bracket_ylim(ylim: tuple[float, float], n_brackets: int) -> tuple[float, float]:
    if n_brackets == 0:
        return ylim
    lo, hi = ylim
    return lo, hi + (hi - lo) * 0.18


def split_touching_bracket(pair: tuple[int, int], pairs: list[tuple[int, int]]) -> tuple[float, float]:
    x1, x2 = map(float, pair)
    if any(other != pair and other[1] == pair[0] for other in pairs):
        x1 += SIGNIFICANCE_JOIN_GAP
    if any(other != pair and other[0] == pair[1] for other in pairs):
        x2 -= SIGNIFICANCE_JOIN_GAP
    return x1, x2


def box_scatter(ax: plt.Axes, df: pd.DataFrame, y: str, ylabel: str, letter: str, ylim: tuple[float, float], yticks: list[float], yfmt: str, stars: list[tuple[int, int]] | None = None, sig_y_base: float = SIGNIFICANCE_Y_BASE) -> None:
    rng = np.random.default_rng(42)
    data = [df[df["group_ext"] == g][y].dropna().to_numpy() for g in GROUP_ORDER]
    bp = ax.boxplot(data, positions=np.arange(3), widths=BOXPLOT_WIDTH, patch_artist=True, showfliers=False)
    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor((*mcolors.to_rgb(PALETTE[group]), 0.30))
        patch.set_edgecolor(PALETTE[group])
        patch.set_linewidth(0.55)
    for key in ["whiskers", "caps"]:
        for artist in bp[key]:
            artist.set_color(AXIS)
            artist.set_linewidth(0.5)
    for median, group in zip(bp["medians"], GROUP_ORDER):
        median.set_color(PALETTE[group])
        median.set_linewidth(1.2)
    for i, group in enumerate(GROUP_ORDER):
        vals = df[df["group_ext"] == group][y].dropna().to_numpy()
        ax.scatter(rng.normal(i, 0.050, len(vals)), vals, s=3.8, color=PALETTE[group], alpha=0.62, linewidth=0, zorder=3)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Control", "ELS", "ELS\nresilient"], fontsize=4.9)
    ax.set_ylabel(ylabel, fontsize=4.6, labelpad=0.8)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    style_axis(ax, labelsize=4.3)
    ax.tick_params(axis="both", which="major", colors=AXIS, width=0.45, length=2.0)
    ax.tick_params(axis="x", pad=1.0)
    ax.yaxis.set_tick_params(left=True)
    if stars:
        for n, pair in enumerate(stars):
            x1, x2 = split_touching_bracket(pair, stars)
            sig_bracket(ax, x1, x2, n, len(stars), y_base=sig_y_base)
    tag(ax, letter, x=-0.22, y=1.08)


def plot_cumulative(ax: plt.Axes, usage: pd.DataFrame, letter: str) -> None:
    for group in GROUP_ORDER:
        sub = usage[usage["group_ext"] == group]
        cols = [col for col in DISPLAY_ORDER if col in sub.columns]
        row = sub[cols].mean().sort_values(ascending=False).cumsum()
        x = np.arange(len(row))
        ax.plot(x, row, color=PALETTE[group], marker="o", markersize=2.0, linewidth=0.75, label=group)
        ax.errorbar(x, row, yerr=sub[cols].reindex(columns=row.index).cumsum(axis=1).sem(), fmt="none", ecolor=PALETTE[group], elinewidth=0.45, capsize=1.3)
    ax.set_ylim(0.4, 0.92)
    ax.set_yticks(np.arange(0.4, 0.91, 0.1))
    ax.set_xticks(np.arange(len(DISPLAY_ORDER)))
    ax.set_xticklabels([DISPLAY_LABELS.get(c, c) for c in row.index], fontsize=4.7)
    ax.set_ylabel("Cumulative Usage", fontsize=6.0, labelpad=0.5)
    handles = [
        plt.Line2D([0], [0], color=PALETTE[group], marker="o", markersize=2.0, linewidth=0.75, label=group)
        for group in GROUP_ORDER
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=4.4, ncol=3, handlelength=1.0, columnspacing=0.6)
    style_axis(ax, labelsize=4.9)
    ax.yaxis.set_label_coords(K_TAG_X, 0.5)
    ax.spines[["top", "right"]].set_visible(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(0.42)
    tag(ax, letter, x=-0.09, y=1.08)


def plot_group_cumulative(ax: plt.Axes, usage: pd.DataFrame, group: str) -> None:
    sub = usage[usage["group_ext"] == group]
    cols = [col for col in DISPLAY_ORDER if col in sub.columns]
    row = sub[cols].mean().sort_values(ascending=False).cumsum()
    x = np.arange(len(row))
    ax.plot(x, row, color=PALETTE[group], marker="o", markersize=1.8, linewidth=0.65)
    ax.fill_between(x, 0, row, color=PALETTE[group], alpha=0.18)
    ax.set_title(group, fontsize=5.2, pad=2)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS.get(s, s)[:3] for s in row.index], fontsize=4.1)
    ax.set_ylabel("Cumulative Usage", fontsize=4.5)
    style_axis(ax, labelsize=4.2)


def plot_chord(ax: plt.Axes, sequences: dict[str, list[str]], meta: pd.DataFrame, group: str, letter: str) -> None:
    mats = []
    group_map = meta.set_index("Animal")["group_ext"].to_dict()
    for animal, seq in sequences.items():
        if group_map.get(str(animal)) == group:
            mats.append(transition_matrix_flow(seq))
    mat = np.mean(mats, axis=0)
    labels = [DISPLAY_LABELS[c] for c in DISPLAY_ORDER]
    color_map = {DISPLAY_LABELS[c]: COLORS[c] for c in DISPLAY_ORDER}
    plot_chord_diagram(mat, labels, color_map, title=group, ax=ax, r=0.98, gap=0.0, arc_width=0.047, label_fontsize=5.6, title_fontsize=6.4, title_pad=7, limit_pad=1.12, label_radius_factor=1.20, flip_labels=False, edge_lw_scale=6.5, edge_lw_offset=0.16, edge_alpha=0.90, edge_alpha_min=None, delta_angle=0.15)
    tag(ax, letter, x=0.00, y=0.98)


def shift_axes(axes: list[plt.Axes], dx: float = 0.0, dy: float = 0.0, scale_w: float = 1.0, scale_h: float = 1.0) -> None:
    for ax in axes:
        pos = ax.get_position()
        cx = pos.x0 + pos.width / 2 + dx
        cy = pos.y0 + pos.height / 2 + dy
        ax.set_position([cx - pos.width * scale_w / 2, cy - pos.height * scale_h / 2, pos.width * scale_w, pos.height * scale_h])


def set_panel_tag_position(ax: plt.Axes, letter: str, x: float, y: float | None = None) -> None:
    for text in ax.texts:
        if text.get_text() == letter:
            _, old_y = text.get_position()
            text.set_position((x, old_y if y is None else y))
            return


def align_panel_tags_to_titles(fig: plt.Figure, axes_and_letters: list[tuple[plt.Axes, str]]) -> None:
    """Place panel letters at the same vertical height as the axes titles."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    y_offset = 0.006
    for ax, letter in axes_and_letters:
        tag_text = next((text for text in ax.texts if text.get_text() == letter), None)
        if tag_text is None:
            continue
        title_box = ax.title.get_window_extent(renderer=renderer)
        tag_x_display = tag_text.get_transform().transform(tag_text.get_position())[0]
        x_fig = fig_inv.transform((tag_x_display, title_box.y1))[0]
        y_fig = fig_inv.transform((tag_x_display, title_box.y1))[1] + y_offset
        tag_text.set_transform(fig.transFigure)
        tag_text.set_position((x_fig, y_fig))
        tag_text.set_ha("center")
        tag_text.set_va("top")


def set_boxplot_axis_width(fig: plt.Figure, axes: list[plt.Axes], width_in: float) -> None:
    target_width = width_in / fig.get_size_inches()[0]
    for ax in axes:
        pos = ax.get_position()
        center = pos.x0 + pos.width / 2
        ax.set_position([center - target_width / 2, pos.y0, target_width, pos.height])
        x_center = np.mean(ax.get_xlim())
        x_range = BOXPLOT_WIDTH * width_in / (BOXPLOT_WIDTH / 3 * FIGURE4_BOXPLOT_AXIS_WIDTH_IN)
        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)


def widen_metric_row_gap(top_axes: list[plt.Axes], bottom_axes: list[plt.Axes], extra_gap: float = METRIC_ROW_EXTRA_GAP) -> None:
    for ax in top_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + extra_gap / 2, pos.width, pos.height])
    for ax in bottom_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - extra_gap / 2, pos.width, pos.height])


def align_axes_row(axes: list[plt.Axes], left: float, right: float) -> None:
    if not axes:
        return
    width = axes[0].get_position().width
    gap = (right - left - len(axes) * width) / (len(axes) - 1)
    for i, ax in enumerate(axes):
        pos = ax.get_position()
        ax.set_position([left + i * (width + gap), pos.y0, width, pos.height])


def set_cumulative_layout(fig: plt.Figure, main_ax: plt.Axes, side_axes: list[plt.Axes], top_ref_ax: plt.Axes, bottom_ref_ax: plt.Axes) -> None:
    fig_width, _ = fig.get_size_inches()
    main_width = FIGURE4_CUMULATIVE_AXIS_WIDTH_IN / fig_width
    top_pos = top_ref_ax.get_position()
    bottom_pos = bottom_ref_ax.get_position()
    main_y0 = bottom_pos.y0
    main_height = top_pos.y1 - bottom_pos.y0
    main_ax.set_position([FIGURE4_CUMULATIVE_X0, main_y0, main_width, main_height])

    side_width = FIGURE4_CUMULATIVE_SIDE_WIDTH_IN * SIDE_CUMULATIVE_WIDTH_SCALE / fig_width
    side_height = main_height / 3 * 0.74
    side_gap = (main_height - 3 * side_height) / 2
    y0s = [
        main_y0 + 2 * (side_height + side_gap),
        main_y0 + side_height + side_gap,
        main_y0,
    ]
    side_x0 = FIGURE4_CUMULATIVE_SIDE_RIGHT - side_width
    for ax, y0 in zip(side_axes, y0s):
        ax.set_position([side_x0, y0, side_width, side_height])


def equalize_boxplot_heights(axes: list[plt.Axes]) -> None:
    if not axes:
        return
    fig_width, fig_height = axes[0].figure.get_size_inches()
    target_width = FIGURE4_BOXPLOT_AXIS_WIDTH_IN / fig_width
    target_height = FIGURE4_BOXPLOT_AXIS_HEIGHT_IN / fig_height
    for ax in axes:
        pos = ax.get_position()
        y_center = pos.y0 + pos.height / 2
        x_center = pos.x0 + pos.width / 2
        ax.set_position([x_center - target_width / 2, y_center - target_height / 2, target_width, target_height])


def add_section_label(fig: plt.Figure, axes: list[plt.Axes], label: str) -> None:
    positions = [ax.get_position() for ax in axes]
    y0 = min(pos.y0 for pos in positions) - 0.006
    y1 = max(pos.y1 for pos in positions) + 0.006
    label_ax = fig.add_axes([0.001, y0, 0.038, y1 - y0])
    label_ax.set_facecolor("#E0E0E0")
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    for spine in label_ax.spines.values():
        spine.set_visible(False)
    label_ax.text(0.5, 0.5, label, rotation=90, ha="center", va="center", fontsize=7.7, color=AXIS)


def export_source_data(metrics: pd.DataFrame, bouts: pd.DataFrame, transitions: pd.DataFrame, output_dir: Path) -> None:
    """Export Figure 6 source data: resilience-stratified diversity, bouts, and transitions."""
    rows = []

    # Diversity metrics (simpson, shannon, evenness, cui) stratified by resilience group
    for metric_name in ["simpson", "shannon", "evenness", "cui"]:
        if metric_name in metrics.columns:
            for group in ["Control", "ELS", "ELS resilient"]:
                data = metrics[metrics["group_ext"] == group][metric_name]
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

    # Bout durations (per cluster, per resilience group)
    for cluster in DISPLAY_ORDER:
        cluster_bouts = bouts[bouts["cluster"] == cluster]
        if not cluster_bouts.empty:
            for group in ["Control", "ELS", "ELS resilient"]:
                group_bouts = cluster_bouts[cluster_bouts["group_ext"] == group]["bout_duration"]
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

    # Transition metrics (lz, recurrence, determinism, markov) stratified by resilience group
    for metric_name in ["lz", "recurrence", "determinism", "markov"]:
        if metric_name in transitions.columns:
            for group in ["Control", "ELS", "ELS resilient"]:
                data = transitions[transitions["group_ext"] == group][metric_name]
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
        output_csv = output_dir / "source_data_figure6.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")


def main() -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred, pred_sequences, full_sequences, meta = load_sequences()
    metrics, usage = compute_frequency_metrics(full_sequences, meta)
    bouts = bout_table(full_sequences, meta)
    transitions = transition_metrics(full_sequences, meta)
    reps = {group: representative(usage, group) for group in GROUP_ORDER}

    fig = plt.figure(figsize=(8.22, 11.28), dpi=300, facecolor="white")
    gs = fig.add_gridspec(6, 14, left=0.075, right=0.960, top=0.982, bottom=0.045, wspace=0.58, hspace=0.54, height_ratios=[0.54, 0.54, 0.54, 0.94, 0.82, 1.02])

    top = [("A", "B", "Control"), ("C", "D", "ELS"), ("E", "F", "ELS resilient")]
    top_axes: dict[str, list[plt.Axes]] = {}
    for row, (l1, l2, group) in enumerate(top):
        ax1 = fig.add_subplot(gs[row, 0:7])
        ax2 = fig.add_subplot(gs[row, 8:14])
        plot_ethogram(ax1, pred, reps[group], l1, show_legend=True)
        plot_barcode(ax2, pred, reps[group], l2, show_legend=True)
        top_axes[group] = [ax1, ax2]
    e_pos = top_axes["ELS resilient"][0].get_position()
    top_axes["ELS resilient"][0].set_position([e_pos.x0, e_pos.y0, e_pos.width - TOP_ETHOGRAM_SPACING_TRIM, e_pos.height])

    metric_grid = gs[3, 0:4].subgridspec(2, 2, hspace=0.50, wspace=0.58)
    box_axes: list[plt.Axes] = []
    axG = fig.add_subplot(metric_grid[0, 0])
    axH = fig.add_subplot(metric_grid[0, 1])
    axI = fig.add_subplot(metric_grid[1, 0])
    axJ = fig.add_subplot(metric_grid[1, 1])
    box_axes.extend([axG, axH, axI, axJ])
    box_scatter(axG, metrics, "simpson", "Simpson index", "G", (0.58, 0.80), [0.60, 0.65, 0.70, 0.75, 0.80], "%.2f", stars=[(0, 1), (1, 2)])
    box_scatter(axH, metrics, "shannon", "Shannon entropy index", "H", (1.20, 1.70), [1.30, 1.40, 1.50, 1.60, 1.70], "%.1f")
    box_scatter(axI, metrics, "evenness", "Evenness index", "I", (0.58, 0.85), [0.60, 0.65, 0.70, 0.75, 0.80, 0.85], "%.2f")
    # CUI resilient-vs-vulnerable p=0.393 -> not significant; only vuln-vs-control is starred.
    box_scatter(axJ, metrics, "cui", "Cumulative usage index", "J", (-0.25, 0.80), [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8], "%.1f", stars=[(0, 1)])

    axK = fig.add_subplot(gs[3, 4:11])
    plot_cumulative(axK, usage, "K")
    side = gs[3, 11:14].subgridspec(3, 1, hspace=0.38)
    side_axes: list[plt.Axes] = []
    for i, group in enumerate(GROUP_ORDER):
        side_ax = fig.add_subplot(side[i])
        side_axes.append(side_ax)
        plot_group_cumulative(side_ax, usage, group)
    frequency_axes = [axG, axH, axI, axJ, axK] + side_axes

    bout_grid = gs[4, :].subgridspec(1, 8, wspace=0.46)
    bout_axes: list[plt.Axes] = []
    overall = bouts.groupby(["Animal", "group_ext"])["bout_duration"].mean().reset_index()
    cluster_means = bouts[bouts["cluster"].isin(DISPLAY_ORDER)].groupby(["Animal", "group_ext", "cluster"], as_index=False)["bout_duration"].mean()
    bout_specs = [
        ("Overall", overall, (0, 3.0), [0, 1, 2, 3], "%.0f", [(0, 1)]),
        ("Freeze", cluster_means[cluster_means["cluster"] == "Freezing"], (0, 2.5), [0, 0.5, 1, 1.5, 2, 2.5], "%.1f", [(0, 1), (1, 2)]),
        ("Sniff", cluster_means[cluster_means["cluster"] == "Sniffing"], (0, 4.0), [0, 1, 2, 3, 4], "%.0f", [(0, 1)]),
        ("Groom", cluster_means[cluster_means["cluster"] == "Grooming"], (0, 0.6), [0, 0.2, 0.4, 0.6], "%.1f", [(0, 1)]),
        ("Turn", cluster_means[cluster_means["cluster"] == "Turn"], (0, 3.0), [0, 0.5, 1, 1.5, 2, 2.5, 3], "%.1f", [(0, 1), (1, 2)]),
        ("Locomotion", cluster_means[cluster_means["cluster"] == "Locomotion"], (0, 0.8), [0, 0.2, 0.4, 0.6, 0.8], "%.1f", None),
        ("Climb", cluster_means[cluster_means["cluster"] == "Climbing"], (0, 2.0), [0, 0.5, 1, 1.5, 2], "%.1f", None),
        ("Jump", cluster_means[cluster_means["cluster"] == "Jump"], (0, 1.20), [0, 0.4, 0.8, 1.2], "%.1f", None),
    ]
    for i, (title, data, ylim, yticks, yfmt, stars) in enumerate(bout_specs):
        ax = fig.add_subplot(bout_grid[i])
        bout_axes.append(ax)
        box_axes.append(ax)
        box_scatter(ax, data.rename(columns={"bout_duration": "value"}), "value", "Mean Bout Duration (s)", chr(ord("L") + i), ylim, yticks, yfmt, stars=stars)
        ax.set_title(title, fontsize=5.4, color=AXIS, pad=3)

    axT = fig.add_subplot(gs[5, 0:3])
    axU = fig.add_subplot(gs[5, 3:6])
    axV = fig.add_subplot(gs[5, 6:9])
    for ax, group, letter in [(axT, "Control", "T"), (axU, "ELS", "U"), (axV, "ELS resilient", "V")]:
        plot_chord(ax, full_sequences, meta, group, letter)
    shift_axes([axT], dx=-0.012, scale_w=1.04, scale_h=1.04)
    shift_axes([axU], scale_w=1.04, scale_h=1.04)
    shift_axes([axV], dx=0.012, scale_w=1.04, scale_h=1.04)

    trans_grid = gs[5, 10:14].subgridspec(2, 2, hspace=0.72, wspace=0.64)
    axW = fig.add_subplot(trans_grid[0, 0])
    axX = fig.add_subplot(trans_grid[0, 1])
    axY = fig.add_subplot(trans_grid[1, 0])
    axZ = fig.add_subplot(trans_grid[1, 1])
    box_axes.extend([axW, axX, axY, axZ])
    box_scatter(axW, transitions, "lz", "Lempel-Ziv complexity", "W", (100, 260), list(range(100, 261, 40)), "%.0f")
    box_scatter(axX, transitions, "recurrence", "Recurrence Rate", "X", (0.24, 0.450), [0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450], "%.3f", stars=[(0, 1), (1, 2)])
    box_scatter(axY, transitions, "determinism", "Determinism", "Y", (0.70, 1.00), [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], "%.2f", stars=[(0, 1)])
    box_scatter(axZ, transitions, "markov", "Markov Entropy", "Z", (0.7, 1.5), np.arange(0.7, 1.51, 0.2).round(1).tolist(), "%.1f", stars=[(0, 1)], sig_y_base=0.930)

    equalize_boxplot_heights(box_axes)
    set_boxplot_axis_width(fig, [axG, axH, axI, axJ], FREQUENCY_METRIC_AXIS_WIDTH_IN)
    set_boxplot_axis_width(fig, bout_axes, BOUT_METRIC_AXIS_WIDTH_IN)
    shift_axes(bout_axes, dy=BOUT_BLOCK_DY)
    shift_axes([axW, axX], dy=0.010)
    shift_axes([axY, axZ], dy=-0.010)
    shift_axes([axG, axH, axI, axJ], dy=METRIC_BLOCK_DY)
    widen_metric_row_gap([axG, axH], [axI, axJ])
    aligned_left = top_axes["Control"][0].get_position().x0
    for ax, letter in [(axG, "G"), (axI, "I")]:
        pos = ax.get_position()
        ax.set_position([aligned_left, pos.y0, pos.width, pos.height])
    align_axes_row(bout_axes, left=aligned_left, right=0.9550)
    for ax, letter in [(axG, "G"), (axH, "H"), (axI, "I"), (axJ, "J")]:
        set_panel_tag_position(ax, letter, FREQUENCY_TAG_X)
    for ax in [axG, axH, axI, axJ]:
        ax.yaxis.set_label_coords(FREQUENCY_YLABEL_X, 0.5)
    for ax, letter in zip(bout_axes, [chr(ord("L") + i) for i in range(len(bout_axes))]):
        ax.yaxis.set_label_coords(BOUT_YLABEL_X, 0.5)
        set_panel_tag_position(ax, letter, BOUT_TAG_X, BOUT_TAG_Y)
    for ax, letter in [(axW, "W"), (axX, "X"), (axY, "Y"), (axZ, "Z")]:
        ax.yaxis.set_label_coords(TRANSITION_YLABEL_X, 0.5)
        set_panel_tag_position(ax, letter, TRANSITION_TAG_X, TRANSITION_TAG_Y)
    set_cumulative_layout(fig, axK, side_axes, axH, axJ)
    set_panel_tag_position(axK, "K", K_TAG_X, K_TAG_Y)
    align_panel_tags_to_titles(fig, list(zip(bout_axes, [chr(ord("L") + i) for i in range(len(bout_axes))])))
    align_panel_tags_to_titles(fig, [(axW, "W"), (axX, "X"), (axY, "Y"), (axZ, "Z")])

    pdf = FIGURE_OUTPUT_DIR / "figure_6_resilience_diversity.pdf"
    svg = FIGURE_OUTPUT_DIR / "figure_6_resilience_diversity.svg"
    png = FIGURE_OUTPUT_DIR / "figure_6_resilience_diversity.png"
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEGACY_FIGURES_DIR / "figure6.pdf")
    fig.savefig(LEGACY_FIGURES_DIR / "figure6.svg")
    fig.savefig(LEGACY_FIGURES_DIR / "figure6.png", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {svg}")
    print(f"Saved {png}")
    print(f"Saved {LEGACY_FIGURES_DIR / 'figure6.pdf'}")
    print("Representatives: " + "; ".join(f"{g}: {a}" for g, a in reps.items()))

    # Export source data
    print("Computing Figure 6 source statistics...")
    export_source_data(metrics, bouts, transitions, SOURCE_OUTPUT_DIR)


if __name__ == "__main__":
    main()
