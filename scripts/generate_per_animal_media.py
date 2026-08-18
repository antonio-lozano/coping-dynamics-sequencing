# SPDX-License-Identifier: MIT
"""Per-animal ethogram and behaviour barplot, one folder per mouse.

The manuscript shows one representative animal per condition. Reviewers and
readers who want to judge how representative that choice was need the whole
set, so this writes every animal's ethogram and cluster-frequency barplot to
supplementary_media/per_animal/<animal>/ together with the underlying counts.

    python scripts/generate_per_animal_media.py
    python scripts/generate_per_animal_media.py --animals 2.4 11.4
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "supplementary_media" / "per_animal"

# Ordered least to most active, matching Figure 4.
CLUSTERS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump"]
COLORS = {
    "Freeze": "#C37BA0", "Sniff": "#5B9AAA", "Groom": "#8EC6DE", "Turn": "#B7DB45",
    "Locomotion": "#F1C232", "Climb": "#F4A259", "Jump": "#E45756",
    "Unassigned": "#DCDCDC",
}
AXIS = "#4D4D4D"
BIN_SECONDS = 0.25
EVENT_STARTS = [3.0, 4.5, 6.0]          # tone onsets, minutes
EVENT_DURATION = 0.5


def load_figure4():
    spec = importlib.util.spec_from_file_location(
        "fig4", ROOT / "scripts" / "generate_figures" / "figure_4_diversity_dynamics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(label: str) -> str:
    """Map the sequence labels onto the seven manuscript clusters."""
    if not label:
        return "Unassigned"
    lookup = {"Freezing": "Freeze", "Sniffing": "Sniff", "Grooming": "Groom",
              "Climbing": "Climb"}
    return lookup.get(label, label if label in CLUSTERS else "Unassigned")


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelsize=8, width=0.8, length=3)
    ax.xaxis.label.set_color(AXIS)
    ax.yaxis.label.set_color(AXIS)
    ax.title.set_color(AXIS)


def shade_events(ax: plt.Axes) -> None:
    for start in EVENT_STARTS:
        ax.axvspan(start, start + EVENT_DURATION, color="#F0F0F0", zorder=0)


def draw_ethogram(ax: plt.Axes, seq: list[str], title: str) -> None:
    order = CLUSTERS + ["Unassigned"]
    index = {name: i for i, name in enumerate(order)}
    codes = np.array([index[canonical(s)] for s in seq])
    minutes = np.arange(len(codes)) * BIN_SECONDS / 60.0

    shade_events(ax)
    for name in order:
        hit = codes == index[name]
        if not hit.any():
            continue
        ax.scatter(minutes[hit], np.full(hit.sum(), index[name]),
                   s=1.6, marker="s", color=COLORS[name], linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, minutes.max() if len(minutes) else 1)
    ax.set_xlabel("Time (minutes)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    style(ax)


def draw_barcode(ax: plt.Axes, seq: list[str]) -> None:
    order = CLUSTERS + ["Unassigned"]
    index = {name: i for i, name in enumerate(order)}
    codes = np.array([index[canonical(s)] for s in seq])
    cmap = ListedColormap([COLORS[name] for name in order])
    ax.imshow(codes[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(order) - 1,
              interpolation="nearest",
              extent=(0, len(codes) * BIN_SECONDS / 60.0, 0, 1))
    ax.set_yticks([])
    ax.set_xlabel("Time (minutes)", fontsize=9)
    ax.set_title("Behavioural barcode", fontsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelsize=8)


def draw_barplot(ax: plt.Axes, seq: list[str]) -> pd.DataFrame:
    labels = [canonical(s) for s in seq]
    counts = pd.Series(labels).value_counts()
    seconds = {name: counts.get(name, 0) * BIN_SECONDS for name in CLUSTERS}
    total = sum(seconds.values()) or 1.0
    ax.bar(range(len(CLUSTERS)), [seconds[c] for c in CLUSTERS],
           color=[COLORS[c] for c in CLUSTERS], edgecolor="white", linewidth=0.6)
    for i, name in enumerate(CLUSTERS):
        ax.text(i, seconds[name], f"{seconds[name]/total*100:.0f}%", ha="center",
                va="bottom", fontsize=7.5, color=AXIS)
    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels(CLUSTERS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Time in cluster (s)", fontsize=9)
    ax.set_title("Total time per behaviour", fontsize=10, pad=6)
    style(ax)
    return pd.DataFrame({"cluster": CLUSTERS,
                         "seconds": [seconds[c] for c in CLUSTERS],
                         "percent": [seconds[c] / total * 100 for c in CLUSTERS]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animals", nargs="*", default=None,
                        help="animal ids to render; default is all")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig4 = load_figure4()
    _p, _ps, sequences, meta = fig4.load_sequences()
    info = meta.set_index("Animal").to_dict("index")

    resilient = {"2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
                 "123.3", "134.2", "148.4", "150.3", "150.5", "159.4"}
    wanted = args.animals or sorted(sequences, key=lambda a: float(a))
    OUT.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for animal in wanted:
        if animal not in sequences:
            print(f"  skipped {animal}: no sequence")
            continue
        seq = sequences[animal]
        group = info[animal]["group"]
        profile = ("Control" if group == "Control"
                   else "ELS resilient" if animal in resilient else "ELS vulnerable")
        folder = OUT / animal
        folder.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(11, 7.5), dpi=args.dpi, facecolor="white")
        grid = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.32, 1.25],
                                width_ratios=[2.1, 1.0], hspace=0.55, wspace=0.28,
                                left=0.08, right=0.97, top=0.90, bottom=0.09)
        title = f"Animal {animal}  ·  {profile}  ·  experiment {info[animal]['Experiment']}"
        fig.suptitle(title, fontsize=12, color=AXIS, y=0.965)

        draw_ethogram(fig.add_subplot(grid[0, :]), seq, "Ethogram (250 ms resolution)")
        draw_barcode(fig.add_subplot(grid[1, :]), seq)
        table = draw_barplot(fig.add_subplot(grid[2, 0]), seq)

        legend_ax = fig.add_subplot(grid[2, 1])
        legend_ax.axis("off")
        legend_ax.legend(handles=[Patch(facecolor=COLORS[c], label=c)
                                  for c in CLUSTERS + ["Unassigned"]],
                         loc="center", frameon=False, fontsize=9, ncol=1,
                         title="Behaviour", title_fontsize=9.5)

        fig.savefig(folder / f"{animal}_ethogram_and_barplot.png",
                    facecolor="white", bbox_inches="tight")
        fig.savefig(folder / f"{animal}_ethogram_and_barplot.pdf",
                    facecolor="white", bbox_inches="tight")
        plt.close(fig)
        table.insert(0, "animal", animal)
        table.to_csv(folder / f"{animal}_cluster_totals.csv", index=False)

        index_rows.append({"animal": animal, "group": group, "profile": profile,
                           "experiment": info[animal]["Experiment"],
                           "litter": animal.split(".")[0],
                           "folder": f"per_animal/{animal}"})

    pd.DataFrame(index_rows).to_csv(OUT / "per_animal_index.csv", index=False)
    print(f"wrote {len(index_rows)} animal folders under {OUT}")
    print(f"wrote {OUT / 'per_animal_index.csv'}")


if __name__ == "__main__":
    main()
