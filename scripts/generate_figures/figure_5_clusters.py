"""
Figure 5 — Behavioral cluster frequencies and trajectories.

Uses the clustered results (`new_results_clusters.pkl`) to compute total usage
and time-course (30 s bins) per behavioral cluster for Control vs ELS.

Includes:
- Panel A: Total time per behavior (bar chart with error bars)
- Panels B–H: Time-course per behavior (mean ± SEM)
"""

#%%
# Imports and paths
print("Loading dependencies and setting paths...")
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `src` imports work from any CWD
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS, RESULTS_DIR, PALETTE

results_pkl = RESULTS_CLUSTERS_PKL
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS
bin_size = fps * bin_seconds

behavior_mapping = {
    1: "Freezing",
    2: "Sniffing",
    3: "Grooming",
    4: "Turn",
    5: "Locomotion",
    6: "Climbing",
    7: "Jump",
}

#%%
# Load clustered results and group labels
print("Loading clustered MoSeq results and group assignments...")
with open(results_pkl, "rb") as f:
    results_dict = pickle.load(f)

index_df = pd.read_csv(index_csv)

def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")

group_map = {}
for _, row in index_df.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
    group_map[raw] = row["group"]
    group_map[norm] = row["group"]
    group_map[prefix] = row["group"]

print(f"Loaded {len(results_dict)} recordings with cluster labels.")

#%%
# Build per-frame dataframe with group and time bins
print("Building per-frame dataset...")
records = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    syll = np.array(data["syllable"])
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    n = len(syll)
    frames = np.arange(n)
    bins = (frames // bin_size).astype(int)
    records.append(
        pd.DataFrame(
            {"recording": rec, "group": group, "cluster": syll, "bin": bins, "frame": frames}
        )
    )

frame_df = pd.concat(records, ignore_index=True)
frame_df = frame_df[frame_df["cluster"].isin(behavior_mapping.keys())]
frame_df["behavior"] = frame_df["cluster"].map(behavior_mapping)
print(f"Frame rows after filtering: {len(frame_df):,}")

#%%
# Panel A: total time per behavior
print("Computing total time per behavior and group...")
total = (
    frame_df.groupby(["group", "behavior"])
    .size()
    .reset_index(name="frames")
)
total["seconds"] = total["frames"] / fps

#%%
# Panel B-H: time course per behavior (mean ± SEM per bin)
print("Computing time-course (mean ± SEM) per behavior...")
per_bin = (
    frame_df.groupby(["group", "recording", "behavior", "bin"])
    .size()
    .reset_index(name="frames")
)
per_bin["percent"] = per_bin.groupby(["recording", "bin"])["frames"].transform(
    lambda x: 100 * x / x.sum()
)

summary = (
    per_bin.groupby(["group", "behavior", "bin"])["percent"]
    .agg(["mean", "sem"])
    .reset_index()
)
summary["time_sec"] = (summary["bin"] + 0.5) * bin_seconds
summary["time_min"] = summary["time_sec"] / 60.0

#%%
print("Plotting Figure 5...")
sns.set_theme(style="ticks", context="paper", font_scale=1.0)

palette = PALETTE

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)

def _panel_tag(ax, tag):
    ax.text(-0.12, 1.08, tag, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#333333")

# Create figure with proper landscape proportions for 4x3 layout
# Wide format to accommodate bar chart + time courses
fig = plt.figure(figsize=(15, 14), dpi=150, facecolor="#FFFFFF")

# Use GridSpec with 3 rows x 3 columns
gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    height_ratios=[1.0, 1.0, 1.0],
    width_ratios=[1.0, 1.0, 1.0],
    hspace=0.40,
    wspace=0.35,
    left=0.08,
    right=0.95,
    top=0.95,
    bottom=0.06,
)

# Row 0: Panel A (bar chart) spans 2 columns, Panel B (Freezing) in column 2
axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[0, 2])

# Row 1: Panels C, D, E (3 time-courses: Sniffing, Grooming, Turn)
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[1, 2])

# Row 2: Panels F, G, H (remaining time-courses: Locomotion, Climbing, Jump)
axF = fig.add_subplot(gs[2, 0])
axG = fig.add_subplot(gs[2, 1])
axH = fig.add_subplot(gs[2, 2])

# =============================================================================
# Panel A: total usage barplot with error bars
# =============================================================================
order = list(behavior_mapping.values())
total_plot = total[total["group"].isin(["Control", "ELS"])].copy()
stats = (
    frame_df.groupby(["group", "recording", "behavior"]).size().reset_index(name="frames")
)
stats["seconds"] = stats["frames"] / fps
stats_summary = stats.groupby(["group", "behavior"])["seconds"].agg(["mean", "sem"]).reset_index()

x = np.arange(len(order))
w = 0.35
ctrl = stats_summary[stats_summary["group"] == "Control"].set_index("behavior").reindex(order)
els = stats_summary[stats_summary["group"] == "ELS"].set_index("behavior").reindex(order)
axA.bar(x - w / 2, ctrl["mean"], yerr=ctrl["sem"], width=w, color=palette["Control"], edgecolor="#FFFFFF",
        linewidth=0.4, capsize=3, label="Control")
axA.bar(x + w / 2, els["mean"], yerr=els["sem"], width=w, color=palette["ELS"], edgecolor="#FFFFFF",
        linewidth=0.4, capsize=3, label="ELS")
axA.set_ylabel("Frequency (s)", fontsize=10, labelpad=6)
axA.set_xticks(x)
axA.set_xticklabels(order, rotation=0, fontsize=9)
axA.set_ylim(0, 300)
axA.set_yticks([0, 50, 100, 150, 200, 250, 300])
axA.legend(frameon=False, loc="upper right", fontsize=9)
axA.set_title("Total Behavior Frequency", fontsize=11, color="#333333", pad=8)

def _sig_bracket(ax, x_center, y, h=8, text="*"):
    ax.plot([x_center - w / 2, x_center - w / 2, x_center + w / 2, x_center + w / 2],
            [y, y + h, y + h, y], color="#4b4b4b", linewidth=0.8)
    ax.text(x_center, y + h + 4, text, ha="center", va="bottom", color="#4b4b4b", fontsize=10)

for beh in ["Freezing", "Sniffing", "Turn"]:
    idx = order.index(beh)
    y_max = max(ctrl.loc[beh, "mean"] + ctrl.loc[beh, "sem"], els.loc[beh, "mean"] + els.loc[beh, "sem"])
    _sig_bracket(axA, idx, y_max + 5)

_style_axes(axA)
_panel_tag(axA, "A")

# =============================================================================
# Time-series panels B–H
# =============================================================================
# Layout: 3x3 grid
# Row 0: A (bar chart, 2 cols), B (Freezing)
# Row 1: C (Sniffing), D (Grooming), E (Turn)
# Row 2: F (Locomotion), G (Climbing), H (Jump)
time_axes = {
    "Freezing": axB,
    "Sniffing": axC,
    "Grooming": axD,
    "Turn": axE,
    "Locomotion": axF,
    "Climbing": axG,
    "Jump": axH,
}

shade_starts = [3.0, 4.5, 6.0]
shade_width = 0.5

def _nice_ceil(val):
    """Round up to a nice number for axis limits."""
    if val <= 0:
        return 1
    magnitude = 10 ** np.floor(np.log10(val))
    normalized = val / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude

def _plot_time(ax, beh, legend_loc="upper right", star=False):
    data = summary[summary["behavior"] == beh]
    for start in shade_starts:
        ax.axvspan(start, start + shade_width, color="#F0F0F0", zorder=0)
    for group, gdata in data.groupby("group"):
        if group not in ("Control", "ELS"):
            continue
        ax.plot(
            gdata["time_min"],
            gdata["mean"],
            label=group,
            color=palette.get(group, "#4d4d4d"),
            linewidth=1.8,
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            gdata["time_min"],
            gdata["mean"] - gdata["sem"],
            gdata["mean"] + gdata["sem"],
            color=palette.get(group, "#4d4d4d"),
            alpha=0.2,
            linewidth=0,
        )
    ax.set_title(beh, fontsize=10, color="#333333", pad=6)
    ax.set_xlabel("Time (min)", fontsize=8, labelpad=3)
    ax.set_ylabel("% time", fontsize=8, labelpad=3)
    ax.set_xlim(1, 7)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    
    # Dynamic y-limits based on data
    if not data.empty:
        y_max_data = (data["mean"] + data["sem"]).max()
        y_max = _nice_ceil(y_max_data * 1.1)  # Add 10% padding and round up
    else:
        y_max = 10  # fallback
    
    ax.set_ylim(0, y_max)
    # Generate nice tick marks
    if y_max <= 1:
        step = 0.2
    elif y_max <= 5:
        step = 1
    elif y_max <= 20:
        step = 5
    elif y_max <= 50:
        step = 10
    else:
        step = 20
    yticks = np.arange(0, y_max + step/2, step)
    ax.set_yticks(yticks)
    
    ax.legend(frameon=False, loc=legend_loc, fontsize=7)
    if star:
        ax.text(4.0, y_max * 0.95, "*", ha="center", va="top", color="#4b4b4b", fontsize=10)
    _style_axes(ax)

_plot_time(axB, "Freezing", legend_loc="lower right", star=True)
_plot_time(axC, "Sniffing", legend_loc="upper right", star=False)
_plot_time(axD, "Grooming", legend_loc="upper right", star=True)
_plot_time(axE, "Turn", legend_loc="lower right", star=True)
_plot_time(axF, "Locomotion", legend_loc="upper right", star=True)
_plot_time(axG, "Climbing", legend_loc="upper right", star=False)
_plot_time(axH, "Jump", legend_loc="upper right", star=False)

for ax, tag in zip([axB, axC, axD, axE, axF, axG, axH], list("BCDEFGH")):
    _panel_tag(ax, tag)

# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_5_clusters.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_5_clusters.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 5 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 5 ready.")

# %%
