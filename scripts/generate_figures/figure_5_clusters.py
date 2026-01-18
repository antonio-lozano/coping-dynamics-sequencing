"""
Figure 5 — Behavioral cluster frequencies, trajectories, and syllable clustering.

Uses the clustered results (`new_results_clusters.pkl`) to compute total usage
and time-course (30 s bins) per behavioral cluster for Control vs ELS.

Includes:
- Panel A: Total time per behavior (bar chart with error bars)
- Panels B–H: Time-course per behavior (mean ± SEM)
- Panel I: Syllable distance heatmap (cosine similarity)
- Panel J: Hierarchical clustering dendrogram
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

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
# Build syllable usage profiles for clustering (from reference code)
print("Building syllable usage profiles for hierarchical clustering...")
syllable_ixs = sorted(behavior_mapping.keys())

# Build profiles: for each syllable, compute usage percentage per recording
syllable_profiles = {code: [] for code in syllable_ixs}
recording_names = []

for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, syllable_ixs)]
    if len(seq) == 0:
        continue
    recording_names.append(rec)
    total_frames = len(seq)
    for code in syllable_ixs:
        pct = np.sum(seq == code) / total_frames
        syllable_profiles[code].append(pct)

# Stack into matrix: rows = syllables, columns = recordings
profile_matrix = np.array([syllable_profiles[code] for code in syllable_ixs])
print(f"Profile matrix shape: {profile_matrix.shape} (syllables x recordings)")

# Compute pairwise distances between syllables using cosine distance
metric = 'cosine'
dists = pdist(profile_matrix, metric=metric)
distance_matrix = squareform(dists)

# Hierarchical clustering using complete linkage
Z = linkage(dists, method='complete')

# Find best threshold using silhouette score
candidate_thresholds = np.linspace(0.01, 0.3, 30)
best_threshold = None
best_score = -1
best_labels = None

for t in candidate_thresholds:
    cluster_labels = fcluster(Z, t=t, criterion='distance')
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters < 2 or n_clusters >= len(syllable_ixs):
        continue
    score = silhouette_score(profile_matrix, cluster_labels, metric=metric)
    if score > best_score:
        best_score = score
        best_threshold = t
        best_labels = cluster_labels

if best_threshold is not None:
    print(f"Best clustering threshold: {best_threshold:.3f} with silhouette score: {best_score:.3f}")
else:
    best_threshold = 0.1  # fallback
    print("No valid clustering found, using default threshold")

cluster_labels = fcluster(Z, t=best_threshold, criterion='distance')
print(f"Number of clusters: {len(np.unique(cluster_labels))}")

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
# Wide format to accommodate bar chart + time courses + clustering panels
fig = plt.figure(figsize=(18, 20), dpi=150, facecolor="#FFFFFF")

# Use GridSpec with explicit margins for clean separation
gs = fig.add_gridspec(
    nrows=4,
    ncols=4,
    height_ratios=[1.0, 1.0, 1.0, 1.2],
    width_ratios=[1.0, 1.0, 1.0, 1.0],
    hspace=0.45,
    wspace=0.40,
    left=0.06,
    right=0.94,
    top=0.96,
    bottom=0.05,
)

# Row 0: Panel A (bar chart) spans 3 columns, Panel B (Freezing) in column 3
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3])

# Row 1: Panels C, D, E, F (4 time-courses: Sniffing, Grooming, Turn, Locomotion)
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[1, 2])
axF = fig.add_subplot(gs[1, 3])

# Row 2: Panels G, H (remaining time-courses), I (heatmap - 2 cols)
axG = fig.add_subplot(gs[2, 0])
axH = fig.add_subplot(gs[2, 1])
axI = fig.add_subplot(gs[2, 2:4])

# Row 3: Panel J (dendrogram - full width)
axJ = fig.add_subplot(gs[3, :])

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
# New mapping with 4-column layout:
# Row 0: A (bar chart), B (Freezing)
# Row 1: C (Sniffing), D (Grooming), E (Turn), F (Locomotion)
# Row 2: G (Climbing), H (Jump), I (heatmap spanning 2 cols)
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

def _plot_time(ax, beh, ylim, yticks, legend_loc="upper right", star=False):
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
    # Auto y-limits with padding
    if not data.empty:
        y_max = (data["mean"] + data["sem"]).max()
        y_max = max(y_max * 1.15, y_max + 1e-6)
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.legend(frameon=False, loc=legend_loc, fontsize=7)
    if star:
        ax.text(4.0, ylim[1] * 0.95, "*", ha="center", va="top", color="#4b4b4b", fontsize=10)
    _style_axes(ax)

_plot_time(axB, "Freezing", (0, 60), [0, 10, 20, 30, 40, 50, 60], legend_loc="lower right", star=True)
_plot_time(axC, "Sniffing", (0, 20), np.arange(0, 20.1, 2.5), legend_loc="upper right", star=False)
_plot_time(axD, "Grooming", (0, 0.40), np.round(np.arange(0, 0.401, 0.05), 2), legend_loc="upper right", star=True)
_plot_time(axE, "Turn", (0, 60), [0, 10, 20, 30, 40, 50, 60], legend_loc="lower right", star=True)
_plot_time(axF, "Locomotion", (0, 12), [0, 2, 4, 6, 8, 10, 12], legend_loc="upper right", star=True)
_plot_time(axG, "Climbing", (0, 12), [0, 2, 4, 6, 8, 10, 12], legend_loc="upper right", star=False)
_plot_time(axH, "Jump", (0, 4.0), np.arange(0, 4.1, 0.5), legend_loc="upper right", star=False)

for ax, tag in zip([axB, axC, axD, axE, axF, axG, axH], list("BCDEFGH")):
    _panel_tag(ax, tag)

# =============================================================================
# Panel I: Syllable distance heatmap (cosine similarity)
# =============================================================================
print("  Panel I: Syllable distance heatmap...")
syllable_names = [behavior_mapping[code] for code in syllable_ixs]

# Plot distance matrix as heatmap with square aspect
im = axI.imshow(distance_matrix, cmap='RdYlBu_r', aspect='equal')
axI.set_xticks(np.arange(len(syllable_names)))
axI.set_yticks(np.arange(len(syllable_names)))
axI.set_xticklabels(syllable_names, rotation=45, ha='right', fontsize=9)
axI.set_yticklabels(syllable_names, fontsize=9)
axI.set_title("Syllable Distance Matrix (Cosine)", fontsize=10, color="#333333", pad=8)

# Add colorbar
cbar = fig.colorbar(im, ax=axI, shrink=0.7, pad=0.02, aspect=15)
cbar.set_label("Cosine Distance", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Add distance values as text
for i in range(len(syllable_names)):
    for j in range(len(syllable_names)):
        axI.text(j, i, f'{distance_matrix[i, j]:.2f}',
                ha="center", va="center", 
                color="white" if distance_matrix[i, j] > 0.35 else "black",
                fontsize=7, fontweight='normal')

_panel_tag(axI, "I")

# =============================================================================
# Panel J: Dendrogram (full width at bottom)
# =============================================================================
print("  Panel J: Hierarchical clustering dendrogram...")

# Plot dendrogram with horizontal orientation for better readability
dendro = dendrogram(Z, labels=syllable_names, ax=axJ, leaf_rotation=0, 
                    leaf_font_size=10, color_threshold=best_threshold,
                    orientation='top')
axJ.set_title("Syllable Hierarchical Clustering", fontsize=11, color="#333333", pad=10)
axJ.set_ylabel("Cosine Distance", fontsize=9)
axJ.set_xlabel("Behavior", fontsize=9)
axJ.axhline(y=best_threshold, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.8)
axJ.text(0.02, best_threshold + 0.008, f'optimal threshold = {best_threshold:.3f}', 
         transform=axJ.get_yaxis_transform(), fontsize=9, color='#E63946', ha='left',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

_style_axes(axJ)
_panel_tag(axJ, "J")

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
