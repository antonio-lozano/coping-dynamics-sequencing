"""
Figure 7 — Behavioral dynamics score and resilient subgroup identification.

Multi-panel figure:
  Row 1: MDS embedding (A), BFL score boxplot (B), Bootstrap null distribution (C)
  Row 2: Mean transition matrix Control (D), Mean transition matrix ELS (E), 
         Mean stabilized transition matrix ELS (F)
  Row 3: BFL score histogram by group (G), Freezing time-course (H), Bar chart (I)
  Row 4-5: Behavior time-courses (J–O)

Uses transition matrices with consecutive duplicate removal (as in reference code).
"""

#%% Imports and paths
print("Loading dependencies and setting paths...")
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from scipy.interpolate import griddata
from scipy import stats
from sklearn.manifold import MDS

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS, BEHAVIOR_MAPPING, CODES, RESULTS_DIR, PALETTE
from src.plotting import plot_chord_diagram, plot_difference_network, add_difference_colorbar
from src.transition_utils import (
    compute_transition_matrix, 
    compute_stabilized_transition_matrices,
    compute_bfl_scores,
    bootstrap_intergroup_distance,
    compute_mean_transition_matrices_by_group,
    relabel_matrix
)

results_pkl = RESULTS_CLUSTERS_PKL
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS
bin_size = fps * bin_seconds

behavior_mapping = BEHAVIOR_MAPPING
codes = CODES

#%% Load clustered syllables and groups
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

#%% Compute transition matrices for each recording (with consecutive duplicate removal)
print("Computing transition matrices (removing consecutive duplicates)...")
transition_matrices = {}
group_assignments = {}

for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
    if len(seq) == 0:
        continue
    
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    
    if group == "Unknown":
        continue
    
    # Compute transition matrix (removes consecutive duplicates)
    TM = compute_transition_matrix(seq)
    # Reindex to ensure consistent shape
    TM = TM.reindex(index=codes, columns=codes, fill_value=0)
    
    transition_matrices[rec_raw] = TM
    group_assignments[rec_raw] = group

print(f"Computed transition matrices for {len(transition_matrices)} recordings")

#%% Compute stabilized transition matrices (subtract control average)
print("Computing stabilized transition matrices...")
stabilized_matrices = compute_stabilized_transition_matrices(
    transition_matrices, group_assignments, control_value="Control"
)

#%% Compute BFL scores from stabilized matrices
print("Computing BFL (Behavioral Flow Likeness) scores...")
bfl_scores = compute_bfl_scores(stabilized_matrices, group_assignments, "Control", "ELS")

# Create DataFrame with BFL scores
bfl_df = pd.DataFrame([
    {"recording": rec, "group": group_assignments[rec], "bfl_score": score}
    for rec, score in bfl_scores.items()
])

# Identify resilient ELS (negative BFL score = closer to control)
bfl_df["resilient"] = (bfl_df["group"] == "ELS") & (bfl_df["bfl_score"] < 0)
bfl_df["group_extended"] = bfl_df.apply(
    lambda r: "ELS resilient" if r["resilient"] else r["group"], axis=1
)

print(f"Resilient ELS count: {bfl_df['resilient'].sum()} / {len(bfl_df[bfl_df['group']=='ELS'])}")

#%% Bootstrap permutation test for intergroup distance
print("Running bootstrap permutation test...")
bootstrap_stats = bootstrap_intergroup_distance(
    stabilized_matrices, group_assignments, "Control", "ELS", n_bootstraps=1000
)
print(f"  True distance: {bootstrap_stats['true_distance']:.3f}")
print(f"  Bootstrap mean: {bootstrap_stats['bootstrap_mean']:.3f} ± {bootstrap_stats['bootstrap_std']:.3f}")
print(f"  Sigma: {bootstrap_stats['sigma']:.3f}")
print(f"  p-value: {bootstrap_stats['p_value']:.4f}")

#%% Compute mean transition matrices per group
print("Computing mean transition matrices per group...")
mean_trans_mats = compute_mean_transition_matrices_by_group(transition_matrices, group_assignments)
mean_stab_mats = compute_mean_transition_matrices_by_group(stabilized_matrices, group_assignments)

#%% Build fixed-length feature vectors for MDS (original approach for visualization)
print("Building time-binned behavioral profiles for MDS...")
bin_counts = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
    if len(seq) == 0:
        continue
    bins = len(seq) // bin_size
    if bins > 0:
        bin_counts.append(bins)

if not bin_counts:
    raise SystemExit("No recordings with sufficient length for binning.")
common_bins = min(bin_counts)
print(f"Using common bin count across recordings: {common_bins} bins of {bin_seconds}s each.")

profiles = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
    if len(seq) == 0:
        continue
    bins = len(seq) // bin_size
    if bins < common_bins:
        continue
    seq = seq[: common_bins * bin_size]
    arr = seq.reshape(common_bins, bin_size)
    feats = []
    for b in arr:
        counts = np.bincount(b, minlength=max(codes) + 1)
        pct = counts[codes] / float(bin_size)
        feats.append(pct)
    feat_vec = np.concatenate(feats)
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    
    # Get BFL-based group_extended
    bfl_row = bfl_df[bfl_df["recording"] == rec_raw]
    if len(bfl_row) > 0:
        group_ext = bfl_row.iloc[0]["group_extended"]
    else:
        group_ext = group
    
    profiles.append({"recording": rec_raw, "group": group, "group_extended": group_ext, "feature": feat_vec})

profile_df = pd.DataFrame(profiles)
feature_matrix = np.vstack(profile_df["feature"].to_numpy())
print(f"Built profiles for {len(profile_df)} recordings; feature dim {feature_matrix.shape[1]}")

#%% Distance matrix and MDS embedding (Euclidean)
print("Computing pairwise distances and MDS embedding...")
dist = pairwise_distances(feature_matrix, metric="euclidean")
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
coords = mds.fit_transform(dist)
profile_df[["mds1", "mds2"]] = coords

#%% LOOCV accuracy for MDS embedding
print("Computing LOOCV accuracy...")
binary_labels = np.array([0 if g == "Control" else 1 for g in profile_df["group"]])
clf = LogisticRegression()
loocv_acc = np.mean(cross_val_score(clf, coords, binary_labels, cv=LeaveOneOut()))
print(f"LOOCV Accuracy: {loocv_acc*100:.1f}%")

#%% Build time-course data (30-second bins)
print("Building 30-second binned time-course data...")
bin_frames = fps * bin_seconds
time_course_records = []

# Build a lookup for group_extended from bfl_df
resilient_lookup = dict(zip(bfl_df["recording"], bfl_df["group_extended"]))

for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    original_seq = np.array(data["syllable"])
    if len(original_seq) == 0:
        continue
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    group_ext = resilient_lookup.get(rec_raw, group)
    
    # Use original sequence length for proper time mapping
    frames = np.arange(len(original_seq))
    bins = frames // bin_frames
    
    for bin_idx in range(bins.max() + 1):
        mask = bins == bin_idx
        if not mask.any():
            continue
        chunk = original_seq[mask]
        # Only count frames with valid codes for percentage
        valid_chunk = chunk[np.isin(chunk, codes)]
        time_min = (bin_idx + 0.5) * bin_seconds / 60.0
        for code in codes:
            if len(valid_chunk) > 0:
                pct = 100.0 * np.sum(valid_chunk == code) / len(valid_chunk)
            else:
                pct = 0.0
            time_course_records.append({
                "recording": rec_raw,
                "group": group,
                "group_extended": group_ext,
                "time_min": time_min,
                "behavior": behavior_mapping[code],
                "pct": pct
            })

time_course_df = pd.DataFrame(time_course_records)
print(f"Time-course data: {len(time_course_df)} rows")

#%% Compute frequency (total seconds) per behavior per group
print("Computing frequency data for bar chart...")
freq_records = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
    if len(seq) == 0:
        continue
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    group_ext = resilient_lookup.get(rec_raw, group)
    
    for code in codes:
        freq_seconds = np.sum(seq == code) / fps
        freq_records.append({
            "recording": rec_raw,
            "group": group,
            "group_extended": group_ext,
            "behavior": behavior_mapping[code],
            "frequency": freq_seconds
        })

freq_df = pd.DataFrame(freq_records)
print(f"Frequency data: {len(freq_df)} rows")

# =============================================================================
# FIGURE 7 — Multi-panel layout
# =============================================================================
print("Rendering Figure 7...")
sns.set_theme(style="ticks", context="paper", font_scale=1.0)

palette = PALETTE

# Custom colormap for MDS background
cmap_dynamics = mcolors.LinearSegmentedColormap.from_list(
    "dynamics", ["#C37B9F", "#FFFFFF", "#F8C650"]
)

# Epoch windows for time-course shading (in minutes)
epoch_windows = [(3.0, 3.5), (4.5, 5.0), (6.0, 6.5)]

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)

def _panel_tag(ax, tag):
    ax.text(-0.10, 1.05, tag, transform=ax.transAxes, fontsize=11, 
            fontweight="bold", color="#4b4b4b", va="top", ha="right")

def _add_epoch_bands(ax, y_range=None):
    if y_range is None:
        y_range = ax.get_ylim()
    for start, end in epoch_windows:
        ax.axvspan(start, end, color="#E8E8E8", zorder=0, alpha=0.7)

def _add_significance_star(ax, x_pos, y_pos):
    ax.text(x_pos, y_pos, "*", ha="center", va="bottom", fontsize=12, 
            fontweight="bold", color="#333333")

def _add_bracket_star(ax, x1, x2, y, star="*"):
    h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#4b4b4b", lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.5, star, ha="center", va="bottom", 
            fontsize=10, color="#4b4b4b")

# -----------------------------------------------------------------------------
# Create figure with custom layout (5 rows)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 20), dpi=150, facecolor="#FFFFFF")

gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 0.9, 0.8, 0.8], 
                      width_ratios=[1, 1, 1],
                      hspace=0.50, wspace=0.40,
                      left=0.07, right=0.95, top=0.96, bottom=0.04)

# Row 0: MDS (A), BFL boxplot (B), Bootstrap histogram (C)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# Row 1: Transition matrices - Control (D), ELS (E), Stabilized ELS (F)
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])

# Row 2: BFL histogram (G), Freezing time-course (H), Bar chart (I)
axG = fig.add_subplot(gs[2, 0])
axH = fig.add_subplot(gs[2, 1])
axI = fig.add_subplot(gs[2, 2])

# Rows 3-4: Behavior time-courses (J-O)
axJ = fig.add_subplot(gs[3, 0])
axK = fig.add_subplot(gs[3, 1])
axL = fig.add_subplot(gs[3, 2])
axM = fig.add_subplot(gs[4, 0])
axN = fig.add_subplot(gs[4, 1])
axO = fig.add_subplot(gs[4, 2])

# =============================================================================
# Panel A — MDS Plot
# =============================================================================
print("  Panel A: MDS embedding...")

# Compute dynamics score for coloring
median_ctrl = profile_df[profile_df["group"] == "Control"][["mds1", "mds2"]].median().to_numpy()
median_els = profile_df[profile_df["group"] == "ELS"][["mds1", "mds2"]].median().to_numpy()
d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
d_els = np.linalg.norm(coords - median_els, axis=1)
eps = 1e-9
dynamics_score = np.log((d_els + eps) / (d_ctrl + eps))

# Background contour
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
margin = 0.12 * max(x_max - x_min, y_max - y_min)
grid_x, grid_y = np.mgrid[x_min - margin : x_max + margin : 100j, 
                          y_min - margin : y_max + margin : 100j]
grid_z = griddata(coords, dynamics_score, (grid_x, grid_y), method="cubic")
grid_z_nn = griddata(coords, dynamics_score, (grid_x, grid_y), method="nearest")
grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)

vmin, vmax = np.nanpercentile(dynamics_score, [5, 95])
vabs = max(abs(vmin), abs(vmax))
cf = axA.contourf(grid_x, grid_y, grid_z, levels=20, cmap=cmap_dynamics, 
                  vmin=-vabs, vmax=vabs, alpha=0.85)

# Scatter points
for grp, color in [("Control", palette["Control"]), ("ELS", palette["ELS"])]:
    mask = profile_df["group"] == grp
    axA.scatter(coords[mask, 0], coords[mask, 1], c=color, s=50, 
                edgecolor="#333333", linewidth=0.5, label=grp, zorder=3, alpha=0.9)

axA.legend(loc="upper right", frameon=False, fontsize=8)
axA.text(0.98, 0.02, f"LOOCV Acc: {loocv_acc*100:.1f}%", transform=axA.transAxes,
         fontsize=7, ha="right", va="bottom", 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888", linewidth=0.5))
axA.set_xlabel("MDS Dimension 1", fontsize=9)
axA.set_ylabel("MDS Dimension 2", fontsize=9)
axA.set_title("MDS Plot (Euclidean)", fontsize=10, color="#4b4b4b")
_style_axes(axA)
_panel_tag(axA, "A")

# =============================================================================
# Panel B — BFL Score Boxplot
# =============================================================================
print("  Panel B: BFL score boxplot...")

box_width = 0.5
sns.boxplot(data=bfl_df, x="group", y="bfl_score", order=["Control", "ELS"],
            ax=axB, width=box_width, fliersize=0,
            boxprops=dict(facecolor="none", edgecolor="#444444", linewidth=1.2),
            medianprops=dict(color="#222222", linewidth=1.5),
            whiskerprops=dict(color="#444444", linewidth=1),
            capprops=dict(color="#444444", linewidth=1))

sns.stripplot(data=bfl_df, x="group", y="bfl_score", order=["Control", "ELS"],
              palette=palette, ax=axB, size=5, alpha=0.7, jitter=0.12, 
              edgecolor="#333333", linewidth=0.3)

axB.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
y_max = bfl_df["bfl_score"].max()
_add_bracket_star(axB, 0, 1, y_max * 1.05, "*")

# Highlight resilient region
els_scores = bfl_df[bfl_df["group"] == "ELS"]["bfl_score"]
if (els_scores < 0).any():
    resilient_y_min = els_scores.min() - 0.1
    rect = FancyBboxPatch((1 - box_width/2, resilient_y_min), box_width, 0 - resilient_y_min + 0.1,
                          boxstyle="round,pad=0.02", fill=False, 
                          edgecolor=palette["ELS resilient"], linestyle=":", linewidth=1.5, alpha=0.8)
    axB.add_patch(rect)
    axB.text(1, resilient_y_min - 0.15, "resilient", ha="center", fontsize=7, 
             color=palette["ELS resilient"], style="italic")

axB.set_xlabel("")
axB.set_ylabel("BFL Score", fontsize=9)
axB.set_title("Behavioral Flow Likeness", fontsize=10, color="#4b4b4b")
_style_axes(axB)
_panel_tag(axB, "B")

# =============================================================================
# Panel C — Bootstrap Null Distribution
# =============================================================================
print("  Panel C: Bootstrap null distribution...")

axC.hist(bootstrap_stats['bootstrap_distances'], bins=30, color='gray', 
         edgecolor='white', alpha=0.7, label='Null distribution')
axC.axvline(bootstrap_stats['true_distance'], color='red', linewidth=2, 
            label=f'True distance')
axC.set_xlabel("Manhattan Distance", fontsize=9)
axC.set_ylabel("Frequency", fontsize=9)
axC.set_title("Bootstrap Permutation Test", fontsize=10, color="#4b4b4b")
axC.legend(loc="upper right", frameon=False, fontsize=7)

# Add stats text
stats_text = f"σ = {bootstrap_stats['sigma']:.2f}\np = {bootstrap_stats['p_value']:.4f}"
axC.text(0.95, 0.95, stats_text, transform=axC.transAxes, fontsize=8, 
         ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                          edgecolor="#888888", linewidth=0.5))
_style_axes(axC)
_panel_tag(axC, "C")

# =============================================================================
# Panel D — Chord Diagram: Control
# =============================================================================
print("  Panel D: Chord diagram (Control)...")

tm_ctrl = mean_trans_mats["Control"]
behavior_names = [behavior_mapping[c] for c in codes]

# Behavior colors for chord diagram
behavior_colors = {
    "Freezing": "#C37B9F",
    "Sniffing": "#4CB7A5",
    "Grooming": "#7EC8E3",
    "Turn": "#8DC63F",
    "Locomotion": "#F8C650",
    "Climbing": "#F4A259",
    "Jump": "#E4572E",
}

plot_chord_diagram(tm_ctrl.values, behavior_names, behavior_colors, 
                   title="Control", ax=axD)
_panel_tag(axD, "D")

# =============================================================================
# Panel E — Chord Diagram: ELS
# =============================================================================
print("  Panel E: Chord diagram (ELS)...")

tm_els = mean_trans_mats["ELS"]
plot_chord_diagram(tm_els.values, behavior_names, behavior_colors, 
                   title="ELS", ax=axE)
_panel_tag(axE, "E")

# =============================================================================
# Panel F — Difference Network: ELS - Control
# =============================================================================
# Panel F — Chord Diagram: ELS - Control Difference
# =============================================================================
print("  Panel F: Chord diagram difference (ELS - Control)...")

tm_stab_els = mean_stab_mats["ELS"]  # This is already ELS - Control avg
plot_chord_diagram(tm_stab_els.values, behavior_names, behavior_colors,
                   title="ELS − Control", ax=axF)
_panel_tag(axF, "F")

# =============================================================================
# Panel G — BFL Score Histogram by Group
# =============================================================================
print("  Panel G: BFL score histogram...")

for grp in ["Control", "ELS"]:
    scores = bfl_df[bfl_df["group"] == grp]["bfl_score"]
    axG.hist(scores, bins=10, alpha=0.6, label=grp, color=palette[grp], edgecolor="white")

axG.axvline(0, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
axG.set_xlabel("BFL Score", fontsize=9)
axG.set_ylabel("Count", fontsize=9)
axG.set_title("BFL Score Distribution", fontsize=10, color="#4b4b4b")
axG.legend(loc="upper right", frameon=False, fontsize=8)
_style_axes(axG)
_panel_tag(axG, "G")

# =============================================================================
# Panel H — Freezing Time-Course
# =============================================================================
print("  Panel H: Freezing time-course...")

def _plot_time_course(ax, behavior, title, tag, show_star=False, legend_loc="lower right"):
    beh_data = time_course_df[time_course_df["behavior"] == behavior]
    _add_epoch_bands(ax)
    
    for grp in ["Control", "ELS", "ELS resilient"]:
        grp_data = beh_data[beh_data["group_extended"] == grp]
        if len(grp_data) == 0:
            continue
        agg = grp_data.groupby("time_min")["pct"].agg(["mean", "sem"]).reset_index()
        x = agg["time_min"].values
        y = agg["mean"].values
        err = agg["sem"].fillna(0).values
        
        ax.plot(x, y, color=palette[grp], marker="o", markersize=3, 
                linewidth=1.2, label=grp, zorder=3)
        ax.fill_between(x, y - err, y + err, color=palette[grp], alpha=0.2, zorder=2)
    
    if show_star:
        y_max = beh_data.groupby("time_min")["pct"].mean().max()
        _add_significance_star(ax, 5.5, y_max * 1.05)
    
    ax.set_xlabel("Time (minutes)", fontsize=8)
    ax.set_ylabel("% of time", fontsize=8)
    ax.set_xlim(0, 7.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_title(title, fontsize=10, color="#4b4b4b")
    ax.legend(loc=legend_loc, frameon=False, fontsize=6)
    _style_axes(ax)
    _panel_tag(ax, tag)

_plot_time_course(axH, "Freezing", "Freezing", "H", show_star=True, legend_loc="lower right")

# =============================================================================
# Panel I — Frequency Bar Chart
# =============================================================================
print("  Panel I: Frequency bar chart...")

freq_agg = freq_df.groupby(["behavior", "group_extended"])["frequency"].agg(["mean", "sem"]).reset_index()
freq_agg.columns = ["behavior", "group", "mean", "sem"]

behavior_order = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
group_order = ["Control", "ELS", "ELS resilient"]

x = np.arange(len(behavior_order))
width = 0.25

for i, grp in enumerate(group_order):
    grp_data = freq_agg[freq_agg["group"] == grp].set_index("behavior").reindex(behavior_order)
    means = grp_data["mean"].fillna(0).values
    sems = grp_data["sem"].fillna(0).values
    offset = (i - 1) * width
    axI.bar(x + offset, means, width, label=grp, color=palette[grp], 
            edgecolor="#444444", linewidth=0.5, yerr=sems, capsize=2,
            error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="#444444"))

axI.set_xticks(x)
axI.set_xticklabels(behavior_order, fontsize=7, rotation=45, ha="right")
axI.set_ylabel("Frequency (s)", fontsize=9)
axI.set_title("Behavior Frequency", fontsize=10, color="#4b4b4b")
axI.legend(loc="upper right", frameon=False, fontsize=6, ncol=1)
_style_axes(axI)
_panel_tag(axI, "I")

# =============================================================================
# Panels J–O — Behavior Time-Courses
# =============================================================================
print("  Panels J-O: Behavior time-courses...")

_plot_time_course(axJ, "Sniffing", "Sniffing", "J", show_star=True, legend_loc="upper right")
_plot_time_course(axK, "Grooming", "Grooming", "K", show_star=False, legend_loc="upper right")
_plot_time_course(axL, "Turn", "Turn", "L", show_star=True, legend_loc="lower right")
_plot_time_course(axM, "Locomotion", "Locomotion", "M", show_star=True, legend_loc="upper right")
_plot_time_course(axN, "Climbing", "Climbing", "N", show_star=False, legend_loc="upper right")
_plot_time_course(axO, "Jump", "Jump", "O", show_star=False, legend_loc="upper right")

# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_7_dynamics.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_7_dynamics.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 7 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 7 ready.")

# %%
