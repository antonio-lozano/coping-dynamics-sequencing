"""
Figure 6 — Diversity, bout structure, and transition metrics of behavioral clusters.

Computes diversity indices, bout durations, cumulative usage, and simple
transition-derived metrics (recurrence, determinism, Markov entropy) from
clustered MoSeq labels, then plots groupwise comparisons.
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
import matplotlib.colors as mcolors
import seaborn as sns
import pickle

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BEHAVIOR_MAPPING, MANUSCRIPT_FIGURES_DIR as RESULTS_DIR, PALETTE
from src.plotting import plot_chord_diagram

results_pkl = RESULTS_CLUSTERS_PKL
index_csv = INDEX_CSV
fps = FPS

behavior_mapping = BEHAVIOR_MAPPING

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

#%%
# Helper functions
def diversity_scores(seq):
    vals, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    simpson = 1 - np.sum(p**2)
    shannon = -np.sum(p * np.log(p))
    evenness = shannon / np.log(len(vals)) if len(vals) > 0 else np.nan
    # cumulative usage index (mean cumulative sum of sorted p)
    p_sorted = np.sort(p)[::-1]
    cumsum = np.cumsum(p_sorted)
    cui = cumsum.mean()
    return simpson, shannon, evenness, cui


def bout_lengths(seq):
    lengths = []
    prev = seq[0]
    run = 1
    for x in seq[1:]:
        if x == prev:
            run += 1
        else:
            lengths.append((prev, run))
            prev = x
            run = 1
    lengths.append((prev, run))
    return lengths


def transition_matrix(seq, codes):
    mat = np.zeros((len(codes), len(codes)), dtype=float)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    for a, b in zip(seq[:-1], seq[1:]):
        if a in code_to_idx and b in code_to_idx:
            mat[code_to_idx[a], code_to_idx[b]] += 1
    return mat


def markov_entropy(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    prob = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    stationary = row_sums.flatten() / row_sums.sum() if row_sums.sum() > 0 else np.zeros(len(row_sums))
    ent = 0.0
    for i, pi in enumerate(stationary):
        if pi <= 0:
            continue
        row = prob[i]
        nz = row[row > 0]
        ent_row = -np.sum(nz * np.log2(nz))
        ent += pi * ent_row
    return ent


def recurrence_and_determinism(seq):
    # Approximate: recurrence as probability of same-state transitions across time
    n = len(seq)
    if n < 2:
        return np.nan, np.nan
    rec = np.mean(seq[:-1] == seq[1:])
    # Determinism approximated as fraction of frames in bouts of length >=2
    bouts = bout_lengths(seq)
    det = sum(l for _, l in bouts if l >= 2) / n
    return rec, det


def lz_complexity(seq):
    # Lempel-Ziv (LZ76) complexity for integer sequences
    if len(seq) == 0:
        return np.nan
    s = " ".join(map(str, seq))
    i, k, l = 0, 1, 1
    c = 1
    n = len(s)
    while True:
        if i + k == n:
            c += 1
            break
        substring = s[i : i + k]
        if substring in s[0:l]:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            l = i
            k = 1
        if i + k > n:
            break
    return c


#%%
# Compute metrics per recording
print("Computing diversity, cumulative usage, bouts, and transition metrics per recording...")
records = []
bout_records = []
trans_records = []
usage_records = []
codes = list(behavior_mapping.keys())

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
    simpson, shannon, evenness, cui = diversity_scores(seq)
    bouts = bout_lengths(seq)
    mean_bout = np.mean([l for _, l in bouts]) / fps
    for beh_code, length in bouts:
        bout_records.append(
            {
                "recording": rec,
                "group": group,
                "behavior": behavior_mapping[beh_code],
                "bout_seconds": length / fps,
            }
        )
    rec_rate, det = recurrence_and_determinism(seq)
    tm = transition_matrix(seq, codes)
    tm_entropy = markov_entropy(tm)
    lz = lz_complexity(seq)

    vals, counts = np.unique(seq, return_counts=True)
    usage = dict(zip(vals, counts / counts.sum()))
    usage_records.append({"recording": rec_raw, "group": group, **{f"m{c}": usage.get(c, 0.0) for c in codes}})
    records.append(
        {
            "recording": rec_raw,
            "group": group,
            "simpson": simpson,
            "shannon": shannon,
            "evenness": evenness,
            "cui": cui,
            "mean_bout_seconds": mean_bout,
            "recurrence": rec_rate,
            "determinism": det,
            "markov_entropy": tm_entropy,
            "lz_complexity": lz,
        }
    )

metrics_df = pd.DataFrame(records)
bouts_df = pd.DataFrame(bout_records)
usage_df = pd.DataFrame(usage_records)
print(f"Metrics for {len(metrics_df)} recordings.")

#%%
# Plot Figure 6 panels
print("Plotting Figure 6 (diversity, CUI, bout durations, transition metrics)...")
sns.set_theme(style="ticks", context="paper", font_scale=1.15)

palette = PALETTE
behavior_colors = {
    "Freezing": "#C37B9F",
    "Sniffing": "#4CB7A5",
    "Grooming": "#7EC8E3",
    "Turn": "#8DC63F",
    "Locomotion": "#F8C650",
    "Climbing": "#F4A259",
    "Jump": "#E4572E",
}

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.8, length=3, labelsize=9)

def _panel_tag(ax, tag):
    ax.text(-0.12, 1.04, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#4b4b4b")

def _box_scatter(ax, df, ycol, ylabel, title, star=False):
    sns.boxplot(data=df, x="group", y=ycol, palette=palette, ax=ax, width=0.5, fliersize=0)
    sns.stripplot(data=df, x="group", y=ycol, color="#2f2f2f", alpha=0.5, size=3, jitter=True, ax=ax)
    ax.set_title(title, fontsize=10, color="#4b4b4b")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    _style_axes(ax)
    if star:
        y_max = df[ycol].max()
        ax.plot([0, 0, 1, 1], [y_max * 1.02, y_max * 1.06, y_max * 1.06, y_max * 1.02], color="#4b4b4b", lw=0.8)
        ax.text(0.5, y_max * 1.07, "*", ha="center", va="bottom", color="#4b4b4b")

def _select_representative(group_label):
    use_cols = [f"m{c}" for c in codes]
    group_df = usage_df[usage_df["group"] == group_label]
    if group_df.empty:
        return None
    mean_vec = group_df[use_cols].mean().to_numpy()
    dists = np.linalg.norm(group_df[use_cols].to_numpy() - mean_vec[None, :], axis=1)
    return group_df.iloc[np.argmin(dists)]["recording"]

def _get_seq(rec_name):
    for rec, data in results_dict.items():
        if str(rec).strip() == rec_name:
            return np.array(data["syllable"])
    return None

def _plot_ethogram(ax, seq, title, tag):
    if seq is None:
        ax.set_axis_off()
        return
    # Keep original sequence to preserve actual time mapping
    original_seq = seq.copy()
    bin_frames = int(round(0.25 * fps))
    # Use original sequence length for proper time mapping
    times = np.arange(0, len(original_seq), bin_frames)
    time_min = times / fps / 60.0
    behavior_names = [behavior_mapping[c] for c in codes]
    for i, beh in enumerate(behavior_names):
        # Check each subsampled frame against the behavior code
        subsampled = original_seq[::bin_frames]
        mask = subsampled == codes[i]
        x = time_min[mask]
        ax.vlines(x, i - 0.35, i + 0.35, color=behavior_colors[beh], linewidth=0.8)
    ax.set_yticks(np.arange(len(behavior_names)))
    ax.set_yticklabels(behavior_names, fontsize=7)
    ax.set_xlim(0, 7.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xlabel("Time (minutes)")
    ax.set_title(title, fontsize=10, color="#4b4b4b")
    for start in [3.0, 4.5, 6.0]:
        ax.axvspan(start, start + 0.5, color="#E6E6E6", zorder=0)
    _style_axes(ax)
    _panel_tag(ax, tag)

def _plot_barcode(ax, seq, title, tag):
    if seq is None:
        ax.set_axis_off()
        return
    # Keep original sequence to preserve actual time mapping
    original_seq = seq.copy()
    if len(original_seq) == 0:
        ax.set_axis_off()
        return
    bin_frames = int(round(0.25 * fps))
    # Use original sequence length for proper time
    total_time_min = len(original_seq) / fps / 60.0
    bins = original_seq[: (len(original_seq) // bin_frames) * bin_frames].reshape(-1, bin_frames)
    # Mode of each bin - if no valid codes, use -1 for gray
    def bin_mode(x):
        valid = x[np.isin(x, codes)]
        if len(valid) == 0:
            return -1
        return np.bincount(valid, minlength=max(codes)+1)[codes].argmax() + 1  # +1 because codes start at 1
    mode = np.apply_along_axis(lambda x: np.bincount(x[np.isin(x, codes)], minlength=max(codes)+1).argmax() if np.any(np.isin(x, codes)) else -1, 1, bins)
    # Map modes to colors
    color_map = {c: behavior_colors[behavior_mapping[c]] for c in codes}
    color_map[-1] = "#cccccc"  # Gray for unknown/invalid
    rgb = np.array([mcolors.to_rgb(color_map.get(c, "#cccccc")) for c in mode])[None, :, :]
    ax.imshow(rgb, aspect="auto", extent=[0, total_time_min, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(0, 7.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xlabel("Time (minutes)")
    ax.set_title(title, fontsize=10, color="#4b4b4b")
    for start in [3.0, 4.5, 6.0]:
        ax.axvspan(start, start + 0.5, color="#E6E6E6", zorder=0, alpha=0.6)
    _style_axes(ax)
    _panel_tag(ax, tag)

# Representative animals
rep_ctrl = _select_representative("Control")
rep_els = _select_representative("ELS")

# Cumulative usage helper (defined early for use in plots)
motif_cols = [f"m{c}" for c in codes]
motif_totals = usage_df[motif_cols].sum().sort_values(ascending=False)
rank_order = motif_totals.index.tolist()

def _cumulative_usage(group_label):
    grp = usage_df[usage_df["group"] == group_label]
    if grp.empty:
        return None
    mean_usage = grp[rank_order].mean()
    cum = np.cumsum(mean_usage)
    sem = grp[rank_order].sem().fillna(0).cumsum()
    return cum, sem

def _bout_panel(ax, data, title, tag, star=False):
    _box_scatter(ax, data, "bout_seconds", "Bout Duration (s)", title, star=star)
    _panel_tag(ax, tag)

def _transition_chord(ax, group_label, tag):
    """Draw a chord diagram for transitions within a group."""
    mats = []
    for rec, data in results_dict.items():
        seq = np.array(data.get("syllable", []))
        seq = seq[np.isin(seq, codes)]
        if len(seq) == 0:
            continue
        rec_raw = str(rec).strip()
        rec_norm = _normalize(rec_raw)
        rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
        group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
        if group != group_label:
            continue
        mats.append(transition_matrix(seq, codes))
    
    if mats:
        mat = np.mean(mats, axis=0)
        # Normalize to get transition probabilities (row-wise)
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_prob = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
        
        # Plot chord diagram
        behavior_names = [behavior_mapping[c] for c in codes]
        plot_chord_diagram(mat_prob, behavior_names, behavior_colors, 
                          title=group_label, ax=ax)
    
    _panel_tag(ax, tag)
    _panel_tag(ax, tag)

# =============================================================================
# FIGURE LAYOUT - Clean 7-row design
# =============================================================================
fig = plt.figure(figsize=(14, 22), dpi=150, facecolor="#FFFFFF")

# Define main grid: 7 rows for logical grouping
gs = fig.add_gridspec(
    7, 1,
    height_ratios=[0.8, 0.8, 1.0, 1.0, 0.9, 0.9, 1.2],
    hspace=0.55,
    left=0.08, right=0.95, top=0.97, bottom=0.04
)

# -----------------------------------------------------------------------------
# ROW 1: Ethogram + Barcode for Control (A, B)
# -----------------------------------------------------------------------------
row1 = gs[0].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.25)
axA = fig.add_subplot(row1[0])
axB = fig.add_subplot(row1[1])

_plot_ethogram(axA, _get_seq(rep_ctrl) if rep_ctrl else None, "Control — Ethogram", "A")
_plot_barcode(axB, _get_seq(rep_ctrl) if rep_ctrl else None, "Control — Barcode", "B")

# -----------------------------------------------------------------------------
# ROW 2: Ethogram + Barcode for ELS (C, D)
# -----------------------------------------------------------------------------
row2 = gs[1].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.25)
axC = fig.add_subplot(row2[0])
axD = fig.add_subplot(row2[1])

_plot_ethogram(axC, _get_seq(rep_els) if rep_els else None, "ELS — Ethogram", "C")
_plot_barcode(axD, _get_seq(rep_els) if rep_els else None, "ELS — Barcode", "D")

# -----------------------------------------------------------------------------
# ROW 3: Diversity indices (E, F, G, H) + Cumulative Usage (I)
# -----------------------------------------------------------------------------
row3 = gs[2].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6], wspace=0.4)
axE = fig.add_subplot(row3[0])
axF = fig.add_subplot(row3[1])
axG = fig.add_subplot(row3[2])
axH = fig.add_subplot(row3[3])
axI = fig.add_subplot(row3[4])

_box_scatter(axE, metrics_df, "simpson", "Simpson", "Simpson Index", star=True)
_panel_tag(axE, "E")
_box_scatter(axF, metrics_df, "shannon", "Shannon", "Shannon Entropy", star=False)
_panel_tag(axF, "F")
_box_scatter(axG, metrics_df, "evenness", "Evenness", "Evenness Index", star=False)
_panel_tag(axG, "G")
_box_scatter(axH, metrics_df, "cui", "CUI", "Cumulative Usage Index", star=True)
_panel_tag(axH, "H")

# Cumulative usage plot (I)
ctrl_cum = _cumulative_usage("Control")
els_cum = _cumulative_usage("ELS")
x = np.arange(len(rank_order))
if ctrl_cum and els_cum:
    axI.errorbar(x, ctrl_cum[0], yerr=ctrl_cum[1], color=palette["Control"], marker="o", 
                 markersize=5, linewidth=1.5, capsize=2, label="Control")
    axI.errorbar(x, els_cum[0], yerr=els_cum[1], color=palette["ELS"], marker="o", 
                 markersize=5, linewidth=1.5, capsize=2, label="ELS")
axI.set_title("Cumulative Usage by Rank", fontsize=10, color="#4b4b4b")
axI.set_ylabel("Cumulative Usage")
axI.set_xlabel("Behavior (ranked)")
axI.set_xticks(x)
axI.set_xticklabels([behavior_mapping[int(c[1:])] for c in rank_order], rotation=45, ha="right", fontsize=7)
axI.legend(frameon=False, fontsize=8, loc="lower right")
_style_axes(axI)
_panel_tag(axI, "I")

# -----------------------------------------------------------------------------
# ROW 4: Bout durations - first row (J, K, L, M)
# -----------------------------------------------------------------------------
row4 = gs[3].subgridspec(1, 4, wspace=0.4)
axJ = fig.add_subplot(row4[0])
axK = fig.add_subplot(row4[1])
axL = fig.add_subplot(row4[2])
axM = fig.add_subplot(row4[3])

overall_df = bouts_df.groupby(["recording", "group"])["bout_seconds"].mean().reset_index()
_bout_panel(axJ, overall_df, "Overall", "J", star=True)
_bout_panel(axK, bouts_df[bouts_df["behavior"] == "Freezing"], "Freezing", "K", star=True)
_bout_panel(axL, bouts_df[bouts_df["behavior"] == "Sniffing"], "Sniffing", "L", star=False)
_bout_panel(axM, bouts_df[bouts_df["behavior"] == "Grooming"], "Grooming", "M", star=True)

# -----------------------------------------------------------------------------
# ROW 5: Bout durations - second row (N, O, P, Q)
# -----------------------------------------------------------------------------
row5 = gs[4].subgridspec(1, 4, wspace=0.4)
axN = fig.add_subplot(row5[0])
axO = fig.add_subplot(row5[1])
axP = fig.add_subplot(row5[2])
axQ = fig.add_subplot(row5[3])

_bout_panel(axN, bouts_df[bouts_df["behavior"] == "Turn"], "Turn", "N", star=False)
_bout_panel(axO, bouts_df[bouts_df["behavior"] == "Locomotion"], "Locomotion", "O", star=False)
_bout_panel(axP, bouts_df[bouts_df["behavior"] == "Climbing"], "Climbing", "P", star=False)
_bout_panel(axQ, bouts_df[bouts_df["behavior"] == "Jump"], "Jump", "Q", star=False)

# -----------------------------------------------------------------------------
# ROW 6: Transition metrics (T, U, V, W)
# -----------------------------------------------------------------------------
row6 = gs[5].subgridspec(1, 4, wspace=0.4)
axT = fig.add_subplot(row6[0])
axU = fig.add_subplot(row6[1])
axV = fig.add_subplot(row6[2])
axW = fig.add_subplot(row6[3])

_box_scatter(axT, metrics_df, "lz_complexity", "LZ Complexity", "Lempel-Ziv", star=False)
_panel_tag(axT, "T")
_box_scatter(axU, metrics_df, "recurrence", "Recurrence", "Recurrence Rate", star=True)
_panel_tag(axU, "U")
_box_scatter(axV, metrics_df, "determinism", "Determinism", "Determinism", star=True)
_panel_tag(axV, "V")
_box_scatter(axW, metrics_df, "markov_entropy", "Markov Entropy", "Markov Entropy", star=True)
_panel_tag(axW, "W")

# -----------------------------------------------------------------------------
# ROW 7: Transition chord diagrams (R, S)
# -----------------------------------------------------------------------------
row7 = gs[6].subgridspec(1, 2, wspace=0.3)
axR = fig.add_subplot(row7[0])
axS = fig.add_subplot(row7[1])

_transition_chord(axR, "Control", "R")
_transition_chord(axS, "ELS", "S")

# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_6_diversity.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_6_diversity.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 6 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 6 ready.")

# %%

