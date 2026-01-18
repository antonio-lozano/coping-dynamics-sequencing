"""
Figure 8 — Resilient vs vulnerable vs control motif characteristics and transitions.

Uses BFL (Behavioral Flow Likeness) score from transition matrices to label resilient
ELS animals, then compares diversity, usage, bout durations, and transition metrics.

Includes time-split analysis: First 3 minutes vs rest of experiment (as in reference code).

Panels:
  Row 1: Simpson diversity (A), Cumulative usage index (B), Mean bout duration (C)
  Row 2: Recurrence rate (D), Determinism (E), Markov entropy (F)
  Row 3: Bout duration by behavior (G), BFL First 3min (H), BFL Rest (I)
  Row 4: Time-split comparison (J), ROC curve for classifier (K), empty or additional
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
from sklearn.metrics import pairwise_distances, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import scipy.stats as st
import warnings

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS, BEHAVIOR_MAPPING, CODES, RESULTS_DIR, PALETTE
from src.transition_utils import (
    compute_transition_matrix, 
    compute_stabilized_transition_matrices,
    compute_bfl_scores,
    relabel_matrix
)

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

results_pkl = RESULTS_CLUSTERS_PKL
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS
bin_size = fps * bin_seconds

behavior_mapping = BEHAVIOR_MAPPING
codes = CODES

#%%
# Load data
print("Loading clustered MoSeq results and group assignments...")
with open(results_pkl, "rb") as f:
    results_dict = pickle.load(f)

index_df = pd.read_csv(index_csv)

def _normalize(name: str) -> str:
    return str(name).strip().replace(" ", "_")

# Robust group map
group_map = {}
for _, row in index_df.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
    group_map[raw] = row["group"]
    group_map[norm] = row["group"]
    group_map[prefix] = row["group"]

#%%
# Compute transition matrices and BFL scores (whole recording)
print("Computing transition matrices and BFL scores (whole recording)...")

transition_matrices = {}
group_assignments = {}
sequences = {}  # Store sequences for later analysis

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
    
    TM = compute_transition_matrix(seq)
    TM = TM.reindex(index=codes, columns=codes, fill_value=0)
    
    transition_matrices[rec_raw] = TM
    group_assignments[rec_raw] = group
    sequences[rec_raw] = seq

# Compute stabilized matrices and BFL scores
stabilized_matrices = compute_stabilized_transition_matrices(
    transition_matrices, group_assignments, control_value="Control"
)
bfl_scores = compute_bfl_scores(stabilized_matrices, group_assignments, "Control", "ELS")

# Create main BFL DataFrame
bfl_df = pd.DataFrame([
    {"recording": rec, "group": group_assignments[rec], "bfl_score": score}
    for rec, score in bfl_scores.items()
])

# Define subgroups based on BFL score
bfl_df["subgroup"] = bfl_df.apply(
    lambda r: "ELS resilient" if (r["group"] == "ELS" and r["bfl_score"] < 0)
    else ("ELS vulnerable" if r["group"] == "ELS" else "Control"),
    axis=1
)

print(f"Subgroup counts:\n{bfl_df['subgroup'].value_counts()}")

#%%
# Time-split analysis: First 3 minutes vs rest (as in reference code)
print("Computing time-split BFL scores (First 3 min vs Rest)...")

frame_rate = fps
frames_first3 = frame_rate * 180  # 3 minutes = 180 seconds

trans_mats_first3 = {}
trans_mats_rest = {}
group_assignments_first3 = {}
group_assignments_rest = {}

for rec, seq in sequences.items():
    group = group_assignments[rec]
    
    # First 3 minutes
    if len(seq) >= frames_first3:
        first_seq = seq[:frames_first3]
        TM_first = compute_transition_matrix(first_seq)
        TM_first = TM_first.reindex(index=codes, columns=codes, fill_value=0)
        trans_mats_first3[rec] = TM_first
        group_assignments_first3[rec] = group
    
    # Rest of recording
    if len(seq) > frames_first3:
        rest_seq = seq[frames_first3:]
        if len(rest_seq) > 0:
            TM_rest = compute_transition_matrix(rest_seq)
            TM_rest = TM_rest.reindex(index=codes, columns=codes, fill_value=0)
            trans_mats_rest[rec] = TM_rest
            group_assignments_rest[rec] = group

# Compute BFL for each time segment
if trans_mats_first3:
    stab_first3 = compute_stabilized_transition_matrices(trans_mats_first3, group_assignments_first3, "Control")
    bfl_first3 = compute_bfl_scores(stab_first3, group_assignments_first3, "Control", "ELS")
    bfl_first3_df = pd.DataFrame([
        {"recording": rec, "group": group_assignments_first3[rec], "bfl_score": score, "segment": "First 3 min"}
        for rec, score in bfl_first3.items()
    ])
else:
    bfl_first3_df = pd.DataFrame()

if trans_mats_rest:
    stab_rest = compute_stabilized_transition_matrices(trans_mats_rest, group_assignments_rest, "Control")
    bfl_rest = compute_bfl_scores(stab_rest, group_assignments_rest, "Control", "ELS")
    bfl_rest_df = pd.DataFrame([
        {"recording": rec, "group": group_assignments_rest[rec], "bfl_score": score, "segment": "Rest"}
        for rec, score in bfl_rest.items()
    ])
else:
    bfl_rest_df = pd.DataFrame()

# Combine time-split data
bfl_time_split = pd.concat([bfl_first3_df, bfl_rest_df], ignore_index=True)
print(f"Time-split BFL data: {len(bfl_time_split)} rows")

#%%
# Helper metrics
def diversity_scores(seq):
    vals, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    simpson = 1 - np.sum(p**2)
    shannon = -np.sum(p * np.log(p + 1e-10))
    evenness = shannon / np.log(len(vals)) if len(vals) > 1 else np.nan
    p_sorted = np.sort(p)[::-1]
    cui = np.cumsum(p_sorted).mean()
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


def transition_matrix_raw(seq, codes):
    mat = np.zeros((len(codes), len(codes)), dtype=float)
    idx = {c: i for i, c in enumerate(codes)}
    for a, b in zip(seq[:-1], seq[1:]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1
    return mat


def markov_entropy(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    prob = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    stationary = row_sums.flatten() / row_sums.sum() if row_sums.sum() > 0 else np.zeros(len(row_sums))
    ent = 0.0
    for i, pi in enumerate(stationary):
        if pi <= 0:
            continue
        nz = prob[i][prob[i] > 0]
        ent += pi * (-np.sum(nz * np.log2(nz + 1e-10)))
    return ent


def recurrence_and_determinism(seq):
    n = len(seq)
    rec = np.mean(seq[:-1] == seq[1:]) if n > 1 else np.nan
    bouts = bout_lengths(seq)
    det = sum(l for _, l in bouts if l >= 2) / n if n > 0 else np.nan
    return rec, det


#%%
# Compute metrics per recording using subgroup labels
print("Computing motif metrics per subgroup...")

# Create subgroup lookup
subgroup_lookup = dict(zip(bfl_df["recording"], bfl_df["subgroup"]))

records = []
bout_records = []

for rec, seq in sequences.items():
    subgroup = subgroup_lookup.get(rec, "Unknown")
    if subgroup == "Unknown":
        continue
    
    simpson, shannon, evenness, cui = diversity_scores(seq)
    bouts = bout_lengths(seq)
    mean_bout = np.mean([l for _, l in bouts]) / fps
    
    for beh_code, length in bouts:
        if beh_code not in behavior_mapping:
            continue
        bout_records.append({
            "recording": rec,
            "subgroup": subgroup,
            "behavior": behavior_mapping[beh_code],
            "bout_seconds": length / fps,
        })
    
    rec_rate, det = recurrence_and_determinism(seq)
    tm = transition_matrix_raw(seq, codes)
    tm_entropy = markov_entropy(tm)
    
    records.append({
        "recording": rec,
        "subgroup": subgroup,
        "simpson": simpson,
        "shannon": shannon,
        "evenness": evenness,
        "cui": cui,
        "mean_bout_seconds": mean_bout,
        "recurrence": rec_rate,
        "determinism": det,
        "markov_entropy": tm_entropy,
    })

metrics_df = pd.DataFrame(records)
bouts_df = pd.DataFrame(bout_records)

print(f"Metrics computed for {len(metrics_df)} recordings")

#%%
# Build feature vectors for classifier (time-binned dynamics)
print("Building feature vectors for classifier...")

feature_vectors = []
labels_full = []
animal_ids = []

for rec, seq in sequences.items():
    group = group_assignments.get(rec, "Unknown")
    if group == "Unknown":
        continue
    
    n_bins = int(np.ceil(len(seq) / bin_size))
    animal_feature = []
    
    for code in sorted(codes):
        binned = [
            np.sum(seq[i * bin_size:(i + 1) * bin_size] == code) / bin_size * 100
            for i in range(n_bins)
        ]
        animal_feature.extend(binned)
    
    feature_vectors.append(animal_feature)
    labels_full.append(1 if group.lower() == "els" else 0)
    animal_ids.append(rec)

# Pad to same length
max_len = max(len(f) for f in feature_vectors)
features_padded = np.array([f + [0] * (max_len - len(f)) for f in feature_vectors])
labels_full = np.array(labels_full)

print(f"Feature matrix shape: {features_padded.shape}")

#%%
# Train classifier and compute ROC
print("Training RandomForest classifier with cross-validation...")

clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = cross_val_score(clf_rf, features_padded, labels_full, cv=cv_rf, scoring='accuracy')
print(f"Cross-validated accuracy: {accuracy_scores.mean():.2f} (+/- {accuracy_scores.std() * 2:.2f})")

pred_probas = cross_val_predict(clf_rf, features_padded, labels_full, cv=cv_rf, method='predict_proba')
roc_auc = roc_auc_score(labels_full, pred_probas[:, 1])
print(f"ROC AUC: {roc_auc:.2f}")

fpr, tpr, thresholds = roc_curve(labels_full, pred_probas[:, 1])

#%%
# Plot Figure 8
print("Plotting Figure 8...")
sns.set_theme(style="ticks", context="paper", font_scale=1.0)

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)

def _panel_tag(ax, tag):
    ax.text(-0.12, 1.05, tag, transform=ax.transAxes, fontsize=11, 
            fontweight="bold", color="#4b4b4b", va="top", ha="right")

fig = plt.figure(figsize=(15, 16), dpi=150, facecolor="#FFFFFF")

gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.2, 1.0],
                      width_ratios=[1, 1, 1],
                      hspace=0.45, wspace=0.35,
                      left=0.07, right=0.95, top=0.96, bottom=0.05)

# Row 0: Diversity metrics
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# Row 1: Transition metrics
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])

# Row 2: Bout duration, BFL First 3min, BFL Rest
axG = fig.add_subplot(gs[2, 0])
axH = fig.add_subplot(gs[2, 1])
axI = fig.add_subplot(gs[2, 2])

# Row 3: Time-split comparison, ROC curve
axJ = fig.add_subplot(gs[3, 0])
axK = fig.add_subplot(gs[3, 1])
axL = fig.add_subplot(gs[3, 2])

subgroup_order = ["Control", "ELS vulnerable", "ELS resilient"]

def _plot_metric(ax, col, title, tag):
    """Plot a single metric as boxplot with stripplot overlay."""
    plot_data = metrics_df[metrics_df["subgroup"].isin(subgroup_order)]
    
    sns.boxplot(data=plot_data, x="subgroup", y=col, order=subgroup_order,
                hue="subgroup", legend=False, palette=PALETTE, ax=ax,
                boxprops=dict(alpha=0.7), fliersize=0)
    sns.stripplot(data=plot_data, x="subgroup", y=col, order=subgroup_order,
                  color="black", alpha=0.5, size=4, ax=ax)
    
    ax.set_title(title, fontsize=10, color="#4b4b4b")
    ax.set_xlabel("")
    ax.set_xticklabels(["Control", "Vulnerable", "Resilient"], fontsize=8, rotation=15)
    _style_axes(ax)
    _panel_tag(ax, tag)

# =============================================================================
# Row 0: Diversity metrics
# =============================================================================
print("  Row 0: Diversity metrics...")
_plot_metric(axA, "simpson", "Simpson Diversity", "A")
_plot_metric(axB, "cui", "Cumulative Usage Index", "B")
_plot_metric(axC, "mean_bout_seconds", "Mean Bout Duration (s)", "C")

# =============================================================================
# Row 1: Transition metrics
# =============================================================================
print("  Row 1: Transition metrics...")
_plot_metric(axD, "recurrence", "Recurrence Rate", "D")
_plot_metric(axE, "determinism", "Determinism", "E")
_plot_metric(axF, "markov_entropy", "Markov Entropy", "F")

# =============================================================================
# Panel G: Bout duration by behavior
# =============================================================================
print("  Panel G: Bout duration by behavior...")

plot_bouts = bouts_df[bouts_df["subgroup"].isin(subgroup_order)]
sns.boxplot(data=plot_bouts, x="behavior", y="bout_seconds", hue="subgroup",
            hue_order=subgroup_order, palette=PALETTE, ax=axG, fliersize=0)
axG.set_title("Bout Duration by Behavior", fontsize=10, color="#4b4b4b")
axG.set_xlabel("")
axG.set_ylabel("Duration (s)", fontsize=9)
axG.tick_params(axis="x", rotation=45)
axG.legend(title="", loc="upper right", frameon=False, fontsize=7)
_style_axes(axG)
_panel_tag(axG, "G")

# =============================================================================
# Panel H: BFL Score - First 3 minutes
# =============================================================================
print("  Panel H: BFL score (First 3 min)...")

if not bfl_first3_df.empty:
    sns.boxplot(data=bfl_first3_df, x="group", y="bfl_score", order=["Control", "ELS"],
                palette=PALETTE, ax=axH, fliersize=0,
                boxprops=dict(alpha=0.7))
    sns.stripplot(data=bfl_first3_df, x="group", y="bfl_score", order=["Control", "ELS"],
                  palette=PALETTE, ax=axH, size=5, alpha=0.7)
    axH.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)

axH.set_title("BFL Score: First 3 Minutes", fontsize=10, color="#4b4b4b")
axH.set_xlabel("")
axH.set_ylabel("BFL Score", fontsize=9)
_style_axes(axH)
_panel_tag(axH, "H")

# =============================================================================
# Panel I: BFL Score - Rest of experiment
# =============================================================================
print("  Panel I: BFL score (Rest)...")

if not bfl_rest_df.empty:
    sns.boxplot(data=bfl_rest_df, x="group", y="bfl_score", order=["Control", "ELS"],
                palette=PALETTE, ax=axI, fliersize=0,
                boxprops=dict(alpha=0.7))
    sns.stripplot(data=bfl_rest_df, x="group", y="bfl_score", order=["Control", "ELS"],
                  palette=PALETTE, ax=axI, size=5, alpha=0.7)
    axI.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)

axI.set_title("BFL Score: Rest of Experiment", fontsize=10, color="#4b4b4b")
axI.set_xlabel("")
axI.set_ylabel("BFL Score", fontsize=9)
_style_axes(axI)
_panel_tag(axI, "I")

# =============================================================================
# Panel J: Time-split BFL comparison (paired)
# =============================================================================
print("  Panel J: Time-split BFL comparison...")

if not bfl_time_split.empty:
    # Pivot to get paired data
    pivot_df = bfl_time_split.pivot_table(index=["recording", "group"], 
                                           columns="segment", 
                                           values="bfl_score").reset_index()
    
    for grp, color in [("Control", PALETTE["Control"]), ("ELS", PALETTE["ELS"])]:
        grp_data = pivot_df[pivot_df["group"] == grp]
        if "First 3 min" in grp_data.columns and "Rest" in grp_data.columns:
            # Plot individual lines
            for _, row in grp_data.iterrows():
                axJ.plot([0, 1], [row["First 3 min"], row["Rest"]], 
                        color=color, alpha=0.3, linewidth=0.8)
            # Plot means
            mean_first = grp_data["First 3 min"].mean()
            mean_rest = grp_data["Rest"].mean()
            axJ.plot([0, 1], [mean_first, mean_rest], color=color, 
                    linewidth=2.5, marker="o", markersize=8, label=grp)
    
    axJ.set_xticks([0, 1])
    axJ.set_xticklabels(["First 3 min", "Rest"], fontsize=9)
    axJ.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    axJ.legend(loc="upper right", frameon=False, fontsize=8)

axJ.set_title("BFL Score: Time-Split Comparison", fontsize=10, color="#4b4b4b")
axJ.set_ylabel("BFL Score", fontsize=9)
_style_axes(axJ)
_panel_tag(axJ, "J")

# =============================================================================
# Panel K: ROC Curve
# =============================================================================
print("  Panel K: ROC curve...")

axK.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', linewidth=2.5, color='#2a9d8f')
axK.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
axK.fill_between(fpr, 0, tpr, color='#2a9d8f', alpha=0.1)
axK.set_xlabel("False Positive Rate", fontsize=9)
axK.set_ylabel("True Positive Rate", fontsize=9)
axK.set_title("ROC Curve (RandomForest)", fontsize=10, color="#4b4b4b")
axK.legend(loc="lower right", frameon=False, fontsize=9)
axK.set_xlim([0, 1])
axK.set_ylim([0, 1.02])
_style_axes(axK)
_panel_tag(axK, "K")

# =============================================================================
# Panel L: ANOVA results summary
# =============================================================================
print("  Panel L: Statistical summary...")

# Perform one-way ANOVA for BFL scores
ctrl_scores = bfl_df[bfl_df["group"] == "Control"]["bfl_score"].values
els_scores = bfl_df[bfl_df["group"] == "ELS"]["bfl_score"].values

t_stat, p_val = st.ttest_ind(ctrl_scores, els_scores)
u_stat, u_pval = st.mannwhitneyu(ctrl_scores, els_scores, alternative='two-sided')

# Group means
ctrl_mean = np.mean(ctrl_scores)
els_mean = np.mean(els_scores)
ctrl_sem = st.sem(ctrl_scores)
els_sem = st.sem(els_scores)

# Text summary
summary_text = (
    f"BFL Score Analysis\n"
    f"─────────────────────\n\n"
    f"Control (n={len(ctrl_scores)}):\n"
    f"  Mean: {ctrl_mean:.3f} ± {ctrl_sem:.3f}\n\n"
    f"ELS (n={len(els_scores)}):\n"
    f"  Mean: {els_mean:.3f} ± {els_sem:.3f}\n\n"
    f"Statistics:\n"
    f"  t-test: t={t_stat:.3f}, p={p_val:.4f}\n"
    f"  Mann-Whitney: U={u_stat:.0f}, p={u_pval:.4f}\n\n"
    f"Classifier Performance:\n"
    f"  Accuracy: {accuracy_scores.mean():.2f} ± {accuracy_scores.std():.2f}\n"
    f"  ROC AUC: {roc_auc:.2f}"
)

axL.text(0.05, 0.95, summary_text, transform=axL.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', edgecolor='#cccccc'))
axL.axis('off')
axL.set_title("Statistical Summary", fontsize=10, color="#4b4b4b")
_panel_tag(axL, "L")

# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_8_resilience.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_8_resilience.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 8 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 8 ready.")

# %%
