"""
Figure 6 — Diversity, bout structure, and transition metrics of behavioral clusters.

Computes diversity indices, bout durations, cumulative usage, and simple
transition-derived metrics (recurrence, determinism, Markov entropy) from
clustered MoSeq labels, then plots groupwise comparisons.
"""

#%%
# Imports and paths
print("Loading dependencies and setting paths...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BEHAVIOR_MAPPING

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
group_map = dict(zip(index_df["name"], index_df["group"]))

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


#%%
# Compute metrics per recording
print("Computing diversity, cumulative usage, bouts, and transition metrics per recording...")
records = []
bout_records = []
trans_records = []
codes = list(behavior_mapping.keys())

for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
    if len(seq) == 0:
        continue
    group = group_map.get(rec, "Unknown")
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
    records.append(
        {
            "recording": rec,
            "group": group,
            "simpson": simpson,
            "shannon": shannon,
            "evenness": evenness,
            "cui": cui,
            "mean_bout_seconds": mean_bout,
            "recurrence": rec_rate,
            "determinism": det,
            "markov_entropy": tm_entropy,
        }
    )

metrics_df = pd.DataFrame(records)
bouts_df = pd.DataFrame(bout_records)
print(f"Metrics for {len(metrics_df)} recordings.")

#%%
# Plot Figure 6 panels
print("Plotting Figure 6 (diversity, CUI, bout durations, transition metrics)...")
sns.set_theme(style="ticks", context="talk")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))

panels = [
    ("simpson", "Simpson diversity"),
    ("shannon", "Shannon entropy"),
    ("evenness", "Evenness"),
    ("cui", "Cumulative usage index"),
    ("mean_bout_seconds", "Mean bout duration (s)"),
    ("recurrence", "Recurrence rate"),
    ("determinism", "Determinism"),
    ("markov_entropy", "Markov entropy"),
]

for (col, title), ax in zip(panels, axes.flatten()):
    sns.boxplot(data=metrics_df, x="group", y=col, palette={"Control": "#f9c74f", "ELS": "#c37ba0"}, ax=ax)
    sns.stripplot(data=metrics_df, x="group", y=col, color="black", alpha=0.4, dodge=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Behavior-specific bout durations
ax_bout = axes.flatten()[-1]
sns.boxplot(
    data=bouts_df,
    x="behavior",
    y="bout_seconds",
    hue="group",
    palette={"Control": "#f9c74f", "ELS": "#c37ba0"},
    ax=ax_bout,
)
ax_bout.set_title("Bout duration by behavior")
ax_bout.set_xlabel("")
ax_bout.set_ylabel("Seconds")
ax_bout.legend(title="Group")
ax_bout.tick_params(axis="x", rotation=45)
ax_bout.spines["top"].set_visible(False)
ax_bout.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

print("Figure 6 ready.")
