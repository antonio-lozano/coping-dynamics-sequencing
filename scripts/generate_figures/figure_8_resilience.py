"""
Figure 8 — Resilient vs vulnerable vs control motif characteristics and transitions.

Reuses the dynamics score (Fig 7) to label resilient ELS animals, then compares
diversity, usage, bout durations, and transition metrics across subgroups.
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
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import warnings

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS, BEHAVIOR_MAPPING, CODES

# Silence upcoming seaborn/pandas FutureWarnings for palette-without-hue and length-1 grouping
warnings.filterwarnings("ignore", category=FutureWarning, message=".*palette.*hue.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*length-1 tuple.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*SeriesGroupBy.grouper.*")

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

# robust group map across raw, normalized, and DLC-stripped names
group_map = {}
for _, row in index_df.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
    group_map[raw] = row["group"]
    group_map[norm] = row["group"]
    group_map[prefix] = row["group"]

#%%
# Build time-binned profiles and dynamics score (reuse Fig 7 logic)
print("Building time-binned profiles and dynamics scores...")

# Determine a common bin count so feature vectors share the same length
bin_counts = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    seq = np.array(data["syllable"])
    seq = seq[np.isin(seq, codes)]
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
    bins = len(seq) // bin_size
    if bins < common_bins:
        continue
    seq = seq[: common_bins * bin_size]
    arr = seq.reshape(common_bins, bin_size)
    bin_feats = []
    for b in arr:
        counts = np.bincount(b, minlength=max(codes) + 1)
        pct = counts[codes] / bin_size
        bin_feats.append(pct)
    feat_vec = np.concatenate(bin_feats)
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    profiles.append({"recording": rec_raw, "group": group, "feature": feat_vec, "seq": seq})

profile_df = pd.DataFrame(profiles)
if profile_df.empty:
    raise SystemExit("No recordings met the common bin requirement.")

feature_matrix = np.stack(profile_df["feature"].to_numpy())
dist = pairwise_distances(feature_matrix, metric="euclidean")
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
coords = mds.fit_transform(dist)
profile_df[["mds1", "mds2"]] = coords

median_ctrl = profile_df[profile_df["group"] == "Control"][["mds1", "mds2"]].median().to_numpy()
median_els = profile_df[profile_df["group"] == "ELS"][["mds1", "mds2"]].median().to_numpy()
eps = 1e-9
d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
d_els = np.linalg.norm(coords - median_els, axis=1)
profile_df["dynamics_score"] = np.log((d_els + eps) / (d_ctrl + eps))
profile_df["subgroup"] = profile_df.apply(
    lambda r: "ELS resilient" if (r["group"] == "ELS" and r["dynamics_score"] < 0)
    else ("ELS vulnerable" if r["group"] == "ELS" else "Control"),
    axis=1,
)

print(profile_df["subgroup"].value_counts())

#%%
# Helper metrics
def diversity_scores(seq):
    vals, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    simpson = 1 - np.sum(p**2)
    shannon = -np.sum(p * np.log(p))
    evenness = shannon / np.log(len(vals)) if len(vals) > 0 else np.nan
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


def transition_matrix(seq, codes):
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
        ent += pi * (-np.sum(nz * np.log2(nz)))
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
records = []
bout_records = []
for _, row in profile_df.iterrows():
    seq = row["seq"]
    subgroup = row["subgroup"]
    simpson, shannon, evenness, cui = diversity_scores(seq)
    bouts = bout_lengths(seq)
    mean_bout = np.mean([l for _, l in bouts]) / fps
    for beh_code, length in bouts:
        if beh_code not in behavior_mapping:
            continue
        bout_records.append(
            {
                "recording": row["recording"],
                "subgroup": subgroup,
                "behavior": behavior_mapping[beh_code],
                "bout_seconds": length / fps,
            }
        )
    rec_rate, det = recurrence_and_determinism(seq)
    tm = transition_matrix(seq, codes)
    tm_entropy = markov_entropy(tm)
    records.append(
        {
            "recording": row["recording"],
            "subgroup": subgroup,
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

#%%
# Plot Figure 8 panels (selected)
print("Plotting Figure 8 comparisons across subgroups...")
sns.set(style="ticks")

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

panels = [
    ("simpson", "Simpson diversity"),
    ("cui", "Cumulative usage index"),
    ("mean_bout_seconds", "Mean bout duration (s)"),
    ("recurrence", "Recurrence rate"),
    ("determinism", "Determinism"),
    ("markov_entropy", "Markov entropy"),
]

for (col, title), ax in zip(panels, axes.flatten()):
    sns.boxplot(
        data=metrics_df,
        x="subgroup",
        y=col,
        hue="subgroup",
        legend=False,
        palette={
            "Control": "#f9c74f",
            "ELS vulnerable": "#c37ba0",
            "ELS resilient": "#4d4d4d",
        },
        ax=ax,
    )
    sns.stripplot(
        data=metrics_df,
        x="subgroup",
        y=col,
        color="black",
        alpha=0.5,
        dodge=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Behavior-specific bout durations (bottom center)
ax_bout = axes[2, 1]
sns.boxplot(
    data=bouts_df,
    x="behavior",
    y="bout_seconds",
    hue="subgroup",
    palette={
        "Control": "#f9c74f",
        "ELS vulnerable": "#c37ba0",
        "ELS resilient": "#4d4d4d",
    },
    ax=ax_bout,
)
ax_bout.set_title("Bout duration by behavior and subgroup")
ax_bout.set_xlabel("")
ax_bout.set_ylabel("Seconds")
ax_bout.legend(title="Subgroup")
ax_bout.tick_params(axis="x", rotation=45)
ax_bout.grid(False)
ax_bout.spines["top"].set_visible(False)
ax_bout.spines["right"].set_visible(False)

# Hide unused axes to avoid empty panels
axes[2, 0].set_visible(False)
axes[2, 2].set_visible(False)

sns.despine(fig=fig, top=True, right=True)

plt.tight_layout()
plt.show()

print("Figure 8 ready.")

# %%
