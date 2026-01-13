"""
Figure 7 — Behavioral dynamics score and resilient subgroup identification.

Computes flattened per-bin behavioral feature vectors from clustered syllables,
builds distance matrices across multiple metrics, and renders MDS plots with
dynamic similarity backgrounds and group/resilient overlays.
"""

#%% Imports and paths
print("Loading dependencies and setting paths...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
from sklearn.metrics import pairwise_distances
# from sklearn.manifold import MDS
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.manifold import MDS

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS, BEHAVIOR_MAPPING, CODES

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

#%% Build fixed-length feature vectors (common_bins)
print("Building time-binned behavioral profiles...")
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
    profiles.append({"recording": rec_raw, "group": group, "feature": feat_vec})

profile_df = pd.DataFrame(profiles)
feature_matrix = np.vstack(profile_df["feature"].to_numpy())
print(f"Built profiles for {len(profile_df)} recordings; feature dim {feature_matrix.shape[1]}")

#%% Distance matrix and MDS embedding for base plot
print("Computing pairwise distances and MDS embedding...")
dist = pairwise_distances(feature_matrix, metric="euclidean")
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
coords = mds.fit_transform(dist)
profile_df[["mds1", "mds2"]] = coords

#%% Dynamics score: log(distance to ELS median / distance to Control median)
print("Computing dynamics scores and resilient subset...")
median_ctrl = profile_df[profile_df["group"] == "Control"][["mds1", "mds2"]].median().to_numpy()
median_els = profile_df[profile_df["group"] == "ELS"][["mds1", "mds2"]].median().to_numpy()
d_ctrl = np.linalg.norm(coords - median_ctrl, axis=1)
d_els = np.linalg.norm(coords - median_els, axis=1)
eps = 1e-9
dynamics_score = np.log((d_els + eps) / (d_ctrl + eps))
profile_df["dynamics_score"] = dynamics_score
profile_df["resilient"] = (profile_df["group"] == "ELS") & (profile_df["dynamics_score"] < 0)

print(f"Resilient ELS count: {profile_df['resilient'].sum()} / {len(profile_df[profile_df['group']=='ELS'])}")

#%% Build distance_matrices from features (multiple metrics)
print("Preparing distance matrices from feature vectors...")
X = feature_matrix
animal_keys = profile_df["recording"].tolist()

def minkowski_p3(u, v): return np.sum(np.abs(u - v) ** 3) ** (1 / 3)
def squared_euclidean(u, v): return np.sum((u - v) ** 2)
def sorensen(u, v):
    denom = (np.sum(u) + np.sum(v))
    return 1 - (2 * np.sum(np.minimum(u, v)) / (denom if denom != 0 else 1))
def ruzicka(u, v):
    num = np.sum(np.minimum(u, v)); den = np.sum(np.maximum(u, v))
    return 1 - (num / den if den != 0 else 1)

distance_funcs = {
    "Cosine": "cosine",
    "Correlation": "correlation",
    "Jensen-Shannon": "jensen",
    "Manhattan": "cityblock",
    "Euclidean": "euclidean",
    "Chebyshev": "chebyshev",
    "Bray-Curtis": "braycurtis",
    "Minkowski_p3": minkowski_p3,
    "Canberra": "canberra",
    "Squared Euclidean": squared_euclidean,
    "Sorensen": sorensen,
    "Ruzicka": ruzicka,
}

distance_matrices = {}
n = X.shape[0]
eps_small = 1e-12
for name, metric in distance_funcs.items():
    if isinstance(metric, str):
        if metric == "jensen":
            Xn = X.copy()
            row_sums = Xn.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            Xn = (Xn + eps_small) / row_sums[:, None]
            js_pdist = pdist(Xn, metric="jensenshannon")
            mat = squareform(js_pdist)
        else:
            mat = cdist(X, X, metric=metric)
    else:
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi = X[i]
            for j in range(i + 1, n):
                v = metric(xi, X[j])
                mat[i, j] = v
                mat[j, i] = v
    distance_matrices[name] = mat
print(f"Built distance_matrices for {len(distance_matrices)} metrics using X.shape={X.shape}")

#%% Plot all distance_matrices in Figure 7 style (MDS + dynamics scores)
print("Rendering MDS plots for all distance matrices...")
epsilon = 1e-9
cmap = mcolors.LinearSegmentedColormap.from_list("yellow_white_purple", ["#c37ba0", "white", "#f9c74f"])  # purple->white->yellow
colors = {"Control": "#f9c74f", "ELS": "#c37ba0", "ELS resilient": "#4d4d4d"}

conditions = [
    profile_df.loc[profile_df["recording"] == k, "group"].iloc[0] if k in profile_df["recording"].values else None
    for k in animal_keys
]
binary_labels = np.array([0 if c == "Control" else 1 for c in conditions])

for metric_name, matrix in distance_matrices.items():
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    embedding = mds.fit_transform(matrix)
    clf = LogisticRegression()
    acc = np.mean(cross_val_score(clf, embedding, binary_labels, cv=LeaveOneOut()))

    control_idx = [i for i, c in enumerate(conditions) if c == "Control"]
    els_idx = [i for i, c in enumerate(conditions) if c == "ELS"]
    if len(control_idx) == 0 or len(els_idx) == 0:
        print(f"Skipping {metric_name}: missing groups.")
        continue

    ctrl_med = np.median(embedding[control_idx, :], axis=0)
    els_med = np.median(embedding[els_idx, :], axis=0)
    d_ctrl = np.linalg.norm(embedding - ctrl_med, axis=1)
    d_els = np.linalg.norm(embedding - els_med, axis=1)
    scores = np.log((d_els + epsilon) / (d_ctrl + epsilon))

    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    margin = 0.05 * max(x_max - x_min, y_max - y_min)
    grid_x, grid_y = np.mgrid[x_min - margin:x_max + margin:200j, y_min - margin:y_max + margin:200j]
    grid_z = griddata(embedding, scores, (grid_x, grid_y), method="cubic")
    grid_z_nn = griddata(embedding, scores, (grid_x, grid_y), method="nearest")
    grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    vmin, vmax = np.nanmin(scores), np.nanmax(scores)
    cf = ax1.contourf(
        grid_x, grid_y, grid_z, levels=30, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9
    )
    cs = ax1.contour(grid_x, grid_y, grid_z, levels=10, colors="k", linewidths=0.7, alpha=0.6)
    ax1.clabel(cs, inline=True, fontsize=8)
    fig.colorbar(cf, ax=ax1, label="Dynamics score (log dist ELS / dist Ctrl)")

    for grp in ["Control", "ELS"]:
        idx = [i for i, c in enumerate(conditions) if c == grp]
        if idx:
            ax1.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                color=colors[grp],
                s=60,
                label=grp,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
    resilient_mask = (np.array(conditions) == "ELS") & (scores < 0)
    if resilient_mask.any():
        ax1.scatter(
            embedding[resilient_mask, 0],
            embedding[resilient_mask, 1],
            facecolors="none",
            edgecolors="black",
            s=120,
            linewidths=1.4,
            alpha=0.9,
            label="Resilient ELS",
        )

    ax1.set_title(f"MDS ({metric_name}) — LOOCV Acc: {acc*100:.1f}%", fontsize=14)
    ax1.set_xlabel("MDS1")
    ax1.set_ylabel("MDS2")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False)

    df_plot = pd.DataFrame({"recording": animal_keys, "group": conditions, "dynamics_score": scores})
    df_plot["group_plot"] = df_plot.apply(
        lambda r: "ELS resilient"
        if (r["group"] == "ELS" and r["dynamics_score"] < 0)
        else ("ELS" if r["group"] == "ELS" else "Control"),
        axis=1,
    )

    sns.boxplot(
        data=df_plot,
        x="group_plot",
        y="dynamics_score",
        order=["Control", "ELS", "ELS resilient"],
        palette={"Control": colors["Control"], "ELS": colors["ELS"], "ELS resilient": colors["ELS resilient"]},
        ax=ax2,
        boxprops=dict(edgecolor="#444444"),
        medianprops=dict(color="#222222"),
    )
    sns.stripplot(
        data=df_plot,
        x="group_plot",
        y="dynamics_score",
        color="black",
        alpha=0.6,
        ax=ax2,
        jitter=True,
    )
    ax2.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax2.set_title("Dynamics score by group")
    ax2.set_xlabel("")
    ax2.set_ylabel("Dynamics score")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    ctrl_animals = [animal_keys[i] for i in control_idx]
    els_typical = [animal_keys[i] for i in els_idx if scores[i] >= 0]
    els_resilient = [animal_keys[i] for i in els_idx if scores[i] < 0]
    print(f"\nMetric: {metric_name}")
    print("Control animals:", ctrl_animals)
    print("ELS typical animals:", els_typical)
    print("ELS resilient animals:", els_resilient)

# %%
