"""
Figure 5 — Behavioral cluster frequencies and trajectories (MoSeq clusters).

Uses the clustered results (`new_results_clusters.pkl`) to compute total usage
and time-course (30 s bins) per behavioral cluster for Control vs ELS.
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

from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BIN_SECONDS

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
group_map = dict(zip(index_df["name"], index_df["group"]))

print(f"Loaded {len(results_dict)} recordings with cluster labels.")

#%%
# Build per-frame dataframe with group and time bins
print("Building per-frame dataset...")
records = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    syll = np.array(data["syllable"])
    group = group_map.get(rec, "Unknown")
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
# Plot Figure 5
print("Plotting Figure 5...")
sns.set_theme(style="ticks", context="talk")

# Panel A: total usage barplot
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
ax_total = axes[0, 0]
palette = {"Control": "#f9c74f", "ELS": "#c37ba0"}
sns.barplot(data=total, x="behavior", y="seconds", hue="group", palette=palette, ax=ax_total)
ax_total.set_title("Total time per behavior")
ax_total.set_ylabel("Seconds")
ax_total.set_xlabel("")
ax_total.legend(title="Group")
for label in ax_total.get_xticklabels():
    label.set_rotation(45)

# Panels B-H: time courses for each behavior
behaviors = list(behavior_mapping.values())
plot_axes = axes.flatten()[1:]
for beh, ax in zip(behaviors, plot_axes):
    data = summary[summary["behavior"] == beh]
    for group, gdata in data.groupby("group"):
        ax.plot(
            gdata["time_min"],
            gdata["mean"],
            label=group,
            color=palette.get(group, "#4d4d4d"),
            linewidth=2,
        )
        ax.fill_between(
            gdata["time_min"],
            gdata["mean"] - gdata["sem"],
            gdata["mean"] + gdata["sem"],
            color=palette.get(group, "#4d4d4d"),
            alpha=0.25,
            linewidth=0,
        )
        ax.scatter(
            gdata["time_min"],
            gdata["mean"],
            color=palette.get(group, "#4d4d4d"),
            s=18,
            zorder=3,
        )
    # shaded periods at 2, 4.5, and 6 minutes
    marks_min = [2.0, 4.5, 6.0]
    half_bin_min = (bin_seconds / 60.0) / 2
    for m in marks_min:
        ax.axvspan(m - half_bin_min, m + half_bin_min, color="#4d4d4d", alpha=0.08, linewidth=0)
    ax.set_title(beh)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("% of time")
    ax.legend(title="Group")

# Hide unused axes if any
for ax in plot_axes[len(behaviors):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Figure 5 ready.")

# %%
