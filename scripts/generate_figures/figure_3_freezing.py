"""
Figure 3 — Supervised freezing over time (SimBA predictions).

Loads SimBA freezing predictions, aligns them with group labels from index.csv,
bins over time, and plots group mean ± SEM across the FC session.
"""

#%%
# Imports and paths
print("Loading dependencies and setting paths...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import FREEZING_DIR, INDEX_CSV, FPS, BIN_SECONDS

freezing_dir = FREEZING_DIR
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS  # bins to match FC tone/shock structure
bin_size = fps * bin_seconds

#%%
# Load group labels and freezing CSVs
print("Loading index.csv for group assignments...")
index_df = pd.read_csv(index_csv)

def _normalize(name: str) -> str:
    return name.strip().replace(" ", "_")

# build a robust lookup with raw, normalized, and truncated (prefix-before-DLC) keys
group_map = {}
prefix_lookup = {}
for _, row in index_df.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else None
    group_map[raw] = row["group"]
    group_map[norm] = row["group"]
    if prefix:
        group_map[prefix] = row["group"]
        prefix_lookup[prefix] = row["group"]
    # also map entries that already end with '_freezing_predictions_only'
    if raw.endswith("_freezing_predictions_only"):
        trimmed = raw.replace("_freezing_predictions_only", "")
        group_map[trimmed] = row["group"]
        group_map[_normalize(trimmed)] = row["group"]
    if norm.endswith("_freezing_predictions_only"):
        trimmed = norm.replace("_freezing_predictions_only", "")
        group_map[trimmed] = row["group"]
        group_map[_normalize(trimmed)] = row["group"]

print("Loading SimBA freezing prediction CSVs...")
freezing_records = []
for csv_path in freezing_dir.glob("*_freezing_predictions_only.csv"):
    base = csv_path.name.replace("_freezing_predictions_only.csv", "")
    df = pd.read_csv(csv_path)
    if "Freezing_Jen_0-125_threshold" not in df.columns:
        continue
    df = df.rename(columns={"Freezing_Jen_0-125_threshold": "freezing"})
    df["frame"] = df["Unnamed: 0"]
    df["animal"] = base
    norm_base = _normalize(base)
    prefix_base = norm_base.split("DLC")[0].rstrip("_") if "DLC" in norm_base else norm_base.split("_freezing")[0]
    grp = group_map.get(base) or group_map.get(norm_base) or group_map.get(prefix_base) or prefix_lookup.get(prefix_base)
    df["group"] = grp if grp is not None else "Unknown"
    freezing_records.append(df[["animal", "group", "frame", "freezing"]])

if not freezing_records:
    raise FileNotFoundError("No freezing prediction CSVs loaded from SimBA output.")

freezing_df = pd.concat(freezing_records, ignore_index=True)
# report unknowns if any
unknown_animals = freezing_df.loc[freezing_df["group"] == "Unknown", "animal"].unique()
if len(unknown_animals) > 0:
    print(f"Warning: {len(unknown_animals)} animals missing group assignment:", unknown_animals)
else:
    print("All animals matched to groups.")
print(f"Loaded freezing predictions for {freezing_df['animal'].nunique()} animals.")

#%%
# Bin freezing over time and summarize by group
print("Computing freezing fraction per time bin and group...")
freezing_df["bin"] = (freezing_df["frame"] // bin_size).astype(int)
agg = (
    freezing_df.groupby(["group", "bin", "animal"])["freezing"]
    .mean()
    .reset_index(name="freeze_frac")
)
summary = (
    agg.groupby(["group", "bin"])["freeze_frac"]
    .agg(["mean", "sem", "count"])
    .reset_index()
)
summary["time_sec"] = (summary["bin"] + 0.5) * bin_seconds
summary["time_min"] = summary["time_sec"] / 60.0
print("Summary rows:", len(summary))
print("Group counts in summary:")
print(summary["group"].value_counts())

#%%
# Plot Figure 3
print("Plotting Figure 3 (freezing over time, mean ± SEM)...")
sns.set_theme(style="ticks", context="talk")
fig, ax = plt.subplots(figsize=(8, 5))

palette = {"Control": "#f9c74f", "ELS": "#c37ba0"}
for group, data in summary.groupby("group"):
    if group not in ("Control", "ELS"):
        continue
    ax.plot(
        data["time_min"],
        data["mean"],
        label=group,
        color=palette.get(group, "#4d4d4d"),
        linewidth=2,
    )
    ax.fill_between(
        data["time_min"],
        data["mean"] - data["sem"],
        data["mean"] + data["sem"],
        alpha=0.25,
        color=palette.get(group, "#4d4d4d"),
        linewidth=0,
    )
    ax.scatter(
        data["time_min"],
        data["mean"],
        color=palette.get(group, "#4d4d4d"),
        s=20,
        zorder=3,
    )

# shaded periods at 2, 4.5, and 6 minutes (± half-bin width)
marks_min = [2.0, 4.5, 6.0]
half_bin_min = (bin_seconds / 60.0) / 2
for m in marks_min:
    ax.axvspan(m - half_bin_min, m + half_bin_min, color="#4d4d4d", alpha=0.08, linewidth=0)

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Freezing (% of frames per bin)")
ax.set_title("Freezing acquisition during FC (SimBA predictions)")
ax.legend(title="Group")
sns.despine()
plt.tight_layout()
plt.show()

print("Figure 3 ready.")

# %%
