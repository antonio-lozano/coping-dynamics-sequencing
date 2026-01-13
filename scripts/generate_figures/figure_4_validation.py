"""
Figure 4 — Validation of MoSeq freezing syllables against SimBA freezing.

Loads raw MoSeq syllables and SimBA freezing labels, computes per-syllable
precision/recall for freezing, and summarizes overlap for key syllables.
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

from src.config import RESULTS_RAW_PKL, FREEZING_DIR, INDEX_CSV, FPS

results_pkl = RESULTS_RAW_PKL
freezing_dir = FREEZING_DIR
index_csv = INDEX_CSV
fps = FPS

#%%
# Load data
print("Loading MoSeq results (syllables) and group assignments...")
with open(results_pkl, "rb") as f:
    results_dict = pickle.load(f)

index_df = pd.read_csv(index_csv)

def _normalize(name: str) -> str:
    return name.strip().replace(" ", "_")

# build robust group lookup: raw, normalized, and prefix-before-DLC
group_map = {}
for _, row in index_df.iterrows():
    raw = str(row["name"]).strip()
    norm = _normalize(raw)
    prefix = norm.split("DLC")[0].rstrip("_") if "DLC" in norm else norm
    group_map[raw] = row["group"]
    group_map[norm] = row["group"]
    group_map[prefix] = row["group"]

print("Loading SimBA freezing prediction CSVs...")
freezing_records = {}
for csv_path in freezing_dir.glob("*_freezing_predictions_only.csv"):
    base = csv_path.name.replace("_freezing_predictions_only.csv", "")
    df = pd.read_csv(csv_path)
    if "Freezing_Jen_0-125_threshold" not in df.columns:
        continue
    freeze_vec = df["Freezing_Jen_0-125_threshold"].to_numpy(dtype=int)
    base_norm = _normalize(base)
    base_prefix = base_norm.split("DLC")[0].rstrip("_")
    freezing_records[base] = freeze_vec
    freezing_records[base_norm] = freeze_vec
    freezing_records[base_prefix] = freeze_vec

print(f"MoSeq recordings: {len(results_dict)}; Freezing files: {len(freezing_records)}")

#%%
# Compute syllable-level precision/recall vs freezing
print("Computing per-syllable precision and recall relative to freezing labels...")
metrics = []
matched_rec = 0
missing_rec = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    freeze = (
        freezing_records.get(rec_raw)
        or freezing_records.get(rec_norm)
        or freezing_records.get(rec_prefix)
    )
    if freeze is None:
        missing_rec.append(rec_raw)
        continue
    matched_rec += 1
    syll = np.array(data["syllable"])
    n = min(len(syll), len(freeze))
    syll = syll[:n]
    freeze = freeze[:n]
    unique_syll, counts = np.unique(syll, return_counts=True)
    freezing_frames = freeze == 1
    for s, c in zip(unique_syll, counts):
        pred_mask = syll == s
        tp = np.sum(pred_mask & freezing_frames)
        fp = np.sum(pred_mask & ~freezing_frames)
        fn = np.sum(~pred_mask & freezing_frames)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        metrics.append(
            {
                "recording": rec_raw,
                "group": group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown"),
                "syllable": s,
                "precision": precision,
                "recall": recall,
                "frames": c,
            }
        )

metrics_df = pd.DataFrame(metrics)
if metrics_df.empty:
    print("No recordings matched between MoSeq results and freezing predictions.")
    print("Missing recordings (no freeze file found):")
    print(missing_rec)
    raise SystemExit("Aborting: no matches for precision/recall computation.")

print(f"Matched recordings: {matched_rec}; Missing: {len(missing_rec)}")
print("Group counts in matched metrics:")
print(metrics_df["group"].value_counts())
print(f"Computed metrics for {metrics_df['syllable'].nunique()} syllables.")

#%%
# Aggregate across recordings
print("Aggregating precision/recall across recordings...")
agg = (
    metrics_df.groupby("syllable")[["precision", "recall", "frames"]]
    .agg({"precision": "mean", "recall": "mean", "frames": "sum"})
    .reset_index()
)
agg = agg.sort_values("precision", ascending=False)

#%%
# Plot Figure 4 panels (precision and recall)
print("Plotting Figure 4 precision and recall barplots...")
sns.set_theme(style="ticks", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

top_n = min(20, len(agg))
show = agg.head(top_n)

sns.barplot(data=show, x="syllable", y="precision", ax=axes[0], color="#c37ba0")
axes[0].set_title("Precision vs SimBA freezing (top syllables)")
axes[0].set_ylabel("Precision")
axes[0].set_xlabel("Syllable ID")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

sns.barplot(data=show, x="syllable", y="recall", ax=axes[1], color="#4d4d4d")
axes[1].set_title("Recall vs SimBA freezing (top syllables)")
axes[1].set_ylabel("Recall")
axes[1].set_xlabel("Syllable ID")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

sns.despine(fig=fig, top=True, right=True)

plt.tight_layout()
plt.show()

print("Figure 4 ready.")

# %%
