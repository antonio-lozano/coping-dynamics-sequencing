"""
Figure 3 — Supervised freezing over time (SimBA predictions).

Loads SimBA freezing predictions, aligns them with group labels from index.csv,
bins over time, and plots group mean ± SEM across the FC session.
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
import warnings

from src.config import FREEZING_DIR, INDEX_CSV, FPS, BIN_SECONDS, MANUSCRIPT_FIGURES_DIR as RESULTS_DIR, PALETTE

freezing_dir = FREEZING_DIR
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS  # keep original script setting (30-second bins)
bin_size = fps * bin_seconds
warnings.warn(
    f"Figure 3 uses bin_seconds={bin_seconds} (original-script compatible, expected 30s).",
    RuntimeWarning,
)

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
print("Freezing dir:", freezing_dir)
print("Freezing dir exists:", freezing_dir.exists())
freezing_records = []
csv_candidates = list(freezing_dir.glob("*_freezing_predictions_only.csv"))
print(f"Found {len(csv_candidates)} candidate CSV(s) matching *_freezing_predictions_only.csv")
for csv_path in csv_candidates:
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
    raise FileNotFoundError(
        "No freezing prediction CSVs loaded from SimBA output. "
        f"Checked: {freezing_dir} (exists={freezing_dir.exists()}). "
        "Expected files like *_freezing_predictions_only.csv."
    )

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

# Explicit omission requested for Figure 3 analysis.
excluded_animals_fig3 = {"Animal_48_6", "48_6"}
excluded_tokens = {_normalize(a).lower() for a in excluded_animals_fig3}
pre_exclude_n_animals = agg["animal"].nunique()
agg = agg[
    ~agg["animal"].map(lambda a: _normalize(str(a)).lower() in excluded_tokens)
].copy()
post_exclude_n_animals = agg["animal"].nunique()
if post_exclude_n_animals != pre_exclude_n_animals:
    warnings.warn(
        f"Figure 3 exclusion applied for known bad subject(s): {sorted(excluded_animals_fig3)}",
        RuntimeWarning,
    )
    print(
        f"Figure 3 exclusion applied: removed {pre_exclude_n_animals - post_exclude_n_animals} animal(s) "
        f"matching {sorted(excluded_animals_fig3)}."
    )
else:
    warnings.warn(
        f"Figure 3 exclusion note: no rows matched exclusion list {sorted(excluded_animals_fig3)}.",
        RuntimeWarning,
    )
    print(f"Figure 3 exclusion note: no animals matched {sorted(excluded_animals_fig3)}.")

# Figure 3 analysis must only use Control/ELS (never Unknown).
unknown_animals_analysis = sorted(agg.loc[agg["group"] == "Unknown", "animal"].unique().tolist())
if unknown_animals_analysis:
    warnings.warn(
        "Figure 3 excludes Unknown-group animals from analysis: "
        + ", ".join(unknown_animals_analysis),
        RuntimeWarning,
    )
agg = agg[agg["group"].isin(["Control", "ELS"])].copy()

def _summary_for_animals(animals):
    data = agg[agg["animal"].isin(animals)]
    summary = (
        data.groupby(["group", "bin"])["freeze_frac"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    # Use bin end-time so 30s bins are labeled 0.5, 1.0, ..., 7.5 minutes.
    summary["time_sec"] = (summary["bin"] + 1.0) * bin_seconds
    summary["time_min"] = summary["time_sec"] / 60.0
    return summary

def _split_animals_by_group(group_label):
    animals = sorted(agg.loc[agg["group"] == group_label, "animal"].unique().tolist())
    if len(animals) >= 41:
        return animals[:25], animals[25:41]
    if len(animals) >= 26:
        return animals[: len(animals) - 16], animals[len(animals) - 16 :]
    mid = max(1, len(animals) // 2)
    return animals[:mid], animals[mid:]

ctrl_a, ctrl_b = _split_animals_by_group("Control")
els_a, els_b = _split_animals_by_group("ELS")
dataset_a_animals = set(ctrl_a + els_a)
dataset_b_animals = set(ctrl_b + els_b)

all_loaded_animals = sorted(agg["animal"].unique().tolist())
analysis_animals = sorted(agg.loc[agg["group"].isin(["Control", "ELS"]), "animal"].unique().tolist())
excluded_unknown_animals = unknown_animals_analysis
print(f"Figure 3 animals used (Control + ELS): {len(analysis_animals)}")

print("\n=== Animal usage audit for Figure 3 ===")
print(f"All animals with freezing traces loaded: {len(all_loaded_animals)}")
print(all_loaded_animals)
print(f"\nAnimals included in analysis (Control/ELS only): {len(analysis_animals)}")
print(analysis_animals)
print(f"\nAnimals excluded from group-based analysis (Unknown group): {len(excluded_unknown_animals)}")
print(excluded_unknown_animals)
print(f"\nControl animals total used: {len(sorted(ctrl_a + ctrl_b))}")
print(sorted(ctrl_a + ctrl_b))
print(f"ELS animals total used: {len(sorted(els_a + els_b))}")
print(sorted(els_a + els_b))
print(f"\nPanel A animals (n={len(dataset_a_animals)}):")
print(sorted(dataset_a_animals))
print(f"  Control in A (n={len(ctrl_a)}): {ctrl_a}")
print(f"  ELS in A (n={len(els_a)}): {els_a}")
print(f"\nPanel B animals (n={len(dataset_b_animals)}):")
print(sorted(dataset_b_animals))
print(f"  Control in B (n={len(ctrl_b)}): {ctrl_b}")
print(f"  ELS in B (n={len(els_b)}): {els_b}")
print("=== End animal usage audit ===\n")

summary_a = _summary_for_animals(dataset_a_animals)
summary_b = _summary_for_animals(dataset_b_animals)
summary_all = _summary_for_animals(analysis_animals)

#%%
# Plot Figure 3 (three panels)
print("Plotting Figure 3 (freezing over time, mean +/- SEM)...")
sns.set_theme(style="ticks", context="paper", font_scale=1.0)

palette = PALETTE  # Use centralized palette

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)

def _panel_tag(ax, tag):
    ax.text(-0.15, 1.08, tag, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#4b4b4b")

def _plot_panel(ax, data, title, tag):
    for start in [3.0, 4.5, 6.0]:
        ax.axvspan(start, start + 0.5, color="#F6F6F6", zorder=0)
    for group, gdata in data.groupby("group"):
        if group not in ("Control", "ELS"):
            continue
        ax.plot(
            gdata["time_min"],
            gdata["mean"] * 100.0,
            label=group,
            color=palette.get(group, "#4d4d4d"),
            linewidth=1.5,
            marker="o",
            markersize=2.5,
        )
        ax.fill_between(
            gdata["time_min"],
            (gdata["mean"] - gdata["sem"]) * 100.0,
            (gdata["mean"] + gdata["sem"]) * 100.0,
            color=palette.get(group, "#4d4d4d"),
            alpha=0.2,
            linewidth=0,
        )
    ax.set_title(title, fontsize=9, color="#4b4b4b", pad=4)
    ax.set_xlabel("Time (minutes)", fontsize=8)
    ax.set_ylabel("Freezing (% of time)", fontsize=8)
    ax.set_xlim(0, 7.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(frameon=False, loc="upper right", fontsize=7, ncol=1)
    ax.text(4.0, 72, "*", ha="center", va="center", color="#4b4b4b", fontsize=10)
    _style_axes(ax)
    _panel_tag(ax, tag)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=300, facecolor="#FFFFFF")
plt.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.85, bottom=0.18)

_plot_panel(axes[0], summary_b, "Sanguino-Gómez and Krugers, 2024", "A")
_plot_panel(axes[1], summary_a, "Sanguino-Gómez et al., 2024", "B")
_plot_panel(axes[2], summary_all, "Combined datasets", "C")

# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_3_freezing.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_3_freezing.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 3 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 3 ready.")

# %%

