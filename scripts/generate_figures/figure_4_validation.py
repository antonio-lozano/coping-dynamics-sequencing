"""
Figure 4 — Validation of MoSeq freezing syllables against SimBA freezing.

Loads raw MoSeq syllables and SimBA freezing labels, computes per-syllable
precision/recall for freezing, and summarizes overlap for key syllables.
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
import re
import warnings

from src.config import RESULTS_RAW_PKL, FREEZING_DIR, INDEX_CSV, FPS, BIN_SECONDS, MANUSCRIPT_FIGURES_DIR as RESULTS_DIR, PALETTE

results_pkl = RESULTS_RAW_PKL
freezing_dir = FREEZING_DIR
index_csv = INDEX_CSV
fps = FPS
bin_seconds = BIN_SECONDS
bin_size = fps * bin_seconds
excluded_animals_fig4 = {"Animal_48_6", "48_6"}
excluded_tokens_fig4 = {_normalize_name.strip().replace(" ", "_").lower() for _normalize_name in excluded_animals_fig4}
warnings.warn(
    f"Figure 4 uses bin_seconds={bin_seconds} (from config/original pipeline setting).",
    RuntimeWarning,
)
warnings.warn(
    "Figure 4 excludes known bad subject(s): " + ", ".join(sorted(excluded_animals_fig4)),
    RuntimeWarning,
)
warnings.warn(
    "Precision/recall uses matched MoSeq+freezing frames with per-recording truncation to min(len(syllable), len(freezing)).",
    RuntimeWarning,
)


def _is_excluded_recording(name: str) -> bool:
    norm = _normalize(str(name)).lower()
    return norm in excluded_tokens_fig4

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
total_freeze_frames = 0
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    rec_raw = str(rec).strip()
    if _is_excluded_recording(rec_raw):
        continue
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
    total_freeze_frames += int(np.sum(freeze == 1))
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
                "tp": tp,
                "fp": fp,
                "fn": fn,
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
if (metrics_df["group"] == "Unknown").any():
    unknown_n = int(metrics_df.loc[metrics_df["group"] == "Unknown", "recording"].nunique())
    warnings.warn(
        f"Figure 4 includes {unknown_n} Unknown-group recording(s) in overall syllable metrics (panel A).",
        RuntimeWarning,
    )

#%%
# Aggregate across recordings
print("Aggregating precision/recall across recordings...")
agg = (
    metrics_df.groupby("syllable")[["precision", "recall", "frames", "tp", "fp", "fn"]]
    .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
    .reset_index()
)
total_frames = agg["frames"].sum()
agg["freezing_contrib"] = agg["tp"] / total_freeze_frames if total_freeze_frames > 0 else np.nan
agg["frame_frac"] = agg["frames"] / total_frames if total_frames > 0 else np.nan
agg = agg.sort_values("precision", ascending=False)

group_agg = (
    metrics_df.groupby(["syllable", "group"])[["precision", "recall", "frames", "tp", "fp", "fn"]]
    .agg({"precision": "mean", "recall": "mean", "frames": "sum", "tp": "sum", "fp": "sum", "fn": "sum"})
    .reset_index()
)

#%%
# Debug cell: inspect syllable 61 precision/recall statistics
debug_syll = 61
print(f"=== DEBUG S{debug_syll} ===")

s_row = agg[agg["syllable"] == debug_syll]
if s_row.empty:
    print(f"S{debug_syll} not found in aggregated metrics.")
else:
    r = s_row.iloc[0]
    print(
        f"S{debug_syll} overall: "
        f"precision_mean={r['precision']:.6f}, recall_mean={r['recall']:.6f}, "
        f"frames={int(r['frames'])}, tp={int(r['tp'])}, fp={int(r['fp'])}, fn={int(r['fn'])}"
    )
    weighted_precision = (r["tp"] / (r["tp"] + r["fp"])) if (r["tp"] + r["fp"]) > 0 else np.nan
    weighted_recall = (r["tp"] / (r["tp"] + r["fn"])) if (r["tp"] + r["fn"]) > 0 else np.nan
    print(
        f"S{debug_syll} weighted from summed counts: "
        f"precision={weighted_precision:.6f}, recall={weighted_recall:.6f}"
    )

s_group = group_agg[group_agg["syllable"] == debug_syll].copy()
if s_group.empty:
    print(f"S{debug_syll} group rows: none")
else:
    s_group = s_group.sort_values("group")
    print(f"S{debug_syll} by group:")
    print(
        s_group[
            ["group", "precision", "recall", "frames", "tp", "fp", "fn"]
        ].to_string(index=False)
    )

s_rec = metrics_df[metrics_df["syllable"] == debug_syll].copy()
print(f"S{debug_syll} recording-level rows: {len(s_rec)}")
if not s_rec.empty:
    print(
        "S61 recording precision summary: "
        f"mean={s_rec['precision'].mean():.6f}, median={s_rec['precision'].median():.6f}, "
        f"min={s_rec['precision'].min():.6f}, max={s_rec['precision'].max():.6f}"
    )
print("=== END DEBUG S61 ===")

#%%
# Debug cell: frame-level freezing labels at S61 frames
print(f"=== DEBUG S{debug_syll} FRAME-LEVEL MATCHES ===")
s61_frame_rows = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    rec_raw = str(rec).strip()
    if _is_excluded_recording(rec_raw):
        continue
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    freeze = (
        freezing_records.get(rec_raw)
        or freezing_records.get(rec_norm)
        or freezing_records.get(rec_prefix)
    )
    if freeze is None:
        continue
    syll = np.array(data["syllable"])
    n = min(len(syll), len(freeze))
    if n <= 0:
        continue
    syll = syll[:n]
    freeze = np.array(freeze[:n], dtype=int)
    idx = np.flatnonzero(syll == debug_syll)
    if idx.size == 0:
        continue
    grp = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    for f in idx:
        s61_frame_rows.append(
            {
                "recording": rec_raw,
                "group": grp,
                "frame": int(f),
                "freeze_label": int(freeze[f]),
            }
        )

s61_frames_df = pd.DataFrame(s61_frame_rows)
if s61_frames_df.empty:
    print(f"No frame-level matches found for S{debug_syll}.")
else:
    print(f"S{debug_syll} frame-level rows: {len(s61_frames_df)}")
    print(f"Recordings containing S{debug_syll}: {s61_frames_df['recording'].nunique()}")
    print("Freeze label counts at S61 frames:")
    print(s61_frames_df["freeze_label"].value_counts(dropna=False).sort_index())
    print("Freeze label fractions at S61 frames:")
    print((s61_frames_df["freeze_label"].value_counts(normalize=True).sort_index()).round(6))

    rec_summary = (
        s61_frames_df.groupby(["recording", "group"])["freeze_label"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "s61_frames", "mean": "freeze_rate_at_s61"})
        .sort_values(["group", "recording"])
    )
    print("Per-recording S61 summary (first 20 rows):")
    print(rec_summary.head(20).to_string(index=False))

    print("Sample frame-level rows (first 40):")
    print(s61_frames_df.sort_values(["recording", "frame"]).head(40).to_string(index=False))

    out_csv = RESULTS_DIR / f"debug_s{debug_syll}_frame_freeze_matches.csv"
    s61_frames_df.sort_values(["recording", "frame"]).to_csv(out_csv, index=False)
    print(f"Saved full S{debug_syll} frame-level table to: {out_csv}")
print(f"=== END DEBUG S{debug_syll} FRAME-LEVEL MATCHES ===")

# Syllables exclusive to ELS (present in ELS recordings but never in Control)
group_counts = (
    metrics_df.groupby(["syllable", "group"])["recording"].nunique().unstack(fill_value=0)
    if not metrics_df.empty
    else pd.DataFrame()
)
els_only = set()
if not group_counts.empty and "ELS" in group_counts.columns:
    control_counts = group_counts["Control"] if "Control" in group_counts.columns else 0
    els_only = set(group_counts[(control_counts == 0) & (group_counts["ELS"] > 0)].index.tolist())

# Overlap per animal: % of SimBA freezing frames covered by S0/S28
print("Computing per-animal overlap between SimBA freezing and S0/S28...")
overlap_rows = []
target_syll = {0, 28}
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    rec_raw = str(rec).strip()
    if _is_excluded_recording(rec_raw):
        continue
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    freeze = (
        freezing_records.get(rec_raw)
        or freezing_records.get(rec_norm)
        or freezing_records.get(rec_prefix)
    )
    if freeze is None:
        continue
    syll = np.array(data["syllable"])
    n = min(len(syll), len(freeze))
    syll = syll[:n]
    freeze = np.array(freeze[:n])
    freeze_mask = freeze == 1
    if freeze_mask.sum() == 0:
        continue
    overlap = np.mean((np.isin(syll, list(target_syll)) & freeze_mask)[freeze_mask])
    overlap_rows.append(
        {
            "recording": rec_raw,
            "group": group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown"),
            "overlap": overlap * 100.0,
        }
    )

overlap_df = pd.DataFrame(overlap_rows)

# Time-course of S0+S28 percentage per bin
print("Computing S0+S28 time-course per animal...")
bin_counts = []
seq_by_rec = []
for rec, data in results_dict.items():
    if "syllable" not in data:
        continue
    if _is_excluded_recording(str(rec).strip()):
        continue
    seq = np.array(data["syllable"])
    bins = len(seq) // bin_size
    if bins > 0:
        bin_counts.append(bins)
        seq_by_rec.append((rec, seq))
if not bin_counts:
    raise SystemExit("No recordings with sufficient length for binning.")
common_bins = min(bin_counts)

time_rows = []
for rec, seq in seq_by_rec:
    seq = seq[: common_bins * bin_size]
    arr = seq.reshape(common_bins, bin_size)
    rec_raw = str(rec).strip()
    rec_norm = _normalize(rec_raw)
    rec_prefix = rec_norm.split("DLC")[0].split("_resnet")[0].rstrip("_")
    group = group_map.get(rec_raw) or group_map.get(rec_norm) or group_map.get(rec_prefix, "Unknown")
    if group not in ("Control", "ELS"):
        continue
    for b, block in enumerate(arr):
        pct = np.mean(np.isin(block, list(target_syll))) * 100.0
        time_rows.append({"recording": rec_raw, "group": group, "bin": b, "pct": pct})

time_df = pd.DataFrame(time_rows)
time_summary = (
    time_df.groupby(["group", "bin"])["pct"]
    .agg(["mean", "sem", "count"])
    .reset_index()
)
# Use bin end-time so first 30s bin is plotted at 0.5 min.
time_summary["time_min"] = (time_summary["bin"] + 1.0) * (bin_seconds / 60.0)

#%%
# Plot Figure 4 panels (precision and recall)
print("Plotting Figure 4 panels (A–E)...")
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Build custom layout using normalized axes positions (from top-origin spec)
fig = plt.figure(figsize=(13, 12), dpi=300, facecolor="#FFFFFF")
axA = fig.add_axes([0.07, 0.72, 0.88, 0.24])
axB = fig.add_axes([0.07, 0.40, 0.42, 0.22])
axC = fig.add_axes([0.58, 0.40, 0.36, 0.22])
axD = fig.add_axes([0.07, 0.07, 0.54, 0.22])
axE = fig.add_axes([0.70, 0.07, 0.24, 0.22])

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.8, length=3, labelsize=7)

def _panel_tag(ax, tag):
    ax.text(-0.08, 1.04, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", color="#4b4b4b")

# Panel A: Total frames covered by syllable (%) sorted by count
aggA = agg.sort_values("frames", ascending=False)
syll_labels_A = [f"S{s}" for s in aggA["syllable"].astype(int)]
axA.bar(
    np.arange(len(aggA)),
    aggA["frame_frac"] * 100,
    color="#ACDBCA",
    edgecolor="#FFFFFF",
    linewidth=0.6,
)
axA.set_title("Total frames covered by syllable (%)", fontsize=11, color="#4b4b4b")
axA.set_ylabel("Total frames covered by syllable (%)")
axA.set_xlabel("Syllables (sorted by count)")
axA.set_xticks(np.arange(len(aggA)))
axA.set_xticklabels(syll_labels_A, rotation=90, fontsize=6)
axA.set_ylim(0, 20)
axA.set_yticks([0, 5, 10, 15, 20])

# Threshold at 0.05 (5%)
threshold = 0.05
cut_idx = int(np.sum(aggA["frame_frac"] >= threshold))
axA.axvline(cut_idx - 0.5, color="#cfcfcf", linestyle="--", linewidth=1)
axA.legend([
    plt.Line2D([0], [0], color="#cfcfcf", linestyle="--", linewidth=1)
], ["Threshold = 0.05"], frameon=True, facecolor="#FFFFFF", edgecolor="#cfcfcf", loc="upper right")

_style_axes(axA)
_panel_tag(axA, "A")

# Panel B: Precision (All vs Control vs ELS)
precision_order = [
    0, 28, 40, 61, 68, 51, 48, 53, 38, 44, 4, 13, 56, 79, 39, 36, 60, 76, 42, 20, 35, 50, 85, 1, 47, 26
]
syll_order = [f"S{s}" for s in precision_order if s in agg["syllable"].values]
show_prec = agg[agg["syllable"].isin(precision_order)].set_index("syllable").reindex(precision_order).dropna().reset_index()
s61_in_agg = bool((agg["syllable"] == 61).any())
s61_in_show = bool((show_prec["syllable"] == 61).any())
if not s61_in_agg:
    print("S61 diagnostic: absent from aggregated metrics (not present in matched/retained recordings).")
elif not s61_in_show:
    s61_row = agg.loc[agg["syllable"] == 61].iloc[0]
    print(
        "S61 diagnostic: present in agg but dropped from panel B view "
        f"(precision={s61_row['precision']}, recall={s61_row['recall']})."
    )
else:
    print("S61 diagnostic: present and plotted in panel B.")

def _group_vals(metric):
    overall = show_prec.set_index("syllable")[metric].reindex(show_prec["syllable"]).to_numpy()
    control = (
        group_agg[group_agg["group"] == "Control"]
        .set_index("syllable")[metric]
        .reindex(show_prec["syllable"])
        .to_numpy()
    )
    els = (
        group_agg[group_agg["group"] == "ELS"]
        .set_index("syllable")[metric]
        .reindex(show_prec["syllable"])
        .to_numpy()
    )
    return overall, control, els

prec_all, prec_ctrl, prec_els = _group_vals("precision")
prec_all = np.array(prec_all) * 100
prec_ctrl = np.array(prec_ctrl) * 100
prec_els = np.array(prec_els) * 100
x = np.arange(len(show_prec))
w = 0.25
axB.bar(x - w, prec_all, width=w, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.4, label="All Groups")
axB.bar(x, prec_els, width=w, color="#BE7896", edgecolor="#FFFFFF", linewidth=0.4, label="ELS")
axB.bar(x + w, prec_ctrl, width=w, color="#F0BE50", edgecolor="#FFFFFF", linewidth=0.4, label="Control")
axB.set_title("Mean Precision (%)", fontsize=11, color="#4b4b4b")
axB.set_ylabel("Mean Precision (%)")
axB.set_xlabel("Syllable")
axB.set_xticks(x)
axB.set_xticklabels([f"S{s}" for s in show_prec["syllable"].astype(int)], rotation=90, fontsize=7)
axB.set_ylim(0, 100)
axB.set_yticks([0, 20, 40, 60, 80, 100])
axB.legend(frameon=False, fontsize=9, loc="upper right")
_style_axes(axB)
_panel_tag(axB, "B")

# Panel C: Recall (All vs Control vs ELS)
recall_order = [0, 1, 28, 13, 4, 40]
show_rec = agg[agg["syllable"].isin(recall_order)].set_index("syllable").reindex(recall_order).dropna().reset_index()
syll_order_rec = [f"S{s}" for s in show_rec["syllable"].astype(int)]
rec_all, rec_ctrl, rec_els = _group_vals("recall")
rec_all = np.array(rec_all) * 100
rec_ctrl = np.array(rec_ctrl) * 100
rec_els = np.array(rec_els) * 100
xr = np.arange(len(show_rec))
axC.bar(xr - w, rec_all[: len(show_rec)], width=w, color="#4B4B4B", edgecolor="#FFFFFF", linewidth=0.4, label="All Groups")
axC.bar(xr, rec_els[: len(show_rec)], width=w, color="#BE7896", edgecolor="#FFFFFF", linewidth=0.4, label="ELS")
axC.bar(xr + w, rec_ctrl[: len(show_rec)], width=w, color="#F0BE50", edgecolor="#FFFFFF", linewidth=0.4, label="Control")
axC.set_title("Mean Recall (%)", fontsize=11, color="#4b4b4b")
axC.set_ylabel("Mean Recall (%)")
axC.set_xlabel("Syllable")
axC.set_xticks(xr)
axC.set_xticklabels(syll_order_rec, rotation=90, fontsize=7)
axC.set_ylim(0, 80)
axC.set_yticks([0, 20, 40, 60, 80])
axC.legend(frameon=False, fontsize=9, loc="upper right")
_style_axes(axC)
_panel_tag(axC, "C")

# Panel D: Overlap per animal bars split by group
palette = PALETTE  # Use centralized palette
if not overlap_df.empty:
    ctrl = overlap_df[overlap_df["group"] == "Control"].sort_values("recording")
    els = overlap_df[overlap_df["group"] == "ELS"].sort_values("recording")
    overlap_plot = pd.concat([ctrl, els], ignore_index=True)
    colors = [palette.get(g, "#cccccc") for g in overlap_plot["group"]]
    axD.bar(np.arange(len(overlap_plot)), overlap_plot["overlap"], color=colors, edgecolor="#FFFFFF", linewidth=0.3)
    axD.set_xticks(np.arange(len(overlap_plot)))
    def _short_id(name: str) -> str:
        norm = _normalize(str(name))
        m = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", norm, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        parts = norm.split("_")
        return "_".join(parts[-2:]) if len(parts) >= 2 else norm
    axD.set_xticklabels([_short_id(n) for n in overlap_plot["recording"].tolist()], rotation=90, fontsize=6)
axD.set_title("D. Overlap (%)", fontsize=11, color="#4b4b4b")
axD.set_ylabel("Overlap between unsupervised and supervised\nfreezing labels (% of frames)")
axD.set_xlabel("")
axD.set_ylim(0, 100)
axD.set_yticks([0, 20, 40, 60, 80, 100])
axD.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=palette["Control"], label="Control"),
    plt.Rectangle((0, 0), 1, 1, color=palette["ELS"], label="ELS"),
], frameon=False, loc="upper right", fontsize=9)
_style_axes(axD)
_panel_tag(axD, "D")

# Panel E: Time course of S0+S28 percentage
span_width_min = 0.5  # 30 seconds wide in minutes
for span in [3.0, 4.5, 6.0]:
    axE.axvspan(span, span + span_width_min, color="#F2F2F2", zorder=0)

for group, data in time_summary.groupby("group"):
    if group not in ("Control", "ELS"):
        continue
    axE.plot(
        data["time_min"],
        data["mean"],
        label=group,
        color=palette.get(group, "#4d4d4d"),
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axE.fill_between(
        data["time_min"],
        data["mean"] - data["sem"],
        data["mean"] + data["sem"],
        color=palette.get(group, "#4d4d4d"),
        alpha=0.2,
        linewidth=0,
    )
axE.set_title("Syllables 0, 28", fontsize=11, color="#4b4b4b")
axE.set_xlabel("Time (minutes)")
axE.set_ylabel("Freezing (% of time)")
axE.set_xlim(0, 8)
axE.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
axE.set_ylim(0, 80)
axE.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
axE.legend(frameon=False, loc="lower right", fontsize=9)
axE.text(4.0, 76, "*", ha="center", va="center", color="#4b4b4b", fontsize=12)
_style_axes(axE)
_panel_tag(axE, "E")

sns.despine(fig=fig)
# Save high-quality outputs (before show to avoid blank figures)
fig.savefig(RESULTS_DIR / "figure_4_validation.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_4_validation.pdf", bbox_inches="tight", facecolor="white")
print(f"Figure 4 saved to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("Figure 4 ready.")

# %%

