#!/usr/bin/env python
"""Quick visuals for aya_robust_v3 model results.

Generates:
1. JEN OOF confusion matrix (counts + normalised)
2. AYA prediction distribution (bar chart per class)
3. AYA confidence histogram
4. Feature shift comparison: v2 vs v3 (before/after coord preprocessing)
5. Per-recording ethogram strips
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_DIR = repo_root / "results" / "model_training" / "models" / "xgb_behavior" / "aya_robust_v3"
PRED_DIR = repo_root / "results" / "predictions" / "matched_dlc_aya_to_jen" / "aya_robust_v3"
OUT_DIR = repo_root / "results" / "manuscript_figures" / "v3_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_COLORS = {
    "Freezing":    "#3B82F6",
    "Sniffing":    "#F59E0B",
    "Grooming":    "#10B981",
    "Turn":        "#EF4444",
    "Locomotion":  "#8B5CF6",
    "Climbing":    "#EC4899",
    "Jump":        "#06B6D4",
    "Unassigned":  "#9CA3AF",
}
CLASS_ORDER = ["Freezing", "Climbing", "Sniffing", "Grooming", "Locomotion", "Turn", "Jump", "Unassigned"]


def fig1_confusion_matrix():
    """JEN out-of-fold confusion matrix."""
    counts_path = MODEL_DIR / "cv_confusion_matrix_oof_counts.csv"
    norm_path = MODEL_DIR / "cv_confusion_matrix_oof_normalized.csv"
    if not counts_path.exists():
        print("  Skipping confusion matrix (no counts CSV)")
        return
    cm_counts = pd.read_csv(counts_path, index_col=0)
    cm_norm = pd.read_csv(norm_path, index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200, facecolor="white")
    sns.heatmap(cm_counts, annot=True, fmt="g", cmap="Blues", cbar=False,
                linewidths=0.3, ax=axes[0])
    axes[0].set_title("JEN OOF Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                linewidths=0.3, vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("JEN OOF Confusion Matrix (Row-normalised)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"jen_oof_confusion_matrix.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: jen_oof_confusion_matrix.png/.pdf")


def fig2_aya_prediction_distribution():
    """Bar chart: predicted class distribution across all AYA recordings."""
    pred_path = PRED_DIR / "predictions_frame.parquet"
    if not pred_path.exists():
        print("  Skipping prediction distribution (no parquet)")
        return
    df = pd.read_parquet(pred_path)

    counts = df["pred_class"].value_counts()
    counts = counts.reindex([c for c in CLASS_ORDER if c in counts.index])
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor="white")
    bars = ax.bar(pcts.index, pcts.values,
                  color=[BEHAVIOR_COLORS.get(c, "#999") for c in pcts.index],
                  edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of total frames")
    ax.set_title("AYA Predicted Behavior Distribution (v3 model)", fontweight="bold")
    ax.set_ylim(0, max(pcts.values) * 1.15)
    sns.despine()
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"aya_prediction_distribution.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: aya_prediction_distribution.png/.pdf")


def fig3_confidence_histogram():
    """Histogram of prediction confidence across AYA recordings."""
    pred_path = PRED_DIR / "predictions_frame.parquet"
    if not pred_path.exists():
        return
    df = pd.read_parquet(pred_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200, facecolor="white")

    # Overall confidence
    axes[0].hist(df["pred_confidence"], bins=50, color="#3B82F6", alpha=0.8, edgecolor="white")
    axes[0].axvline(0.5, color="red", ls="--", lw=1, label="0.5 threshold")
    low_pct = (df["pred_confidence"] < 0.5).mean() * 100
    axes[0].set_title(f"Overall Confidence ({low_pct:.1f}% < 0.5)", fontweight="bold")
    axes[0].set_xlabel("Prediction Confidence")
    axes[0].set_ylabel("Frame Count")
    axes[0].legend()

    # Per-class confidence
    for cls in CLASS_ORDER:
        subset = df.loc[df["pred_class"] == cls, "pred_confidence"]
        if len(subset) > 0:
            axes[1].hist(subset, bins=30, alpha=0.5, label=cls,
                         color=BEHAVIOR_COLORS.get(cls, "#999"))
    axes[1].set_title("Confidence by Predicted Class", fontweight="bold")
    axes[1].set_xlabel("Prediction Confidence")
    axes[1].legend(fontsize=7, ncol=2)
    sns.despine()
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"aya_confidence_histogram.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: aya_confidence_histogram.png/.pdf")


def fig4_feature_shift():
    """Compare JEN vs AYA normalised feature distributions for v3.

    Uses the saved model-aligned features for AYA and recomputes JEN features
    from the cached training data.
    """
    aya_path = PRED_DIR / "computed_features_model_aligned.parquet"
    jen_cache = repo_root / "results" / "model_training" / "feature_cache"
    jen_parquets = sorted(jen_cache.glob("features_*.parquet"))

    if not aya_path.exists():
        print("  Skipping feature shift (no AYA features)")
        return
    if not jen_parquets:
        print("  Skipping feature shift (no JEN cached features)")
        return

    aya_feat = pd.read_parquet(aya_path)
    jen_all = pd.read_parquet(jen_parquets[0])
    # Keep only non-augmented JEN recordings for fair comparison
    jen_orig = jen_all[~jen_all["recording"].str.contains("_aug", na=False)].copy()
    meta_cols = {"behavior_cluster", "recording", "group", "frame"}
    feat_cols = [c for c in aya_feat.columns if c not in meta_cols and c in jen_orig.columns]

    if not feat_cols:
        print("  Skipping feature shift (no overlapping feature columns)")
        return

    # Compute IQR-normalised shift for each feature
    shifts = {}
    for col in feat_cols:
        jen_vals = jen_orig[col].dropna().to_numpy(dtype=float)
        aya_vals = aya_feat[col].dropna().to_numpy(dtype=float)
        if len(jen_vals) < 10 or len(aya_vals) < 10:
            continue
        jen_med = np.median(jen_vals)
        aya_med = np.median(aya_vals)
        jen_iqr = np.subtract(*np.percentile(jen_vals, [75, 25]))
        scale = jen_iqr if jen_iqr > 1e-6 else 1e-6
        shifts[col] = abs(jen_med - aya_med) / scale

    if not shifts:
        print("  Skipping feature shift (no computable shifts)")
        return

    shift_df = pd.DataFrame([
        {"feature": k, "iqr_shift": v} for k, v in shifts.items()
    ]).sort_values("iqr_shift", ascending=False)

    # Summary stats
    well_aligned = (shift_df["iqr_shift"] < 0.5).sum()
    moderate = ((shift_df["iqr_shift"] >= 0.5) & (shift_df["iqr_shift"] < 2)).sum()
    severe = (shift_df["iqr_shift"] >= 2).sum()
    total = len(shift_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200, facecolor="white")

    # Histogram of shifts
    axes[0].hist(shift_df["iqr_shift"].clip(upper=10), bins=40, color="#3B82F6",
                 alpha=0.8, edgecolor="white")
    axes[0].axvline(0.5, color="green", ls="--", lw=1.5, label=f"< 0.5 (aligned): {well_aligned}/{total}")
    axes[0].axvline(2.0, color="red", ls="--", lw=1.5, label=f"> 2.0 (severe): {severe}/{total}")
    axes[0].set_xlabel("Median Shift (IQR units)")
    axes[0].set_ylabel("Feature Count")
    axes[0].set_title("v3 Feature Distribution Shift (JEN vs AYA)", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Top-20 worst shifts
    top20 = shift_df.head(20)
    colors = ["#EF4444" if v >= 2 else "#F59E0B" if v >= 0.5 else "#10B981"
              for v in top20["iqr_shift"]]
    axes[1].barh(range(len(top20)), top20["iqr_shift"].values, color=colors)
    axes[1].set_yticks(range(len(top20)))
    axes[1].set_yticklabels(top20["feature"].values, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median Shift (IQR units)")
    axes[1].set_title("Top 20 Most-Shifted Features", fontweight="bold")
    sns.despine()
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"feature_shift_v3.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save CSV
    shift_df.to_csv(OUT_DIR / "feature_shift_v3.csv", index=False)
    print(f"  Saved: feature_shift_v3.png/.pdf/.csv  "
          f"(aligned={well_aligned}, moderate={moderate}, severe={severe} / {total})")


def fig5_ethograms():
    """Per-recording ethogram strips showing predicted behavior over time."""
    pred_path = PRED_DIR / "predictions_frame.parquet"
    if not pred_path.exists():
        return
    df = pd.read_parquet(pred_path)
    recordings = df["recording"].unique()

    fig, axes = plt.subplots(len(recordings), 1, figsize=(14, 1.2 * len(recordings)),
                             dpi=200, facecolor="white")
    if len(recordings) == 1:
        axes = [axes]

    for ax, rec in zip(axes, recordings):
        rec_df = df[df["recording"] == rec].sort_values("frame")
        classes = rec_df["pred_class"].values
        n = len(classes)
        # Build colour array
        colors_arr = np.array([
            tuple(int(BEHAVIOR_COLORS.get(c, "#9CA3AF").lstrip("#")[i:i+2], 16) / 255
                  for i in (0, 2, 4))
            for c in classes
        ])  # (N, 3)
        # Plot as image
        img = colors_arr[np.newaxis, :, :]  # (1, N, 3)
        ax.imshow(img, aspect="auto", interpolation="nearest",
                  extent=[0, n, 0, 1])
        ax.set_yticks([])
        short_name = rec.split("DLC")[0].replace("_", " ").strip()
        ax.set_ylabel(short_name, fontsize=7, rotation=0, ha="right", va="center")
        if ax != axes[-1]:
            ax.set_xticks([])
    axes[-1].set_xlabel("Frame")

    # Legend
    patches = [mpatches.Patch(color=BEHAVIOR_COLORS[c], label=c) for c in CLASS_ORDER
               if c in BEHAVIOR_COLORS]
    fig.legend(handles=patches, loc="upper center", ncol=len(patches), fontsize=7,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("AYA Predicted Behavior Ethograms (v3)", fontweight="bold", y=1.05)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"aya_ethograms.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: aya_ethograms.png/.pdf")


def fig6_normalization_qc():
    """Normalization QC from feature-store manifest diagnostics."""
    store_root = repo_root / "results" / "model_training" / "feature_store"
    manifests = sorted(store_root.glob("*/manifest.parquet"))
    if not manifests:
        print("  Skipping normalization QC (no feature-store manifests found)")
        return

    rows = []
    for m in manifests:
        try:
            df = pd.read_parquet(m)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "dataset_id", m.parent.name)
        rows.append(df)
    if not rows:
        print("  Skipping normalization QC (manifest read failed)")
        return

    all_df = rows[0].copy() if len(rows) == 1 else pd.concat(rows, ignore_index=True)
    all_df["ref_dist_px"] = pd.to_numeric(all_df.get("ref_dist_px"), errors="coerce")
    all_df["finite_ratio"] = pd.to_numeric(all_df.get("finite_ratio"), errors="coerce")
    all_df["x_range_px"] = pd.to_numeric(all_df.get("x_range_px"), errors="coerce")
    all_df["y_range_px"] = pd.to_numeric(all_df.get("y_range_px"), errors="coerce")

    inf_df = all_df[all_df["variant"].astype(str) == "inference_raw"].copy()
    if inf_df.empty:
        inf_df = all_df.copy()

    summary = (
        inf_df.groupby("dataset_id")[["ref_dist_px", "finite_ratio", "x_range_px", "y_range_px"]]
        .agg(["count", "mean", "median", "min", "max"])
    )
    summary.columns = ["_".join([str(x) for x in col]).strip("_") for col in summary.columns]
    summary = summary.reset_index().sort_values("dataset_id")
    summary_csv = OUT_DIR / "normalization_qc_summary_v3.csv"
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=220, facecolor="white")
    hist_vals = inf_df["ref_dist_px"].dropna()
    if len(hist_vals) > 0:
        axes[0].hist(hist_vals, bins=40, color="#3B82F6", alpha=0.82, edgecolor="white")
    axes[0].set_title("Reference Body Distance (px) Distribution", fontweight="bold")
    axes[0].set_xlabel("ref_dist_px")
    axes[0].set_ylabel("Count")

    scatter_df = inf_df.dropna(subset=["x_range_px", "y_range_px"])
    if not scatter_df.empty:
        sc = axes[1].scatter(
            scatter_df["x_range_px"],
            scatter_df["y_range_px"],
            c=scatter_df["finite_ratio"].fillna(0.0),
            cmap="viridis",
            s=36,
            alpha=0.85,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label("finite_ratio")
    axes[1].set_title("Coordinate Range Before Feature Extraction", fontweight="bold")
    axes[1].set_xlabel("x_range_px")
    axes[1].set_ylabel("y_range_px")

    sns.despine()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"normalization_qc_v3.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: normalization_qc_v3.png/.pdf")
    print(f"  Saved: {summary_csv.name}")


def main():
    print(f"Output directory: {OUT_DIR}")
    print()
    print("1. JEN OOF confusion matrix")
    fig1_confusion_matrix()
    print("2. AYA prediction distribution")
    fig2_aya_prediction_distribution()
    print("3. AYA confidence histogram")
    fig3_confidence_histogram()
    print("4. Feature shift analysis (v3)")
    fig4_feature_shift()
    print("5. AYA ethograms")
    fig5_ethograms()
    print("6. Normalization QC")
    fig6_normalization_qc()
    print("\nDone.")


if __name__ == "__main__":
    main()
