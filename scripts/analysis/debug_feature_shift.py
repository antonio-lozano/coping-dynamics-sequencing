"""
Debug script: compare feature distributions between JEN (training) and AYA (inference)
to assess domain shift and generalization risk.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd

# ============================================================
# 1. Load AYA raw features (before normalization)
# ============================================================
aya_raw_path = REPO_ROOT / "results/predictions/aya_robust_v2/robust_light_no_tail_v2/computed_features_raw.csv"
aya_aligned_path = REPO_ROOT / "results/predictions/aya_robust_v2/robust_light_no_tail_v2/computed_features_model_aligned.csv"
norm_stats_path = REPO_ROOT / "results/model_training/models/xgb_behavior/robust_light_no_tail_v2/normalization_stats.json"

print("=" * 70)
print("FEATURE DISTRIBUTION SHIFT ANALYSIS: JEN (train) vs AYA (inference)")
print("=" * 70)

aya_raw = pd.read_csv(aya_raw_path)
aya_aligned = pd.read_csv(aya_aligned_path)
print(f"\nAYA raw features:     {aya_raw.shape}")
print(f"AYA aligned features: {aya_aligned.shape}")

# ============================================================
# 2. Load JEN normalization stats (median/IQR from training)
# ============================================================
with open(norm_stats_path) as f:
    norm_stats = json.load(f)

# norm_stats is {feature_name: {median, q25, q75, iqr, scale_used}, ...}
feature_names = list(norm_stats.keys())
jen_medians = {k: v["median"] for k, v in norm_stats.items()}
jen_iqrs = {k: v["iqr"] for k, v in norm_stats.items()}
print(f"JEN normalization stats for {len(feature_names)} features")

# ============================================================
# 3. Also extract JEN features live for a deeper comparison
# ============================================================
from scripts.analysis.extended_features import compute_extended_features
from src.ml.pose_features import compute_kinematic_features, load_dlc_csv

jen_csv_dir = REPO_ROOT / "data" / "to_predict" / "test"
jen_csvs = sorted(jen_csv_dir.glob("*.csv"))[:5]  # sample 5 JEN recordings

print(f"\nExtracting JEN features from {len(jen_csvs)} recordings for direct comparison...")
jen_frames = []
for csv_path in jen_csvs:
    pose_df = load_dlc_csv(csv_path)
    base_feats = compute_kinematic_features(pose_df, fps=25, window=15)
    ext_feats = compute_extended_features(pose_df, base_feats, fps=25)
    combined = pd.concat([base_feats, ext_feats], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined["recording"] = csv_path.stem
    jen_frames.append(combined)
    print(f"  {csv_path.stem}: {len(combined)} frames, {combined.shape[1]} features")

jen_raw = pd.concat(jen_frames, ignore_index=True)
print(f"\nJEN raw features (sampled): {jen_raw.shape}")

# ============================================================
# 4. Compare distributions feature-by-feature
# ============================================================
meta_cols = {"frame", "time_s", "recording"}
# Use features present in both
common_features = [c for c in feature_names if c in aya_raw.columns and c in jen_raw.columns]
print(f"\nCommon features for comparison: {len(common_features)}")

# For each feature, compute: median, IQR, and Wasserstein-like shift
results = []
for feat in common_features:
    jen_vals = jen_raw[feat].dropna().to_numpy()
    aya_vals = aya_raw[feat].dropna().to_numpy()
    
    if len(jen_vals) == 0 or len(aya_vals) == 0:
        continue
    
    jen_med = np.median(jen_vals)
    aya_med = np.median(aya_vals)
    jen_iqr = np.subtract(*np.percentile(jen_vals, [75, 25]))
    aya_iqr = np.subtract(*np.percentile(aya_vals, [75, 25]))
    
    # Normalized shift: how many JEN IQRs is the AYA median away from JEN median
    if jen_iqr > 1e-10:
        median_shift_iqr = abs(aya_med - jen_med) / jen_iqr
    else:
        median_shift_iqr = float("inf") if abs(aya_med - jen_med) > 1e-10 else 0.0
    
    # IQR ratio: how different is the spread
    if jen_iqr > 1e-10:
        iqr_ratio = aya_iqr / jen_iqr
    else:
        iqr_ratio = float("inf") if aya_iqr > 1e-10 else 1.0
    
    # After normalization: what are AYA values in the normalized space?
    if feat in aya_aligned.columns:
        aya_norm_vals = aya_aligned[feat].dropna().to_numpy()
        aya_norm_med = np.median(aya_norm_vals)
        aya_norm_iqr = np.subtract(*np.percentile(aya_norm_vals, [75, 25]))
        frac_clipped = np.mean((aya_norm_vals <= -4) | (aya_norm_vals >= 4)) * 100
    else:
        aya_norm_med = np.nan
        aya_norm_iqr = np.nan
        frac_clipped = np.nan
    
    results.append({
        "feature": feat,
        "jen_median": jen_med,
        "aya_median": aya_med,
        "jen_iqr": jen_iqr,
        "aya_iqr": aya_iqr,
        "median_shift_iqr": median_shift_iqr,
        "iqr_ratio": iqr_ratio,
        "aya_norm_median": aya_norm_med,
        "aya_norm_iqr": aya_norm_iqr,
        "pct_clipped_at_boundary": frac_clipped,
    })

shift_df = pd.DataFrame(results).sort_values("median_shift_iqr", ascending=False)

# ============================================================
# 5. Report
# ============================================================
print("\n" + "=" * 70)
print("TOP 30 FEATURES WITH LARGEST DISTRIBUTION SHIFT (median shift in IQR units)")
print("=" * 70)
print(f"{'Feature':<45} {'JEN med':>10} {'AYA med':>10} {'Shift(IQR)':>10} {'IQR ratio':>10} {'%clipped':>8}")
print("-" * 95)
for _, row in shift_df.head(30).iterrows():
    print(f"{row['feature']:<45} {row['jen_median']:>10.3f} {row['aya_median']:>10.3f} {row['median_shift_iqr']:>10.2f} {row['iqr_ratio']:>10.2f} {row['pct_clipped_at_boundary']:>8.1f}%")

print("\n\n" + "=" * 70)
print("TOP 20 FEATURES WITH MOST CLIPPING IN NORMALIZED SPACE")
print("=" * 70)
clip_sorted = shift_df.sort_values("pct_clipped_at_boundary", ascending=False)
for _, row in clip_sorted.head(20).iterrows():
    print(f"{row['feature']:<45} clipped={row['pct_clipped_at_boundary']:>6.1f}% | norm_med={row['aya_norm_median']:>7.2f} | norm_iqr={row['aya_norm_iqr']:>7.2f}")

print("\n\n" + "=" * 70)
print("FEATURES WITH GOOD ALIGNMENT (shift < 0.5 IQR, IQR ratio 0.5-2.0)")
print("=" * 70)
good = shift_df[(shift_df["median_shift_iqr"] < 0.5) & (shift_df["iqr_ratio"] > 0.5) & (shift_df["iqr_ratio"] < 2.0)]
print(f"Well-aligned features: {len(good)} / {len(shift_df)} ({100*len(good)/max(1,len(shift_df)):.1f}%)")
for _, row in good.head(20).iterrows():
    print(f"  {row['feature']:<45} shift={row['median_shift_iqr']:.2f}  iqr_ratio={row['iqr_ratio']:.2f}")

print("\n\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total features compared:       {len(shift_df)}")
print(f"Median shift (IQR units):       {shift_df['median_shift_iqr'].median():.2f}")
print(f"Mean shift (IQR units):         {shift_df['median_shift_iqr'].mean():.2f}")
print(f"Features with shift > 2 IQR:    {(shift_df['median_shift_iqr'] > 2).sum()}")
print(f"Features with shift > 5 IQR:    {(shift_df['median_shift_iqr'] > 5).sum()}")
print(f"Features with shift > 10 IQR:   {(shift_df['median_shift_iqr'] > 10).sum()}")
print(f"Features with >10% clipping:    {(shift_df['pct_clipped_at_boundary'] > 10).sum()}")
print(f"Features with >25% clipping:    {(shift_df['pct_clipped_at_boundary'] > 25).sum()}")
print(f"Well-aligned (shift<0.5, IQR ratio 0.5-2x): {len(good)}")

# Group by feature type
def _feature_group(name):
    if name.startswith("ext_freq_"):
        return "frequency"
    if name.startswith("ext_postural_"):
        return "postural"
    if name.startswith("ext_temporal_"):
        return "temporal"
    if any(k in name for k in ["_roll_mean", "_roll_sd"]):
        return "rolling_stats"
    if "velocity" in name.lower() or "speed" in name.lower():
        return "velocity"
    if "distance" in name.lower() or "dist_" in name.lower():
        return "distance"
    if "angle" in name.lower() or "angular" in name.lower():
        return "angular"
    if "center" in name.lower():
        return "centroid"
    return "other"

shift_df["group"] = shift_df["feature"].map(_feature_group)
print(f"\n\nSHIFT BY FEATURE GROUP:")
print(f"{'Group':<20} {'Count':>6} {'Med shift':>10} {'Mean shift':>10} {'Pct>2IQR':>10} {'Med %clip':>10}")
for grp, gdf in shift_df.groupby("group"):
    n = len(gdf)
    med_s = gdf["median_shift_iqr"].median()
    mean_s = gdf["median_shift_iqr"].mean()
    pct_big = (gdf["median_shift_iqr"] > 2).sum() / n * 100
    med_clip = gdf["pct_clipped_at_boundary"].median()
    print(f"{grp:<20} {n:>6} {med_s:>10.2f} {mean_s:>10.2f} {pct_big:>9.0f}% {med_clip:>10.1f}%")

# Save full comparison
out_path = REPO_ROOT / "results" / "mapping_qc" / "feature_shift_jen_vs_aya.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
shift_df.to_csv(out_path, index=False)
print(f"\nFull comparison saved to {out_path}")
