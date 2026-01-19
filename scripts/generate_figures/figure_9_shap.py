"""
Figure 9 — SHAP-based explainability for kinematic feature contributions to
unsupervised behavior classification.

Multi-panel publication figure:
  Piece 1 (top): A (accuracy vs chance), B (global SHAP), C (confusion matrix), D (SHAP Freezing), E (SHAP Sniffing)
  Piece 2 (bottom): F (SHAP Grooming), G (SHAP Turn), H (SHAP Locomotion), I (SHAP Climbing), J (SHAP Jump), K (SHAP Unassigned)

Pipeline:
1) Load DLC pose data directly and compute kinematic features per frame.
2) Merge with MoSeq-derived behavior_cluster labels.
3) Train XGBoost multi-class classifier.
4) Report per-class accuracy with chance baseline.
5) Plot normalized confusion matrix.
6) Compute SHAP values; plot global top-20 feature importances (stacked by behavior).
7) Plot class-specific SHAP beeswarm/violin (top 10 features) for each behavior.
"""

#%%

# ==============================================================================
# QUICK MODE - Controlled by QUICK_MODE environment variable
# Set by run_all_figures.py --quick flag, or manually for testing
# ==============================================================================
import os
QUICK_MODE = os.environ.get("QUICK_MODE", "0") == "1"

#%% Imports and paths
print("Loading dependencies and setting paths...")
from pathlib import Path
import sys
import re
import warnings
warnings.filterwarnings("ignore")

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
import shap
import pickle

from src.config import (
    RESULTS_CLUSTERS_PKL, INDEX_CSV, FPS, BEHAVIOR_MAPPING,
    RESULTS_DIR, PALETTE, DATA_DIR, DLC_DIR
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
fps = FPS
label_col = "behavior_cluster"

# Behavior palette matching the visual spec
BEHAVIOR_PALETTE = {
    "Freezing": "#C37B9F",      # magenta/pink
    "Sniffing": "#4CB7A5",      # teal/blue-green  
    "Grooming": "#7EC8E3",      # light cyan/pale blue
    "Turn": "#8DC63F",          # lime green
    "Locomotion": "#F8C650",    # yellow/amber
    "Climbing": "#F4A259",      # orange
    "Jump": "#E4572E",          # red
    "Unassigned": "#888888",    # gray
}

# Body part mapping from DLC output
BODYPART_NAMES = {
    "nose": "Nose",
    "H1R": "Head Right 1",
    "H2R": "Head Right 2", 
    "H1L": "Head Left 1",
    "H2L": "Head Left 2",
    "B1R": "Right Body 1",
    "B2R": "Right Body 2",
    "B3R": "Right Body 3",
    "B1L": "Left Body 1",
    "B2L": "Left Body 2",
    "B3L": "Left Body 3",
    "tail": "Tail",
    "S2": "Spine 2",
    "S1": "Spine 1",
}

# ==============================================================================
# DLC DATA LOADING AND FEATURE COMPUTATION
# ==============================================================================

def _normalize_name(name: str) -> str:
    """Normalize recording name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def load_dlc_csv(csv_path: Path) -> pd.DataFrame:
    """Load a DLC CSV file and return (x, y) coordinates per bodypart per frame."""
    df = pd.read_csv(csv_path, header=[1, 2], index_col=0)
    # Flatten multi-index columns to bodypart_coord format
    df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]
    # Keep only x, y (drop likelihood)
    keep_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    return df[keep_cols].reset_index(drop=True)

def compute_kinematic_features(pose_df: pd.DataFrame, fps: int = 25) -> pd.DataFrame:
    """
    Compute kinematic features from DLC pose data.
    
    Features computed per frame:
    - Pairwise distances between all body parts
    - Per-bodypart velocities
    - Angular velocity of body axis
    - Summed velocity across all points
    - SD and mean of distances over sliding window
    """
    features = {}
    n_frames = len(pose_df)
    
    # Extract bodypart names from columns
    bodyparts = sorted(set(c.rsplit("_", 1)[0] for c in pose_df.columns))
    
    # Get x, y arrays for each bodypart
    coords = {}
    for bp in bodyparts:
        x_col = f"{bp}_x"
        y_col = f"{bp}_y"
        if x_col in pose_df.columns and y_col in pose_df.columns:
            coords[bp] = pose_df[[x_col, y_col]].values
    
    # 1. Pairwise distances between all body parts
    bp_list = list(coords.keys())
    for i, bp1 in enumerate(bp_list):
        for bp2 in bp_list[i+1:]:
            dist = np.sqrt(np.sum((coords[bp1] - coords[bp2])**2, axis=1))
            bp1_nice = BODYPART_NAMES.get(bp1, bp1)
            bp2_nice = BODYPART_NAMES.get(bp2, bp2)
            features[f"{bp1_nice} - {bp2_nice} Distance"] = dist
    
    # 2. Per-bodypart velocities (frame-to-frame displacement)
    for bp in bp_list:
        xy = coords[bp]
        vel = np.zeros(n_frames)
        vel[1:] = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1)) * fps
        bp_nice = BODYPART_NAMES.get(bp, bp)
        features[f"{bp_nice} Velocity"] = vel
    
    # 3. Angular velocity (using nose-tail axis)
    if "nose" in coords and "tail" in coords:
        nose = coords["nose"]
        tail = coords["tail"]
        angle = np.arctan2(nose[:, 1] - tail[:, 1], nose[:, 0] - tail[:, 0])
        ang_vel = np.zeros(n_frames)
        ang_diff = np.diff(angle)
        # Handle angle wrapping
        ang_diff = np.where(ang_diff > np.pi, ang_diff - 2*np.pi, ang_diff)
        ang_diff = np.where(ang_diff < -np.pi, ang_diff + 2*np.pi, ang_diff)
        ang_vel[1:] = np.abs(ang_diff) * fps
        features["Angular Velocity"] = ang_vel
        features["SD Angular Velocity"] = pd.Series(ang_vel).rolling(15, center=True, min_periods=1).std().values
    
    # 4. Summed velocity across all body points
    all_vel = np.zeros(n_frames)
    for bp in bp_list:
        xy = coords[bp]
        v = np.zeros(n_frames)
        v[1:] = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1)) * fps
        all_vel += v
    features["Summed Velocity"] = all_vel
    
    # 5. Center body position (mean of body points)
    body_bps = [bp for bp in bp_list if bp.startswith("B") or bp.startswith("S")]
    if body_bps:
        center = np.mean([coords[bp] for bp in body_bps], axis=0)
        features["Center Body X"] = center[:, 0]
        features["Center Body Y"] = center[:, 1]
        # Distances from each point to center
        for bp in bp_list:
            dist_to_center = np.sqrt(np.sum((coords[bp] - center)**2, axis=1))
            bp_nice = BODYPART_NAMES.get(bp, bp)
            features[f"{bp_nice} - Center Body Distance"] = dist_to_center
    
    # 6. Rolling statistics (variability)
    window = 15  # ~0.6s at 25fps
    for key in list(features.keys()):
        if "Distance" in key or "Velocity" in key:
            series = pd.Series(features[key])
            features[f"Mean {key}"] = series.rolling(window, center=True, min_periods=1).mean().values
            features[f"SD {key}"] = series.rolling(window, center=True, min_periods=1).std().values
    
    # 7. Tail variability (specific feature mentioned in spec)
    if "tail" in coords:
        tail_xy = coords["tail"]
        tail_vel = np.zeros(n_frames)
        tail_vel[1:] = np.sqrt(np.sum(np.diff(tail_xy, axis=0)**2, axis=1)) * fps
        features["Mean Tail Velocity"] = pd.Series(tail_vel).rolling(window, center=True, min_periods=1).mean().values
        features["Tail Variability"] = pd.Series(tail_vel).rolling(window, center=True, min_periods=1).std().values
        
        # Body 3 to Tail variability (mentioned in spec)
        for side in ["R", "L"]:
            b3 = f"B3{side}"
            if b3 in coords:
                dist = np.sqrt(np.sum((coords[b3] - coords["tail"])**2, axis=1))
                side_name = "Right" if side == "R" else "Left"
                features[f"{side_name} Body 3 - Tail Variability"] = pd.Series(dist).rolling(window, center=True, min_periods=1).std().values
    
    return pd.DataFrame(features)

def load_all_dlc_and_compute_features(dlc_dir: Path, results_pkl: Path, index_csv: Path):
    """
    Load all DLC CSVs, compute features, and merge with MoSeq cluster labels.
    """
    print(f"Loading MoSeq cluster results from {results_pkl}...")
    with open(results_pkl, "rb") as f:
        results_dict = pickle.load(f)
    
    index_df = pd.read_csv(index_csv)
    name_to_group = {_normalize_name(r["name"]): r["group"] for _, r in index_df.iterrows()}
    
    all_rows = []
    
    # Find DLC CSV files
    dlc_csvs = list(dlc_dir.glob("*DLC*.csv"))
    print(f"Found {len(dlc_csvs)} DLC CSV files.")
    
    for csv_path in dlc_csvs:
        # Extract recording name
        fname = csv_path.stem
        # Try to match with results_dict keys
        matched_key = None
        fname_norm = _normalize_name(fname)
        
        for key in results_dict.keys():
            if _normalize_name(key) == fname_norm or fname_norm in _normalize_name(key) or _normalize_name(key) in fname_norm:
                matched_key = key
                break
        
        if matched_key is None:
            # Try partial match on animal ID
            for key in results_dict.keys():
                key_norm = _normalize_name(key)
                # Extract animal number pattern
                animal_match = re.search(r"animal\s*(\d+)", fname_norm)
                session_match = re.search(r"_(\d+)dlc", fname_norm)
                if animal_match:
                    animal_id = animal_match.group(1)
                    if f"animal{animal_id}" in key_norm:
                        # Check session number if present
                        if session_match:
                            session_id = session_match.group(1)
                            if f"_{session_id}" in key_norm or f"_{session_id}." in str(key).lower():
                                matched_key = key
                                break
                        else:
                            matched_key = key
                            break
        
        if matched_key is None:
            continue
        
        # Get cluster labels (key is 'syllable' singular, not 'syllables')
        rec_data = results_dict[matched_key]
        if "syllable" in rec_data:
            labels = rec_data["syllable"]
        elif "syllables_reindexed" in rec_data:
            labels = rec_data["syllables_reindexed"]
        elif "syllables" in rec_data:
            labels = rec_data["syllables"]
        else:
            print(f"  No syllable data for {matched_key}, keys: {list(rec_data.keys())}")
            continue
        
        # Map to behavior names (values 1-7 map to behaviors, others to Unassigned)
        behavior_labels = []
        for s in labels:
            if np.isnan(s) if isinstance(s, float) else False:
                behavior_labels.append("Unassigned")
            else:
                behavior_labels.append(BEHAVIOR_MAPPING.get(int(s), "Unassigned"))
        
        # Load DLC and compute features
        try:
            pose_df = load_dlc_csv(csv_path)
        except Exception as e:
            print(f"  Error loading {csv_path.name}: {e}")
            continue
        
        # Ensure same length
        min_len = min(len(pose_df), len(behavior_labels))
        pose_df = pose_df.iloc[:min_len]
        behavior_labels = behavior_labels[:min_len]
        
        # Compute features
        feat_df = compute_kinematic_features(pose_df, fps=fps)
        feat_df = feat_df.iloc[:min_len]
        feat_df[label_col] = behavior_labels
        feat_df["recording"] = matched_key
        
        # Get group
        for name_key, grp in name_to_group.items():
            if name_key in _normalize_name(matched_key):
                feat_df["group"] = grp
                break
        
        all_rows.append(feat_df)
        print(f"  Processed {csv_path.name} -> {matched_key} ({len(feat_df)} frames)")
    
    if not all_rows:
        return None
    
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total: {len(combined):,} frames with {combined.shape[1]-3} features.")
    return combined

# ==============================================================================
# STYLING FUNCTIONS
# ==============================================================================

def _style_axes(ax):
    """Minimal axis styling: left/bottom spines gray, no top/right."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#4b4b4b", width=0.6, length=3, labelsize=8)

def _panel_tag(ax, tag):
    """Add panel letter near upper-left."""
    ax.text(-0.12, 1.05, tag, transform=ax.transAxes, fontsize=16,
            fontweight="bold", color="#4b4b4b", va="top", ha="right")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

#%% Load or compute features
print("=" * 60)
print("FIGURE 9: SHAP-based Explainability")
print("=" * 60)

# Try to find DLC CSVs directory (prefer config, then fallbacks)
dlc_candidates = [
    DLC_DIR,
    DATA_DIR / "CSVs_all",
    Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\equipo_project_data\CSVs_all"),
    Path(r"C:\Users\admin\keypoint_moseq\keypoint_moseq_project\code\equipo_project\2025_01_20-23_00_27\results"),
]

dlc_dir = None
for candidate in dlc_candidates:
    if candidate.exists():
        dlc_dir = candidate
        break

if dlc_dir is None:
    print("ERROR: Could not find DLC CSV directory. Please set the path.")
    sys.exit(1)

print(f"Using DLC directory: {dlc_dir}")

# Load and compute features
df = load_all_dlc_and_compute_features(dlc_dir, RESULTS_CLUSTERS_PKL, INDEX_CSV)

if df is None or len(df) == 0:
    print("No data loaded. Exiting.")
    sys.exit(1)

# ==============================================================================
# QUICK MODE: Subsample data for fast testing (stratified by class)
# ==============================================================================
if QUICK_MODE:
    print("\n QUICK MODE ENABLED - Using minimal data for fast testing...")
    max_samples_per_class = 250  # Limit samples per class
    # Stratified sampling to ensure all classes are represented
    sampled_dfs = []
    for label in df[label_col].unique():
        class_df = df[df[label_col] == label]
        n_samples = min(len(class_df), max_samples_per_class)
        if n_samples > 0:
            sampled_dfs.append(class_df.sample(n=n_samples, random_state=42))
    df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"   Stratified subsampled to {len(df)} frames ({max_samples_per_class} max per class)")

# Clean up feature columns
feature_cols = [c for c in df.columns if c not in [label_col, "recording", "group"]]
X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X = X_df.to_numpy(dtype=np.float32)

# Handle infinite values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Labels
le = LabelEncoder()
y = le.fit_transform(df[label_col])
classes = le.inverse_transform(np.arange(len(le.classes_)))
num_classes = len(classes)
chance_level = 1.0 / num_classes

print(f"Classes: {list(classes)}")
print(f"Chance level: {chance_level:.3f}")

#%% Cross-validated accuracy and confusion matrix
# XGBoost parameters based on QUICK_MODE
if QUICK_MODE:
    xgb_params = dict(
        n_estimators=10,      # Minimal trees
        max_depth=3,          # Shallow trees
        learning_rate=0.3,    # Fast learning
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    n_cv_folds = 2  # Fewer folds
    shap_sample_size = 100  # Very small SHAP sample
    shap_compute_size = 50  # Even smaller for actual SHAP computation
else:
    xgb_params = dict(
        n_estimators=500,     # Full trees
        max_depth=6,          # Deeper trees
        learning_rate=0.05,   # Slower learning
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    n_cv_folds = 3  # Standard CV
    shap_sample_size = 1000  # Larger SHAP sample
    shap_compute_size = 500  # For actual SHAP computation

print(f"\nRunning {n_cv_folds}-fold cross-validation...")
skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
per_class_acc = {cls: [] for cls in classes}
cms = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model_cv = xgb.XGBClassifier(**xgb_params)
    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_val)
    
    for i, cls in enumerate(classes):
        mask = y_val == i
        if mask.any():
            per_class_acc[cls].append(accuracy_score(y_val[mask], y_pred[mask]))
    
    cm_norm = confusion_matrix(y_val, y_pred, labels=np.arange(num_classes), normalize="true")
    cms.append(cm_norm)
    print(f"  Fold {fold}: Overall accuracy = {accuracy_score(y_val, y_pred):.3f}")

acc_mean = np.array([np.mean(per_class_acc[cls]) if per_class_acc[cls] else 0 for cls in classes])
acc_sem = np.array([np.std(per_class_acc[cls], ddof=1) / np.sqrt(len(per_class_acc[cls])) 
                    if len(per_class_acc[cls]) > 1 else 0 for cls in classes])
cm_mean = np.mean(cms, axis=0)

# Overall accuracy
overall_acc = np.mean([accuracy_score(y[val_idx], model_cv.predict(X[val_idx])) 
                       for _, val_idx in skf.split(X, y)])

#%% Train final model for SHAP
print("\nTraining final XGBoost model...")
final_model = xgb.XGBClassifier(**xgb_params, use_label_encoder=False)
final_model.fit(X, y)

#%% SHAP computation
print("\nComputing SHAP values (sampling for speed)...")
sample_size = min(shap_sample_size, len(X))
sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
X_sample = X[sample_idx]

# Use shap.Explainer for better compatibility with multi-class XGBoost
# TreeExplainer has compatibility issues with recent XGBoost multi-class models
# So we use KernelExplainer with predict_proba which is more robust
n_background = 20 if QUICK_MODE else 50
print(f"  Using KernelExplainer for multi-class compatibility (background={n_background})...")
background = shap.kmeans(X_sample, n_background)  # Summarize background data
explainer = shap.KernelExplainer(final_model.predict_proba, background)

#%% Compute SHAP values
compute_size = min(shap_compute_size, len(X_sample))
print(f"  Computing SHAP for {compute_size} samples...")
shap_values = explainer.shap_values(X_sample[:compute_size])
X_sample = X_sample[:compute_size]  # Match the sample size used for SHAP

if isinstance(shap_values, list):
    # List of (n_samples, n_features) arrays, one per class
    shap_array = np.stack(shap_values)  # (n_classes, n_samples, n_features)
elif shap_values.ndim == 3:
    # Shape is (n_samples, n_features, n_classes) - transpose to (n_classes, n_samples, n_features)
    shap_array = np.transpose(shap_values, (2, 0, 1))
else:
    # Binary case: (n_samples, n_features)
    shap_array = np.expand_dims(shap_values, axis=0)

print(f"  SHAP array shape: {shap_array.shape}")  # Should be (n_classes, n_samples, n_features)

feature_names = X_df.columns.tolist()

#%%
# Global SHAP importance (stacked by class)
mean_abs_shap = np.mean(np.abs(shap_array), axis=1)  # (n_classes, n_features)
mean_abs_overall = mean_abs_shap.mean(axis=0)  # (n_features,)
print(f"  mean_abs_overall shape: {mean_abs_overall.shape}")
top20_idx = [int(i) for i in np.argsort(mean_abs_overall)[::-1][:20]]  # Explicit int conversion

# ==============================================================================
# FIGURE 9 - SINGLE COMPLETE FIGURE
# ==============================================================================
print("\n" + "=" * 60)
print("Rendering Figure 9 (Panels A-K)...")
print("=" * 60)

sns.set_theme(style="ticks", context="paper", font_scale=1.1)

# Single large figure: 4 rows x 3 cols
# Row 0: A (accuracy), B (SHAP importance, spans 2 cols)
# Row 1: C (confusion matrix), D (Freezing), E (Sniffing)
# Row 2: F (Grooming), G (Turn), H (Locomotion)
# Row 3: I (Climbing), J (Jump), K (Unassigned)

fig = plt.figure(figsize=(22, 24), dpi=150, facecolor="white")
gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.2, 1.2, 1.2], 
              hspace=0.35, wspace=0.50,
              left=0.08, right=0.95, top=0.97, bottom=0.04)

# --------------------------------------------------------------------------
# Panel A: Accuracy vs Chance (horizontal bar chart)
# --------------------------------------------------------------------------
axA = fig.add_subplot(gs[0, 0])

# Order: behaviors top to bottom, Overall at BOTTOM
behavior_order = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump", "Unassigned"]
display_classes = [c for c in behavior_order if c in classes]
n_classes = len(display_classes)
y_positions = np.arange(n_classes + 1)  # 0=Overall at bottom, 1..n=behaviors

# Plot behavior bars (positions 1 to n, top to bottom visually)
for i, cls in enumerate(display_classes):
    cls_idx = list(classes).index(cls)
    acc = acc_mean[cls_idx]
    color = BEHAVIOR_PALETTE.get(cls, "#888888")
    ypos = n_classes - i  # Top behavior at highest y
    axA.barh(ypos, acc, color=color, edgecolor="white", linewidth=0.5, height=0.7)
    
    # Annotation: accuracy (+delta, fold)
    delta = acc - chance_level
    fold = acc / chance_level if chance_level > 0 else 0
    axA.text(acc + 0.02, ypos, f"{acc:.2f} (+{delta:.2f}, {fold:.1f}×)", 
             va="center", ha="left", fontsize=8, color="#4b4b4b")

# Overall bar at bottom (y=0) with smooth continuous gradient
overall_ypos = 0
# Create smooth gradient using imshow
from matplotlib.colors import LinearSegmentedColormap
gradient_colors = [BEHAVIOR_PALETTE.get(cls, "#888888") for cls in display_classes]
cmap_gradient = LinearSegmentedColormap.from_list("behavior_gradient", gradient_colors, N=256)
# Draw gradient bar using imshow
gradient_data = np.linspace(0, 1, 256).reshape(1, -1)
axA.imshow(gradient_data, aspect='auto', cmap=cmap_gradient, 
           extent=[0, overall_acc, overall_ypos - 0.35, overall_ypos + 0.35], zorder=2)
axA.text(overall_acc + 0.02, overall_ypos, f"{overall_acc:.2f}", 
         va="center", ha="left", fontsize=8, fontweight="bold", color="#4b4b4b")

# Chance line
axA.axvline(chance_level, color="gray", linestyle="--", linewidth=1.2, zorder=5)
axA.text(chance_level + 0.01, -0.7, f"Chance = {chance_level:.2f}", 
         fontsize=8, color="gray", va="top")

# Axis setup
axA.set_yticks([0] + list(range(n_classes, 0, -1)))
axA.set_yticklabels(["Overall"] + display_classes, fontsize=9)
axA.set_xlim(0, 1.05)
axA.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_xlabel("Accuracy", fontsize=11)
axA.set_title("Accuracy vs. Chance", fontsize=13, fontweight="bold", color="#4b4b4b")
_style_axes(axA)
_panel_tag(axA, "A")

# --------------------------------------------------------------------------
# Panel B: Global SHAP Feature Importance (stacked horizontal bars)
# --------------------------------------------------------------------------
axB = fig.add_subplot(gs[0, 1:])

top_features = [feature_names[i] for i in top20_idx]
y_pos_shap = np.arange(20)[::-1]

# Create stacked bars
for feat_idx, (global_idx, ypos) in enumerate(zip(top20_idx, y_pos_shap)):
    left = 0
    for cls_idx, cls in enumerate(classes):
        val = mean_abs_shap[cls_idx, global_idx]
        color = BEHAVIOR_PALETTE.get(cls, "#888888")
        axB.barh(ypos, val, left=left, color=color, edgecolor="none", height=0.7)
        left += val

# Truncate long feature names
def _truncate_name(name, max_len=40):
    return name[:max_len-3] + "..." if len(name) > max_len else name

axB.set_yticks(y_pos_shap)
axB.set_yticklabels([_truncate_name(f, 35) for f in top_features], fontsize=9)
axB.set_xlabel("Mean |SHAP value|", fontsize=11)
axB.set_title("SHAP Global Feature Importance", fontsize=13, fontweight="bold", color="#4b4b4b")
_style_axes(axB)
_panel_tag(axB, "B")

# Legend for behaviors
legend_patches = [mpatches.Patch(color=BEHAVIOR_PALETTE.get(cls, "#888888"), label=cls) 
                  for cls in behavior_order if cls in classes]
axB.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=7, ncol=2)

# --------------------------------------------------------------------------
# Panel C: Normalized Confusion Matrix
# --------------------------------------------------------------------------
axC = fig.add_subplot(gs[1, 0])

# Reorder confusion matrix to match behavior_order
cm_order_idx = [list(classes).index(c) for c in display_classes if c in classes]
cm_reordered = cm_mean[np.ix_(cm_order_idx, cm_order_idx)]

# Purple/mauve colormap
cmap_cm = mcolors.LinearSegmentedColormap.from_list("purples", ["#FFFFFF", "#E8D4E8", "#C37B9F", "#8B4B6B"])
im = axC.imshow(cm_reordered, cmap=cmap_cm, vmin=0, vmax=0.9, aspect="auto")

# Add text annotations
for i in range(len(cm_order_idx)):
    for j in range(len(cm_order_idx)):
        val = cm_reordered[i, j]
        text_color = "white" if val > 0.5 else "#4b4b4b"
        axC.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

axC.set_xticks(np.arange(len(cm_order_idx)))
axC.set_yticks(np.arange(len(cm_order_idx)))
axC.set_xticklabels([display_classes[i] for i in range(len(cm_order_idx))], rotation=45, ha="right", fontsize=9)
axC.set_yticklabels([display_classes[i] for i in range(len(cm_order_idx))], fontsize=9)
axC.set_xlabel("Predicted Label", fontsize=11)
axC.set_ylabel("True Label", fontsize=11)
axC.set_title("Normalized Confusion Matrix", fontsize=13, fontweight="bold", color="#4b4b4b")

# Colorbar
cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
_panel_tag(axC, "C")

# --------------------------------------------------------------------------
# SHAP beeswarm plot function (dots form violin shape via density-based jitter)
# --------------------------------------------------------------------------
def plot_shap_violin_dots(ax, class_name, panel_letter, shap_array, X_sample, feature_names, classes):
    """Plot SHAP beeswarm where dots form violin shape via density-based y-jitter."""
    from scipy.stats import gaussian_kde
    
    cls_idx = list(classes).index(class_name)
    class_shap = shap_array[cls_idx]  # (n_samples, n_features)
    
    # Top 10 features for this class
    mean_abs = np.mean(np.abs(class_shap), axis=0)
    top10_idx = [int(i) for i in np.argsort(mean_abs)[::-1][:10]]
    
    top10_shap = class_shap[:, top10_idx]
    top10_features = X_sample[:, top10_idx]
    top10_names = [feature_names[i] for i in top10_idx]
    
    # y positions (reversed so highest importance at top)
    y_positions = np.arange(10)[::-1]
    
    # Normalize feature values for coloring (0=low, 1=high)
    feat_normalized = np.zeros_like(top10_features)
    for j in range(10):
        fmin, fmax = top10_features[:, j].min(), top10_features[:, j].max()
        if fmax > fmin:
            feat_normalized[:, j] = (top10_features[:, j] - fmin) / (fmax - fmin)
        else:
            feat_normalized[:, j] = 0.5
    
    # Blue (low) to Red (high) colormap (like SHAP default)
    cmap_shap = plt.cm.coolwarm
    
    # Compute x-limits from all SHAP values for this class's top 10 features
    all_shap_min = top10_shap.min()
    all_shap_max = top10_shap.max()
    shap_range = all_shap_max - all_shap_min
    if shap_range == 0:
        shap_range = 0.1
    # Add 15% padding
    x_lim_min = all_shap_min - 0.15 * shap_range
    x_lim_max = all_shap_max + 0.15 * shap_range
    
    # Plot dots for each feature - density-based y-jitter creates violin shape
    for j, ypos in enumerate(y_positions):
        shap_vals = top10_shap[:, j]
        feat_vals = feat_normalized[:, j]
        
        # Subsample points for scatter
        n_plot = min(500, len(shap_vals))
        plot_idx = np.random.choice(len(shap_vals), n_plot, replace=False)
        shap_plot = shap_vals[plot_idx]
        feat_plot = feat_vals[plot_idx]
        
        # Compute density-based y-jitter (points spread according to local density)
        try:
            if len(np.unique(shap_plot)) > 1:
                kde = gaussian_kde(shap_plot, bw_method=0.5)
                density = kde(shap_plot)
                # Normalize density to [0, 1]
                density_norm = (density - density.min()) / (density.max() - density.min() + 1e-10)
                # Higher density = more spread in y
                max_jitter = 0.4
                y_jitter = ypos + (np.random.uniform(-1, 1, n_plot) * density_norm * max_jitter)
            else:
                y_jitter = ypos + np.random.uniform(-0.1, 0.1, n_plot)
        except:
            y_jitter = ypos + np.random.uniform(-0.2, 0.2, n_plot)
        
        # Sort by SHAP value so overlapping dots show correctly
        sort_idx = np.argsort(np.abs(shap_plot))
        
        colors = cmap_shap(feat_plot[sort_idx])
        ax.scatter(shap_plot[sort_idx], y_jitter[sort_idx], c=colors, s=5, alpha=0.8, edgecolors="none", zorder=3)
    
    # Zero reference line
    ax.axvline(0, color="#CCCCCC", linewidth=1, zorder=0)
    
    # Set x-axis limits based on the data for THIS class
    ax.set_xlim(x_lim_min, x_lim_max)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_truncate_name(n, 25) for n in top10_names], fontsize=8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title(f"{class_name}", fontsize=12, fontweight="bold", color=BEHAVIOR_PALETTE.get(class_name, "#4b4b4b"))
    _style_axes(ax)
    _panel_tag(ax, panel_letter)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_shap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=8)
    cbar.ax.set_ylabel("Feature value", fontsize=8, rotation=270, labelpad=12)

# --------------------------------------------------------------------------
# Panels D-K: SHAP violin+dots for each behavior
# --------------------------------------------------------------------------
all_classes = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump", "Unassigned"]
panel_letters = ["D", "E", "F", "G", "H", "I", "J", "K"]
# Positions: D,E in row 1 (cols 1,2); F,G,H in row 2; I,J,K in row 3
positions = [(1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]

for (row, col), cls, letter in zip(positions, all_classes, panel_letters):
    if cls in classes:
        ax = fig.add_subplot(gs[row, col])
        plot_shap_violin_dots(ax, cls, letter, shap_array, X_sample, feature_names, classes)

plt.tight_layout()

# Save single figure
fig.savefig(RESULTS_DIR / "figure_9_shap.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(RESULTS_DIR / "figure_9_shap.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved Figure 9 to {RESULTS_DIR}")

# Show only if not in batch mode
import os
if not os.environ.get("BATCH_MODE"):
    plt.show()

print("\n" + "=" * 60)
print("Figure 9 complete!")
print("=" * 60)

# %%
