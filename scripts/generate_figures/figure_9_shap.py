"""
Figure 9 — SHAP-based explainability for kinematic feature contributions to
unsupervised behavior classification.

Pipeline:
1) Load kinematic features with MoSeq-derived behavior_cluster labels.
2) Train XGBoost multi-class classifier.
3) Report per-class accuracy (mean +/- SEM), chance line.
4) Plot normalized confusion matrix.
5) Compute SHAP values; plot global top-20 feature importances color-coded by
   the behavior they most strongly predict.
6) Plot class-specific SHAP violins (top 10 features) for each behavior.
"""

#%% Imports and paths
print("Loading dependencies and setting paths...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
import shap

from src.config import POSE_FEATURES_PARQUET

#%% Paths
features_path = POSE_FEATURES_PARQUET
label_col = "behavior_cluster"

#%% Load data
print("Loading pose-derived feature table...")
df = pd.read_parquet(features_path, engine="fastparquet")
print(f"Loaded {len(df):,} rows, {df.shape[1]-1} feature columns.")

# Ensure numeric feature matrix
feature_cols = [c for c in df.columns if c != label_col]
X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
if X_df.isna().any().any():
    X_df = X_df.fillna(0)
X = X_df.to_numpy(dtype=np.float32)

# Labels
le = LabelEncoder()
y = le.fit_transform(df[label_col])
classes = le.inverse_transform(np.arange(len(le.classes_)))
num_classes = len(classes)
chance_level = 1.0 / num_classes

#%% Cross-validated accuracy and confusion matrix
print("Running cross-validated accuracy and confusion matrices...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
per_class_acc = {cls: [] for cls in classes}
cms = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model_cv = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=0,
    )
    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_val)
    for i, cls in enumerate(classes):
        mask = y_val == i
        if mask.any():
            per_class_acc[cls].append(accuracy_score(y_val[mask], y_pred[mask]))
    cm_norm = confusion_matrix(y_val, y_pred, labels=np.arange(num_classes), normalize="true")
    cms.append(cm_norm)

acc_mean = np.array([np.mean(per_class_acc[cls]) for cls in classes])
acc_sem = np.array([np.std(per_class_acc[cls], ddof=1) / np.sqrt(len(per_class_acc[cls])) for cls in classes])
cm_mean = np.mean(cms, axis=0)

#%% Train final model on full data (for SHAP and plots)
print("Training final XGBoost model on full dataset...")
final_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=0,
)
final_model.fit(X, y)

# Sanitize base_score attribute for SHAP (workaround for string formatting)
booster = final_model.get_booster()
bs_attr = booster.attr("base_score")
if bs_attr is not None:
    try:
        clean_bs = str(float(bs_attr.strip("[]")))
        booster.set_attr(base_score=clean_bs)
    except ValueError:
        pass

#%% Accuracy bar + confusion matrix
print("Plotting accuracy bar and confusion matrix...")
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette = sns.color_palette("husl", n_colors=num_classes)

# Panel A: per-class accuracy (mean +/- SEM)
ax_acc = axes[0]
ax_acc.bar(classes, acc_mean, yerr=acc_sem, color=palette, edgecolor="black", alpha=0.85, capsize=4)
ax_acc.axhline(chance_level, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance_level:.2f})")
ax_acc.set_ylim(0, 1.05)
ax_acc.set_ylabel("Accuracy (mean +/- SEM)")
ax_acc.set_title("A. Per-class accuracy (5-fold CV)")
ax_acc.tick_params(axis="x", rotation=45)
ax_acc.legend(frameon=False)

# Panel C: normalized confusion matrix
ax_cm = axes[1]
im = ax_cm.imshow(cm_mean, cmap="viridis", vmin=0, vmax=1)
ax_cm.set_xticks(np.arange(num_classes))
ax_cm.set_yticks(np.arange(num_classes))
ax_cm.set_xticklabels(classes, rotation=90)
ax_cm.set_yticklabels(classes)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
ax_cm.set_title("C. Normalized confusion matrix (mean CV)")
fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

#%% SHAP computation on a subset for speed
print("Computing SHAP values on a validation subset...")
sample_idx = np.random.choice(len(X), size=min(4000, len(X)), replace=False)
X_sample = X[sample_idx]

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

# For multi-class, shap_values is list[num_classes] each (n_samples, n_features)
if isinstance(shap_values, list):
    shap_array = np.stack(shap_values)  # shape (num_classes, n_samples, n_features)
else:
    shap_array = np.expand_dims(shap_values, axis=0)

feature_names = X_df.columns.tolist()

#%% Global SHAP bar (top 20), color-coded by dominant class contribution
print("Plotting global SHAP bar (top 20)...")
mean_abs_shap = np.mean(np.abs(shap_array), axis=1)  # (num_classes, n_features)
mean_abs_overall = mean_abs_shap.sum(axis=0) / num_classes
top_idx = np.argsort(mean_abs_overall)[::-1][:20]
top_features = [feature_names[i] for i in top_idx]

dominant_class_idx = np.argmax(mean_abs_shap[:, top_idx], axis=0)
bar_colors = [palette[i] for i in dominant_class_idx]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(top_idx))[::-1], mean_abs_overall[top_idx][::-1], color=bar_colors[::-1], edgecolor="black", alpha=0.9)
ax.set_yticks(range(len(top_idx))[::-1])
ax.set_yticklabels(top_features[::-1])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("B. Global SHAP feature importances (top 20)\nColor = class with highest contribution")
plt.tight_layout()
plt.show()

#%% Class-specific SHAP violins (top 10 per class)
print("Plotting class-specific SHAP violins...")
for idx, cls in enumerate(classes):
    class_shap = shap_array[idx]  # (n_samples, n_features)
    mean_abs = np.mean(np.abs(class_shap), axis=0)
    top10_idx = np.argsort(mean_abs)[::-1][:10]
    shap.summary_plot(
        class_shap[:, top10_idx],
        features=X_sample[:, top10_idx],
        feature_names=[feature_names[i] for i in top10_idx],
        plot_type="violin",
        color=palette[idx],
        show=False,
        max_display=10,
    )
    plt.title(f"{chr(ord('D') + idx)}. SHAP violin: {cls}")
    plt.tight_layout()
    plt.show()

print("Figure 9 ready.")
