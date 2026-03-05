"""Reusable XGBoost training/inference helpers for behavior classification."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def get_xgb_params(quick: bool = False) -> dict[str, Any]:
    """Return XGBoost hyperparameters matching the figure pipeline defaults."""
    if quick:
        return {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


def n_cv_folds(quick: bool = False) -> int:
    """Return default number of CV folds for the selected mode."""
    return 2 if quick else 3


def build_feature_matrix(
    df: pd.DataFrame,
    label_col: str = "behavior_cluster",
    non_feature_cols: tuple[str, ...] = ("recording", "group"),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert frame-level table into numeric feature DataFrame + float32 matrix."""
    excluded = {label_col, *non_feature_cols}
    feature_cols = [c for c in df.columns if c not in excluded]
    x_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = x_df.to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x_df, x


def stratified_subsample_by_label(
    df: pd.DataFrame,
    label_col: str,
    max_samples_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample up to max_samples_per_class rows for each class label."""
    sampled = []
    for label in sorted(df[label_col].astype(str).unique()):
        class_df = df[df[label_col].astype(str) == label]
        n_samples = min(len(class_df), max_samples_per_class)
        if n_samples > 0:
            sampled.append(class_df.sample(n=n_samples, random_state=random_state))
    if not sampled:
        return df.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True)


def encode_labels(labels: pd.Series | np.ndarray) -> tuple[np.ndarray, LabelEncoder, np.ndarray]:
    """Encode string labels to numeric ids for multi-class training."""
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(labels))
    classes = le.inverse_transform(np.arange(len(le.classes_)))
    return y, le, classes


def compute_cv_metrics(
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    xgb_params: dict[str, Any],
    folds: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute out-of-fold CV metrics and confusion matrices."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    per_class_acc: dict[str, list[float]] = {cls: [] for cls in classes}
    fold_overall = []
    fold_balanced = []
    cms = []
    n_classes = len(classes)
    y_oof_pred = np.zeros(len(y), dtype=int)
    y_oof_proba = np.zeros((len(y), n_classes), dtype=np.float32)

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_cv = xgb.XGBClassifier(**xgb_params)
        model_cv.fit(x_train, y_train)
        y_pred = model_cv.predict(x_val)
        y_proba = model_cv.predict_proba(x_val)

        y_oof_pred[val_idx] = y_pred
        y_oof_proba[val_idx] = y_proba

        fold_overall.append(float(accuracy_score(y_val, y_pred)))
        fold_balanced.append(float(balanced_accuracy_score(y_val, y_pred)))

        for i, cls in enumerate(classes):
            mask = y_val == i
            if mask.any():
                per_class_acc[str(cls)].append(float(accuracy_score(y_val[mask], y_pred[mask])))

        cm_norm = confusion_matrix(y_val, y_pred, labels=np.arange(len(classes)), normalize="true")
        cms.append(cm_norm)

    acc_mean = {}
    acc_sem = {}
    for cls in classes:
        vals = np.asarray(per_class_acc[str(cls)], dtype=float)
        acc_mean[str(cls)] = float(vals.mean()) if vals.size else 0.0
        if vals.size > 1:
            acc_sem[str(cls)] = float(vals.std(ddof=1) / np.sqrt(vals.size))
        else:
            acc_sem[str(cls)] = 0.0

    labels = np.arange(n_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        y_oof_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y,
        y_oof_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y,
        y_oof_pred,
        average="weighted",
        zero_division=0,
    )
    cm_counts = confusion_matrix(y, y_oof_pred, labels=labels)
    cm_norm = confusion_matrix(y, y_oof_pred, labels=labels, normalize="true")

    per_class_prf = {
        str(cls): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }

    metrics = {
        "fold_overall_accuracy": fold_overall,
        "fold_balanced_accuracy": fold_balanced,
        "overall_accuracy_mean": float(np.mean(fold_overall)) if fold_overall else 0.0,
        "overall_accuracy_oof": float(accuracy_score(y, y_oof_pred)),
        "balanced_accuracy_oof": float(balanced_accuracy_score(y, y_oof_pred)),
        "macro_precision_oof": float(macro_precision),
        "macro_recall_oof": float(macro_recall),
        "macro_f1_oof": float(macro_f1),
        "weighted_precision_oof": float(weighted_precision),
        "weighted_recall_oof": float(weighted_recall),
        "weighted_f1_oof": float(weighted_f1),
        "chance_level": float(1.0 / max(1, len(classes))),
        "per_class_accuracy_mean": acc_mean,
        "per_class_accuracy_sem": acc_sem,
        "per_class_prf_oof": per_class_prf,
        "confusion_matrix_mean": np.mean(cms, axis=0).tolist() if cms else [],
        "confusion_matrix_oof_counts": cm_counts.tolist(),
        "confusion_matrix_oof_normalized": cm_norm.tolist(),
        "class_names": [str(c) for c in classes],
        "oof_pred_sample_count": int(len(y_oof_pred)),
        "folds": int(folds),
        "random_state": int(random_state),
        "log_loss_oof": float(log_loss(y, y_oof_proba, labels=labels)),
    }
    return metrics


def train_xgb_classifier(
    x: np.ndarray,
    y: np.ndarray,
    xgb_params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    """Fit final XGB classifier on all samples."""
    params = xgb_params or get_xgb_params(quick=False)
    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(x, y)
    return model


def align_features_for_inference(frame_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reindex to training feature schema; missing cols -> 0.0, extra cols dropped."""
    numeric = frame_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    aligned = numeric.reindex(columns=feature_names, fill_value=0.0)
    aligned = aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return aligned


def predict_probabilities(
    model: xgb.XGBClassifier,
    frame_features: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Return per-class probabilities plus argmax class and confidence."""
    x = frame_features.to_numpy(dtype=np.float32)
    probas = model.predict_proba(x)
    if probas.ndim == 1:
        probas = probas[:, None]
    if probas.shape[1] != len(class_names):
        raise ValueError(
            f"Model returned {probas.shape[1]} classes but metadata has {len(class_names)} classes."
        )

    pred_idx = np.argmax(probas, axis=1)
    pred_class = [class_names[i] for i in pred_idx]
    pred_conf = probas[np.arange(len(probas)), pred_idx]

    out = pd.DataFrame(
        {
            "pred_class": pred_class,
            "pred_confidence": pred_conf.astype(float),
        }
    )
    for i, cls in enumerate(class_names):
        out[f"p_{cls}"] = probas[:, i].astype(float)
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_model_artifacts(
    model: xgb.XGBClassifier,
    model_dir: Path,
    metadata: dict[str, Any],
    cv_metrics: dict[str, Any] | None = None,
    save_pickle: bool = True,
) -> dict[str, Path]:
    """Persist model as JSON + pickle + metadata (+ optional metrics)."""
    model_dir.mkdir(parents=True, exist_ok=True)

    model_json = model_dir / "model.json"
    model.save_model(str(model_json))

    artifact_paths = {"model_json": model_json}

    if save_pickle:
        model_pkl = model_dir / "model.pkl"
        with open(model_pkl, "wb") as f:
            pickle.dump(model, f)
        artifact_paths["model_pkl"] = model_pkl

    meta = dict(metadata)
    meta.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())
    meta_path = model_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(meta), f, indent=2)
    artifact_paths["metadata"] = meta_path

    if cv_metrics is not None:
        metrics_path = model_dir / "cv_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(cv_metrics), f, indent=2)
        artifact_paths["cv_metrics"] = metrics_path

    return artifact_paths


def load_model_artifacts(model_dir: Path) -> tuple[xgb.XGBClassifier, dict[str, Any], str]:
    """Load metadata and model (JSON primary, pickle fallback)."""
    model_dir = Path(model_dir)
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_json = model_dir / "model.json"
    model_pkl = model_dir / "model.pkl"

    if model_json.exists():
        params = metadata.get("xgb_params", {})
        model = xgb.XGBClassifier(**params)
        model.load_model(str(model_json))
        return model, metadata, "json"

    if model_pkl.exists():
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        return model, metadata, "pickle"

    raise FileNotFoundError(
        f"No loadable model artifact found in {model_dir}. Expected model.json or model.pkl."
    )
