"""Train and apply the experimental centroid-feature behavior classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotations import BEHAVIOR_CLUSTER_MAP, BEHAVIOR_ORDER, behavior_lookup
from .features import add_behavior_labels, build_frame_features, require_columns


@dataclass
class BehaviorClassifierBundle:
    model: Any
    label_encoder: Any
    feature_columns: list[str]
    behavior_cluster_map: dict[str, list[int]]
    metadata: dict[str, Any]


def load_moseq_table(path: str | Path) -> pd.DataFrame:
    """Load a MoSeq per-frame table."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _make_model(
    *,
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    deps = _classifier_dependencies()
    return deps["XGBClassifier"](
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=n_jobs,
        random_state=random_state,
        eval_metric="mlogloss",
    )


def _classifier_dependencies() -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Behavior classifier dependencies are missing. Install the updated "
            "requirements first, for example: uv pip install -r requirements.txt"
        ) from exc

    return {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "StratifiedKFold": StratifiedKFold,
        "LabelEncoder": LabelEncoder,
        "compute_sample_weight": compute_sample_weight,
        "XGBClassifier": XGBClassifier,
    }


def _prepare_training_data(
    df: pd.DataFrame,
    *,
    include_unassigned: bool,
    max_frames: int | None = None,
    random_state: int = 42,
    label_col: str = "behavior_label",
) -> tuple[pd.DataFrame, list[str], np.ndarray, Any]:
    deps = _classifier_dependencies()
    lookup = behavior_lookup(BEHAVIOR_CLUSTER_MAP)
    labeled = add_behavior_labels(df, lookup, label_col=label_col)
    # Temporal features describe the original recording, not adjacency among
    # sampled/labeled training rows. Compute before either exclusion or random
    # subsampling, just as predict_behaviors computes on a complete recording.
    # This corrects experimental retraining only; archived manuscript models
    # and classifier results are not regenerated or claimed to be reproduced.
    featured, feature_cols = build_frame_features(labeled)
    if not include_unassigned:
        featured = featured[featured[label_col] != "Unassigned"].copy()
    featured = _sample_labeled_frames(
        featured,
        label_col=label_col,
        max_frames=max_frames,
        random_state=random_state,
    )

    require_columns(featured, [label_col])
    featured = (
        featured[featured[label_col].notna()]
        .sort_values(["name", "frame_index"])
        .reset_index(drop=True)
    )

    label_order = [label for label in BEHAVIOR_ORDER if label in set(featured[label_col])]
    label_encoder = deps["LabelEncoder"]()
    label_encoder.fit(label_order)
    y = label_encoder.transform(featured[label_col].to_numpy())
    return featured, feature_cols, y, label_encoder


def _sample_labeled_frames(
    df: pd.DataFrame,
    *,
    label_col: str,
    max_frames: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_frames is None or max_frames <= 0 or len(df) <= max_frames:
        return df

    counts = df[label_col].value_counts()
    target_counts = np.maximum(
        1,
        np.floor(counts / counts.sum() * max_frames).astype(int),
    )
    while target_counts.sum() > max_frames:
        largest = target_counts.idxmax()
        if target_counts[largest] <= 1:
            break
        target_counts[largest] -= 1
    while target_counts.sum() < max_frames:
        room = counts - target_counts
        room = room[room > 0]
        if room.empty:
            break
        target_counts[room.idxmax()] += 1

    sampled = []
    for label, target in target_counts.items():
        group = df[df[label_col] == label]
        sampled.append(group.sample(n=min(int(target), len(group)), random_state=random_state))
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=random_state)


def cross_validate(
    df: pd.DataFrame,
    *,
    include_unassigned: bool = True,
    balanced_weights: bool = True,
    n_splits: int = 3,
    random_state: int = 42,
    max_frames: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified CV and return summary and confusion-matrix tables."""

    deps = _classifier_dependencies()
    featured, feature_cols, y, label_encoder = _prepare_training_data(
        df,
        include_unassigned=include_unassigned,
        max_frames=max_frames,
        random_state=random_state,
    )
    X = featured[feature_cols].to_numpy()
    cv = deps["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = np.zeros_like(y)
    model_kwargs = model_kwargs or {}

    for train_idx, test_idx in cv.split(X, y):
        model = _make_model(random_state=random_state, **model_kwargs)
        sample_weight = (
            deps["compute_sample_weight"]("balanced", y[train_idx]) if balanced_weights else None
        )
        model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight)
        pred[test_idx] = model.predict(X[test_idx])

    labels = list(label_encoder.classes_)
    cm = deps["confusion_matrix"](y, pred, labels=np.arange(len(labels)))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    support = cm.sum(axis=1)

    summary_rows = [
        {
            "metric": "overall_accuracy",
            "value": deps["accuracy_score"](y, pred),
            "n_frames": int(len(y)),
            "n_classes": int(len(labels)),
        },
        {
            "metric": "balanced_accuracy",
            "value": deps["balanced_accuracy_score"](y, pred),
            "n_frames": int(len(y)),
            "n_classes": int(len(labels)),
        },
    ]
    summary_rows.extend(
        {
            "metric": f"recall_{label}",
            "value": float(score),
            "n_frames": int(n),
            "n_classes": int(len(labels)),
        }
        for label, score, n in zip(labels, recall, support)
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_behavior"
    cm_df = cm_df.reset_index()
    return pd.DataFrame(summary_rows), cm_df


def train_behavior_classifier(
    df: pd.DataFrame,
    *,
    include_unassigned: bool = True,
    balanced_weights: bool = True,
    random_state: int = 42,
    max_frames: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> BehaviorClassifierBundle:
    """Fit the final classifier on all annotated frames."""

    deps = _classifier_dependencies()
    featured, feature_cols, y, label_encoder = _prepare_training_data(
        df,
        include_unassigned=include_unassigned,
        max_frames=max_frames,
        random_state=random_state,
    )
    X = featured[feature_cols].to_numpy()
    model_kwargs = model_kwargs or {}
    model = _make_model(random_state=random_state, **model_kwargs)
    sample_weight = deps["compute_sample_weight"]("balanced", y) if balanced_weights else None
    model.fit(X, y, sample_weight=sample_weight)

    metadata = {
        "n_frames": int(len(y)),
        "n_classes": int(len(label_encoder.classes_)),
        "classes": list(label_encoder.classes_),
        "include_unassigned": bool(include_unassigned),
        "balanced_weights": bool(balanced_weights),
        "random_state": int(random_state),
        "max_frames": max_frames,
        "temporal_feature_context": "full_recording_before_training_row_selection",
        "model_kwargs": model_kwargs,
    }
    return BehaviorClassifierBundle(
        model=model,
        label_encoder=label_encoder,
        feature_columns=feature_cols,
        behavior_cluster_map=BEHAVIOR_CLUSTER_MAP,
        metadata=metadata,
    )


def save_bundle(bundle: BehaviorClassifierBundle, path: str | Path) -> Path:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "joblib is required to save classifier bundles. Install the updated "
            "requirements first, for example: uv pip install -r requirements.txt"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> BehaviorClassifierBundle:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "joblib is required to load classifier bundles. Install the updated "
            "requirements first, for example: uv pip install -r requirements.txt"
        ) from exc

    try:
        bundle = joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name not in {"src", "src.behavior_classifier", "src.behavior_classifier.pipeline"}:
            raise
        # The trusted bundled, uncompressed archive predates the package rename.
        # Remap only its exact class reference, without installing a global
        # ``src`` alias or rewriting any model bytes. Other globals are resolved
        # normally; this is compatibility, not a safe unpickler for untrusted data.
        import inspect

        from joblib.numpy_pickle import NumpyUnpickler

        class LegacyBundleUnpickler(NumpyUnpickler):
            def find_class(self, module, name):
                if (module, name) == (
                    "src.behavior_classifier.pipeline",
                    "BehaviorClassifierBundle",
                ):
                    return BehaviorClassifierBundle
                return super().find_class(module, name)

        kwargs = {}
        if "ensure_native_byte_order" in inspect.signature(NumpyUnpickler).parameters:
            kwargs["ensure_native_byte_order"] = True
        with Path(path).open("rb") as handle:
            bundle = LegacyBundleUnpickler(str(path), handle, **kwargs).load()
    if not isinstance(bundle, BehaviorClassifierBundle):
        raise TypeError(f"Unexpected classifier bundle type: {type(bundle)!r}")
    return bundle


def predict_behaviors(
    df: pd.DataFrame,
    bundle: BehaviorClassifierBundle,
    *,
    id_columns: tuple[str, ...] = ("name", "frame_index"),
) -> pd.DataFrame:
    """Predict behavior labels for a per-frame MoSeq table."""

    featured, _ = build_frame_features(df)
    X = featured[bundle.feature_columns].to_numpy()
    pred = bundle.model.predict(X)
    labels = bundle.label_encoder.inverse_transform(pred.astype(int))

    out_cols = [column for column in id_columns if column in featured.columns]
    out = featured[out_cols].copy()
    out["predicted_behavior"] = labels

    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(X)
        out["predicted_probability"] = proba.max(axis=1)
        for idx, label in enumerate(bundle.label_encoder.classes_):
            out[f"prob_{label}"] = proba[:, idx]

    return out
