"""Machine-learning utilities for behavior modeling."""

from .behavior_xgb import (
    align_features_for_inference,
    build_feature_matrix,
    compute_cv_metrics,
    encode_labels,
    get_xgb_params,
    load_model_artifacts,
    n_cv_folds,
    predict_probabilities,
    save_model_artifacts,
    stratified_subsample_by_label,
    train_xgb_classifier,
)
from .pose_features import (
    BODYPART_NAMES,
    compute_kinematic_features,
    load_all_dlc_and_compute_features,
    load_dlc_csv,
    normalize_recording_name,
)

__all__ = [
    "BODYPART_NAMES",
    "align_features_for_inference",
    "build_feature_matrix",
    "compute_cv_metrics",
    "compute_kinematic_features",
    "encode_labels",
    "get_xgb_params",
    "load_all_dlc_and_compute_features",
    "load_dlc_csv",
    "load_model_artifacts",
    "n_cv_folds",
    "normalize_recording_name",
    "predict_probabilities",
    "save_model_artifacts",
    "stratified_subsample_by_label",
    "train_xgb_classifier",
]
