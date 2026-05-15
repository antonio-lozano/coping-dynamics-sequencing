"""Machine-learning utilities for behavior modeling.

Expose helpers lazily so importing one utility does not force all ML and
plotting dependencies to be installed.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "align_features_for_inference": (".behavior_xgb", "align_features_for_inference"),
    "build_feature_matrix": (".behavior_xgb", "build_feature_matrix"),
    "compute_cv_metrics": (".behavior_xgb", "compute_cv_metrics"),
    "encode_labels": (".behavior_xgb", "encode_labels"),
    "get_xgb_params": (".behavior_xgb", "get_xgb_params"),
    "load_model_artifacts": (".behavior_xgb", "load_model_artifacts"),
    "n_cv_folds": (".behavior_xgb", "n_cv_folds"),
    "predict_probabilities": (".behavior_xgb", "predict_probabilities"),
    "save_model_artifacts": (".behavior_xgb", "save_model_artifacts"),
    "stratified_subsample_by_label": (".behavior_xgb", "stratified_subsample_by_label"),
    "train_xgb_classifier": (".behavior_xgb", "train_xgb_classifier"),
    "BODYPART_NAMES": (".pose_features", "BODYPART_NAMES"),
    "compute_kinematic_features": (".pose_features", "compute_kinematic_features"),
    "load_all_dlc_and_compute_features": (".pose_features", "load_all_dlc_and_compute_features"),
    "load_dlc_csv": (".pose_features", "load_dlc_csv"),
    "normalize_recording_name": (".pose_features", "normalize_recording_name"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
