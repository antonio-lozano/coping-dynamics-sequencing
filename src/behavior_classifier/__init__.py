"""Behavior classifier used by Figure 7A-C and Supplementary Figure 5."""

from .annotations import BEHAVIOR_CLUSTER_MAP, behavior_lookup

__all__ = [
    "BEHAVIOR_CLUSTER_MAP",
    "BehaviorClassifierBundle",
    "behavior_lookup",
    "cross_validate",
    "load_bundle",
    "load_moseq_table",
    "predict_behaviors",
    "save_bundle",
    "train_behavior_classifier",
]


def __getattr__(name: str):
    if name in {
        "BehaviorClassifierBundle",
        "cross_validate",
        "load_bundle",
        "load_moseq_table",
        "predict_behaviors",
        "predict_from_dlc_file",
        "run_from_raw_video",
        "save_bundle",
        "train_behavior_classifier",
    }:
        if name in {"predict_from_dlc_file", "run_from_raw_video"}:
            from . import raw_pipeline

            return getattr(raw_pipeline, name)
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
