from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import joblib


@dataclass(frozen=True)
class LoadedModel:
    classifier: Any
    feature_names: list[str]
    feature_mode: str
    threshold: Optional[float] = None


def model_from_payload(obj: Any) -> LoadedModel:
    """A saved payload as a LoadedModel, deriving the fields older files omit.

    The one place that knows those derivations; the evaluator used to carry a
    copy and the two had already started to drift.
    """
    if isinstance(obj, dict) and "classifier" in obj:
        feature_names = list(obj.get("feature_names", ["velocity"]))
        feature_mode = str(obj.get("feature_mode", "")).strip().lower()
        if not feature_mode:
            feature_mode = "legacy4k" if len(feature_names) >= 1000 else "compact"
        stored_threshold = obj.get("threshold")
        threshold = float(stored_threshold) if stored_threshold is not None else None
        return LoadedModel(obj["classifier"], feature_names, feature_mode, threshold)
    if hasattr(obj, "predict_proba"):
        return LoadedModel(obj, ["velocity"], "compact", None)
    raise TypeError("Model must be dict(classifier, feature_names) or estimator with predict_proba")


def load_model(model_path: Union[str, Path]) -> LoadedModel:
    # The shipped models were written by scikit-learn 1.7, so newer releases warn
    # on every tree in the forest. Predictions were checked against 1.6.1, 1.7.2
    # and 1.8.0 and matched to the bit, and the `ml` extra caps the version at
    # the tested range, so that warning is noise rather than a signal here — but
    # only that warning; anything else a model file says while loading is real.
    with warnings.catch_warnings():
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.simplefilter("ignore", InconsistentVersionWarning)
        except ImportError:  # pragma: no cover - sklearn always present with models
            warnings.simplefilter("ignore")
        obj: Any = joblib.load(model_path)
    return model_from_payload(obj)


def resolve_threshold(model: LoadedModel, requested: Optional[float], fallback: float) -> float:
    """Pick the decision threshold: an explicit request wins, then the value stored
    in the model file, then the caller's fallback."""
    if requested is not None:
        return float(requested)
    if model.threshold is not None:
        return float(model.threshold)
    return float(fallback)
