from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import LoadedModel


def predict_freezing(features: pd.DataFrame, model: LoadedModel, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in model.feature_names if name not in features.columns]
    if missing:
        # Reindexing fills absent columns with zeros, and a forest given zeros
        # still returns confident-looking probabilities. Left unchecked that
        # yields a full set of freezing percentages computed from nothing, so
        # anything beyond a stray column is treated as the wrong tracking data.
        raise ValueError(
            f"The tracking data is missing {len(missing)} of the "
            f"{len(model.feature_names)} features this model expects, including "
            f"{', '.join(missing[:5])}. This usually means the body parts differ "
            "from the ones the model was trained on, or the feature mode does not "
            "match. Retrain on your own tracking rather than predicting from it."
        )

    feat = features.reindex(columns=model.feature_names)
    probs = model.classifier.predict_proba(feat.to_numpy(dtype=float))[:, 1]
    pred = (probs >= threshold).astype(int)
    return probs, pred


def save_frame_predictions(out_csv: str | Path, probs: np.ndarray, pred: np.ndarray) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    freezing_confidence = np.clip(probs.astype(float), 0.0, 1.0) * 100.0
    moving_confidence = 100.0 - freezing_confidence
    pd.DataFrame(
        {
            "prob": probs,
            "pred": pred,
            "freezing_confidence": freezing_confidence,
            "moving_confidence": moving_confidence,
        }
    ).to_csv(out_csv, index=False)
