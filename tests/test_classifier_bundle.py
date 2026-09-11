"""The archived model must load after the src -> coping_dynamics package rename."""

from pathlib import Path

import numpy as np
import pandas as pd

from coping_dynamics.behavior_classifier.pipeline import load_bundle, predict_behaviors, save_bundle


def test_archived_bundle_loads_predicts_and_roundtrips_without_source_alias(tmp_path):
    import sys

    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root / "classifier/figure7_behavior_classifier.joblib")
    data = pd.read_csv(root / "data/raw/moseq_syllables_per_frame.csv.gz", nrows=100)
    prediction = predict_behaviors(data, bundle)
    assert len(prediction) == 100
    probabilities = prediction.filter(regex="^prob_").to_numpy()
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert "src" not in sys.modules
    saved = save_bundle(bundle, tmp_path / "current.joblib")
    restored = load_bundle(saved)
    pd.testing.assert_frame_equal(prediction, predict_behaviors(data, restored))
