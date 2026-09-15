import numpy as np
import pandas as pd

from freezing_dlc.predict import predict_freezing, save_frame_predictions


def test_save_frame_predictions_includes_freezing_and_moving_confidence(tmp_path):
    out_csv = tmp_path / "pred.csv"

    save_frame_predictions(out_csv, np.array([0.25, 0.8]), np.array([0, 1]))

    saved = pd.read_csv(out_csv)
    assert saved["freezing_confidence"].tolist() == [25.0, 80.0]
    assert saved["moving_confidence"].tolist() == [75.0, 20.0]



def test_predict_refuses_data_missing_the_models_features():
    """Zero-filling absent columns yields confident numbers from nothing."""
    import pytest

    from freezing_dlc.model import LoadedModel

    class _Always:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    model = LoadedModel(_Always(), ["velocity", "x_spread", "y_spread"], "compact", 0.5)
    features = pd.DataFrame({"velocity": [0.1, 0.2]})

    with pytest.raises(ValueError, match="missing 2 of the 3 features"):
        predict_freezing(features, model, threshold=0.5)


def test_predict_accepts_features_with_extra_columns():
    from freezing_dlc.model import LoadedModel

    class _Always:
        def predict_proba(self, X):
            assert X.shape[1] == 1
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    model = LoadedModel(_Always(), ["velocity"], "compact", 0.5)
    features = pd.DataFrame({"velocity": [0.1, 0.2], "unused": [9.0, 9.0]})

    probs, pred = predict_freezing(features, model, threshold=0.5)
    assert pred.tolist() == [1, 1]
